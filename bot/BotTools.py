import keyword
from typing import final
from langchain import Prompt
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# 工具
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import OpenAIEmbeddings
import requests
import json


with open('config.json', 'r') as f:
    config = json.load(f)

YUANFENJU_API_KEY = config["YUANFENJU_API_KEY"]

@tool
def test():
    """Test tool"""
    return "test"

@tool
def search(query:str):
    """只有需要了解实时信息或不知道的事情的时候才会使用这个工具"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    print("实时搜索结果：", result)
    return result

@tool
def get_info_from_local_db(query:str):
    """只有回答与2024年运势或者龙年运势相关的问题时，会使用这个工具。"""
    # 导入 Qdrant 客户端并初始化，指定数据存储的路径和模型
    client = Qdrant(
        QdrantClient(path="./local_qdrand"),
        "local_documents",
        # 向量引擎
        OpenAIEmbeddings()
    )
    # 将客户端配置为检索器，使用最大边际相关性（MMR）作为搜索类型
    retriever = client.as_retriever(search_type="mmr")
    # 使用检索器根据提供的查询（query）检索相关文档
    result = retriever.get_relevant_documents(query)
    return result

@tool
def bazi_cesuan(query: str):
    '''
        https://doc.yuanfenju.com/bazi/paipan.html
        $request_data = [
        'api_key' => 'FsF1CsVevk3N17w7oBkSydfSk',
        'name' => '张三',
        'sex' => '1',
        'type' => '1',
        'year' => '1988',
        'month' => '11',
        'day' => '8',
        'hours' => '12',
        'minute' => '20',
        'zhen' => '1',
        'province' => '北京市',
        'city' => '北京',
        ];
    '''
    print("====bazi_cesuan=====")
    url = f"https://api.yuanfenju.com/index.php/v1/Bazi/paipan"
    """只有做八字测算(排盘)的时候才会使用这个工具，需要输入用户姓名和出生年月日时，如果缺少用户姓名和出生年月日时，则不可用。"""
    prompt = ChatPromptTemplate.from_template(
        """你是一个参数查询助手，根据用户输入内容找出相关的参数并按json格式返回(用户不会输入api_key，使用下面的默认值)，JSON字段如下：
        - "api_key" : "uTsQLiVYd9EoBPQ0QEgCtdibw", - "name":"姓名", - "sex":"性别，0表示男，1表示女，根据姓名判断", 
        - "type":"日历类型 0农历 1公历,默认1", - "year":"出生年 例: 1988", - "month":"出生月 例: 8", - "day":"出生日 例: 7", - "hours":"出生时 例: 12", "minute": "0", 如果没有找到相关参数，则需要提醒用户告诉你这些内容，只返回数据结构，不要有其他的评论，用户输入：{query}
        """)
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | ChatOpenAI(temperature=0) | parser
    data = chain.invoke({"query": query})
    print("八字查询结果：", data)
    result = requests.post(url, data=data)
    if result.status_code == 200:
        print("====返回数据====")
        print(result.json())
        try: 
            json = result.json()
            print("json:", json)

            print("bazi:",json["data"]["bazi_info"]["bazi"])
            bazi_str = ' '.join(json["data"]["bazi_info"]["bazi"])
            final_str = "八字为：" + bazi_str
            print("final_str:", final_str)
            return final_str
        except Exception as e:
            return "八字查询失败，可能是你忘记询问用户姓名或者出生年月日时了。"
    else:
        return "技术错误，请用户稍后再试"
    
@tool
def yaoyigua():
    """只有用户想要摇卦占卜的时候才会使用这个工具"""
    '''
        $request_data = [
        'api_key' => 'FsF1CsVevk3N17w7oBkSydfSk',
        ];
    '''
    print("====yaoyigua=====")
    api_key = YUANFENJU_API_KEY
    url = f"https://api.yuanfenju.com/index.php/v1/Zhanbu/yaogua"
    try:
        result = requests.post(url, data={"api_key": api_key})
        if result.status_code == 200:
            print("====返回数据====")
            response_data = result.json()
            print(response_data)
            image = response_data["data"]["image"]
            print("封图片:", image)
            return response_data
        else:
            print("响应错误，状态码:", result.status_code)
            return "技术错误，请用户稍后再试"
    except requests.RequestException as e:
        print("请求失败:", e)
        return "请求异常，请检查网络连接"
    
@tool
def jiemeng(query:str):
    """只有用户想要解梦的时候才会使用这个工具,需要输入用户的梦境内容，如果缺少用户梦境内容，则不可用。"""
    '''
        $request_data = [
        'api_key' => 'FsF1CsVevk3N17w7oBkSydfSk',
        'title_zhougong' => '婴儿',
        ];
    '''
    api_key = YUANFENJU_API_KEY
    url = f"https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    LLM = OpenAI(temperature=0)
    prompt = PromptTemplate.from_template("根据内容提取1个关键词，只返回关键词，内容为:{topic}")
    prompt_value = prompt.invoke({"topic": query})
    keyword = LLM.invoke(prompt_value)
    print("提取的关键词：", keyword)
    result = requests.post(url, data={"api_key":api_key, "title_zhougong":keyword})
    if result.status_code == 200:
        print("====返回数据====")
        print(result.json())
        # 将这个JSON字符串解析成Python能够处理的数据结构（如字典）。
        final_str = json.loads(result.text)
        return final_str
    else:
        return "技术错误，请用户稍后再试"