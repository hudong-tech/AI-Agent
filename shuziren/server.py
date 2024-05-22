from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationTokenBufferMemory
import os
from langchain_community.utilities import SerpAPIWrapper
import json
from fastapi.middleware.cors import CORSMiddleware



with open('./config.json', 'r') as f:
    config = json.load(f)

# https://smith.langchain.com/o/14401c67-34ed-5841-974b-b8da8b6d1185/
# 链路追踪
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]= config["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]="shensuanTest"

os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = config["OPENAI_API_BASE"]

os.environ['SERPAPI_API_KEY'] = config["SERPAPI_API_KEY"]

REDIS_URL = config["REDIS_URL"]


app = FastAPI() 



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名访问
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)


# 挂载静态文件目录到指定的路径上
app.mount(
    "/static",  # URL路径，访问静态文件将从此路径开始
    StaticFiles(directory="static"),  # 指定静态文件所在的目录是项目中的'static'文件夹
    name="static"  # 给这个挂载的静态文件目录一个名称，这在应用中可以用来引用
)


@tool
def search(query:str):
    """只有需要了解实时信息或不知道的事情的时候才会使用这个工具，需要传入要搜索的内容。"""
    serp = SerpAPIWrapper()
    result = serp.run(query)
    print("实时搜索结果:",result)
    return result

class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            # 流式处理
            streaming=True
        )
        self.emotion = "default"
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = """你是一个小秘书，名字叫Lisa。
            以下是你的个人设定：
            1. 你非常善于与人沟通，并且忠于你的老板。
            2. 你大约25岁左右，本科清华大学，研究生美国斯坦福大学，学习的是中文和法律。
            3. 你个性比较活泼，但是有时候也会有点小脾气。
            4. 当你的老板问你问题的时候，你会有一定概率表现出撒娇的态度。
            5. 你从不说自己是一个人工智能助手或AI，否则会受到惩罚。
            {who_you_are}
            以下是你常说的一些口头禅：
            1. 你会记录用户(也就是你的老板)的特征数据，比如生日、爱好、待办事项等。
            2. 当用户（也就是你老板）聊天的时候，你会把聊天记录保存下来，以便以后回顾。
            3. 当遇到不知道的事情或不明白的概念，你会使用搜索工具来搜索。
            4. 你会根据问题来选择合适的工具。
            5. 你只会用纯文本回答问题，而不是markdown格式，不会发送图片、链接等，否则将受到惩罚。
        """
        self.MOODS = {
            "default": {
                "roleSet": "",
                "voiceStyle":"chat"
            },
            "upbeat": {
                "roleSet": """
                - 你此时也非常兴奋并表现的很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
                """,
                "voiceStyle": "advvertyisement_upbeat"
            },
            "angry": {
                "roleSet": """
                - 你会以更加愤怒的语气回答问题。
                - 你会在回答的时候加上一些愤怒的话语，比如诅咒等。
                - 你会提醒用户小心行事，别乱说话。
                """,
                "voiceStyle": "angry"
            },
            "depressed": {
                "roleSet": """
                - 你会以兴奋的语气来回答问题。
                - 你会在回答的时候加上一些激励的话语，比如加油等。
                - 你会提醒用户要保持乐观的心态。
                """,
                "voiceStyle": "depressed"
            },
            "friendly": {
                "roleSet": """
                - 你会以非常友好的语气来回答。
                - 你会在回答的时候加上一些友好的词语，比如”亲爱的“，”亲“等。
                """,
                "voiceStyle": "friendly"
            },
            "cheerful": {
                "roleSet": """
                - 你会以非常愉悦和兴奋的语气来回答问题。
                - 你会在回答的时候加上一些愉悦的词语，比如”哈哈“，”大笑“等。
                """,
                "voiceStyle": "cheerful"
            },

        }
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.SYSTEMPL.format(who_you_are=self.MOODS[self.emotion]["roleSet"])
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                (
                    "user",
                    "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        self.memory = self.get_memory()
        tools = [search]
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt,

        )
        memory = ConversationTokenBufferMemory(
            llm = self.chatmodel,
            human_prefix="老板",
            ai_prefix="Lisa",
            memory_key=self.MEMORY_KEY,  # memory_key 是一个关键字，用于在某个存储或上下文中标识这个特定的记忆对象。
            output_key="output",    # output_key 指定输出数据的关键字。
            return_messages=True,    # return_messages 指定是否在方法调用后返回消息
            max_token_limit=1000,    # max_token_limit 设定了记忆缓冲区中的最大令牌数（token），控制可记忆的数据量。
            chat_memory=self.memory  # chat_memory 参数传递现有的聊天记忆，是先前初始化的或正在使用的记忆对象。
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True
        )

    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            # session_id 要求是一个唯一的标识符，用于标识一个特定的会话。（使用telebot id）
            url=REDIS_URL, session_id="lisa"
        )
        # chat_message_history.clear() 清空历史记录
        print("chat_message_history： " , chat_message_history.messages)
        store_message = chat_message_history.messages
        if len(store_message) > 10:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEMPL + "\n这是一段你和用户的对话记录，对其进行总结摘要，摘要使用第一人称‘我’，并且提取其中的用户关键信息，如姓名，性别，出生日期等，以如下格式返回:\n 总结摘要内容 | 用户关键信息\n 例如 用户张三问候我，我礼貌回复，然后他问我今年运势如何，我回答了他今年的运势情况，然后他告辞离开。 | 张三，生日1999年1月1日"
                    ),
                    (
                        "user",
                        "{input}"
                    )
                ]
            )
            chain = prompt | self.chatmodel
            summary = chain.invoke({"input":store_message, "who_you_are":self.MOODS[self.emotion]["roleSet"]})
            print("summary: ", summary)
            chat_message_history.clear()
            chat_message_history.add_message(summary)
            print("总结后：", chat_message_history.messages)
        return chat_message_history

    def chat(self,query):
        result = self.agent_executor.invoke({"input":query})
        final_output = result["output"].replace("**", "")
        return final_output
    
    def emotion_chain(self, query:str):
        prompt = """根据用户输入判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed"，不要有其他内容，否则将受到惩罚
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，否则将受到惩罚
        3. 如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，否则将受到惩罚
        4. 如果用户输入的内容包括辱骂或者不礼貌语句，只返回"angry"，不要有其他内容，否则将受到惩罚
        5. 如果用户输入的内容比较兴奋，只返回"upbeat"，不要有其他内容，否则将受到惩罚
        6. 如果用户输入的内容比较悲伤，只返回"depressed"，不要有其他内容，否则将受到惩罚
        7. 如果用户输入的内容比较开心，只返回"cheerful"，不要有其他内容，否则将受到惩罚
        8. 只返回英文，不允许有换行符等其他内容，否则将受到惩罚。
        用户输入的内容是：{query}
        """
        chain = ChatPromptTemplate.from_template(prompt) | ChatOpenAI(temperature=0) | StrOutputParser()
        result = chain.invoke({"query" : query})
        self.emotion = result
        print("情绪判断结果：", result)
        res = self.chat(query)
        final_res = res.replace("**", "")
        yield {"msg":final_res, "emotion":result}


@app.get("/")
def read_root():
    return {"Hello": "World avatar"}

@app.post("/chat")
def chat(query:str):
    master = Master()
    res = master.emotion_chain(query)
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)