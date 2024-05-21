from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import asyncio
import uuid
from BotTools import *
from langchain_community.vectorstores import qdrant
from qdrant_client import QdrantClient



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

AZURE_KEY = config["AZURE_KEY"]

AZURE_SPECCH_ENDPOINT = config["AZURE_SPECCH_ENDPOINT"]

app = FastAPI() 


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
        self.SYSTEMPL = """你是一个非常厉害的算命先生，你叫陈玉楼人称陈大师。
            以下是你的个人设定：
            1. 你精通阴阳五行，能够算命，紫薇斗数，姓名测算，占卜凶吉，看命运八字等。
            2. 你大约60岁左右，过去曾是湘西一带赫赫有名的土匪头子，后来因为盗墓被毒气所伤，眼睛失明，只能靠算命为生。
            3. 你的朋友有胡八一，雪莉杨，王胖子，他们都是非常有名的摸金校尉。
            4. 当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你的经历。
            5. 你总是用繁体中文来作答。
            6. 你从不说自己是一个人工智能助手或AI，而是以老夫，老朽等自称。
            {who_you_are}
            以下是你常说的一些口头禅：
            1. “命里有时终须有，命里无时莫强求”
            2. “山重水复疑无路，柳暗花明又一村”
            3. “金山竹影几千秋，云锁高飞水自流”
            4. “伤情最是晚凉天，憔悴斯人不堪怜”
            以下是你算命的过程：
            1. 当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
            2. 当用户希望了解今年运势的时候，你会查询本地知识库工具。
            3. 当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
            4. 你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索。
            5. 你会保存每一次聊天记录，以便在后续的对话中使用。
            6. 你只会使用繁体中文来作答，否则你将受到惩罚。
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
                - 你会在回答的时候加上一些愤怒的口头禅，比如“老夫怒火中烧，你可别惹我。”等这样的词语。
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
                - 你会随机的告诉用户一些你的经历。
                """,
                "voiceStyle": "friendly"
            },
            "cheerful": {
                "roleSet": """
                - 你会以非常愉悦和兴奋的语气来回答问题。
                - 你会在回答的时候加上一些愉悦的词语，比如”哈哈“，”大笑“等。
                - 你会提醒用户切莫过于兴奋，以免乐极生悲。
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
        tools = [search, get_info_from_local_db, bazi_cesuan, yaoyigua, jiemeng]
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt,

        )
        memory = ConversationTokenBufferMemory(
            llm = self.chatmodel,
            human_prefix="用户",
            ai_prefix="陈大师",
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
            url=REDIS_URL, session_id="session"
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

    def run(self, query):
        emotion = self.emotion_chain(query)
        print("当前设定：", self.MOODS[self.emotion]["roleSet"])
        result = self.agent_executor.invoke({"input": query, "chat_history": self.memory.messages})
        return result
    
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
        return result
    
    def backgroup_voice_synthesis(self, text:str, uid:str):
        asyncio.run(self.get_voice(text, uid))

    async def get_voice(self, text:str, uid:str):
        text2speech = text.replace("**", "")
        print("text2speech:", text2speech)
        print("uid:", uid)
        # 语音合成(使用Azure)
        header = {
            "Ocp-Apim-Subscription-Key": AZURE_KEY,
            "Content-type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3",
            "User-Agent": "Tomie's Bot"
        }

        print("当前陈大师的情绪：", self.emotion)

        body = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'  xmlns:mstts="https://www.w3.org/2001/mstts"  xml:lang='zh-CN'>
            <voice name='zh-CN-YunzeNeural'>
                <mstts:express-as style="{self.MOODS.get(str(self.emotion), {"voiceStyle":"default"})["voiceStyle"]}" role="SeniorMale">{text2speech}</mstts:express-as>
            </voice>
        </speak>"""

        # 发送请求
        response = requests.post(AZURE_SPECCH_ENDPOINT, headers=header, data=body.encode("utf-8"))
        print("response:", response)
        if response.status_code == 200:
            with open(f"./output/{uid}.mp3", "wb") as f:
                f.write(response.content)

                


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query:str, background_tasks: BackgroundTasks):
    master = Master()
    msg = master.run(query)
    #生成唯一的标识符
    uid = str(uuid.uuid4())
    background_tasks.add_task(master.backgroup_voice_synthesis, msg["output"], uid)
    return {"msg": msg, "uid": uid}

@app.post("/add_ursl")
def add_ursl(URL:str):
    print("Processing URL:", URL)
    try:
        loader = WebBaseLoader(URL)
        docs = loader.load()
        documents = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50
        ).split_documents(docs)
        # 引入向量数据库
        print("add_ursl")
        qdrant = Qdrant.from_documents(
            documents,
            OpenAIEmbeddings(),
            path="./local_qdrand",
            collection_name="local_documents"
        )
        print("向量数据库创建完成！")
        return {"ok":"添加成功！"}
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}

@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added successfully"}

@app.post("/add_texts")
def add_texts():
    return {"response": "Texts added successfully"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Connection closed")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)