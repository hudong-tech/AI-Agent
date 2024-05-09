from doctest import master
from email.errors import MessageParseError
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
import os

from zmq import Message

app = FastAPI()

os.environ["OPENAI_API_KEY"] = "sk-sWo0SNwz7Zrt0ATm246851735c7e4eD19330E4E5Be9bE56c"
os.environ["OPENAI_API_BASE"] = "https://ai-yyds.com/v1"


@tool
def test():
    """Test tool"""
    return "test"

class Master:
    def __init__(self):
        self.chatmodel = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            # 流式处理
            streaming=True
        )
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPL = ""
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个助手"
                ),
                (
                    "user",
                    "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        self.memory = ""
        tools = [test]
        agent = create_openai_tools_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )

    def run(self, query):
        result = self.agent_executor.invoke({"input": query})
        return result


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query:str):
    master = Master()
    return master.run(query)

@app.post("/add_ursl")
def add_ursl():
    return {"response": "URLs added successfully"}

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