#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import os

os.environ["OPENAI_API_KEY"] = "sk-sWo0SNwz7Zrt0ATm246851735c7e4eD19330E4E5Be9bE56c"
os.environ["OPENAI_API_BASE"] = "https://ai-yyds.com/v1"

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="LangChain Server is a server for LangChain, a chain of language models."
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

add_routes(
    app,
    prompt | model,
    path="/joke"
)

if __name__ == "__main__":
    import uvicorn
    # http://localhost:8001/docs
    # http://localhost:8001/joke/playground/
    uvicorn.run(app, host="localhost", port=8001)