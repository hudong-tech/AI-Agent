# AI Agent

## 运行
运行环境
 - python 3.10.12
``` shell
pip install -r requirements.txt

pip show fastapi

python server.py
```

## 入口
- 服务器端：接口 -> langchain -> openai\ollama。。
- 客户端：电报机器人、微信机器人、website。
- 接口：http,https,websocket


## 服务器
1. 接口访问，python 选型 fastapi
2. /chat 的接口，post请求
3. /add_ursl 从url中学习知识
4. /add_pdfs 从pdf里学习知识
5. /add_texts 从txt文本里学习

