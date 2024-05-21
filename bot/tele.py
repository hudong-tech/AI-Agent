import telebot
import urllib.parse
import requests
import json
import os
import asyncio

with open('./config.json', 'r') as f:
    config = json.load(f)

bot = telebot.TeleBot(config['TELEBOT_KEY'])

@bot.message_handler(commands=["start"])
def start_message(message):
    bot.send_message(message.chat.id, "你好，我是陈瞎子，精通阴阳五行，能够算命，紫薇斗数，姓名测算，占卜凶吉，看命运八字等，请问有什么能帮助你？")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    # bot.reply_to(message, message.text)
    try:
        # 将用户输入的文本进行URL编码
        encoded_text = urllib.parse.quote(message.text)
        # 使用requests库向后端发送请求
        response = requests.post(config['BACKEND_SERVICE_API'] + encoded_text, timeout=100)
        if response.status_code == 200:
            # 将后端返回的json数据转换为python字典
            aiSay = json.loads(response.text)
            print("aiSay", aiSay)

            if "msg" in aiSay:
                bot.reply_to(message, aiSay["msg"]["output"])
                # 创建output目录如果它不存在
                if not os.path.exists('output'):
                    os.makedirs('output')
                audio_path = f"output/{aiSay['uid']}.mp3"
                asyncio.run(check_audio(message, audio_path))
            else:
                bot.reply_to(message, "对不起,我不知道怎么回答你")
    except requests.RequestException as e:
        print("echo_all error:", e)
        bot.reply_to(message, "对不起，我不知道怎么回答你")

async def check_audio(message, audio_path):
    while True:
        if os.path.exists(audio_path):
            print("audio_path", audio_path)
            with open(audio_path, 'rb') as audio:
                bot.send_audio(message.chat.id, audio)
            os.remove(audio_path)
            break
        else:
            print("watting for audio...")
            # 使用asyncio.sleep(1)来等待1秒
            await asyncio.sleep(1)

# 机器人回复消息
bot.infinity_polling()