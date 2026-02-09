import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import dotenv

dotenv.load_dotenv()

system_message = SystemMessage(content='你是一个著名的小说作家')
human_message = HumanMessage(content='帮我写一个有点雷雨式悲剧的家庭伦理的短篇小说')

messages = [system_message, human_message]

client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_URL'),
                    model='qwen3-max-2026-01-23')
response = client.invoke(messages)

print(response)
print(type(response))
print(response.content)
