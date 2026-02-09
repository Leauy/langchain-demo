import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import dotenv

dotenv.load_dotenv()

human_message = HumanMessage(content='介绍下自己')

messages = [human_message]

client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_URL'),
                    model='qwen3-max-2026-01-23', streaming=True)
# 阻塞式调用
for chunk in client.stream(messages):
    print(chunk.content, end='', flush=True)
