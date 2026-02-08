import os
import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

llm = ChatOpenAI(model='kimi-k2.5', api_key=os.getenv('DASHSCOPE_API_KEY'),base_url=os.getenv('DASHSCOPE_BASE_URL'))

messages = [
    SystemMessage(content='我是网络运维助手，我叫marvelnet'),
    HumanMessage(content='我叫刘洋，华三的驻场运维工程师')
]

from pprint import pprint

def pretty_ai_message(msg):
    print("🤖 AI Response:")
    print(msg.content)
    print("\n--- Metadata ---")
    pprint(msg.response_metadata)

response = llm.invoke(messages)

print(type(response))
print(response)
pretty_ai_message(response)


