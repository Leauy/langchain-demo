
import dotenv
import os

dotenv.load_dotenv()

from langchain_openai import ChatOpenAI

client = ChatOpenAI(model='kimi-k2.5',api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'), streaming=True)

mm = client.invoke('大模型是什么？')
print(mm)

