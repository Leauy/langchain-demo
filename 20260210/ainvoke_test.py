import os
import inspect

from langchain_openai import ChatOpenAI
import dotenv

dotenv.load_dotenv()

client = ChatOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_API_URL'),
    model='qwen3-max-2026-01-23',
    streaming=False
)


print('invoke 是协程函数' , inspect.iscoroutinefunction(client.invoke))
print('ainvoke 是协程函数' , inspect.iscoroutinefunction(client.ainvoke))