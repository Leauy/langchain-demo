import os
import dotenv

dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import Message, SystemMessage, AIMessage

llm = ChatOpenAI(model='kimi-k2.5', api_key=os.getenv('DASHSCOPE_API_KEY'),base_url=os.getenv('DASHSCOPE_BASE_URL'))
print(llm.invoke('家庭要不要一起管钱？统一都给媳妇管着？'))




