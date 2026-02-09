import os
import dotenv

from langchain_openai import ChatOpenAI
# 前提 加载配置
dotenv.load_dotenv()

# 获取对话模型
chat_model = ChatOpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url=os.getenv('DASHSCOPE_BASE_URL'),
    model='deepseek-v3.2'
)
# 调用模型
response = chat_model.invoke('帮我解释下什么是langchain？')

#
print(response.content)
