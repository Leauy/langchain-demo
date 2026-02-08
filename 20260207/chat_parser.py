import os
import dotenv
from langchain_core.output_parsers import JsonOutputParser

dotenv.load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='kimi-k2.5',api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是网络设备运维专家，熟悉各种厂商、型号的设备配置, 用JSON格式回复，问题用question，回答用answer'),
    ('user', '{input}')
])

output_parser = JsonOutputParser()

chain = prompt | llm | output_parser
message = chain.invoke({'input': '华三的防火墙增加一个新的防火墙策略'})
print(message)