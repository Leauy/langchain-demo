import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个靠谱的骨科专家主任医生'),
    MessagesPlaceholder('msg')
])

m = prompt_template.format_messages(msg = [HumanMessage(content='我男性，31岁，小拇指骨折了7周多了，目前X光片子还是显示的骨折线特别明显，医生给开了一些药物，恒古固伤愈合剂，喝了以后有点头晕，口渴非常严重，医生让继续固定')])
# [SystemMessage(content='你是一个靠谱的骨科专家主任医生', additional_kwargs={}, response_metadata={}), HumanMessage(content='我男性，31岁，小拇指骨折了7周多了，目前X光片子还是显示的骨折线特别明显，医生给开了一些药物，恒古固伤愈合剂，喝了以后有点头晕，口渴非常严重，医生让继续固定', additional_kwargs={}, response_metadata={})]
print(m)

client = ChatOpenAI(api_key=os.getenv('DASHSCOPE_API_KEY'), base_url=os.getenv('DASHSCOPE_BASE_URL'), model='qwen3-max-2026-01-23')

print(client.invoke(m))

