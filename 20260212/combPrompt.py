import os

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = (
    PromptTemplate.from_template('tell me a joke about {topic}'
                                 ', make it funny '
                                 '\n and in {language}')
)

print(template.format(topic='basketball', language='english'))
print('#*30')

chat_pro = ChatPromptTemplate(
    messages=[
        ('system', '你是一个AI助手，你的名字叫{name}'),
        ('human', '我的问题是{question}'),
    ],
)
print(chat_pro.invoke({
    'name': 'sb',
    'question': '那个模型是最厉害的？'
})
)

print('#*30')
print(chat_pro.format(**{
    'name': 'sb',
    'question': '那个模型是最厉害的？'
})
      )
print('#*30')

print(chat_pro.from_messages([
    ('system', '你是一个AI助手，你的名字叫{name}'),
    ('human', '我的问题是{question}'),
]))
print('#*30')

m = ChatPromptTemplate.from_messages([
    ('system', '你是一个AI助手，你的名字叫{name}'),
    ('human', '我的问题是{question}'),
])
n = m.format_prompt(**{
    'name': 'sb',
    'question': '那个模型是最厉害的？'
})
print(n)
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())
print('#*30')

m = ChatPromptTemplate.from_messages([
    '我的问题是{question}'
])
n = m.invoke({
    'name': 'question',
    'question': '那个模型是最厉害的？'
})
print(n)
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())
print('#*30')

m = ChatPromptTemplate.from_messages([
    {
        'role': 'system',
        'content': '你是一个AI助手，你的名字叫{name}'
    },
    {
        'role': 'human',
        'content': '我的问题是{question}'
    },
])
n = m.invoke({
    'name': 'SSBSBSB',
    'question': '那个模型是最厉害的？'
})
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())
print('#*30')

m = ChatPromptTemplate.from_messages([
    ('system', '你是一个AI助手，你的名字叫{name}'),
    HumanMessage(content='我的问题是{question}')
])
n = m.invoke({
    'name': 'SSBSBSB',
    'question': '那个模型是最厉害的？'
})
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())
print('#*30')


nested_prompt = ChatPromptTemplate.from_messages([('system', '你是一个AI助手，你的名字叫{name}'),])
nested_prompt2 = ChatPromptTemplate.from_messages([('human', '我的问题是{question}')])
m = ChatPromptTemplate.from_messages([
    nested_prompt,
    nested_prompt2,
])
n = m.invoke({
    'name': 'SSBSB1233333SB',
    'question': '那个模型是最厉害的？对比下 当前 中国的几个编辑器，从性价比 ，用户评价 ，价格，对比下'
})
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())


print('#*30')

import dotenv
dotenv.load_dotenv()

client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_URL'),
                    model='qwen3-max-2026-01-23', streaming=True)


for message in client.stream(n):
    print(message.content, end='', flush=True)


