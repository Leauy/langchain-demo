import pprint

from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="deepseek-v3.2", api_key='sk-346d6c9bc67d4a25af518ea75236f03e', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
pprint.pprint(llm.invoke('你好，我在学习langchain，未来会被淘汰吗？'))