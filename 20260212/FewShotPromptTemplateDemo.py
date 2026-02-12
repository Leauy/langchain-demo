import os

import  dotenv
from langchain_core.prompts import FewShotPromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI


dotenv.load_dotenv()


chat_model = ChatOpenAI(api_key=os.getenv('DASHSCOPE_API_KEY'), base_url=os.getenv('DASHSCOPE_BASE_URL'),  model='qwen3-max-2026-01-23')

prompt = PromptTemplate.from_template(
    template='输入如下：{input}, 输出如下：{output}',
)

examples = [
    {
        'input': '北京天气如何？', 'output': '北京',
    },
    {
        'input': '南京下雨吗？', 'output': '南京',
    },
    {
         'input': '武汉热吗？', 'output': '武汉',
    }
]

f = FewShotPromptTemplate(
    example_prompt=prompt,
    examples=examples,
    input_variables=['input'],
    suffix='输入如下：{input}, 输出如下：',
)

print(f.invoke({'input': '西安刮风不'}))

print(chat_model.invoke(f.invoke({'input': '西安刮风不'})))
print(chat_model.invoke(f.invoke({'input': '西安刮风不'})).content)