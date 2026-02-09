import os
import dotenv
from dashscope import api_key
from openai import OpenAI, base_url

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv('DASHSCOPE_API_KEY'), base_url=os.getenv('DASHSCOPE_BASE_URL'))

response = client.chat.completions.create(
    model='kimi-k2.5',
    messages=[
        {'role': 'system', 'content': '你是一个历史钻研特别深的老教授，特别熟悉明朝的相关历史'},
        {'role': 'user', 'content': '明朝灭亡的原因?'}
    ]
)

from pprint import pprint


def pretty_ai_message(msg):
    print("🤖 AI Response:")
    print(msg.content)
    print("\n--- Metadata ---")
    pprint(msg.response_metadata)
pprint('思考过程')
pprint(response.choices[0].message.reasoning_content)

pprint('结果')
pprint(response.choices[0].message.content)
