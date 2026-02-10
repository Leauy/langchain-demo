import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

import dotenv

dotenv.load_dotenv()


client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_URL'),
                    model='qwen3-max-2026-01-23', streaming=True)

message1 = [SystemMessage(content='你是一个专业的骨科主任'), HumanMessage(content='小拇指骨折了，已经7周了，目前骨折线还是很明显，该怎么办？')]
message2 = [SystemMessage(content='你是一个专业的骨科主任'), HumanMessage(content='恒古骨伤愈合剂这个能行吗？')]
message3 = [SystemMessage(content='你是一个专业的骨科主任'), HumanMessage(content='恒古骨伤愈合剂这个能行吗，每次喝完口特别渴，还有点头晕，正常吗？')]

messages = [message1, message2, message3]
for message in client.batch(messages):
    print(message.content, end='', flush=True)

