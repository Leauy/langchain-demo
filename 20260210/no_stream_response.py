import os

from langchain_openai import ChatOpenAI
import dotenv

dotenv.load_dotenv()


client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_URL'),
                    model='qwen3-max-2026-01-23', streaming=False)
respnse = client.invoke('小拇指骨折了，已经7周了，目前骨折线还是很明显，该怎么办？医生开了一些药促进骨骼愈合的')
print(respnse.content)


# human_message = HumanMessage(content='介绍下自己')
#
# messages = [human_message]
#
# client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_URL'),
#                     model='qwen3-max-2026-01-23', streaming=True)
# # 阻塞式调用
# for chunk in client.stream(messages):
#     print(chunk.content, end='', flush=True)
