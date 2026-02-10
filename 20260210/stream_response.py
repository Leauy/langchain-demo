import os

from langchain_openai import ChatOpenAI
import dotenv

dotenv.load_dotenv()


client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_URL'),
                    model='qwen3-max-2026-01-23', streaming=True)


for message in client.stream('小拇指骨折了，已经7周了，目前骨折线还是很明显，该怎么办？医生开了一些药促进骨骼愈合的,目前有吃一些钙片，牛奶，鸡蛋，平时只有在敲键盘的时候偶尔会去 把支架拆掉，平时都是带着的，几乎没怎么用力'):
    print(message.content, end='', flush=True)

