import dashscope
from dashscope import Generation
import os
import dotenv

dotenv.load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

resp = Generation.call(
    model="qwen-plus-2025-12-01",
    prompt="把下面的一段话翻译成中文：Actions speak louder than words."
)

print(resp.output.text)