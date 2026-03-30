import os

import dotenv
from langchain_core.prompts import load_prompt





if __name__ == '__main__':
    dotenv.load_dotenv()
    prompt = load_prompt("./prompt.yaml", encoding="utf-8")
    print(prompt.format_prompt(name="刘启陌", what="sb"))

    prompt1 = load_prompt("./prompt.json", encoding="utf-8")
    print(prompt1.format_prompt(name="liuyang", what="123456"))