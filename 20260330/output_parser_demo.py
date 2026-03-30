import os

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI





if __name__ == '__main__':
    dotenv.load_dotenv()
    prompt = load_prompt("./prompt.yaml", encoding="utf-8")
    print(prompt.format_prompt(name="刘启陌", what="sb"))

    prompt1 = load_prompt("./prompt.json", encoding="utf-8")
    ss = prompt1.format_prompt(name="liuyang", what="123456")
    print(ss)
    chat_model = ChatOpenAI(model="qwen3.5-flash", api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_URL"))
    r = chat_model.invoke(ss)
    parser = StrOutputParser()
    print(parser.invoke(r))

