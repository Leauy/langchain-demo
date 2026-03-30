import os

import dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

if __name__ == '__main__':
    dotenv.load_dotenv()

    chat_model = ChatOpenAI(model="qwen3.5-flash", api_key=os.getenv("OPENAI_API_KEY"),
                            base_url=os.getenv("OPENAI_API_URL"))
    json_output = JsonOutputParser()
    prompt_template = PromptTemplate.from_template(
        template="回答用户问题\n 以{format_instruction} 格式输出 \n 问题是{question}",
        partial_variables={"format_instruction": json_output.get_format_instructions()}
    )
    # s = prompt_template.invoke(input={"question": "陕西的地级市名称和代号"})
    # print(json_output.invoke(chat_model.invoke(s)))
    chain = prompt_template | chat_model | json_output
    json_result = chain.invoke(input={"question": "陕西的地级市名称和代号"})
    print(json_result)

    
