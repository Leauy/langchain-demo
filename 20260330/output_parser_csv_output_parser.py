import os
from itertools import chain
import defusedxml
import dotenv
from langchain_core.output_parsers import JsonOutputParser, XMLOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

if __name__ == '__main__':
    dotenv.load_dotenv()

    chat_model = ChatOpenAI(model="qwen3.5-flash", api_key=os.getenv("OPENAI_API_KEY"),
                            base_url=os.getenv("OPENAI_API_URL"))
    response = chat_model.invoke("陕西省，各个地级市的名称和代号,以csv格式数据返回")
    print(response.content)
    csv_output = CommaSeparatedListOutputParser()
    print(csv_output.invoke(response))

    prompt_template = PromptTemplate.from_template(
        template="回答用户问题\n 以{format_instruction} 格式输出 \n 问题是{question}",
        partial_variables={"format_instruction": csv_output.get_format_instructions()}
    )

    # s = prompt_template.invoke(input={"question": "陕西的地级市名称和代号"})
    # print(csv_output.invoke(chat_model.invoke(s)))
    
    chain = prompt_template | chat_model | csv_output
    response = chain.invoke(input={"question": "陕西的地级市名称和代号"})
    print(response)


