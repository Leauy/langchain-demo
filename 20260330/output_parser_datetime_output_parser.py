import os
from itertools import chain
import defusedxml
import dotenv
from langchain_classic.output_parsers import DatetimeOutputParser
from langchain_core.output_parsers import JsonOutputParser, XMLOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

if __name__ == '__main__':
    dotenv.load_dotenv()

    chat_model = ChatOpenAI(model="qwen3.5-flash", api_key=os.getenv("OPENAI_API_KEY"),
                            base_url=os.getenv("OPENAI_API_URL"))
    response = chat_model.invoke("中华人民共和国什么时间成立的？返回日期时间格式 年月日时分秒")
    print(response.content)
    datetime_output = DatetimeOutputParser()
    print(datetime_output.invoke(response))

    prompt_template = PromptTemplate.from_template(
        template="回答用户问题\n 以{format_instruction} 格式输出 \n 问题是{question}",
        partial_variables={"format_instruction": datetime_output.get_format_instructions()}
    )

    # s = prompt_template.invoke(input={"question": "陕西的地级市名称和代号"})
    # print(datetime_output.invoke(chat_model.invoke(s)))
    
    chain = prompt_template | chat_model | datetime_output
    response = chain.invoke(input={"question": "中华人民共和国什么时间成立的？"})
    print(response)


