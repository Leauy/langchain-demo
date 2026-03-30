# 第4章：Model I/O之Output Parsers

语言模型返回的内容通常是字符串文本格式，但是在实际AI应用开发过程中，往往希望model可以返回更直观、更加格式化内容，以确保应用能够顺利进行后续的逻辑处理。此时，LangChain提供的输出解析器就派上用场了。

输出解析器（outputParser）负责获取LLM的输出并将其转换成更加合适的格式。这在应用开发中极其重要。

## 4.1 输出解析器的分类

LangChain有许多不同类型的输出解析器

- StrOutput Parser：字符串解析器
- JsonOutputParser：JSON解析器，确保输出符合特定JSON对象格式
- XMLOutputParser：XML解析器，允许以流行的XML格式从LLM获取结果
- CommaSeparatedListOutputParser：CSV解析器，模型的输出以逗号隔开，以列表形式返回输出
- DatetimeOutputParser：日期时间解析器，可用于将LLM输出解析为日期时间格式

除了上述常用的输出解析器外，还有：

- EnumOutputParser：枚举解析器，将LLM的输出，解析为预定义的枚举值
- StructuredOutputParser：将非结构化文本转换成预定一格式的结构化数据如字典
- OutputFixingParser：输出修复解析器，用于自动修复格式错误的解析器，比如将返回的不符合预期格式的输出，尝试修正为正确的结构化数据如JSON
- RetryOutput Parser：重试解析器，当主解析器因格式错误无法解析LLM的输出时，通过调用另一个LLM自动修正错误，并重新尝试解析

## 4.2 具体解析器的使用

### 4.2.1 字符串解析器StrOutputParser

StrOutputParser简单地将任何输入转化成字符串。他是一个简单的解析器，从结果中提取content字段。

```python
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

```



### 4.2.2 Json解析器 JsonOutputParser

JsonOutputParser，即JSON输出解析器，是一种用于将大模型的自由文本输出转换为结构化JSON数据的工具

使用场景：特别适用于需要严格结构化输出的场景，比如API调用、数据存储或者下游任务处理。

实现方式：

- 用户自己通过提示词指明返回JSON格式
- 借助JsonOutputParser的格条_format_instructions() 生成格式说明，指导模型输出JSON结构



```python
import os

import dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

if __name__ == '__main__':
    dotenv.load_dotenv()

    chat_model = ChatOpenAI(model="qwen3.5-flash", api_key=os.getenv("OPENAI_API_KEY"),
                            base_url=os.getenv("OPENAI_API_URL"))
    response = chat_model.invoke("陕西省，各个地级市的名称和代号,以json格式数据返回")
    print(response.content)
    json_output = JsonOutputParser()
    print(json_output.invoke(response))

    prompt_template = PromptTemplate.from_template(
        template="回答用户问题\n 以{format_instruction} 格式输出 \n 问题是{question}",
        partial_variables={"format_instruction": json_output.get_format_instructions()}
    )

    s = prompt_template.invoke(input={"question": "陕西的地级市名称和代号"})
    print(json_output.invoke(chat_model.invoke(s)))
# C:\Users\99291\AppData\Local\Microsoft\WindowsApps\python3.exe E:\learn\AI-demo\langchain-demo\20260330\output_parser_json_output_parser.py 
{
  "province": "陕西省",
  "cities": [
    {
      "name": "西安市",
      "code": "610100",
      "note": "省级行政中心，车牌简称：陕 A"
    },
    {
      "name": "铜川市",
      "code": "610200",
      "note": "车牌简称：陕 B"
    },
    {
      "name": "宝鸡市",
      "code": "610300",
      "note": "车牌简称：陕 C"
    },
    {
      "name": "咸阳市",
      "code": "610400",
      "note": "车牌简称：陕 D"
    },
    {
      "name": "渭南市",
      "code": "610500",
      "note": "车牌简称：陕 E"
    },
    {
      "name": "延安市",
      "code": "610600",
      "note": "车牌简称：陕 F"
    },
    {
      "name": "汉中市",
      "code": "610700",
      "note": "车牌简称：陕 G"
    },
    {
      "name": "榆林市",
      "code": "610800",
      "note": "车牌简称：陕 K"
    },
    {
      "name": "安康市",
      "code": "610900",
      "note": "车牌简称：陕 S"
    },
    {
      "name": "商洛市",
      "code": "611000",
      "note": "车牌简称：陕 U"
    }
  ],
  "info": {
    "code_type": "国家标准行政区划代码（市级）",
    "description": "以上代码为中国民政部发布的行政区划代码前六位。如需车牌代码请参考 note 字段。"
  }
}
{'province': '陕西省', 'cities': [{'name': '西安市', 'code': '610100', 'note': '省级行政中心，车牌简称：陕 A'}, {'name': '铜川市', 'code': '610200', 'note': '车牌简称：陕 B'}, {'name': '宝鸡市', 'code': '610300', 'note': '车牌简称：陕 C'}, {'name': '咸阳市', 'code': '610400', 'note': '车牌简称：陕 D'}, {'name': '渭南市', 'code': '610500', 'note': '车牌简称：陕 E'}, {'name': '延安市', 'code': '610600', 'note': '车牌简称：陕 F'}, {'name': '汉中市', 'code': '610700', 'note': '车牌简称：陕 G'}, {'name': '榆林市', 'code': '610800', 'note': '车牌简称：陕 K'}, {'name': '安康市', 'code': '610900', 'note': '车牌简称：陕 S'}, {'name': '商洛市', 'code': '611000', 'note': '车牌简称：陕 U'}], 'info': {'code_type': '国家标准行政区划代码（市级）', 'description': '以上代码为中国民政部发布的行政区划代码前六位。如需车牌代码请参考 note 字段。'}}

```

知识扩展链

```python
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
    # 挨个调用invoke方法
    chain = prompt_template | chat_model | json_output
    json_result = chain.invoke(input={"question": "陕西的地级市名称和代号"})
    print(json_result)

    

```



### 4.2.3 XML解析器XMLOutputParser

XMLOutputParser， 将模型的自由文本输出转换为可以编程处理的XML数据。

任何实现：在PromptTemplate中指定XML格式要求，让模型返回<tag>content</tag>形式数据。

注意XMLOutputParser不会直接将模型的输出保持为原始XML字符串，而是会解析XML并转换成Python字典。目的是未来方便程序后续处理数据，而不是单纯的保留XML格式。

举例1：不采用XMLOutputParser，通过大模型的能力，返回XML格式数据

```python
import os
from itertools import chain
import defusedxml
import dotenv
from langchain_core.output_parsers import JsonOutputParser, XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

if __name__ == '__main__':
    dotenv.load_dotenv()

    chat_model = ChatOpenAI(model="qwen3.5-flash", api_key=os.getenv("OPENAI_API_KEY"),
                            base_url=os.getenv("OPENAI_API_URL"))
    response = chat_model.invoke("陕西省，各个地级市的名称和代号,以xml格式数据返回")
    print(response.content)
    xml_output = XMLOutputParser()
    print(xml_output.invoke(response))

    prompt_template = PromptTemplate.from_template(
        template="回答用户问题\n 以{format_instruction} 格式输出 \n 问题是{question}",
        partial_variables={"format_instruction": xml_output.get_format_instructions()}
    )

    # s = prompt_template.invoke(input={"question": "陕西的地级市名称和代号"})
    # print(xml_output.invoke(chat_model.invoke(s)))
    
    chain = prompt_template | chat_model | xml_output
    response = chain.invoke(input={"question": "陕西的地级市名称和代号"})
    print(response)



```

### 4.2.4 csv解析器CommaSeparatedListOutputParser

```python
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



```

### 4.2.5 日期解析器

```python
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
    response = chat_model.invoke("陕西省，各个地级市的名称和代号,以csv格式数据返回")
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
    response = chain.invoke(input={"question": "中国什么时间成立的？"})
    print(response)



```

