# 第2章：LangChain使用之Model I/O

## 1、Model I/O介绍

Model I/O模块是与语言模型（LLMs）进行交互的核心组件，在整个框架中有很重要的地位。

所谓的Model I/O，包括输入提示（Format）、调用模型（Predict）、输出解析（Parse）。分别对应Prompt Template、Model和Output parser。

简单来说就是输入、模型处理、输出三个步骤

![image-20260208224501403](E:\learn\AI-demo\langchain-demo\LangChain-ModelIO.assets\image-20260208224501403.png)

针对每个环节，LangChain 都提供了模板和工具，可以快捷的调用各种语言模型的接口。

## 2、Model I/O之调用模型1

LangChain作为一个工具，不提供任何的LLMs，而是依赖第三方集成各种大模型。比如，将OpenAI、Anthropic、Hugging Face、Llama、Qwen、等平台的模型无缝接入到你的应用。

### 2.1 模型的不同分类方式

简单来说就是用谁家的API以什么方式调用那种类型的大模型

#### 角度1：按照模型功能的不同

- 非对话模型：LLMs、text Model
- 对话模型：Chat Models（推荐）
- 嵌入模型（Embedding Models）暂不考虑

#### 角度2：模型调用时，几个重要参数的书写位置不同,api_key，base_url,model-name

- 硬编码方式：将参数写在代码
- 使用环境变量的方式
- 使用配置文件的方式（推荐）

#### 角度3：具体API的调用

- 使用LangChain提供的API（推荐）
- 使用OpenAI官方的API
- 使用其他平台提供的API

OpenAI的GPT系列模型影响了大模型技术发展的开发范式和标准。无论是Qwen还是deepseek等模型，他们使用的方法和函数调用逻辑基本上遵循OpenAI定义的规范，没有太大差异。这就使得大部分的开源项目能够通过一个较为通用的接口来接入和使用不同的模型。

### 2.2 角度1出发：按照功能不同举例

#### 类型1：LLMs非对话模型

LLMs，也叫TextModel，非对话模型，是许多语言模型应用程序的支柱。主要特点如下：

- 输入：接受文本字符串或者PromptValue对象
- 输出：总是返回字符串

![image-20260208225943214](E:\learn\AI-demo\langchain-demo\LangChain-ModelIO.assets\image-20260208225943214.png)

- 适用场景：仅需单词文本生成任务（如摘要生成、翻译、代码生成、单次问答）或对既不支持消息结构的就模型（如本地部署模型）
- 不支持多轮对话上下文。每次调用独立处理输入，无法自动关联历史对话（需要手动拼接历史文本）
- 局限性：无法处理角色分工或者复杂对话逻辑

演示代码如下：

```python
import os
import dotenv

dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import Message, SystemMessage, AIMessage

llm = ChatOpenAI(model='kimi-k2.5', api_key=os.getenv('DASHSCOPE_API_KEY'),base_url=os.getenv('DASHSCOPE_BASE_URL'))
print(llm.invoke('家庭要不要一起管钱？统一都给媳妇管着？'))

```



#### 类型2：Chat Models 对话模型

Chat Models 也叫聊天模型、对话模型，底层使用LLMs

大语言模型调用，以ChatModel为主

主要特点：

- 输入：接受消息列表List[BaseMessage] 或者 PromptValue，每条消息需要指定角色如SystemMessage、HumanMessage、AIMessage
- 输出：总是返回带着角色的消息对象（BaseMessage子类）通常是AIMessage

![image-20260208230843357](E:\learn\AI-demo\langchain-demo\LangChain-ModelIO.assets\image-20260208230843357.png)

- 原生支持多轮对话：通过消息列表维护上下文，例如SystemMessage、HumanMessage、AIMessage，。。。）模型可以基于完整对话历史生成回复。
- 适用场景：对话系统（如客服机器人、长期交互的AI助手

演示代码如下：

```python
import os
import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

llm = ChatOpenAI(model='kimi-k2.5', api_key=os.getenv('DASHSCOPE_API_KEY'),base_url=os.getenv('DASHSCOPE_BASE_URL'))

messages = [
    SystemMessage(content='我是网络运维助手，我叫marvelnet'),
    HumanMessage(content='我叫刘洋，华三的驻场运维工程师')
]

from pprint import pprint

def pretty_ai_message(msg):
    print("🤖 AI Response:")
    print(msg.content)
    print("\n--- Metadata ---")
    pprint(msg.response_metadata)

response = llm.invoke(messages)

print(type(response))
print(response)
pretty_ai_message(response)
```

#### 类型3：嵌入模型的调用



```python
import traceback
from pprint import pprint

import dotenv, os

dotenv.load_dotenv()

from langchain_core.embeddings import Embeddings
from dashscope import MultiModalEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class QwenEmbedding(Embeddings):
    def embed_documents(self, texts):
        vectors = []
        BATCH_SIZE = 16  # <=20

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]

            resp = MultiModalEmbedding.call(
                model="multimodal-embedding-v1",
                input=batch,
                api_key=os.getenv("DASHSCOPE_API_KEY")
            )

            embs = resp["output"]["embeddings"]
            vectors.extend([e["embedding"] for e in embs])

        return vectors

    def embed_query(self, text):
        resp = MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=[text],
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        return resp["output"]["embeddings"][0]["embedding"]


def process_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [Document(page_content=f.read())]


docs = process_text_file("E:/learn/AI-demo/langchain-demo/README.md")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)
try:
    embedding_model = QwenEmbedding()
    pprint(embedding_model.embed_query('如何运行游戏？'))
except Exception as e:
    traceback.print_exc()

```

### 2.3 角度2出发：参数位置不同举例

#### 2.3.1 模型调用的主要方法以及参数

相关方法及属性

- OpenAI / ChatOpenAI ：创建一个模型对象（非对话类/对话类）
- model.invode(xxx)：执行调用，将用户输入发送给模型
- .content ：提取模型返回的实际文本内容

模型调用函数使用时需要初始化模型，并设置必要的参数

1、必须设置的参数·

- base_url: 大模型API服务的根地址
- api_key: 用于身份验证的密钥，由大模型服务商提供
- model/model-name: 指定要调用的具体的大模型名称如deepseek-V3，qwen等

2、其他参数

- temperature：温度，控制生成文本的随机性，取值范围0~1
  - 值越低- 输出越确定，保守（适合事实回答）
  - 值越高- 输出越多样，有创意（适合创意写作）

通常根据需要设置如下：

- 精确模式（0.5或者更低）：生成的文本更加安全可靠，但是缺乏创意和多样性
- 平衡模式（0.8）：生成的文本既有一定的多样性，又能保持较好的连贯性和准确性。
- 创意模式（1）：生成的文本更具创意，但是也更容易出现语法错误或者不合逻辑的内容



- max_tokens: 限制生成文本的最大长度，防止输出过长

Token是什么？

基本单位：大模型处理文本的最小单位时token（相当于自然语言中的词或者字），输出时逐个token依次生成

收费依据：大语言模型（LLM）通常是以token数量作为其计量收费的依据

1token大约 1-1.8个中文字，大约3-4个英文字母

token与字符的转化的可视化工具

- https://platform.openai.com/tokenizer
- https://console.bce.baidu.com/support/#/tokenizer

max_tokensL建议设置

- 客户短回复：128-256
- 常规对话、多轮对话：512-1024
- 长内容生成：1024-4096

#### 2.3.2 模型调用推荐平台：closeai

这里推荐使用的平台

考虑到OpenAI等模型在国内服务访问

#### 2.3.3 参数位置

- 硬编码
- 环境变量
- .env配置文件方式

![image-20260209000646263](E:\learn\AI-demo\langchain-demo\LangChain-ModelIO.assets\image-20260209000646263.png)