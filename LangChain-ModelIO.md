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

### 2.4 角度3出发：各平台API的调用举例

#### 2.4.1 OpenAI官方API

考虑到OpenAI在国内访问以及充值的不便，我们仍然使用closeAI网址 https://www.closeai-asia.com 注册和充值，具体费用自理。

##### 调用非对话模型

```python
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
```

##### 调用对话模型

```python
import dashscope
import os
import dotenv

dotenv.load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-plus-2025-12-01",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "把下面的一段话翻译成中文：Actions speak louder than words."},
    ]
)
print(completion.choices[0].message)
```



### 2.5 如何选择合适的大模型？

#### 2.5.1 有没有最好的大模型

凡是问那个大模型最好的？都是不懂得

不妨反问：无论做什么，有都表现更好的员工的吗

没有最好的大模型，只有最适合的大模型

基础模型选型，合规和安全时首要考量因素

为什么不要依赖榜单？

- 榜单以及被应试教育污染，还算值得相信的榜单：LMSYS Chatbot Arena LeaderBoard
- 榜单体现的时整体能力，放到一件具体事情上，排名低的可能反倒更好
- 榜单体现不出成本差异



本课程主要以OpenAI为例展开后续的课程。因为：

- OpenAI最流行，即便国内也是如此
- OpenAI最先进，别的模型有的能力，OpenAI一定都i有。OpenAI有的，其他模型不一定有
- 其他模型都在追赶和模仿OpenAI

学活OpenAI，其他模型触类旁通反之不一定。

#### 2.5.2 小结：获取大模型的标准方式

后续的各种模型测试，都基于以下的模型展开

**非对话模型**

```python
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
```

**对话模型**

```python
import dashscope
import os
import dotenv

dotenv.load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-plus-2025-12-01",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "把下面的一段话翻译成中文：Actions speak louder than words."},
    ]
)
print(completion.choices[0].message)
```

## 3、Model I/O之调用模型2

### 3.1 关于对话模型的Message（消息）

聊天模型，除了将字符串作为输入外，还可以使用聊天消息作为输入，并返回聊天消息作为输出。

Langchain内置消息的类型：

- System Message： 设定AI行为规则或者背景信息，比如设定AI的初始状态、行为模式、或对话的总体目标。比如作为一个代码专家，或者返回JSON格式。通常作为消息序列中的第一个传递
- HumanMessage： 表示来自用户输入，比如实现一个快速排序的方法
- AIMessage：存储AI回复的内容，可以是文本，也可以时调用工具的请求
- ChatMessage：可以自定义角色的通用消息类型
- FuctionMessage/ToolMessage：函数调用/工具消息，用于函数调用结果的消息类型

注意
FuctionMessage/ToolMessage分别是在函数调用和工具调用场景下才会使用的特殊信息类型，HumanMessage、AIMessage和SystemMessage才是最常用的消息类型。

### 3.2 关于上下文记忆

### 3.3 关于模型调用的方法

为了尽可能简化自定义链的创建，我们实现了一个Runnable的协议。许多哟的LangChain组件实现了Runnable协议，包括聊天模型】提示词模板、输出解析器、检索器、代理（智能体）等。

Runnable定义的公共的调用方法如下：

- invoke： 处理单条输入，等待LLM完全推理完成后再返回调用结果
- stream：流式响应，逐字输出LLM的响应结果
- batch：处理批量输入

这些也有相应的异步方法，应该与asyncio 和 await 语法一起以实现并发：

- astream：异步流式响应
- ainvoke：异步处理单条输入
- abatch：异步处理批量输入
- astream_log:异步流式返回中间步骤以及最终响应
- astream_events: 测试版异步流式返回链中发生的事件（在langchain-core 0.1.4 中可用）

#### 3.3.1 流式输出和非流式输出

在langchain中，语言模型是的输出分为了两种主要的模式：流式输出和非流式输出

下面2个场景

- 非流式输出这是langchain与LLM交互时默认的行为，是最简单、最稳定的语言模型调用方式。当用户发出请求后，系统在后台等待模型生成完整响应。然后一次性将全部结果返回
- 流式输出：一种更具交互感的模型输出方式，用户不再需要等待完整答案，而是能看到模型逐个tolen地实时返回内容。

#### 3.3.2 批量调用

```python
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

import dotenv

dotenv.load_dotenv()


client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_URL'),
                    model='qwen3-max-2026-01-23', streaming=True)

message1 = [SystemMessage(content='你是一个专业的骨科主任'), HumanMessage(content='小拇指骨折了，已经7周了，目前骨折线还是很明显，该怎么办？')]
message2 = [SystemMessage(content='你是一个专业的骨科主任'), HumanMessage(content='恒古骨伤愈合剂这个能行吗？')]
message3 = [SystemMessage(content='你是一个专业的骨科主任'), HumanMessage(content='恒古骨伤愈合剂这个能行吗，每次喝完口特别渴，还有点头晕，正常吗？')]

messages = [message1, message2, message3]
for message in client.batch(messages):
    print(message.content, end='', flush=True)


```

#### 3.3.3 同步调用和异步调用

同步调用：阻塞式，顺序执行

异步调用

允许程序在等待某些操作完成时继续执行其他的任务，而不是阻塞等待。这在处理IO操作（如网络请求、文件读写等）时特别有用，可以显著提高程序的效率和响应性。

举例：

```python
import asyncio
import os
import time

from langchain_openai import ChatOpenAI
import dotenv

dotenv.load_dotenv()

async def model_call():

    client = ChatOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_API_URL'),
        model='qwen3-max-2026-01-23',
        streaming=False
    )
    for message in client.stream(
            '小拇指骨折了，已经7周了，目前骨折线还是很明显，该怎么办？医生开了一些药促进骨骼愈合的,目前有吃一些钙片，牛奶，鸡蛋，平时只有在敲键盘的时候偶尔会去 把支架拆掉，平时都是带着的，几乎没怎么用力'):
        print(message.content, end='', flush=True)

async def other_task():
    await asyncio.sleep(1)
    print('other_task finished')


async def main():
    start = time.time()
    await asyncio.gather(model_call(), other_task())
    end = time.time()
    print(end - start)
    return 'cost time: {}'.format(end - start)

if __name__ == '__main__':
    result = asyncio.run(main())
    print(result)
```

使用asyncio.gather()并行执行时，理想情况下，两个任务几乎同时开始，他们的执行时间将重叠。如果两个任务的执行时间相同（5s）那么总的执行时间应该接近单个任务的执行时间，而不是两者之和。



异步调用之ainvoke

验证ainvoke 是否时异步？

```python
import os
import inspect

from langchain_openai import ChatOpenAI
import dotenv

dotenv.load_dotenv()

client = ChatOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_API_URL'),
    model='qwen3-max-2026-01-23',
    streaming=False
)


print('invoke 是协程函数' , inspect.iscoroutinefunction(client.invoke))
print('ainvoke 是协程函数' , inspect.iscoroutinefunction(client.ainvoke))

invoke 是协程函数 False
ainvoke 是协程函数 True
```





