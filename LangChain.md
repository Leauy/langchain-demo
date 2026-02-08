# LangChain

![image-20260111104350174](.\LangChain.assets\image-20260111104350174.png)

## 第一章: LangChain使用概述

### 1、介绍LangChain

#### 1.1 什么是LangChain

LangChain是2022年10月，由哈佛大学的Harrison Chase 发起研发的一个开源开发的开源框架，用于开发由大语言模型LLMs驱动的应用程序。

比如搭建智能体（Agent）、问答系统（QA）、对话机器人、文档搜索系统、企业私有知识库等。

![image-20260111105155631](.\LangChain.assets\image-20260111105155631.png)

简单概括

LangChain 不等于 LLMs

LangChain之于LLMs，类似Spring之于Java

LangChain之于LLMs，类似django、flask之于python

LangChain = Lang + Chain

Language 大语言模型
Chain 链，将大模型与外部数据&各种组件连接成链，以此构建AI应用程序。



![image-20260111105523411](.\LangChain.assets\image-20260111105523411.png)

应用开发是大模型最值得关注的方向：应用为王

学习LangChain框架，高效开发大模型应用。

#### 1.2 有哪些大模型应用开发框架？

sk-f16240a40d4f421880bf04e6e90c5163

![image-20260111105935537](.\LangChain.assets\image-20260111105935537.png)

- Langchain：这些工具里出现最早、最成熟的，适合复杂任务分解和单智能体应用
- LlamaIndex：专注于高效的索引和检索，适合RAG场景。
- Langchain4j：Langchain还出了Java、Javascript （LangChain.js)两个语言的版本，Langchain4j 的功能略少于LangChain，但是主要的核心功都是有的。
- SpringAI/SpringAI Alibaba：有待进一步成熟，此外只是简单的对于一些接口进行了封装
- SementicKernel：也称为sj，微软推出的，对于C#同学来说，那就是5颗星。

#### 1.3 为什么需要LangChain？

问题1：LLMs用的好好的，干嘛还需要LangChain？

在大语言模型LLM如ChatGPT、Claude、DeepSeek等快速发展的今天，开发者不仅希望能使用这些模型，还希望能将它们灵活集成到自己应用中，实现更强大的对话能力，检索增强生成（RAG）、工具调用（tool calling）、多轮推理等功能。

![image-20260111110531121](.\LangChain.assets\image-20260111110531121.png)

#### 

LangChain更为方便解决这些问题而生的。比如：大模型默认不能联网，如果需要联网，用langchain。

问题2：我们可以使用GPT或者Deepseek等模型的API进行开发，为何需要langhain这样的框架？

不使用LangChain，确实开源使用GPT或者Deepseek等模型的API进行开发。比如搭建智能体Agent、问答系统、对话机器人等复杂的LLM应用。

但使用LangChain的好处

- 简化开发难度：更简单、更高效、效果好
- 学习成本更低：不同模型的API不同，调用方式也有区别，切换模型时学习成本更高，使用langChain，可以以统一、规范的方式进行调用，有更好的移植性
- 现成的链式组装：LangChain提供了一些现成的链式组装，用于完成特定的高级任务。让复杂的逻辑变得结构化】易组合、易扩展。

![image-20260111111642897](.\LangChain.assets\image-20260111111642897.png)



问题3：LangChain提供了哪些功能？

LangChain是一个帮助你构建LLM应用的全套工具集。这里涉及到prompt构建、LLM接入、记忆管理、工具调用、RAG、智能体开发等模块。



#### 1.4 LangChain的使用场景

学完LangChain，如下类型的项目，大家都可以实现：

![image-20260111112230623](.\LangChain.assets\image-20260111112230623.png)

比如：医院智能助手

![image-20260111112934466](.\LangChain.assets\image-20260111112934466.png)

比如万象知识库





![image-20260111113522244](.\LangChain.assets\image-20260111113522244.png)

#### 1.5 LangChain资料介绍

![image-20260111114035082](.\LangChain.assets\image-20260111114035082.png)

#### 1.6 架构设计

##### 1.6.1 总体架构图



![image-20260111114310185](.\LangChain.assets\image-20260111114310185.png)

图中展示了LangChain生态系统的主要组件及其分类，分为3个层次：架构（Architecture）、组件（Components）、和部署Deployment。

##### 1.6.2 内部架构详情

**结构1： LangChain**

langchain：构成应用程序认知架构的Chains、Agents、Retrieval strategiesdeng

比如：构成应用程序的链、智能体、RAG

langchain-community：第三方集成

比如：Model I/O、Retrieval、Tool & ToolKit；合作伙伴包langchain-openai、langchain-anthropic等

langchain-Core: 基础抽象和LangChain表达式语言（LCEL）

小结：LangChain，就是AI应用组织套件，封装了一堆的API。langchain框架不大，但是里面琐碎的知识点特别多，就像乐高，提供了很多标准化的乐高零件。

**结构2：LangGraph**

LangGraph可以看做基于LangChain的API进一步封装，能够协调多个Chain、Agent、Tools完成更加复杂的任务，实现更高级的功能。

**结构2：LangSmith**

LangSmith链路追踪。提供了6大功能，涉及Debugging、Playground、Prompt Management、Annotation（注释）、Testing测试、Monitoring监控等。与LangChain无缝集成，帮助你原型阶段过渡到生产阶段。

正因为LangSmith这样的工具存在，才使得LangChain意义更大。

**结构4：LangServe**

将LangChain的可运行项和链部署成Rest API，使得他们可以通过网络进行调用，

Java'怎么调用langchain？就通过这个langserve。将langchain应用包装成一个rest api，对外暴露服务。同时支持更高的并发，稳定性更好。



总结：LangChain当中，最有前途的两个模块：LangGraph和 LangSmith。



### 2、开发前的准备工作

#### 2.1 前置知识

#### 2.2 相关环境安装

### 3、大模型应用开发

大模型应用技术特点：门槛低，天花板高

#### 3.1 基于RAG架构的开发

背景：

- 大模型的知识冻结（过时，训练完成后截至，最新的知识不会更新）
- 大模型幻觉

而RAG就可以非常精确的解决这两个问题。

举例：

LLM在考试的时候面对陌生的领域，答复能力有限，然后就放飞自我了。而此时RAG给了一些提示和思路，让LLM懂了开始往这个提示的方向做，最终考试的正确率从60%到了90%。

![image-20260111210744971](.\LangChain.assets\image-20260111210744971.png)

何为RAG？

Retrieval-Augmented Generation（检索增强生成）

![image-20260111210843174](.\LangChain.assets\image-20260111210843174.png)





检索-增强-生成过程： 检索可以理解为10步

增强理解 12步（这里的提示词包含检索到的数据）

生成理解为第15步

类似的细节图如下

![image-20260111211156032](.\LangChain.assets\image-20260111211156032.png)



强调下难点的步骤：

![image-20260111211230518](.\LangChain.assets\image-20260111211230518.png)



这些过程中的难点：1、文件解析，2、文件切割，3、知识检索，4、知识重排序

Embedding ：将知识向量化

Rerank: 知识重排序



reranker的使用场景

- 适合：追求回答高精度和高相关性的场景中特别适合使用reranker，例如专业知识库或者客服系统等应用。

- 不适合：引入reranker会增加召回时间，增加检索延迟。服务对响应时间要求高时，使用reranker可能不合适。

这里有三个位置涉及大模型的使用

- 第三步向量化的时候，需要使用EmbeddingModels
- 第7步重排序时，需要使用RerankModels
- 第9步生成答案时，需要使用LLM



#### 3.2 基于Agent架构的开发

充分利用LLM的推理决策能力，通过增加规划、记忆、和工具调用的能力，构造一个能够独立思考、逐步完成给定目标的智能体。

举例：传统程序 VS Agent（智能体）

![image-20260111212639106](.\LangChain.assets\image-20260111212639106.png)



OpenAI的元老 Lilian在2023年6月在个人博客首次提出了现代AI Agent架构。



![image-20260111212839137](.\LangChain.assets\image-20260111212839137.png)

![image-20260111212902920](.\LangChain.assets\image-20260111212902920.png)

一个数学公式来表示：
Agent = LLM + Memory + Tools +planning + Action

![image-20260111212944838](.\LangChain.assets\image-20260111212944838.png)

智能体核心要素被细化为以下模块

1、大模型（LLM）作为“大脑”：提供推理、规划和知识理解能力，是AI Agent的决策中枢。

大脑主要由一个大语言模型LLM组成，承担着信息处理和决策等功能，并可以呈现推理和规划的过程，能很好的应对未知任务。

2、记忆（Memory）

记忆机制能让智能体在处理重复工作时调用以前的经验，从而避免用户进行大量重复交互。

- 短期记忆：存储单词对话周期的上下文信息，属于临时信息存储机制。受限于模型的上下文窗口长度。

  ![image-20260111213903362](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260111213903362.png)

  ![image-20260111213922597](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260111213922597.png)

- 长期记忆：可以横跨多个任务或者时间周期，可以存储并调用核心知识，非即时任务。

  长期记忆，可以通过模型参数微调（固化知识）、知识图谱（结构化语义网络）或者向量数据库（相似性检索）方式实现

3、工具使用（tool use）：调用外部工具（如API 数据库）扩展能力边界

![image-20260111214224923](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260111214224923.png)

4、规划决策（Planning）：通过任务分解、反思、自省框架实现复杂任务处理。例如，利用思维链（Chain of Thought）将目标拆解为子任务、并通过反馈优化策略。

![image-20260111214358331](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260111214358331.png)

5、行动（Action）：实际决策的模块，涵盖软件接口操作（如自动订票）和物理交互（如机器人执行搬运），比如：检索、推理、编程等。

智能体会形成完整的计划流程。例如先读取以前工作的经验和记忆，之后规划子目标并使用相应工具去处理问题，最后输出给用户并完成反思。

#### 3.3 大模型应用开发的4个场景

##### 场景1：纯Prompt

- prompt是操作大模型的唯一接口
- 当人看：你说一句，ta回一句，。。。

![image-20260111215039182](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260111215039182.png)

##### 场景2： Agent + Function Calling

- Agent： AI 主动提要求
- Function Calling： 需要对接外部系统时，AI要求执行某个函数
- 当人看：你问ta【我明天去杭州出差，是否需要带伞？】，ta让你先看天气预报，你看了告诉ta，ta再告诉你要不要带伞

![image-20260111215144834](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260111215144834.png)

##### 场景3：RAG（Retrieval - Augmented Generation）

RAG：需要补充领域知识时使用

- Embeddings：把文字转换为更易于相似度计算的向量
- 向量数据库：存储向量，便于查询
- 向量搜索：根据输入向量，找到相似的向量

举例：考试答题时，到书上找相关内容，再结合题目组成答案。

![image-20260111221257388](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260111221257388.png)

##### 场景4：Fine-tuning（精调/微调）

举例：努力学习考试内容，长期记住，活学活用

![image-20260111221512080](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260111221512080.png)

特定：成本最高；在前面的方式解决不了问题的情况下，在使用。

##### 如何选择

面对一个需求，如何开始，如何选择技术方案？

![image-20260111221800792](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260111221800792.png)

注意：其中最容易被忽略的，是准备测试数据。

下面，我们重点介绍下大模型应用的开发两类：基于RAG的架构，基于Agent的架构。



### 4、LangChain的核心组件

学习LangChain的最简单直接的方法就是阅读官方文档。

https://python.langchain.com/v0.1/docs/modules

通过文档目录我们可以看到，LangChain构成的核心组件。

![image-20260113233722320](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260113233722320.png)

#### 4.1 一个问题引发的思考

如果要组织一个AI应用，开发者一般需要什么？

- 提示词模板的构建，不仅仅只包含用户输入
- 模型和返回，参数设置，返回内容的格式化输出
- 知识库查询，这里会包含文档加载，切割，以及转化为词嵌入Embedding向量
- 其他第三方工具调用，一般包括天气查询，Google搜索，一些自定义的接口能力调用
- 记忆获取，每一个对话都有上下文，在开启对话之前总得获取到之前的上下文

#### 4.2 核心组件的概述

LangChain的核心组件涉及六大模块，这六大模块提供了一个全面且强大的框架，使得开发者者能够创建复杂、高效且用户有好的基于大模型的应用。

![image-20260113234324054](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260113234324054.png)

#### 4.3 核心组件的说明

##### 核心模块1： MODEL/IO

这个模块使用的最多，也最简单。

Model/IO：标准化各个大模型的输入和输出，包含输入模板，模型本身和格式化输出。以下是使用语言模型从输入到输出的基本流程。

![image-20260113234558959](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260113234558959.png)

以下是对每一块的总结：

- format（格式化）：即指代prompts template，通过模板管理大模型的输入。将原始数据格式化成模型可以处理的形式，插入到一个模板问题中，然后送入模型进行处理。
- Predict（预测）：即代指Models，使用通用接口调用不同的大语言模型，接受被送进来的问题，然后基于这个问题进行预测或生成回答。
- Parse（生成）：即代指Output Parser部分，用来从模型的推理中提取信息，并按照预先设定好的模板来规范化输出。比如，格式化成一个结构化的JSON对象。

##### 核心模块2： Chains

Chain：链条，用于将多个模块串联起来组成一个完整的流程，是LangChain框架中最重要的模块。

例如：一个Chain可能包括一个Prompt模板，一个语言模型，一个输出解析器，他们一起工作以处理用户输入，生成响应并处理输出。

常见的Chain类型

- LLMChain：最基础的模型调用链
- SequentialChain：多个链串联执行
- RouterChain：自动分析用户的需求，引导最合适的链
- RetrievalQA：结合向量数据库进行问答的链

##### 核心模块3： Memory

Memory：记忆模块，用于保存对话历史或者上下文信息，以便在后续对话中使用

常见的Memory类型：

- ConversationBufferMemory：保存完整的对话历史
- ConversationSummaryMemory：保存精简摘要的对话历史
- ConversationSummaryBufferMemory：混合型记忆机制，兼具上面两个类型的特点
- VectorStoreRetrieverMemory：保存对话历史存储在向量数据库中

##### 核心模块4：Agents

Agents，对应智能体，是LangChain的高阶能力，它可以自主选择工具，并规划执行步骤

Agent的关键组成：

- AgentType：定义决策逻辑的工作量模式
- Tool：一些内置的功能模块，如API调用，引擎搜索，文本处理，数据查询等工具。Agents通过这些工具来执行特定的功能
- AgentExecutor：用来运行智能体并执行其决策的工具，负责协调智能体的决策和实际的工具执行。

目前最热门的智能体开发实践，未来能够真正实现通用人工智能的落地方案。这里的Agent，就会涉及到前面讲的memory以及tools。

##### 核心模块5： Retrieval

Retrieval：对应着RAG，检索外部数据，然后在执行生成步骤时将其传递到LLM。步骤包括文档加载，切割，embedding等。

![image-20260114000313300](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260114000313300.png)



- source：数据源，即大模型可以识别的多种类型的数据：视频、图片、文本、代码、文档等
- Load：负责将来自不同的数据源的非结构化数据，加载为文档对象
- Transform：负责对加载的 文档进行转换处理，比如将文本拆分成具有语义意义的小块。
- Embed：  将文本编码转为向量的能力。一种用于嵌入文档，另一种用于嵌入查询
- Store：将向量化的数据进行存储
- Retrieval：从大规模文本框中检索和查询相关的文本段落

图中绿色表示入库存储前的操作。

##### 核心模块6： Callbacks

Callbacks：回调机制，允许连接到LLM应用程序的各个阶段，可以监控和分析LangChain的运行情况，比如日志记录，监控，流传输以及优化性能。

回调函数，对于程序员不陌生，这个函数允许我们在LLM的各个阶段使用各种各样的钩子，从而实现日志的监控、记录以及流式传输的功能

#### 4.4 小结

- Model/IO 模块使用多，简单
- Chains 模块：最重要
- Retrieval模块、agents模块，大模型的注意落地场景。

在这个基础上，其他组件要么是他们的辅助，要么只是完成常规应用程序的任务。

辅助：比如向量数据库的分块和嵌入，用于追踪、观测的callbacks

任务：比如tools、memory

![image-20260114001443030](E:\learn\AI-demo\langchain-demo\LangChain.assets\image-20260114001443030.png)

我们要做的就是一个一个module的去攻破，最后将他们融会贯通。

### 5、LangChain的helloworld

#### 5.1 获取大模型

.env

```shell
#DEEPSEEK_API_KEY=sk-79a625372f1c47f394b0818709b28216
#DEEPSEEK_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-346d6c9bc67d4a25af518ea75236f03e
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```



```python3

import dotenv
import os

dotenv.load_dotenv()

from langchain_openai import ChatOpenAI

client = ChatOpenAI(model='kimi-k2.5',api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'), streaming=True)

mm = client.invoke('大模型是什么？')
print(mm)
```

#### 5.2 使用提示词模板

```python
import os
import dotenv

dotenv.load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='kimi-k2.5',api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是网络设备运维专家，熟悉各种厂商、型号的设备配置'),
    ('user', '{input}')
])

chain = prompt | llm
message = chain.invoke({'input': '华三的防火墙增加一个新的防火墙策略'})
print(message)
```

#### 5.3 使用输出解析器

```python
import os
import dotenv
from langchain_core.output_parsers import JsonOutputParser

dotenv.load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='kimi-k2.5',api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是网络设备运维专家，熟悉各种厂商、型号的设备配置, 用JSON格式回复，问题用question，回答用answer'),
    ('user', '{input}')
])

output_parser = JsonOutputParser()

chain = prompt | llm | output_parser
message = chain.invoke({'input': '华三的防火墙增加一个新的防火墙策略'})
print(message)
```

#### 5.4 使用向量存储

使用简单的本地向量存储FAISS

安装FAISS

```shell
pip install faiss-cpu
pip install langchain_community
```

可能存在的坑 切片长度，一次能处理的切片数量

```python
import traceback
from http.client import responses

import dotenv, os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

from langchain_core.embeddings import Embeddings
from dashscope import MultiModalEmbedding
from langchain_community.vectorstores import FAISS
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


docs = process_text_file("E:/learn/AI-demo/langchain-demo/LangChain.md")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)
try:
    embedding_model = QwenEmbedding()
    vector = FAISS.from_documents(documents, embedding_model)
    mm = vector.similarity_search("langchain smith")
    print(mm)
except Exception as e:
    traceback.print_exc()

```



#### 5.5 RAG检索增强生成



```python
import traceback
from http.client import responses

import dotenv, os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

from langchain_core.embeddings import Embeddings
from dashscope import MultiModalEmbedding
from langchain_community.vectorstores import FAISS
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


docs = process_text_file("E:/learn/AI-demo/langchain-demo/LangChain.md")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)
try:
    embedding_model = QwenEmbedding()
    vector = FAISS.from_documents(documents, embedding_model)
    mm = vector.similarity_search("langchain smith")
    print(mm)
except Exception as e:
    traceback.print_exc()

```

#### 5.6 使用Agent

```python
import traceback
from http.client import responses

import dotenv, os
from langchain_classic.agents import AgentExecutor
from langchain_community.agent_toolkits import create_openapi_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from torch import hub

dotenv.load_dotenv()

from langchain_core.embeddings import Embeddings
from dashscope import MultiModalEmbedding
from langchain_community.vectorstores import FAISS
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


docs = process_text_file("E:/learn/AI-demo/langchain-demo/LangChain.md")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(docs)
try:
    embedding_model = QwenEmbedding()
    vector = FAISS.from_documents(documents, embedding_model)
    mm = vector.similarity_search("langchain smith")
    print(mm)
    question = 'LangGraph有什么功能？'
    retriver = vector.as_retriever(search_kwargs={'k': 3})
    docs = retriver.invoke(question)

    retriver_tool = create_retriever_tool(
        retriver,
        "CivilCodeRetriver",
        '搜索LangGraph有什么功能，关于LangGraph相关的任何问题，必须使用此工具'

    )
    llm = ChatOpenAI(model='kimi-k2.5', api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))
    tools = [retriver_tool]
    prompt = hub.pull('hwchase17/openai-functions-agent')
    agent = create_openapi_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.invoke(question)
except Exception as e:
    import pdb;pdb.set_trace()
    traceback.print_exc()

```

