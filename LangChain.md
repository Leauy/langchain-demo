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

LangChain更为方便解决浙西问题而生的。比如：大模型默认不能联网，如果需要联网，用langchain。

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



