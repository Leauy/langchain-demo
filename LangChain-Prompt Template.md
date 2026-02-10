# 第3章：LangChain-Prompt Template

Prompt Template，通过模板管理大模型的输入。

## 3.1、Prompt Template介绍与分类

Prompt Template是LangChain中的一个概念，接受用户输入，返回一个传递给LLM的信息（提示词）

在应用开发中，固定的提示词限制了模型的灵活性和适用范围。所以Prompt Template 是一个模板化的字符串，你可以将变量插入到模板中，从而创建出不同的提示词。调用时：

- 以字典作为输入，其中每个键代表要填充的提示模板中的变量
- 输出一个PromptValue。这个PromptValue可以传递给LLM或者ChatModel，并且还可以转化成字符串或者消息列表。



### 不同类型的提示模板

- PromptTemplate： LLM提示模板，用于生成字符串提示，它使用Python的字符串来模板提示。
- ChatPromptTemplate: 聊天提示模板，用于组合各种角色的消息模板，传入聊天模型
- XxxMessagePromptTemplate： 消息提示词模板，包括SystemMessagePromptTemplate，HumanMessagePromptTemplate，AIMessagePromptTemplate、ChatMessagePromptTemplate等
- FewShotPromptTemplate：样本提示词模板，通过示例来教模型如何回答
- PipelinePrompt：管道提示词模板，用于把几个提示词组合在一起使用
- 自定义模板：允许基于其他模板类来定制自己的提示词模板

## 3.2 复习：str.format()

Python的str.format()方法是一种字符串格式化的手段，允许在字符串中插入变量。使用这种方法。可以创建包含占位符的字符串模板，占位符用{}标识。

```python
a = 'name {} age {}'.format('liuyang', 12)
b = 'name {name} age {age}'.format(name='liuyang', age=12)
c = {'name': 'sb', 'age':33}
b.format(**c)
```

## 3.3 具体使用PromptTemplate

### 3.3.1 使用说明

PromptTemplate类，用于快速构建包含变量额提示词模板，并且通过传入不同的参数值生成自定义的提示词。

主要参数介绍

- template：定义提示词模板的字符串，其中包含文本和变量占位符如{name}
- input_variables: 列表，指定模板中使用的变量名称，在调用模板时被替换
- paritial_variables: 字典用于定义模板中一些固定的变量名。这些值不需要再每次调用时被替换，

### 3.3.2 两种实例化方法

#### PromptTemplate如何获取实例

两种方式，

- 采用构造方法

```python

# 参数必选 template, input_variables
a = PromptTemplate(template='你是一个{role}, 名称是{name}', input_variables=['role', 'name'])
print(a)
print(a.format(name='laowang', role='骨科专家'))
```

- from_template()  推荐

  ```python
  from langchain_core.prompts import PromptTemplate
  
  import os
  
  from langchain_openai import ChatOpenAI
  import dotenv
  
  dotenv.load_dotenv()
  
  
  # 参数必选 template, input_variables
  a = PromptTemplate(template='你是一个{role}, 名称是{name}', input_variables=['role', 'name'])
  print(a)
  print(a.format(name='laowang', role='骨科专家'))
  
  b= PromptTemplate.from_template(template='你是一个{role}, 名称是{name}')
  print(b)
  print(b.format(**{'name': 'bianque', 'role': 333}))
  
  mm = 'tell me a joke'
  print(PromptTemplate.from_template(mm).format())
  # for message in client.stream(
  #         '小拇指骨折了，已经7周了，目前骨折线还是很明显，该怎么办？医生开了一些药促进骨骼愈合的,目前有吃一些钙片，牛奶，鸡蛋，平时只有在敲键盘的时候偶尔会去 把支架拆掉，平时都是带着的，几乎没怎么用力'):
  #     print(message.content, end='', flush=True)
  
  ```

#### 两种特殊结构的使用

部分提示词模板的使用，组合提示词的使用

部分提示词模板的使用



#### 给变量赋值的两种方式：format / invoke

#### 结合大模型使用

