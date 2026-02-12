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

```python
from langchain_core.prompts import PromptTemplate

template = (
    PromptTemplate.from_template('tell me a joke about {topic}'
                                 ', make it funny '
                                 '\n and in {language}')
)

print(template.format(topic='basketball', language='english'))
```

部分提示词模板的使用

#### 给变量赋值的两种方式：format / invoke

invoke（）：传入的字典，返回ChatPromptValue

format（）：传入的变量的值，返回str
format_message(): 传入变量的值，返回消息构成的list



本质：不管是使用构造方法、还是使用from_message()来创建ChatPromptTemplate的实例，本质上来讲，传入的消息组成的列表。



从调用上来讲，我们看到，不管使用构造方法，还是from_message().参数类型是多样的。可以是字符串、字典、消息、元组构成的列表

#### 方式1： 元组列表

```python
m = ChatPromptTemplate.from_messages([
    ('system', '你是一个AI助手，你的名字叫{name}'),
    ('human', '我的问题是{question}'),
])
n = m.format_prompt(**{
    'name': 'sb',
    'question': '那个模型是最厉害的？'
})
print(n)
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())
print('#*30')
```



#### 方式2： 字符串列表

```python
m = ChatPromptTemplate.from_messages([
    '我的问题是{question}'
])
n = m.invoke({
    'name': 'question',
    'question': '那个模型是最厉害的？'
})
print(n)
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())
print('#*30')
```



#### 方式3： 字典列表

```python
m = ChatPromptTemplate.from_messages([
    {
        'role': 'system',
        'content': '你是一个AI助手，你的名字叫{name}'
    },
    {
        'role': 'human',
        'content': '我的问题是{question}'
    },
])
n = m.invoke({
    'name': 'SSBSBSB',
    'question': '那个模型是最厉害的？'
})
print(n)
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())
print('#*30')
```

#### 方式4：消息列表

```python

m = ChatPromptTemplate.from_messages([
    ('system', '你是一个AI助手，你的名字叫{name}'),
    HumanMessage(content='我的问题是{question}')
])
n = m.invoke({
    'name': 'SSBSBSB',
    'question': '那个模型是最厉害的？'
})
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())
print('#*30')

```



#### 方式5：Chat提示词模板类型

```python
nested_prompt = ChatPromptTemplate.from_messages([('system', '你是一个AI助手，你的名字叫{name}'),])
nested_prompt2 = ChatPromptTemplate.from_messages([('human', '我的问题是{question}')])
m = ChatPromptTemplate.from_messages([
    nested_prompt,
    nested_prompt2,
])
n = m.invoke({
    'name': 'SSBSB1233333SB',
    'question': '那个模型是最厉害的？'
})
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())
print('#*30')
```

#### 结合大模型使用

```python

nested_prompt = ChatPromptTemplate.from_messages([('system', '你是一个AI助手，你的名字叫{name}'),])
nested_prompt2 = ChatPromptTemplate.from_messages([('human', '我的问题是{question}')])
m = ChatPromptTemplate.from_messages([
    nested_prompt,
    nested_prompt2,
])
n = m.invoke({
    'name': 'SSBSB1233333SB',
    'question': '那个模型是最厉害的？对比下 当前 中国的几个编辑器，从性价比 ，用户评价 ，价格，对比下'
})
# 将ChatPromptValue转换成消息构成的list
print(n.to_messages())
print(n.to_string())


print('#*30')

import dotenv
dotenv.load_dotenv()

client = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_URL'),
                    model='qwen3-max-2026-01-23', streaming=True)


for message in client.stream(n):
    print(message.content, end='', flush=True)

```



#### 插入消息列表： MessagePlaceholder

当你不确定消息提示模板使用什么角色时候，希望在格式化过程中插入消息列表时，该怎么办？这就需要使用MessagePlaceholder，负责在特定位置添加消息列表。

使用场景：多轮对话系统存储历史消息以及Agent的中间步骤处理此功能非常有用。

举例1：

```python
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个靠谱的骨科专家主任医生'),
    MessagesPlaceholder('msg')
])

# [SystemMessage(content='你是一个靠谱的骨科专家主任医生', additional_kwargs={}, response_metadata={}), HumanMessage(content='我男性，31岁，小拇指骨折了7周多了，目前X光片子还是显示的骨折线特别明显，医生给开了一些药物，恒古固伤愈合剂，喝了以后有点头晕，口渴非常严重，医生让继续固定', additional_kwargs={}, response_metadata={})]
print(prompt_template.format_messages(msg = [HumanMessage(content='我男性，31岁，小拇指骨折了7周多了，目前X光片子还是显示的骨折线特别明显，医生给开了一些药物，恒古固伤愈合剂，喝了以后有点头晕，口渴非常严重，医生让继续固定')]))


```



举例3： 存储对话历史记录

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', '你是一个靠谱的骨科专家主任医生'),
        MessagesPlaceholder('history'),
        ('human', '{question}'),
    ]
)

m= prompt.format(
    history=[
        HumanMessage('我男性，31岁，小拇指骨折了7周多了，目前X光片子还是显示的骨折线特别明显，医生给开了一些药物，恒古固伤愈合剂，喝了以后有点头晕，口渴非常严重，医生让继续固定'),
        AIMessage(content='你好，感谢你提供详细的信息。作为骨科主任医生，我来为你分析一下目前的情况，并给出专业建议。\n\n---\n\n### 一、关于小拇指骨折7周后骨折线仍明显的问题\n\n**1. 小指（第五掌骨或指骨）骨折的愈合时间：**  \n一般而言，成人手指骨折（尤其是掌骨或近节指骨）在正常情况下 **4~6周** 内会有明显的骨痂形成，X光上骨折线会逐渐模糊。但个体差异较大，部分人可能需要 **8~10周甚至更久** 才能完全愈合，尤其是以下情况：\n\n- 骨折类型复杂（如粉碎性、斜形、有移位未完全复位）\n- 血供较差（小指末节血供相对较差）\n- 吸烟、营养不良、糖尿病等影响愈合的因素\n- 固定不充分或过早活动\n\n**2. X光片上“骨折线明显”不一定代表没愈合：**  \n有时候即使临床上已有骨痂形成、稳定性良好，X光片上仍可见清晰的骨折线，这可能是由于：\n- 骨痂密度低，X光不易显影（早期骨痂为软骨样组织）\n- 拍片角度问题\n- 骨折端对位良好但尚未完全重塑\n\n✅ 建议：结合**临床症状**判断——是否还有明显压痛？是否能轻微活动而不剧痛？如果疼痛已明显减轻、无异常活动，即使X光线仍可见，也可能处于愈合过程中。\n\n---\n\n###二、关于“恒古固伤愈合剂”引起的副作用\n\n你提到服药后出现**头晕、严重口渴**，这确实值得重视。\n\n**1. 恒古固伤愈合剂** 是一种中成药，常用于促进骨折愈合，成分多含活血化瘀、接骨续筋的中药（如三七、骨碎补、自然铜等）。虽然总体安全性较好，但**个体差异大**，部分人可能出现：\n\n- 口干、口渴（可能与某些温燥药物有关）\n- 头晕（可能与血压波动、脱水或药物成分影响中枢有关）\n- 胃肠不适、上火等\n\n**2. 严重口渴 + 头晕 ≠ 正常反应！**  \n尤其“非常严重”的口渴，需警惕是否：\n- 药物导致轻度脱水或电解质紊乱\n- 掩盖了其他问题（如高血糖？虽然你年轻，但不能完全排除）\n- 对某些成分过敏或不耐受\n\n✅ 建议：\n- **暂停该药1~2天**，观察头晕和口渴是否缓解；\n- 多饮水（但不要过量），监测是否有其他症状（如心慌、乏力、尿多等）；\n- 如停药后症状明显改善，可考虑换用其他促进骨折愈合的方式（如钙+维生素D3、物理治疗等）；\n- **务必告知主治医生这些副作用**，不要自行长期服用引起不适的药物。\n\n---\n\n### 三、关于继续固定的问题\n\n医生建议继续固定是合理的，尤其如果：\n- 骨折尚未达到临床愈合标准（仍有压痛、异常活动）\n- X光确实显示骨痂少、骨折线清晰且无硬化迹象\n\n但也要注意：\n- **过度固定**（超过8~10周）可能导致关节僵硬、肌肉萎缩，尤其手指关节活动度恢复困难；\n- 建议在医生指导下，**逐步开始保护性功能锻炼**（如邻近关节的主动活动），避免“僵住”。\n\n✅ 建议下一步：\n1. **复查X光（正侧位）**，最好与上次片子对比，看是否有骨痂进展；\n2. **评估临床愈合指标**：局部无压痛、无纵向叩击痛、可轻微抗阻活动；\n3. **如仍无愈合迹象**（>10周无骨痂），需考虑“延迟愈合”甚至“骨不连”可能，进一步检查（如CT、骨代谢指标）；\n4. **调整药物**：若恒古固伤愈合剂副作用明显，可改用更温和的方案（如碳酸钙D3、氨基葡萄糖、必要时低强度脉冲超声等辅助治疗）。\n\n---\n\n### 总结建议：\n\n| 项目 | 建议 |\n|------|------|\n| 药物 | 暂停恒古固伤愈合剂，观察副作用是否缓解；咨询医生是否更换 |\n| 固定 | 继续固定合理，但需评估是否可开始部分功能锻炼 |\n| 复查 | 建议1~2周内复查X光 + 临床查体，判断愈合进展 |\n| 营养 | 保证高蛋白、钙、维C、维D摄入，戒烟戒酒 |\n| 警惕 | 若持续无愈合迹象（>10周），需排查延迟愈合原因 |\n\n---\n\n如有条件，建议到**手外科或骨科专科门诊**进一步评估，必要时做**CT三维重建**，比X光更敏感地判断骨痂形成情况。\n\n如果你方便，也可以告诉我：\n- 具体是哪一节骨折（末节？中节？掌骨？）\n- 是否有移位？当初是否手法复位？\n- 目前手指能否轻微弯曲？有无肿胀？\n\n我可以帮你更精准判断。\n\n祝你早日康复！')
    ],
    question='斜形、近掌骨，没有明显压痛，能轻微活动而不剧痛，喝完药严重口渴 + 头晕 大约持续2-3小时后 会缓解。药物开了一周了，上次是7周以后做的检查，这次复查建议 什么时间去？'
)
print(m)

client = ChatOpenAI(api_key=os.getenv('DASHSCOPE_API_KEY'), base_url=os.getenv('DASHSCOPE_BASE_URL'), model='qwen3-max-2026-01-23')

print(client.invoke(m).content)
```

### 3.5 具体使用

#### 3.5.1 使用说明

在构建prompt时，可以通过构建一个少量示例列表去进行格式化prompt，这是一种简单但是强大的指导生成的方式，在某些情况下可以显著提高模型性能。

少量示例提示模板可以由一组示例或者一个负责从定义的集合中选择一部分示例的示例选择器构建。

- 前者：使用FewShotPromptTemplate 或者FewShotChatMessagePromptTemplate
- 后者：使用Example selectors（示例选择器）

每一个示例的结构都是一个字典，其中键时输入变量，值时输入变量的值

体会：zeroshot会导致低质量回答

FewShotPromptTemplate  的使用

```python
import os

import  dotenv
from langchain_core.prompts import FewShotPromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI


dotenv.load_dotenv()


chat_model = ChatOpenAI(api_key=os.getenv('DASHSCOPE_API_KEY'), base_url=os.getenv('DASHSCOPE_BASE_URL'),  model='qwen3-max-2026-01-23')

prompt = PromptTemplate.from_template(
    template='输入如下：{input}, 输出如下：{output}',
)

examples = [
    {
        'input': '北京天气如何？', 'output': '北京',
    },
    {
        'input': '南京下雨吗？', 'output': '南京',
    },
    {
         'input': '武汉热吗？', 'output': '武汉',
    }
]

f = FewShotPromptTemplate(
    example_prompt=prompt,
    examples=examples,
    input_variables=['input'],
    suffix='输入如下：{input}, 输出如下：',
)

print(f.invoke({'input': '西安刮风不'}))

print(chat_model.invoke(f.invoke({'input': '西安刮风不'})))
print(chat_model.invoke(f.invoke({'input': '西安刮风不'})).content)

#text='输入如下：北京天气如何？, 输出如下：北京\n\n输入如下：南京下雨吗？, 输出如下：南京\n\n输入如下：武汉热吗？, 输出如下：武汉\n\n输入如下：西安刮风不, 输出如下：'
#content='西安' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 1, 'prompt_tokens': 58, 'total_tokens': 59, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'qwen3-max-2026-01-23', 'system_fingerprint': None, 'id': 'chatcmpl-ccb97c65-5eeb-9bd2-9ed0-a06809c08453', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019c529e-c8c2-7812-85d8-388d5fed89ad-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 58, 'output_tokens': 1, 'total_tokens': 59, 'input_token_details': {}, 'output_token_details': {}}
#西安
```





