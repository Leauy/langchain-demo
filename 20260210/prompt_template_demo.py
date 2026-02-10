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

print(b.format(**{'name': 'bianque', 'role': 333}))

e= PromptTemplate.from_template(template='请去评价{product}的优缺点，包括{aspect1} 和 {aspect2} 角度',
                                partial_variables={'aspect1':'性能'})

print(e.format(product='电脑', aspect2='外观'))