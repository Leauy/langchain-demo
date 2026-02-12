"""
LangChain 提示词模板 Demo
"""
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage


def demo_basic_template():
    """基础提示词模板"""
    template = """
    你是一个{role}，请用{style}的风格回答以下问题：
    
    问题：{question}
    """
    
    prompt = PromptTemplate(
        input_variables=["role", "style", "question"],
        template=template
    )
    
    # 格式化提示词
    formatted = prompt.format(
        role="专业程序员",
        style="简洁明了",
        question="什么是Python的装饰器？"
    )
    print("=== 基础提示词模板 ===")
    print(formatted)
    print()


def demo_chat_template():
    """聊天提示词模板"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{domain}专家。"),
        ("human", "请解释{topic}的概念。"),
    ])
    
    messages = template.format_messages(
        domain="人工智能",
        topic="深度学习"
    )
    
    print("=== 聊天提示词模板 ===")
    for msg in messages:
        print(f"{msg.type}: {msg.content}")
    print()


def demo_few_shot_template():
    """Few-shot 提示词模板"""
    examples = [
        {
            "input": "今天天气很好",
            "output": "正面"
        },
        {
            "input": "这部电影太糟糕了",
            "output": "负面"
        },
        {
            "input": "快递明天到",
            "output": "中性"
        }
    ]
    
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="文本: {input}\n情感: {output}\n"
    )
    
    # 构建 few-shot 提示
    few_shot_prompt = ""
    for example in examples:
        few_shot_prompt += example_prompt.format(**example)
    
    few_shot_prompt += "文本: {input}\n情感:"
    
    final_prompt = PromptTemplate(
        input_variables=["input"],
        template=few_shot_prompt
    )
    
    print("=== Few-shot 提示词模板 ===")
    print(final_prompt.format(input="这个产品质量不错"))
    print()


def demo_partial_template():
    """部分参数化模板"""
    prompt = PromptTemplate(
        input_variables=["name", "age", "city"],
        template="我叫{name}，今年{age}岁，来自{city}。"
    )
    
    # 预设部分参数
    partial_prompt = prompt.partial(name="张三", city="北京")
    
    print("=== 部分参数化模板 ===")
    print(partial_prompt.format(age=25))
    print()


def demo_pipeline_template():
    """管道式提示词"""
    # 第一个模板：生成标题
    title_template = PromptTemplate(
        input_variables=["topic"],
        template="请为'{topic}'生成3个吸引人的标题："
    )
    
    # 第二个模板：生成大纲
    outline_template = PromptTemplate(
        input_variables=["title"],
        template="基于标题'{title}'，生成文章大纲："
    )
    
    # 使用管道
    topic = "人工智能的未来"
    title_prompt = title_template.format(topic=topic)
    print("=== 管道式提示词 ===")
    print(f"步骤1 - 生成标题提示:\n{title_prompt}\n")
    
    # 模拟获取标题（实际应该调用 LLM）
    selected_title = "AI革命：未来十年的变革"
    outline_prompt = outline_template.format(title=selected_title)
    print(f"步骤2 - 生成大纲提示:\n{outline_prompt}\n")


if __name__ == "__main__":
    print("🚀 LangChain 提示词模板 Demo\n")
    
    demo_basic_template()
    demo_chat_template()
    demo_few_shot_template()
    demo_partial_template()
    demo_pipeline_template()
    
    print("✅ Demo 完成！")
