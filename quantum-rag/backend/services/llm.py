"""LLM service using Alibaba DashScope qwen3.5-flash."""
from typing import List, Iterator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import settings


class LLMService:
    """Service for LLM interactions using DashScope API."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
            temperature=0.7
        )

    def generate(
        self,
        question: str,
        context: str,
        sources_info: List[dict] = None
    ) -> str:
        """Generate answer based on question and context."""
        system_prompt = """你是一个专业的量子网络设备知识库助手。请根据提供的上下文信息回答用户问题。

要求：
1. 回答要准确、专业、简洁
2. 如果上下文中没有相关信息，请诚实告知
3. 在回答中引用相关的来源模块
4. 使用中文回答"""

        user_prompt = f"""上下文信息：
{context}

用户问题：{question}

请根据以上上下文信息回答用户问题："""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        return response.content

    def generate_stream(
        self,
        question: str,
        context: str
    ) -> Iterator[str]:
        """Stream generate answer."""
        system_prompt = """你是一个专业的量子网络设备知识库助手。请根据提供的上下文信息回答用户问题。

要求：
1. 回答要准确、专业、简洁
2. 如果上下文中没有相关信息，请诚实告知
3. 在回答中引用相关的来源模块
4. 使用中文回答"""

        user_prompt = f"""上下文信息：
{context}

用户问题：{question}

请根据以上上下文信息回答用户问题："""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content


# Singleton instance
llm_service = LLMService()
