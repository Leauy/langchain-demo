"""Embedding service using Alibaba DashScope text-embedding-v3."""
from typing import List
from langchain_openai import OpenAIEmbeddings

from backend.config import settings


class EmbeddingService:
    """Service for generating text embeddings using DashScope API."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.DASHSCOPE_API_KEY,
            openai_api_base=settings.DASHSCOPE_BASE_URL
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self.embeddings.embed_documents(texts)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return 1024  # text-embedding-v3 default dimension


# Singleton instance
embedding_service = EmbeddingService()
