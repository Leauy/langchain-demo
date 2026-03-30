"""Rerank service using Alibaba DashScope qwen3-vl-rerank."""
import httpx
from typing import List, Dict
from backend.config import settings


class RerankService:
    """Service for reranking documents using DashScope Rerank API."""

    def __init__(self):
        self.api_key = settings.DASHSCOPE_API_KEY
        self.base_url = settings.DASHSCOPE_BASE_URL
        self.model = settings.RERANK_MODEL

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None
    ) -> List[Dict]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of documents with 'content' field
            top_k: Number of top documents to return

        Returns:
            List of documents sorted by relevance score
        """
        if top_k is None:
            top_k = settings.RERANK_TOP_K

        if not documents:
            return []

        # Prepare documents for reranking
        doc_texts = [doc.get("content", "") for doc in documents]

        # Call DashScope Rerank API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": doc_texts,
                    "top_n": top_k
                },
                timeout=30.0
            )

            if response.status_code != 200:
                # If rerank fails, return original documents (fallback)
                return documents[:top_k]

            result = response.json()

        # Parse rerank results
        reranked_docs = []
        if "results" in result:
            for item in result["results"]:
                index = item.get("index", 0)
                relevance_score = item.get("relevance_score", 0.0)
                if index < len(documents):
                    doc = documents[index].copy()
                    doc["score"] = relevance_score
                    reranked_docs.append(doc)

        return reranked_docs[:top_k]


# Singleton instance
rerank_service = RerankService()
