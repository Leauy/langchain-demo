"""Configuration management for quantum-rag backend."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # DashScope API Configuration
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL: str = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "qwen3-vl-rerank")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "qwen3.5-flash")

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/quantum_rag.db")

    # Vector Store Configuration
    VECTOR_INDEX_PATH: str = os.getenv("VECTOR_INDEX_PATH", "./data/vector_index")

    # RAG Configuration
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))


settings = Settings()
