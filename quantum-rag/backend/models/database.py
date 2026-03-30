"""SQLite database models for quantum-rag."""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import enum

from backend.config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DatasourceStatus(str, enum.Enum):
    """Status of a datasource."""
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


class Conversation(Base):
    """Conversation model for storing chat history."""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), default="新对话")
    datasource_id = Column(Integer, ForeignKey("datasources.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    datasource = relationship("Datasource", back_populates="conversations")


class Message(Base):
    """Message model for storing individual chat messages."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON string of sources
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class Datasource(Base):
    """Datasource model for storing document sources."""
    __tablename__ = "datasources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    file_path = Column(String(512), nullable=True)
    file_type = Column(String(50), nullable=True)  # excel, pdf, txt, markdown
    status = Column(String(20), default=DatasourceStatus.PROCESSING.value)
    document_count = Column(Integer, default=0)
    vector_dimension = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    conversations = relationship("Conversation", back_populates="datasource")
    documents = relationship("Document", back_populates="datasource", cascade="all, delete-orphan")


class Document(Base):
    """Document model for storing individual document chunks."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    datasource_id = Column(Integer, ForeignKey("datasources.id"), nullable=False)
    module = Column(String(255), nullable=True)
    sub_module = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    doc_id = Column(String(255), nullable=False)  # FAISS document ID
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    datasource = relationship("Datasource", back_populates="documents")


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
