"""History router for conversation management."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.models import get_db, Conversation, Message

router = APIRouter(prefix="/api/history", tags=["history"])


class MessageItem(BaseModel):
    """Message item model."""
    id: int
    role: str
    content: str
    sources: str = None
    created_at: str

    class Config:
        from_attributes = True


class ConversationSummary(BaseModel):
    """Conversation summary model."""
    id: int
    title: str
    datasource_id: int = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class ConversationDetail(BaseModel):
    """Conversation detail model."""
    id: int
    title: str
    datasource_id: int = None
    messages: List[MessageItem]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


@router.get("", response_model=List[ConversationSummary])
def get_conversations(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get list of conversations."""
    conversations = db.query(Conversation).order_by(
        Conversation.updated_at.desc()
    ).offset(skip).limit(limit).all()

    return [
        ConversationSummary(
            id=c.id,
            title=c.title,
            datasource_id=c.datasource_id,
            created_at=c.created_at.isoformat(),
            updated_at=c.updated_at.isoformat()
        )
        for c in conversations
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Get a single conversation with all messages."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at).all()

    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        datasource_id=conversation.datasource_id,
        messages=[
            MessageItem(
                id=m.id,
                role=m.role,
                content=m.content,
                sources=m.sources,
                created_at=m.created_at.isoformat()
            )
            for m in messages
        ],
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat()
    )


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Delete a conversation."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conversation)
    db.commit()

    return {"message": "Conversation deleted successfully"}
