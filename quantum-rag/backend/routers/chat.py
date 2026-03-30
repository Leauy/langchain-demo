"""Chat router for QA functionality."""
import json
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.models import get_db, Conversation, Message, Datasource
from backend.services import vectorstore_service, rerank_service, llm_service
from backend.config import settings

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request model."""
    question: str
    conversation_id: Optional[int] = None
    datasource_id: Optional[int] = None
    stream: bool = False


class SourceItem(BaseModel):
    """Source item model."""
    module: str
    sub_module: str
    content: str
    score: float


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    sources: List[SourceItem]
    conversation_id: int


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Process a chat question and return answer with sources.
    """
    # Get or create conversation
    if request.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == request.conversation_id
        ).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Create new conversation
        datasource_id = request.datasource_id
        if datasource_id:
            datasource = db.query(Datasource).filter(
                Datasource.id == datasource_id
            ).first()
            if not datasource:
                raise HTTPException(status_code=404, detail="Datasource not found")

        conversation = Conversation(
            title=request.question[:50] + "..." if len(request.question) > 50 else request.question,
            datasource_id=datasource_id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.question
    )
    db.add(user_message)
    db.commit()

    # Determine datasource
    datasource_id = request.datasource_id or conversation.datasource_id
    if not datasource_id:
        # Get first available datasource
        datasource = db.query(Datasource).filter(
            Datasource.status == "ready"
        ).first()
        if datasource:
            datasource_id = datasource.id

    if not datasource_id:
        raise HTTPException(status_code=400, detail="No datasource available")

    # Step 1: Retrieve documents from vector store
    documents = vectorstore_service.search(
        datasource_id=datasource_id,
        query=request.question,
        top_k=settings.RETRIEVAL_TOP_K
    )

    # Step 2: Rerank documents
    reranked_docs = await rerank_service.rerank(
        query=request.question,
        documents=documents,
        top_k=settings.RERANK_TOP_K
    )

    # Step 3: Build context
    context_parts = []
    for i, doc in enumerate(reranked_docs, 1):
        context_parts.append(
            f"[文档{i}] 模块: {doc.get('module', '')}, 子模块: {doc.get('sub_module', '')}\n"
            f"内容: {doc.get('content', '')}"
        )
    context = "\n\n".join(context_parts)

    # Step 4: Generate answer
    if request.stream:
        return StreamingResponse(
            stream_chat_response(request.question, context, reranked_docs, conversation.id, db),
            media_type="text/event-stream"
        )

    answer = llm_service.generate(
        question=request.question,
        context=context
    )

    # Step 5: Save assistant message
    sources_json = json.dumps([{
        "module": doc.get("module", ""),
        "sub_module": doc.get("sub_module", ""),
        "content": doc.get("content", "")[:500],  # Truncate for storage
        "score": doc.get("score", 0.0)
    } for doc in reranked_docs], ensure_ascii=False)

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        sources=sources_json
    )
    db.add(assistant_message)
    db.commit()

    # Build response
    sources = [
        SourceItem(
            module=doc.get("module", ""),
            sub_module=doc.get("sub_module", ""),
            content=doc.get("content", ""),
            score=doc.get("score", 0.0)
        )
        for doc in reranked_docs
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        conversation_id=conversation.id
    )


async def stream_chat_response(question: str, context: str, sources: list, conversation_id: int, db: Session):
    """Stream chat response."""
    full_answer = ""
    for chunk in llm_service.generate_stream(question, context):
        full_answer += chunk
        yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"

    # Save message after streaming complete
    sources_json = json.dumps([{
        "module": doc.get("module", ""),
        "sub_module": doc.get("sub_module", ""),
        "content": doc.get("content", "")[:500],
        "score": doc.get("score", 0.0)
    } for doc in sources], ensure_ascii=False)

    assistant_message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=full_answer,
        sources=sources_json
    )
    db.add(assistant_message)
    db.commit()

    yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"
