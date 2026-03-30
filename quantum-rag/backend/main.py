"""FastAPI main application for quantum-rag."""
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import init_db
from backend.routers import chat_router, history_router, datasource_router

# Create FastAPI app
app = FastAPI(
    title="Quantum RAG API",
    description="量子网络设备知识库问答系统 API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(history_router)
app.include_router(datasource_router)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/vector_index", exist_ok=True)

    # Initialize database
    init_db()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Quantum RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
