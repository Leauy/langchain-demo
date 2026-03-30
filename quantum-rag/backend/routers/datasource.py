"""Datasource router for document management."""
import os
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.models import get_db, Datasource, DatasourceStatus
from backend.services import vectorstore_service
from backend.services.document_loader import document_loader
from backend.config import settings

router = APIRouter(prefix="/api/datasource", tags=["datasource"])

# File upload directory
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class DatasourceCreate(BaseModel):
    """Datasource creation model."""
    name: str
    description: Optional[str] = None


class DatasourceUpdate(BaseModel):
    """Datasource update model."""
    name: Optional[str] = None
    description: Optional[str] = None


class DatasourceResponse(BaseModel):
    """Datasource response model."""
    id: int
    name: str
    description: str = None
    file_type: str = None
    status: str
    document_count: int
    vector_dimension: int = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


@router.get("", response_model=List[DatasourceResponse])
def get_datasources(
    db: Session = Depends(get_db)
):
    """Get list of all datasources."""
    datasources = db.query(Datasource).order_by(Datasource.created_at.desc()).all()

    return [
        DatasourceResponse(
            id=d.id,
            name=d.name,
            description=d.description,
            file_type=d.file_type,
            status=d.status,
            document_count=d.document_count,
            vector_dimension=d.vector_dimension,
            created_at=d.created_at.isoformat(),
            updated_at=d.updated_at.isoformat()
        )
        for d in datasources
    ]


@router.get("/{datasource_id}", response_model=DatasourceResponse)
def get_datasource(
    datasource_id: int,
    db: Session = Depends(get_db)
):
    """Get a single datasource."""
    datasource = db.query(Datasource).filter(
        Datasource.id == datasource_id
    ).first()

    if not datasource:
        raise HTTPException(status_code=404, detail="Datasource not found")

    return DatasourceResponse(
        id=datasource.id,
        name=datasource.name,
        description=datasource.description,
        file_type=datasource.file_type,
        status=datasource.status,
        document_count=datasource.document_count,
        vector_dimension=datasource.vector_dimension,
        created_at=datasource.created_at.isoformat(),
        updated_at=datasource.updated_at.isoformat()
    )


@router.post("", response_model=DatasourceResponse)
async def create_datasource(
    name: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Create a new datasource with file upload."""
    # Determine file type
    filename = file.filename.lower()
    if filename.endswith((".xlsx", ".xls")):
        file_type = "excel"
    elif filename.endswith(".pdf"):
        file_type = "pdf"
    elif filename.endswith(".md"):
        file_type = "markdown"
    elif filename.endswith(".txt"):
        file_type = "txt"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Save file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Create datasource record
    datasource = Datasource(
        name=name,
        description=description,
        file_path=file_path,
        file_type=file_type,
        status=DatasourceStatus.PROCESSING.value
    )
    db.add(datasource)
    db.commit()
    db.refresh(datasource)

    # Process file in background (simplified - should be async task in production)
    try:
        # Load documents
        documents = document_loader.load_file(file_path, file_type)

        # Create vector store
        success = vectorstore_service.create_store(datasource.id, documents)

        if success:
            datasource.status = DatasourceStatus.READY.value
            datasource.document_count = len(documents)
            datasource.vector_dimension = settings.EMBEDDING_MODEL and 1024 or None
        else:
            datasource.status = DatasourceStatus.ERROR.value
    except Exception as e:
        datasource.status = DatasourceStatus.ERROR.value
        print(f"Error processing datasource: {e}")

    db.commit()
    db.refresh(datasource)

    return DatasourceResponse(
        id=datasource.id,
        name=datasource.name,
        description=datasource.description,
        file_type=datasource.file_type,
        status=datasource.status,
        document_count=datasource.document_count,
        vector_dimension=datasource.vector_dimension,
        created_at=datasource.created_at.isoformat(),
        updated_at=datasource.updated_at.isoformat()
    )


@router.put("/{datasource_id}", response_model=DatasourceResponse)
def update_datasource(
    datasource_id: int,
    update_data: DatasourceUpdate,
    db: Session = Depends(get_db)
):
    """Update datasource metadata."""
    datasource = db.query(Datasource).filter(
        Datasource.id == datasource_id
    ).first()

    if not datasource:
        raise HTTPException(status_code=404, detail="Datasource not found")

    if update_data.name is not None:
        datasource.name = update_data.name
    if update_data.description is not None:
        datasource.description = update_data.description

    db.commit()
    db.refresh(datasource)

    return DatasourceResponse(
        id=datasource.id,
        name=datasource.name,
        description=datasource.description,
        file_type=datasource.file_type,
        status=datasource.status,
        document_count=datasource.document_count,
        vector_dimension=datasource.vector_dimension,
        created_at=datasource.created_at.isoformat(),
        updated_at=datasource.updated_at.isoformat()
    )


@router.delete("/{datasource_id}")
def delete_datasource(
    datasource_id: int,
    db: Session = Depends(get_db)
):
    """Delete a datasource and its vector index."""
    datasource = db.query(Datasource).filter(
        Datasource.id == datasource_id
    ).first()

    if not datasource:
        raise HTTPException(status_code=404, detail="Datasource not found")

    # Delete vector store
    vectorstore_service.delete_store(datasource_id)

    # Delete file
    if datasource.file_path and os.path.exists(datasource.file_path):
        os.remove(datasource.file_path)

    # Delete database record
    db.delete(datasource)
    db.commit()

    return {"message": "Datasource deleted successfully"}


@router.post("/{datasource_id}/reindex", response_model=DatasourceResponse)
async def reindex_datasource(
    datasource_id: int,
    db: Session = Depends(get_db)
):
    """Reindex a datasource's documents."""
    datasource = db.query(Datasource).filter(
        Datasource.id == datasource_id
    ).first()

    if not datasource:
        raise HTTPException(status_code=404, detail="Datasource not found")

    if not datasource.file_path or not os.path.exists(datasource.file_path):
        raise HTTPException(status_code=400, detail="Source file not found")

    datasource.status = DatasourceStatus.PROCESSING.value
    db.commit()

    try:
        # Load documents
        documents = document_loader.load_file(
            datasource.file_path,
            datasource.file_type
        )

        # Delete old index and create new one
        vectorstore_service.delete_store(datasource_id)
        success = vectorstore_service.create_store(datasource_id, documents)

        if success:
            datasource.status = DatasourceStatus.READY.value
            datasource.document_count = len(documents)
        else:
            datasource.status = DatasourceStatus.ERROR.value
    except Exception as e:
        datasource.status = DatasourceStatus.ERROR.value
        print(f"Error reindexing datasource: {e}")

    db.commit()
    db.refresh(datasource)

    return DatasourceResponse(
        id=datasource.id,
        name=datasource.name,
        description=datasource.description,
        file_type=datasource.file_type,
        status=datasource.status,
        document_count=datasource.document_count,
        vector_dimension=datasource.vector_dimension,
        created_at=datasource.created_at.isoformat(),
        updated_at=datasource.updated_at.isoformat()
    )
