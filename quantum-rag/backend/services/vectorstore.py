"""Vector store service using FAISS."""
import os
import json
import pickle
from typing import List, Dict, Optional
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.config import settings
from backend.services.embedding import embedding_service


class VectorStoreService:
    """Service for managing FAISS vector store."""

    def __init__(self):
        self.index_path = Path(settings.VECTOR_INDEX_PATH)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._stores: Dict[int, FAISS] = {}  # datasource_id -> FAISS store
        self._doc_metadata: Dict[int, Dict[str, Dict]] = {}  # datasource_id -> {doc_id: metadata}

    def _get_store_path(self, datasource_id: int) -> Path:
        """Get the path for a datasource's vector store."""
        return self.index_path / f"datasource_{datasource_id}"

    def _get_metadata_path(self, datasource_id: int) -> Path:
        """Get the path for a datasource's metadata."""
        return self.index_path / f"datasource_{datasource_id}_metadata.pkl"

    def create_store(
        self,
        datasource_id: int,
        documents: List[Document]
    ) -> bool:
        """
        Create a new vector store for a datasource.

        Args:
            datasource_id: The datasource ID
            documents: List of LangChain Document objects

        Returns:
            True if successful
        """
        if not documents:
            return False

        try:
            # Create FAISS store
            store = FAISS.from_documents(
                documents,
                embedding_service.embeddings
            )

            # Save store to disk
            store_path = self._get_store_path(datasource_id)
            store.save_local(str(store_path))

            # Save metadata
            metadata = {}
            for i, doc in enumerate(documents):
                doc_id = f"doc_{i}"
                metadata[doc_id] = {
                    "module": doc.metadata.get("module", ""),
                    "sub_module": doc.metadata.get("sub_module", ""),
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "")
                }

            metadata_path = self._get_metadata_path(datasource_id)
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            # Cache in memory
            self._stores[datasource_id] = store
            self._doc_metadata[datasource_id] = metadata

            return True
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False

    def load_store(self, datasource_id: int) -> Optional[FAISS]:
        """Load a vector store for a datasource."""
        if datasource_id in self._stores:
            return self._stores[datasource_id]

        store_path = self._get_store_path(datasource_id)
        if not store_path.exists():
            return None

        try:
            store = FAISS.load_local(
                str(store_path),
                embedding_service.embeddings,
                allow_dangerous_deserialization=True
            )

            # Load metadata
            metadata_path = self._get_metadata_path(datasource_id)
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    self._doc_metadata[datasource_id] = pickle.load(f)

            self._stores[datasource_id] = store
            return store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None

    def search(
        self,
        datasource_id: int,
        query: str,
        top_k: int = None
    ) -> List[Dict]:
        """
        Search for relevant documents.

        Args:
            datasource_id: The datasource ID
            query: Search query
            top_k: Number of results

        Returns:
            List of documents with content and metadata
        """
        if top_k is None:
            top_k = settings.RETRIEVAL_TOP_K

        store = self.load_store(datasource_id)
        if not store:
            return []

        try:
            # Search in FAISS
            docs_and_scores = store.similarity_search_with_score(query, k=top_k)

            results = []
            metadata = self._doc_metadata.get(datasource_id, {})

            for doc, score in docs_and_scores:
                result = {
                    "content": doc.page_content,
                    "module": doc.metadata.get("module", ""),
                    "sub_module": doc.metadata.get("sub_module", ""),
                    "score": float(score),
                    "source": doc.metadata.get("source", "")
                }
                results.append(result)

            return results
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []

    def delete_store(self, datasource_id: int) -> bool:
        """Delete a datasource's vector store."""
        # Remove from memory
        if datasource_id in self._stores:
            del self._stores[datasource_id]
        if datasource_id in self._doc_metadata:
            del self._doc_metadata[datasource_id]

        # Remove from disk
        store_path = self._get_store_path(datasource_id)
        metadata_path = self._get_metadata_path(datasource_id)

        try:
            if store_path.exists():
                import shutil
                shutil.rmtree(store_path)
            if metadata_path.exists():
                metadata_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting vector store: {e}")
            return False

    def get_document_count(self, datasource_id: int) -> int:
        """Get the number of documents in a datasource's store."""
        store = self.load_store(datasource_id)
        if store:
            return store.index.ntotal
        return 0


# Singleton instance
vectorstore_service = VectorStoreService()
