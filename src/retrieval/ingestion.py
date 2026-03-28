from typing import List, Dict
from src.retrieval.vector_store import VectorStore

class DocumentIngestionPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Simple sliding window chunker. For publication, enhance with semantic chunking."""
        chunks = []
        for i in range(0, max(1, len(text)), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return list(set(chunks)) # Naive deduplication

    def ingest_text(self, text: str, source_name: str, topic: str = "general"):
        """Ingest text into the vector store with metadata."""
        chunks = self.chunk_document(text)
        doc_ids = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source_name": source_name,
                "chunk_id": f"{source_name}_chunk_{i}",
                "topic": topic
            }
            doc_id = self.vector_store.add_document(chunk, metadata)
            doc_ids.append(doc_id)
        return doc_ids
