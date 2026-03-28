import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import hashlib

class VectorStore:
    def __init__(self, db_path: str, embedding_model: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model, device="cuda"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_func
        )

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        doc_id = hashlib.md5(content.encode()).hexdigest()
        if not self.collection.get(ids=[doc_id])['ids']:
            self.collection.add(documents=[content], metadatas=[metadata or {}], ids=[doc_id])
            return doc_id
        return doc_id

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        retrieved_docs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "id": results['ids'][0][i]
                })
        return retrieved_docs
