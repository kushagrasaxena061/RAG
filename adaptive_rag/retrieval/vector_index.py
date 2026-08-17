import chromadb
from typing import List, Dict, Any
from adaptive_rag.models.schema import Chunk

class VectorIndex:
    """Persistent dense vector search using ChromaDB and local embeddings."""
    def __init__(self, persist_directory: str = "./data/chroma", collection_name: str = "adaptive_rag_chunks"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks: return
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        # Store metadata for filtering (Document ID, Type, etc.)
        metadatas = [{
            "document_id": c.metadata.document_id,
            "content_type": c.metadata.content_type.value,
            "page_number": c.metadata.page_number
        } for c in chunks]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, top_k: int = 5, where: Dict[str, Any] = None) -> List[dict]:
        """Returns ordered candidates by semantic distance."""
        results = self.collection.query(query_texts=[query], n_results=top_k, where=where)
        if not results['ids'] or not results['ids'][0]: return []
        
        formatted_results = []
        for idx, doc_id in enumerate(results['ids'][0]):
            formatted_results.append({
                "chunk_id": doc_id,
                "distance": results['distances'][0][idx] if 'distances' in results and results['distances'] else 0.0
            })
        return formatted_results
