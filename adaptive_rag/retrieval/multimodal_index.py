import io
import os
import torch
from typing import List, Dict, Any, Optional
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
from adaptive_rag.models.schema import Chunk

class MultimodalVisualIndex:
    """
    Cross-modal vector search indexing image embeddings and text queries 
    into a shared latent space via CLIP (Contrastive Language-Image Pre-training).
    """
    def __init__(
        self,
        persist_directory: str = "./data/chroma_multimodal",
        collection_name: str = "visual_chunks",
        model_name: str = "clip-ViT-B-32"
    ):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = SentenceTransformer(model_name, device=self.device)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.image_store: Dict[str, bytes] = {}

    def add_figure_chunks(self, chunks: List[Chunk], raw_images: Dict[str, bytes]):
        """Encodes PIL Images with CLIP Vision Transformer and stores in ChromaDB."""
        if not chunks:
            return

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            cid = chunk.chunk_id
            img_bytes = raw_images.get(cid)
            if img_bytes:
                try:
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    # Visual embedding via CLIP image encoder
                    img_emb = self.model.encode(pil_img, convert_to_numpy=True).tolist()
                    
                    ids.append(cid)
                    embeddings.append(img_emb)
                    documents.append(chunk.content)
                    metadatas.append({
                        "document_name": chunk.metadata.document_name,
                        "page_number": chunk.metadata.page_number,
                        "content_type": chunk.metadata.content_type.value,
                        "section_title": chunk.metadata.section_title or "Visual"
                    })
                    self.image_store[cid] = img_bytes
                except Exception as e:
                    print(f"[Warning] Failed to visually embed image chunk {cid}: {e}")

        if ids:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def search_visual(self, query: str, top_k: int = 3, target_document: str = "All Documents") -> List[Dict[str, Any]]:
        """Encodes text query via CLIP text encoder and queries the visual vector index."""
        if self.collection.count() == 0:
            return []

        # Text embedding in CLIP space
        query_emb = self.model.encode(query, convert_to_numpy=True).tolist()
        
        where_filter = None
        if target_document != "All Documents":
            where_filter = {"document_name": target_document}

        try:
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=min(top_k, self.collection.count()),
                where=where_filter
            )
        except Exception:
            return []

        if not results['ids'] or not results['ids'][0]:
            return []

        visual_matches = []
        for idx, doc_id in enumerate(results['ids'][0]):
            visual_matches.append({
                "chunk_id": doc_id,
                "content": results['documents'][0][idx],
                "metadata": results['metadatas'][0][idx],
                "distance": results['distances'][0][idx] if 'distances' in results and results['distances'] else 0.0
            })
        return visual_matches
