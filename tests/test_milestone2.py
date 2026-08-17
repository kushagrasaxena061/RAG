import pytest
import time
from adaptive_rag.models.schema import Chunk, ChunkMetadata, ChunkType, ContentType
from adaptive_rag.retrieval.vector_index import VectorIndex
from adaptive_rag.retrieval.bm25_index import BM25Index
from adaptive_rag.retrieval.hybrid_search import HybridRetriever

@pytest.fixture
def sample_chunks():
    base_meta = ChunkMetadata(
        document_id="doc_123", document_name="test.pdf", page_number=1, created_at=time.time()
    )
    
    chunk1 = Chunk(
        chunk_id="c_1", chunk_type=ChunkType.CHILD, content="The company saw massive revenue growth due to AI scaling.",
        token_count=10, content_hash="hash1", metadata=base_meta.model_copy()
    )
    
    chunk2 = Chunk(
        chunk_id="c_2", chunk_type=ChunkType.CHILD, content="Account identifier XYL-9942 was flagged for review.",
        token_count=10, content_hash="hash2", metadata=base_meta.model_copy()
    )
    
    chunk3 = Chunk(
        chunk_id="c_3", chunk_type=ChunkType.CHILD, content="The server rack temperature must not exceed 180°C.",
        token_count=10, content_hash="hash3", metadata=base_meta.model_copy()
    )
    return [chunk1, chunk2, chunk3]

def test_hybrid_search_fusion(sample_chunks, tmp_path):
    vector_dir = str(tmp_path / "chroma")
    bm25_path = str(tmp_path / "bm25.pkl")
    
    vec_idx = VectorIndex(persist_directory=vector_dir)
    bm25_idx = BM25Index(persist_path=bm25_path)
    
    vec_idx.add_chunks(sample_chunks)
    bm25_idx.add_chunks(sample_chunks)
    
    retriever = HybridRetriever(vec_idx, bm25_idx)
    
    # 1. Semantic query should favor chunk 1
    res_semantic = retriever.search("financial profits and scaling", top_k=1)
    assert res_semantic[0].chunk_id == "c_1"
    
    # 2. Exact lexical match should perfectly hit chunk 2 (BM25 power)
    res_lexical = retriever.search("XYL-9942", top_k=1)
    assert res_lexical[0].chunk_id == "c_2"
