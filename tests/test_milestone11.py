import pytest
import time
from adaptive_rag.models.schema import Chunk, ChunkMetadata, ChunkType, ContentType
from adaptive_rag.reasoning.contradiction import ContradictionDetector
from adaptive_rag.reasoning.temporal import TemporalReasoningEngine

@pytest.fixture
def multi_version_chunks():
    meta_v1 = ChunkMetadata(
        document_id="doc_1_v1", document_name="annual_report.pdf", version="1.0",
        page_number=5, created_at=time.time(), content_type=ContentType.TEXT
    )
    meta_v2 = ChunkMetadata(
        document_id="doc_1_v2", document_name="annual_report.pdf", version="2.0",
        page_number=5, created_at=time.time(), content_type=ContentType.TEXT
    )

    c1 = Chunk(
        chunk_id="c_v1", chunk_type=ChunkType.CHILD,
        content="Q4 Revenue was reported as $10.5M in the preliminary release.",
        token_count=12, content_hash="hash_v1", metadata=meta_v1
    )
    c2 = Chunk(
        chunk_id="c_v2", chunk_type=ChunkType.CHILD,
        content="Q4 Revenue was revised to $11.2M in the audited restatement.",
        token_count=12, content_hash="hash_v2", metadata=meta_v2
    )
    return [c1, c2]

def test_contradiction_detection_multi_version(multi_version_chunks):
    """Verify that contradictory statements between document versions are flagged."""
    detector = ContradictionDetector()
    report = detector.detect_conflicts(
        query="What was Q4 revenue?",
        chunks=multi_version_chunks
    )

    assert report.has_contradiction is True
    assert len(report.conflicting_sources) == 2
    assert "1.0" in report.discrepancy_summary
    assert "2.0" in report.discrepancy_summary

def test_temporal_extraction_and_ordering():
    """Verify chronological ordering of time-series financial evidence."""
    engine = TemporalReasoningEngine()
    
    meta = ChunkMetadata(
        document_id="doc_f", document_name="financials.pdf", page_number=1, created_at=0.0
    )
    
    chunk_2025 = Chunk(
        chunk_id="c_2025", chunk_type=ChunkType.CHILD,
        content="In 2025, operating margins expanded by 4%.",
        token_count=8, content_hash="h25", metadata=meta.model_copy()
    )
    chunk_2023 = Chunk(
        chunk_id="c_2023", chunk_type=ChunkType.CHILD,
        content="In 2023, initial expansion into cloud infrastructure began.",
        token_count=8, content_hash="h23", metadata=meta.model_copy()
    )
    chunk_2024 = Chunk(
        chunk_id="c_2024", chunk_type=ChunkType.CHILD,
        content="In 2024, revenue stabilized across enterprise accounts.",
        token_count=8, content_hash="h24", metadata=meta.model_copy()
    )

    sorted_chunks = engine.sort_chunks_by_temporal_order([chunk_2025, chunk_2023, chunk_2024])
    
    assert sorted_chunks[0].chunk_id == "c_2023"
    assert sorted_chunks[1].chunk_id == "c_2024"
    assert sorted_chunks[2].chunk_id == "c_2025"
