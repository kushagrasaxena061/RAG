import pytest
from adaptive_rag.models.schema import Chunk, ChunkMetadata, ChunkType, ContentType
from adaptive_rag.reasoning.generator import AnswerGenerator
from adaptive_rag.reasoning.evaluator import SelfEvaluator, EvaluationResult
from adaptive_rag.reasoning.correction_loop import CorrectionEngine

@pytest.fixture
def mock_chunks():
    meta1 = ChunkMetadata(document_id="d1", document_name="Q3_Report.pdf", page_number=12, created_at=0.0)
    meta2 = ChunkMetadata(document_id="d2", document_name="Market_Analysis.pdf", page_number=5, created_at=0.0)
    
    c1 = Chunk(chunk_id="c1", chunk_type=ChunkType.CHILD, content="Revenue in Q3 was $4.2M.", token_count=10, content_hash="h1", metadata=meta1)
    c2 = Chunk(chunk_id="c2", chunk_type=ChunkType.CHILD, content="Cloud adoption grew 20% globally.", token_count=10, content_hash="h2", metadata=meta2)
    
    return [c1, c2]

def test_generator_formatting(mock_chunks):
    """Verify that context is formatted correctly to inject provenance tags for the LLM."""
    generator = AnswerGenerator()
    formatted = generator.format_context(mock_chunks)
    
    # Ensure strict provenance markers are embedded in the prompt
    assert "[Q3_Report.pdf, p. 12]" in formatted
    assert "Revenue in Q3 was $4.2M." in formatted
    assert "[Market_Analysis.pdf, p. 5]" in formatted

def test_evaluator_parsing():
    """Verify the evaluator correctly interprets strict JSON evaluations."""
    evaluator = SelfEvaluator()
    mock_llm_eval = '''{
        "is_supported_by_evidence": false,
        "addresses_query": true,
        "needs_retrieval": true,
        "feedback": "The answer mentions Q4 profits but the context only contains Q3."
    }'''
    
    result = evaluator.evaluate("What were Q4 profits?", "Q4 profits were $5M.", [], mock_response=mock_llm_eval)
    
    assert not result.is_supported_by_evidence
    assert result.needs_retrieval
    assert "Q4 profits" in result.feedback

def test_correction_loop(mock_chunks):
    """Verify the engine triggers a retrieval callback if evaluation fails."""
    generator = AnswerGenerator()
    evaluator = SelfEvaluator()
    engine = CorrectionEngine(generator, evaluator, max_rounds=2)
    
    retrieval_called = False
    
    def mock_retrieve_more(feedback: str):
        nonlocal retrieval_called
        retrieval_called = True
        # Simulate returning a new chunk that solves the missing info
        meta = ChunkMetadata(document_id="d1", document_name="Q3_Report.pdf", page_number=13, created_at=0.0)
        return [Chunk(chunk_id="c3", chunk_type=ChunkType.CHILD, content="Q4 profits were expected to be $5M.", token_count=10, content_hash="h3", metadata=meta)]

    mock_eval_fail = '''{"is_supported_by_evidence": false, "addresses_query": false, "needs_retrieval": true, "feedback": "Missing Q4"}'''
    
    # Execute loop
    final_answer, final_context = engine.execute(
        query="What about Q4?", 
        initial_chunks=mock_chunks, 
        retrieve_more_callback=mock_retrieve_more,
        mock_gen="Generated Answer",
        mock_eval=mock_eval_fail # Forces failure, triggers loop
    )
    
    assert retrieval_called is True
    assert len(final_context) == 3 # Initial 2 + 1 newly retrieved
