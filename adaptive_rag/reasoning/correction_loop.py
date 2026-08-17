from typing import List, Tuple, Callable
from adaptive_rag.models.schema import Chunk
from adaptive_rag.reasoning.generator import AnswerGenerator
from adaptive_rag.reasoning.evaluator import SelfEvaluator

class CorrectionEngine:
    """Coordinates the Generate -> Evaluate -> Retry loop."""
    
    def __init__(self, generator: AnswerGenerator, evaluator: SelfEvaluator, max_rounds: int = 2):
        self.generator = generator
        self.evaluator = evaluator
        self.max_rounds = max_rounds

    def execute(self, query: str, initial_chunks: List[Chunk], retrieve_more_callback: Callable[[str], List[Chunk]], mock_gen=None, mock_eval=None) -> Tuple[str, List[Chunk]]:
        current_chunks = initial_chunks.copy()
        
        for round_idx in range(self.max_rounds):
            # 1. Generate
            answer = self.generator.generate(query, current_chunks, mock_response=mock_gen)
            
            # 2. Evaluate
            eval_result = self.evaluator.evaluate(query, answer, current_chunks, mock_response=mock_eval)
            
            # 3. Decide
            if eval_result.is_supported_by_evidence and eval_result.addresses_query and not eval_result.needs_retrieval:
                return answer, current_chunks # Success
                
            if eval_result.needs_retrieval and round_idx < (self.max_rounds - 1):
                # We need more info, trigger the callback to the retrieval router
                new_chunks = retrieve_more_callback(eval_result.feedback)
                current_chunks.extend(new_chunks)
                continue # Retry generation with new context
                
        # If we exit the loop, we return the best effort
        return answer, current_chunks
