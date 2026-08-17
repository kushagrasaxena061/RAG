import tiktoken
from pydantic import BaseModel, Field
from adaptive_rag.config import ModelContextConfig

class TokenBudgetReport(BaseModel):
    """Token budget breakdown for Context Manager visibility."""
    context_window: int
    system_prompt_tokens: int
    conversation_memory_tokens: int
    query_tokens: int
    output_budget: int
    safety_margin: int
    max_retrieval_tokens: int
    available_tokens: int

class TokenBudgetManager:
    """Controls context budget and prevents context overflow."""
    def __init__(self, config: ModelContextConfig):
        self.config = config
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = tiktoken.get_encoding("gpt2")

    def count_tokens(self, text: str) -> int:
        """Accurately count tokens for input text."""
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def calculate_budget(
        self,
        system_prompt: str = "",
        conversation_history_tokens: int = 0,
        query: str = "",
    ) -> TokenBudgetReport:
        """Dynamically compute the exact remaining token budget for retrieval evidence."""
        sys_tokens = self.count_tokens(system_prompt) if system_prompt else self.config.system_prompt_reserve
        query_tokens = self.count_tokens(query) if query else self.config.query_reserve

        fixed_consumption = (
            sys_tokens
            + conversation_history_tokens
            + query_tokens
            + self.config.max_output_tokens
            + self.config.safety_margin_tokens
        )

        max_retrieval_tokens = max(0, self.config.context_window_tokens - fixed_consumption)

        return TokenBudgetReport(
            context_window=self.config.context_window_tokens,
            system_prompt_tokens=sys_tokens,
            conversation_memory_tokens=conversation_history_tokens,
            query_tokens=query_tokens,
            output_budget=self.config.max_output_tokens,
            safety_margin=self.config.safety_margin_tokens,
            max_retrieval_tokens=max_retrieval_tokens,
            available_tokens=max_retrieval_tokens,
        )

    def fits_in_budget(self, current_tokens: int, additional_tokens: int, budget: TokenBudgetReport) -> bool:
        """Determine if additional content fits within the designated retrieval budget."""
        return (current_tokens + additional_tokens) <= budget.max_retrieval_tokens
