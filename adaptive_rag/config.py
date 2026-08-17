from pydantic import BaseModel, Field

class ModelContextConfig(BaseModel):
    model_name: str = Field(default="llama-3-70b-instruct")
    context_window_tokens: int = Field(default=32768)
    max_output_tokens: int = Field(default=4096)
    system_prompt_reserve: int = Field(default=1024)
    conversation_memory_reserve: int = Field(default=4096)
    query_reserve: int = Field(default=1024)
    safety_margin_tokens: int = Field(default=512)

class IngestionConfig(BaseModel):
    parent_chunk_size_tokens: int = Field(default=1024)
    child_chunk_size_tokens: int = Field(default=256)
    chunk_overlap_tokens: int = Field(default=32)
    min_chunk_size_tokens: int = Field(default=40)
    enable_deduplication: bool = Field(default=True)

class RetrievalConfig(BaseModel):
    """Configuration for the Hybrid Search Engine."""
    vector_persist_dir: str = Field(default="./data/chroma")
    bm25_persist_path: str = Field(default="./data/bm25_index.pkl")
    rrf_k: int = Field(default=60)
    default_top_k: int = Field(default=5)

class AppConfig(BaseModel):
    model: ModelContextConfig = Field(default_factory=ModelContextConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

global_config = AppConfig()
