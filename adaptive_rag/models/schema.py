from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    DOCUMENT_SUMMARY = "document_summary"
    SECTION_SUMMARY = "section_summary"

class ChunkType(str, Enum):
    DOCUMENT_SUMMARY = "document_summary"
    SECTION_SUMMARY = "section_summary"
    PARENT = "parent"
    CHILD = "child"

class QueryCategory(str, Enum):
    SIMPLE = "simple"
    TABULAR = "tabular"
    MULTI_STEP = "multi_step"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    VISUAL = "visual"
    CONVERSATIONAL = "conversational"

class IngestionStage(str, Enum):
    INITIALIZED = "initialized"
    VALIDATING = "validating"
    PARSING = "parsing"
    CHUNKING = "chunking"
    OCR_FALLBACK = "ocr_fallback"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"

class TableRepresentation(BaseModel):
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    raw_csv: str = ""
    title: Optional[str] = None
    page_number: int = 1

class ImageRepresentation(BaseModel):
    image_hash: str
    bounding_box: Tuple[float, float, float, float]
    caption: str

class ChunkMetadata(BaseModel):
    document_id: str
    document_name: str
    version: str = "1.0"
    page_number: int
    section_title: Optional[str] = None
    section_path: List[str] = Field(default_factory=list)
    content_type: ContentType = ContentType.TEXT
    created_at: float
    user_metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    chunk_id: str
    parent_chunk_id: Optional[str] = None
    chunk_type: ChunkType
    content: str
    structured_data: Optional[Any] = None
    token_count: int
    content_hash: str
    metadata: ChunkMetadata

class Section(BaseModel):
    section_id: str
    title: str
    level: int
    page_start: int
    page_end: int
    content: str
    tables: List[TableRepresentation] = Field(default_factory=list)
    images: List[ImageRepresentation] = Field(default_factory=list)
    token_count: int

class ParsedPage(BaseModel):
    page_number: int
    text: str
    tables: List[TableRepresentation] = Field(default_factory=list)
    images: List[ImageRepresentation] = Field(default_factory=list)
    headings: List[str] = Field(default_factory=list)
    token_count: int
    requires_ocr: bool = False

class Document(BaseModel):
    document_id: str
    document_name: str
    version: str = "1.0"
    document_hash: str
    document_type: str = "pdf"
    total_pages: int
    total_tokens: int
    pages: List[ParsedPage] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    user_metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestionProgress(BaseModel):
    document_id: str
    stage: IngestionStage
    current_page: int = 0
    total_pages: int = 0
    chunks_created: int = 0
    progress_percentage: float = 0.0
    message: str = ""

class QueryPlan(BaseModel):
    intent: str = Field(description="The core user intent.")
    category: QueryCategory = Field(default=QueryCategory.SIMPLE)
    rewritten_query: str = Field(description="Normalized query with co-references resolved.")
    expanded_queries: List[str] = Field(default_factory=list, description="Synonym and lexical expansions.")
    sub_queries: List[str] = Field(default_factory=list, description="Decomposed sub-questions.")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Extracted metadata constraints.")
    adaptive_top_k: int = Field(default=6, description="Dynamically allocated retrieval budget.")
    bm25_weight: float = Field(default=0.5, description="Lexical search weight.")
    vector_weight: float = Field(default=0.5, description="Semantic vector search weight.")
    needs_multi_step: bool = Field(default=False)
    confidence: float = Field(default=0.9)

class EvaluationReport(BaseModel):
    is_sufficient: bool = True
    faithfulness_score: float = 1.0
    relevance_score: float = 1.0
    missing_information: str = ""
    suggested_followup_query: Optional[str] = None
    citation_precision: float = 1.0
