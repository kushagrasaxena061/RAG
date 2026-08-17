from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field

class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

class ChunkType(str, Enum):
    DOCUMENT_SUMMARY = "document_summary"
    SECTION_SUMMARY = "section_summary"
    PARENT = "parent"
    CHILD = "child"

class IngestionStage(str, Enum):
    INITIALIZED = "initialized"
    VALIDATING = "validating"
    OCR_FALLBACK = "ocr_fallback"
    PARSING = "parsing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"

class IngestionProgress(BaseModel):
    document_id: str
    stage: IngestionStage
    current_page: int = 0
    total_pages: int = 0
    chunks_created: int = 0
    progress_percentage: float = 0.0
    message: str = ""

class TableRepresentation(BaseModel):
    """Preserves exact table structure without flattening."""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    raw_csv: str

class ImageRepresentation(BaseModel):
    """Placeholder for multimodal retrieval."""
    image_hash: str
    bounding_box: Tuple[float, float, float, float]
    caption: Optional[str] = None

class ChunkMetadata(BaseModel):
    document_id: str
    document_name: str
    version: str = "1.0"
    page_number: int
    section_title: Optional[str] = None
    section_path: List[str] = Field(default_factory=list)
    document_type: str = "general"
    content_type: ContentType = ContentType.TEXT
    created_at: float
    user_metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    chunk_id: str
    parent_chunk_id: Optional[str] = None
    chunk_type: ChunkType
    content: str  # String representation for LLM context
    structured_data: Optional[Union[TableRepresentation, ImageRepresentation]] = None
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
    document_type: str
    total_pages: int
    total_tokens: int
    pages: List[ParsedPage] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    user_metadata: Dict[str, Any] = Field(default_factory=dict)
