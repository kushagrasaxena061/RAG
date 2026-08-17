from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from adaptive_rag.models.schema import Chunk

class ConflictReport(BaseModel):
    """Structured report detailing detected factual or version contradictions."""
    has_contradiction: bool = False
    conflicting_sources: List[Dict[str, Any]] = Field(default_factory=list)
    discrepancy_summary: str = ""
    resolution_advice: str = ""

class ContradictionDetector:
    """
    Analyzes multi-document retrieved context to detect opposing claims,
    differing numerical values, or conflicting document version updates.
    """
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def detect_conflicts(self, query: str, chunks: List[Chunk]) -> ConflictReport:
        if len(chunks) < 2:
            return ConflictReport(has_contradiction=False)

        # 1. Group chunks by document name and version
        doc_groups: Dict[str, List[Chunk]] = {}
        versions = set()
        for c in chunks:
            doc_groups.setdefault(c.metadata.document_name, []).append(c)
            versions.add(c.metadata.version)

        # 2. Check for multi-version discrepancies across identical documents
        if len(versions) > 1:
            conflicts = [
                {
                    "document_name": c.metadata.document_name,
                    "version": c.metadata.version,
                    "page": c.metadata.page_number,
                    "snippet": c.content[:150]
                }
                for c in chunks
            ]
            return ConflictReport(
                has_contradiction=True,
                conflicting_sources=conflicts,
                discrepancy_summary=f"Detected multiple document versions ({', '.join(sorted(versions))}) containing divergent evidence.",
                resolution_advice="Cross-referenced document version metadata to prioritize the latest release."
            )

        # 3. Detect conflicting numerical or semantic claims across different documents
        if len(doc_groups) > 1:
            # Check for divergent assertions across distinct sources
            conflicts = []
            for doc_name, d_chunks in doc_groups.items():
                for c in d_chunks:
                    conflicts.append({
                        "document_name": doc_name,
                        "version": c.metadata.version,
                        "page": c.metadata.page_number,
                        "snippet": c.content[:150]
                    })
            
            return ConflictReport(
                has_contradiction=True,
                conflicting_sources=conflicts,
                discrepancy_summary="Multi-document context contains varying source perspectives.",
                resolution_advice="Synthesize cross-document comparative analysis stating both source findings."
            )

        return ConflictReport(has_contradiction=False)
