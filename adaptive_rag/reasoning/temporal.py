import re
from typing import List, Dict, Any, Tuple
from adaptive_rag.models.schema import Chunk

class TemporalReasoningEngine:
    """
    Extracts, parses, and aligns temporal references (years, quarters, dates)
    to facilitate accurate historical and time-series comparisons.
    """
    YEAR_PATTERN = re.compile(r"\b(19\d\d|20\d\d)\b")
    QUARTER_PATTERN = re.compile(r"\bQ([1-4])\b|\b([1-4])(?:st|nd|rd|th)?\s*quarter\b", re.IGNORECASE)

    def extract_temporal_anchors(self, text: str) -> List[str]:
        """Extracts chronological entities such as years and quarters from text."""
        years = self.YEAR_PATTERN.findall(text)
        quarters = [f"Q{m[0] or m[1]}" for m in self.QUARTER_PATTERN.findall(text)]
        return sorted(list(set(years + quarters)))

    def sort_chunks_by_temporal_order(self, chunks: List[Chunk], reverse: bool = False) -> List[Chunk]:
        """Sorts evidence chunks chronologically based on text content and section titles."""
        def get_sort_key(c: Chunk) -> Tuple[int, str, int]:
            anchors = self.extract_temporal_anchors(f"{c.content} {c.metadata.section_title or ''}")
            year = int(anchors[0]) if anchors and anchors[0].isdigit() else 0
            return (year, c.metadata.version, c.metadata.page_number)

        return sorted(chunks, key=get_sort_key, reverse=reverse)
