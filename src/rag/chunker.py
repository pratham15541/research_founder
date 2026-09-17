"""
Section-Aware Academic Document Chunker.
Splits academic paper sections into overlapping text chunks with rich metadata tracking.
"""

from typing import List, Dict, Any
import re

class AcademicChunker:
    """Chunks academic papers by section with sliding window overlap and metadata tagging."""

    @classmethod
    def chunk_paper(
        cls,
        paper: Dict[str, Any],
        max_chunk_words: int = 350,
        overlap_words: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Split a paper's sections into retrievable chunks.
        Sections processed: abstract, methods, limitations, future_work, and full_text.
        """
        chunks: List[Dict[str, Any]] = []
        paper_id = paper.get("id", "")
        title = paper.get("title", "Untitled")
        doi = paper.get("doi", "")
        year = paper.get("year", 2024)

        # Process structured sections if available
        sections = paper.get("sections", {})
        if not sections and paper.get("abstract"):
            sections = {"abstract": paper.get("abstract", "")}

        for sec_type, content in sections.items():
            if not content or len(content.strip()) < 30:
                continue

            words = content.split()
            if len(words) <= max_chunk_words:
                chunks.append({
                    "chunk_id": f"{paper_id}_{sec_type}_0",
                    "paper_id": paper_id,
                    "paper_title": title,
                    "doi": doi,
                    "year": year,
                    "section_type": sec_type,
                    "text": content.strip()
                })
            else:
                start = 0
                idx = 0
                step = max_chunk_words - overlap_words
                while start < len(words):
                    end = min(start + max_chunk_words, len(words))
                    chunk_text = " ".join(words[start:end])
                    chunks.append({
                        "chunk_id": f"{paper_id}_{sec_type}_{idx}",
                        "paper_id": paper_id,
                        "paper_title": title,
                        "doi": doi,
                        "year": year,
                        "section_type": sec_type,
                        "text": chunk_text
                    })
                    if end == len(words):
                        break
                    start += step
                    idx += 1

        return chunks

    @classmethod
    def chunk_corpus(
        cls,
        papers: List[Dict[str, Any]],
        max_chunk_words: int = 350,
        overlap_words: int = 50
    ) -> List[Dict[str, Any]]:
        """Chunk all papers in a corpus."""
        all_chunks = []
        for p in papers:
            all_chunks.extend(cls.chunk_paper(p, max_chunk_words, overlap_words))
        return all_chunks

