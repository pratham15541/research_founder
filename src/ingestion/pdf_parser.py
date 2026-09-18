"""
Secure double-column academic PDF parser and section segmenter.
Handles IEEE, ACM, Springer formats with PyMuPDF / fitz, column boundary sorting,
and regex extraction of Abstract, Methods, Limitations, and Future Work sections.
"""

import re
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Preferred PyMuPDF import (eliminates `fitz` deprecation warning)
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

# Section matching regex patterns
SECTION_PATTERNS = {
    "abstract": re.compile(r"^\s*(\d+[\.\s]+)?abstract\b", re.IGNORECASE),
    "methods": re.compile(r"^\s*(\d+[\.\s]+)?(methodology|methods|proposed\s+method|system\s+model|architecture)\b", re.IGNORECASE),
    "limitations": re.compile(r"^\s*(\d+[\.\s]+)?(limitations|weaknesses|threats\s+to\s+validity)\b", re.IGNORECASE),
    "future_work": re.compile(r"^\s*(\d+[\.\s]+)?(future\s+work|future\s+directions|conclusions?\s+and\s+future\s+work)\b", re.IGNORECASE),
    "conclusion": re.compile(r"^\s*(\d+[\.\s]+)?(conclusions?|summary)\b", re.IGNORECASE)
}

class AcademicPDFParser:
    """Parses academic PDFs with column sorting, layout preservation, and section mining."""

    @staticmethod
    def validate_pdf_bytes(content: bytes) -> bool:
        """Fast PDF byte validation that tolerates small binary prefixes before the header."""
        if len(content) < 10 or len(content) > 25 * 1024 * 1024:
            return False
        header_window = content[:1024]
        return b"%PDF-" in header_window

    @staticmethod
    def validate_pdf_file(file_path: Path) -> bool:
        """Verify that the file exists, is non-empty, and has %PDF- magic bytes."""
        if not file_path.exists() or file_path.is_dir():
            return False
        if file_path.stat().st_size < 10:  # Must have at least enough bytes for header
            return False
        if file_path.stat().st_size > 25 * 1024 * 1024:  # Security: reject >25MB
            return False

        try:
            with open(file_path, "rb") as f:
                return AcademicPDFParser.validate_pdf_bytes(f.read())
        except Exception:
            return False

    @staticmethod
    def inspect_pdf_file(file_path: Path, min_text_chars: int = 0) -> Dict[str, Any]:
        """
        Return a detailed validation report for false-positive/false-negative control.
        `is_valid` means the file looks like a parseable PDF with at least one page.
        """
        report: Dict[str, Any] = {
            "is_valid": False,
            "reason": "",
            "size_bytes": 0,
            "has_pdf_header": False,
            "has_eof_marker": False,
            "page_count": 0,
            "pages_with_text": 0,
            "text_chars": 0,
            "text_extraction_ok": False
        }

        if not file_path.exists() or file_path.is_dir():
            report["reason"] = "missing_file"
            return report

        size = file_path.stat().st_size
        report["size_bytes"] = size
        if size < 10:
            report["reason"] = "too_small"
            return report
        if size > 25 * 1024 * 1024:
            report["reason"] = "too_large"
            return report

        try:
            content = file_path.read_bytes()
        except Exception as exc:
            report["reason"] = f"read_error:{exc}"
            return report

        report["has_pdf_header"] = AcademicPDFParser.validate_pdf_bytes(content)
        report["has_eof_marker"] = b"%%EOF" in content[-4096:]
        if not report["has_pdf_header"]:
            report["reason"] = "missing_pdf_header"
            return report

        if fitz is None:
            report["is_valid"] = True
            report["reason"] = "valid_header_parser_unavailable"
            return report

        try:
            doc = fitz.open(file_path)
            try:
                report["page_count"] = len(doc)
                text_chars = 0
                pages_with_text = 0
                for page in doc:
                    text = page.get_text("text").strip()
                    if text:
                        pages_with_text += 1
                        text_chars += len(text)
                report["pages_with_text"] = pages_with_text
                report["text_chars"] = text_chars
                report["text_extraction_ok"] = text_chars >= min_text_chars if min_text_chars else text_chars > 0
                report["is_valid"] = len(doc) > 0
                if not report["is_valid"]:
                    report["reason"] = "zero_pages"
                elif min_text_chars and text_chars < min_text_chars:
                    report["reason"] = "low_text_extractable_pdf"
                else:
                    report["reason"] = "ok"
            finally:
                doc.close()
        except Exception as exc:
            report["reason"] = f"parser_error:{exc}"

        return report

    @classmethod
    def parse_pdf(cls, file_path: Path) -> Dict[str, Any]:
        """
        Extract text, sections, and basic metadata from an academic PDF.
        Supports double-column layout sorting.
        """
        validation = cls.inspect_pdf_file(file_path)
        if not validation["is_valid"]:
            raise ValueError(f"Invalid or untrusted PDF file: {file_path.name} ({validation['reason']})")

        if fitz is None:
            logger.warning("PyMuPDF not installed. Returning empty parse result.")
            return {"title": file_path.stem, "abstract": "", "sections": {}, "full_text": ""}

        doc = fitz.open(file_path)
        full_text_blocks: List[str] = []
        page_texts: List[str] = []
        page_count = 0
        pages_with_text = 0

        try:
            page_count = len(doc)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                # Extract blocks with coordinates: (x0, y0, x1, y1, text, block_no, block_type)
                blocks = page.get_text("blocks")
                
                # Sort blocks by column then vertical position to handle double-column reading order
                # In standard double-column, mid_x is ~ page.rect.width / 2
                page_width = page.rect.width
                mid_x = page_width / 2.0

                # Column-aware sorting
                def block_sort_key(b):
                    x0, y0, x1, y1 = b[:4]
                    # If block is in left column (center < mid_x), column_id = 0, else 1
                    col_id = 0 if ((x0 + x1) / 2.0) < mid_x else 1
                    return (col_id, y0)

                sorted_blocks = sorted(blocks, key=block_sort_key)
                page_content = []
                for b in sorted_blocks:
                    text = b[4].strip()
                    if text:
                        page_content.append(text)
                
                page_str = "\n\n".join(page_content)
                if page_str.strip():
                    pages_with_text += 1
                page_texts.append(page_str)
                full_text_blocks.extend(page_content)
        finally:
            doc.close()

        full_text = "\n\n".join(page_texts)

        # Segment into structured sections
        sections = cls._segment_sections(full_text_blocks)

        # Extract title (usually the first large block of page 1)
        title = file_path.stem
        if full_text_blocks:
            candidate_title = full_text_blocks[0].replace("\n", " ").strip()
            if len(candidate_title) > 10 and len(candidate_title) < 250:
                title = candidate_title

        abstract = sections.get("abstract", "")
        # If abstract not parsed by header, take the first 1-2 blocks after title
        if not abstract and len(full_text_blocks) > 1:
            candidate_abstract = full_text_blocks[1].replace("\n", " ").strip()
            if len(candidate_abstract) > 80:
                abstract = candidate_abstract

        return {
            "title": title,
            "abstract": abstract,
            "sections": sections,
            "full_text": full_text[:50000],  # Bound text length for memory safety
            "is_uploaded": True,
            "source": "uploaded_pdf",
            "pdf_local_path": str(file_path),
            "pdf_validation": validation,
            "pdf_page_count": page_count,
            "pdf_pages_with_text": pages_with_text,
            "pdf_text_chars": len(full_text)
        }

    @classmethod
    def _segment_sections(cls, blocks: List[str]) -> Dict[str, str]:
        """Segment raw text blocks into labeled sections using regex heuristics."""
        sections: Dict[str, List[str]] = {
            "abstract": [],
            "methods": [],
            "limitations": [],
            "future_work": [],
            "conclusion": []
        }

        current_sec: Optional[str] = None

        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if not lines:
                continue

            first_line = lines[0]
            matched_sec = None

            for sec_name, pattern in SECTION_PATTERNS.items():
                if pattern.search(first_line):
                    matched_sec = sec_name
                    break

            if matched_sec:
                current_sec = matched_sec
                # If the heading has content following it on subsequent lines
                remaining_lines = "\n".join(lines[1:])
                if remaining_lines:
                    sections[current_sec].append(remaining_lines)
            elif current_sec:
                # Append block to the active section
                sections[current_sec].append(block)

        return {
            sec: "\n\n".join(parts).strip()
            for sec, parts in sections.items()
            if parts
        }

    @staticmethod
    def extract_explicit_future_work_statements(sections: Dict[str, str]) -> List[str]:
        """
        Extract concise candidate gap seeds from Future Work / Limitations sections.
        Looks for sentences containing future-intent indicators ('we plan to', 'remains open', 'future work should').
        """
        text_corpus = sections.get("future_work", "") + "\n" + sections.get("limitations", "")
        if not text_corpus.strip():
            return []

        # Sentence split heuristic
        sentences = re.split(r"(?<=[.!?])\s+", text_corpus.replace("\n", " "))
        future_indicators = [
            "future work", "future direction", "remains an open", "remains unexplored",
            "we plan to", "we hope to", "could be extended", "promising avenue",
            "unaddressed", "scalable extension", "further investigation"
        ]

        seeds: List[str] = []
        for sent in sentences:
            sent_clean = sent.strip()
            if len(sent_clean) > 30 and len(sent_clean) < 300:
                if any(ind in sent_clean.lower() for ind in future_indicators):
                    seeds.append(sent_clean)

        return seeds[:5]  # Top 5 future work statements
