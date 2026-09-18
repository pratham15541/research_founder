"""
Unit tests for AcademicPDFParser: magic bytes validation and section extraction.
"""

import pytest
from pathlib import Path
from src.ingestion.pdf_parser import AcademicPDFParser


def _write_pdf(path: Path, text: str = "Abstract\nThis is a valid test PDF.") -> None:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_pdf_magic_bytes_validation(tmp_path):
    # Valid PDF header
    valid_pdf = tmp_path / "valid.pdf"
    _write_pdf(valid_pdf)
    assert AcademicPDFParser.validate_pdf_file(valid_pdf) is True
    report = AcademicPDFParser.inspect_pdf_file(valid_pdf)
    assert report["is_valid"] is True
    assert report["page_count"] == 1

    # Invalid header (e.g. text or binary masquerading as PDF)
    invalid_file = tmp_path / "fake.pdf"
    with open(invalid_file, "wb") as f:
        f.write(b"NOT_A_PDF_HEADER")
    assert AcademicPDFParser.validate_pdf_file(invalid_file) is False

def test_future_work_statement_extraction():
    sections = {
        "future_work": (
            "In future work, we plan to evaluate the architecture on ultra-sparse graph benchmarks. "
            "Extending this framework to continuous-time differential equations remains an open challenge. "
            "We hope to deploy on embedded edge hardware."
        ),
        "limitations": (
            "Our benchmark is limited to adult clinical cohorts. "
            "Scalable extension to pediatric genomics requires unaddressed ethical consents."
        )
    }

    seeds = AcademicPDFParser.extract_explicit_future_work_statements(sections)
    assert len(seeds) >= 2
    assert any("remains an open challenge" in s.lower() for s in seeds)
    assert any("we plan to" in s.lower() for s in seeds)
