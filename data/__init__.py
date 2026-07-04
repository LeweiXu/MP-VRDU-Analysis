"""Data loading and rendering package for MMLongBench-Doc."""

from data.binning import (
    BIN_TO_DOC_TYPES,
    DEFAULT_BINS,
    DOC_TYPE_TO_BIN,
    OPTION_A_BIN_COUNTS,
    doc_type_bin,
)
from data.loader import find_mmlongbench_root, load_mmlongbench, resolve_pdf
from data.render import pdf_page_count, render_pdf, render_question_pages, validate_gold_pages

__all__ = [
    "BIN_TO_DOC_TYPES",
    "DEFAULT_BINS",
    "DOC_TYPE_TO_BIN",
    "OPTION_A_BIN_COUNTS",
    "find_mmlongbench_root",
    "doc_type_bin",
    "load_mmlongbench",
    "pdf_page_count",
    "render_pdf",
    "render_question_pages",
    "resolve_pdf",
    "validate_gold_pages",
]
