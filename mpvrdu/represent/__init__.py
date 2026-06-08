"""Stage 2/5: document representation (parsers / OCR -> text + structure).

The Stage-5 sub-study varies the parser (PyMuPDF4LLM / MinerU / Tesseract) and
the chunking granularity behind one interface; everything downstream (indexing,
retrieval, generation) is identical across parsers.
"""

from .base import (Parser, ParsedPage, PyMuPDF4LLMParser, PyMuPDFParser,
                   get_parser, register_parser)  # noqa: F401
from .chunking import Chunk, chunk_pages  # noqa: F401
