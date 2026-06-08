"""Stage 7: consolidate JSONL outputs into tables + figures.

Everything here reads ONLY the JSONL result files (which embed their config in
the meta header), so the whole results section regenerates from raw outputs with
one command — the reproducibility requirement.
"""

from .aggregate import (aggregate_dir, summarize_run, to_markdown_table)  # noqa: F401
