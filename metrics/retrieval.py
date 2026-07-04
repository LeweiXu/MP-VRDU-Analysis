"""Compute page-retrieval metrics against gold evidence pages.

Purpose:
    Reserved for precision, recall, and F1 over page indices returned by text or
    vision retrievers.

Pipeline role:
    Stage M6/F-stage retrieval experiments will score `Retriever` outputs here
    before table builders slice results by doc type and evidence modality.

Arguments:
    None. This module is import-only until retrieval metrics are implemented.
"""
