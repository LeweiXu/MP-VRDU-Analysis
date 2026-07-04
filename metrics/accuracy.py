"""Summarise answer accuracy and confidence intervals.

Purpose:
    Reserved for corpus/bin/rung accuracy summaries, document-level bootstrap
    confidence intervals, and effect-size helpers required by the paper tables.

Pipeline role:
    Stage M5 will aggregate judged `ResultRow` objects here so every headline
    number uses the same document-level resampling rule.

Arguments:
    None. This module is import-only until accuracy helpers are implemented.
"""
