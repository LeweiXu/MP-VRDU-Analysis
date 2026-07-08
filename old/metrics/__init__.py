"""Metrics package for accuracy, retrieval, abstention, cost, and frontiers.

Purpose:
    Groups all measurement code that turns cached predictions/scores into
    comparable paper numbers.

Pipeline role:
    Table builders should import metrics from these submodules rather than
    reimplementing scoring, confidence intervals, token/latency aggregation, or
    frontier rules.

Arguments:
    None. This package initializer is import-only.
"""
