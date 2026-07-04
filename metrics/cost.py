"""Aggregate latency and token costs for representation/model comparisons.

Purpose:
    Reserved for cost summaries using the primary latency@batch=1 metric and
    secondary split text/vision token counts.

Pipeline role:
    Table builders will combine `Prediction` cost fields through this module to
    report frontier latency and routing-policy cost.

Arguments:
    None. This module is import-only until cost helpers are implemented.
"""
