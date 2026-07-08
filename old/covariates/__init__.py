"""Covariate package for retrieval and document-type classification.

Purpose:
    Groups interfaces for factors that feed or annotate the pipeline without
    being the evaluated reasoner: page retrieval and doc-type classification.

Pipeline role:
    Conditioners use retrievers; routing policies use classifiers; metrics and
    table builders report both as measured covariates.

Arguments:
    None. This package initializer is import-only.
"""
