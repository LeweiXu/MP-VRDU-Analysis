"""Pipeline package for the four-stage MP-VRDU architecture.

Purpose:
    Groups the frozen interfaces and orchestration code for input conditioning,
    representation, reasoning, judging, and cached execution.

Pipeline role:
    Submodules implement the paper's A->B->C->D flow plus the orchestrator that
    composes it for each experiment cell.

Arguments:
    None. This package initializer is import-only.
"""
