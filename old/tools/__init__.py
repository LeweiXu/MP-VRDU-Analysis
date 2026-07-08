"""Document extraction tool package.

Purpose:
    Groups modular text, layout, and visual helpers that feed representation
    composers without coupling the pipeline to a specific parser implementation.

Pipeline role:
    `tools.text`, `tools.layout`, and `tools.visual` implement the channel
    functions called by `pipeline.representation`.

Arguments:
    None. This package initializer is import-only.
"""
