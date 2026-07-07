"""Table building: turn judged result rows into the eight paper tables.

Purpose:
    The post-generation reporting layer, split out of `experiments/` because it is
    a separate concern from running the GPU generation tasks. `reporting.tables`
    holds the pure per-table aggregation functions; `reporting.build` routes each
    task's judged rows (and side artifacts) into the right builder and writes the
    CSVs plus a combined markdown report.

Pipeline role:
    `cli.build` calls `reporting.build.build_tables_from_artifacts`. Nothing here
    touches the GPU or the reasoner; it only reads cached `results.jsonl` and side
    artifacts.

Arguments:
    None. Import-only.
"""
