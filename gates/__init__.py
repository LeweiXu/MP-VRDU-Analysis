"""Section-2 go/no-go gates and the cached-cell viewer they share.

Purpose:
    The gate tooling (F1 frontier divergence, F2 judge-human agreement, F3
    classifier feasibility), split out of `experiments/` because it evaluates a run
    rather than generating one. `gates.core` holds the gate logic; `gates.viewer`
    joins cached predictions/results back to the source PDF + rendered pages (used
    both by the F2 agreement packet and the standalone `scripts.inspect_results`
    debug viewer); `gates.__main__` is the CLI (`python -m gates <subcommand>`).

Pipeline role:
    Complements the three experiment roles: after a run has generated and judged
    rows, this package scores the gates and can render the cells for a human to
    inspect. The heavy model work stays in the classifier/reasoner paths.

Arguments:
    None. Import-only. CLI entry point: `python -m gates`.
"""
