"""Phase 2 of the full smoke run: judge cached predictions and build 8 tables.

Purpose:
    Runs the internet-bound half of the Section-1 "full run, tiny data" sweep on
    the Kaya login node: it re-judges the predictions cached by phase 1
    (`kaya/smoke_generate.py`) with a real API judge and writes the eight paper
    table CSVs. It uses no GPU and opens no PDFs; every cell must be a
    prediction-cache hit, so phase 1 has to finish first.

Pipeline role:
    Thin login-node entry point over `experiments.smoke_run.run_judge`. It runs
    online (not offline) so the judge API is reachable; the API key
    (`GEMINI_API_KEY` for the default Gemini judge, or `OPENAI_API_KEY` for
    gpt-4o-mini) is forwarded from the local `.env` by `kaya.kaya` for online
    login runs (see `secrets.forward` in `kaya/config.json`).

CLI:
    `python -m kaya.kaya run kaya/smoke_judge.py -- [--judge SPEC] [--questions N]`

Arguments:
    --judge SPEC: judge for the run: `gemini` (default), `gpt-4o-mini`, `stub`.
    --questions N: run only the first N smoke questions (default: all).
    --k N: top-k for the matched/cross retrieval cells (default: config k).
    --bootstrap N: document-level bootstrap resamples (default: 200).
"""

# kaya: target=login
# kaya: env=true
# kaya: offline=false
# kaya: job-name=smoke-judge

from __future__ import annotations

import argparse

from config import ExperimentConfig
from experiments.smoke import load_smoke_questions
from experiments.smoke_run import run_judge
from pipeline.judge import get_judge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="gemini", help="judge: gemini (default), gpt-4o-mini, or stub")
    parser.add_argument("--questions", type=int, help="number of leading smoke questions")
    parser.add_argument("--k", type=int, help="top-k for matched/cross retrieval cells")
    parser.add_argument("--bootstrap", type=int, default=200, help="document-level bootstrap resamples")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(smoke=True)
    questions = load_smoke_questions(config.paths.data_dir)
    if args.questions is not None:
        questions = questions[: max(1, args.questions)]
    run_judge(config, questions, judge=get_judge(args.judge), k=args.k, n_bootstrap=args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
