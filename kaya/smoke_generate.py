"""Phase 1 of the full smoke run: generate + cache predictions on a GPU.

Purpose:
    Runs the GPU-bound half of the Section-1 "full run, tiny data" sweep on a
    Kaya compute node: the real Qwen3-VL-2B reasoner over the oracle ladder and
    the matched/cross retrieval cells, plus the real doc-type classifier. Every
    reasoner output is written to the shared `PredictionCache`; the throwaway
    judge here is a stub because compute nodes have no internet. The GPT-4o-mini
    judge and the table build happen in phase 2 (`kaya/smoke_judge.py`).

Pipeline role:
    Thin GPU entry point over `experiments.smoke_run.run_generate`. It is
    submitted through SLURM by `kaya.kaya submit` and runs in Hugging Face
    offline mode so it reads prestaged weights instead of phoning home.

CLI:
    `python -m kaya.kaya submit kaya/smoke_generate.py -- [--questions N] [--k N]`

Arguments:
    --questions N: run only the first N smoke questions (default: all).
    --k N: top-k for the matched/cross retrieval cells (default: config k).
"""

# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=smoke-generate

from __future__ import annotations

import argparse

from config import ExperimentConfig
from experiments.smoke import load_smoke_questions
from experiments.smoke_run import run_generate
from kaya.prestage import prepare_tool_cache_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, help="number of leading smoke questions")
    parser.add_argument("--k", type=int, help="top-k for matched/cross retrieval cells")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(smoke=True)
    prepare_tool_cache_env(config.paths.hf_home)
    questions = load_smoke_questions(config.paths.data_dir)
    if args.questions is not None:
        questions = questions[: max(1, args.questions)]
    run_generate(config, questions, k=args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
