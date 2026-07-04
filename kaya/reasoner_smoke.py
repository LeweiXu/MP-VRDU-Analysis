"""Smoke-generate Qwen3-VL answers for the four representation rungs on Kaya.

Purpose:
    Provides the Stage-M3 GPU barrier: load the smoke Qwen3-VL local reasoner
    from prestaged Hugging Face cache and generate one answer for each requested
    representation rung on the frozen MMLongBench smoke corpus.

Pipeline role:
    Runs on a Kaya compute node through `kaya.kaya submit`, with Hugging Face
    offline mode enabled by the runner. It exercises the real
    `LocalVLMBackend` through the normal orchestrator path:
    oracle pages -> representation -> `ModelInput` -> reasoner -> cached row.

CLI:
    `python -m kaya.kaya submit kaya/reasoner_smoke.py -- [options]`

Arguments:
    --questions N: number of answerable smoke questions to run (default: 1).
    --representation R: repeatable rung filter among T, TL, TLV, V. If omitted,
        all four rungs run.
    --fresh-cache: write rows to an isolated M3 smoke cache instead of the
        default orchestrator cache, avoiding old stub rows during validation.
"""

# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=m3-reasoner-smoke

from __future__ import annotations

import argparse
import json

from config import ExperimentConfig
from experiments.smoke import load_smoke_questions
from kaya.prestage import prepare_tool_cache_env
from pipeline.conditioner import OracleConditioner
from pipeline.orchestrator import Orchestrator, ResultCache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=1, help="number of answerable smoke questions")
    parser.add_argument(
        "--representation",
        action="append",
        choices=("T", "TL", "TLV", "V"),
        help="representation rung to run; repeatable",
    )
    parser.add_argument(
        "--fresh-cache",
        action="store_true",
        help="write to results/cache/m3_reasoner_smoke/results.jsonl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(smoke=True)
    prepare_tool_cache_env(config.paths.hf_home)
    questions = [question for question in load_smoke_questions(config.paths.data_dir) if question.evidence_pages]
    selected = questions[: max(1, args.questions)]
    representations = tuple(args.representation or config.representations)
    cache = None
    if args.fresh_cache:
        cache = ResultCache(config.paths.cache_dir / "m3_reasoner_smoke" / "results.jsonl")
    orchestrator = Orchestrator(config, cache=cache)
    conditioner = OracleConditioner()

    print(
        json.dumps(
            {
                "event": "start",
                "questions": len(selected),
                "representations": list(representations),
                "reasoner_spec": orchestrator.reasoner.spec,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for question in selected:
        for representation in representations:
            row = orchestrator.run_cell(question, conditioner, representation)
            print(
                json.dumps(
                    {
                        "event": "row",
                        "question_id": row.question_id,
                        "doc_id": row.doc_id,
                        "representation": row.representation,
                        "answer": row.answer,
                        "input_text_tokens": row.input_text_tokens,
                        "input_visual_tokens": row.input_visual_tokens,
                        "output_tokens": row.output_tokens,
                        "latency_s": row.latency_s,
                        "model_spec": row.model_spec,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    print(json.dumps({"event": "complete"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
