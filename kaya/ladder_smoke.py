"""Run the Stage-M4 oracle ladder through the resumable orchestrator cache.

Purpose:
    Provides the real GPU smoke command for Stage M4. It expands the frozen
    smoke corpus over oracle pages and the `T`/`TL`/`TLV`/`V` representation
    ladder, then runs those cells through `pipeline.orchestrator`.

Pipeline role:
    This is the end-to-end A->B->C barrier after the Qwen reasoner can generate:
    `OracleConditioner` selects gold pages, representation composers produce
    `ModelInput`, the local Qwen backend answers, and `ResultCache` stores one
    row per `(question, rung, model_spec)` cell. A second pass verifies the cache
    is resumable without calling the reasoner again.

CLI:
    `python -m kaya.kaya submit kaya/ladder_smoke.py -- [options]`

Arguments:
    --questions N: optional leading smoke-question limit. Omit to run every
        frozen smoke question.
    --representation R: repeatable rung filter among T, TL, TLV, V. If omitted,
        all four rungs run.
    --fresh-cache: write to `results/cache/m4_ladder_smoke/results.jsonl`
        instead of the default orchestrator cache.
    --no-cache-check: skip the second cache-hit-only pass.
"""

# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=m4-ladder-smoke

from __future__ import annotations

import argparse
import json

from config import ExperimentConfig
from experiments.runner import run_oracle_ladder
from experiments.smoke import load_smoke_questions
from kaya.prestage import prepare_tool_cache_env
from pipeline.orchestrator import Orchestrator, ResultCache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, help="number of leading smoke questions")
    parser.add_argument(
        "--representation",
        action="append",
        choices=("T", "TL", "TLV", "V"),
        help="representation rung to run; repeatable",
    )
    parser.add_argument(
        "--fresh-cache",
        action="store_true",
        help="write to results/cache/m4_ladder_smoke/results.jsonl",
    )
    parser.add_argument(
        "--no-cache-check",
        action="store_true",
        help="do not run the second pass that verifies every cell is a cache hit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(smoke=True)
    prepare_tool_cache_env(config.paths.hf_home)
    questions = load_smoke_questions(config.paths.data_dir)
    if args.questions is not None:
        questions = questions[: max(1, args.questions)]
    representations = tuple(args.representation or config.representations)
    cache = None
    if args.fresh_cache:
        cache = ResultCache(config.paths.cache_dir / "m4_ladder_smoke" / "results.jsonl")
    orchestrator = Orchestrator(config, cache=cache)

    print(
        json.dumps(
            {
                "event": "start",
                "questions": len(questions),
                "representations": list(representations),
                "reasoner_spec": orchestrator.reasoner.spec,
                "cache": str(orchestrator.cache.path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    batch = run_oracle_ladder(
        config,
        questions,
        orchestrator=orchestrator,
        representations=representations,
    )
    for row in batch.rows:
        print(
            json.dumps(
                {
                    "event": "row",
                    "question_id": row.question_id,
                    "doc_id": row.doc_id,
                    "condition": row.condition,
                    "representation": row.representation,
                    "model_spec": row.model_spec,
                    "page_indices": list(row.page_indices),
                    "answer": row.answer,
                    "input_text_tokens": row.input_text_tokens,
                    "input_visual_tokens": row.input_visual_tokens,
                    "output_tokens": row.output_tokens,
                    "latency_s": row.latency_s,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "pass",
                "pass": 1,
                "rows": len(batch.rows),
                "computed": batch.computed,
                "cache_hits": batch.cache_hits,
                "cache_rows": batch.cache_rows,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.no_cache_check:
        repeat = run_oracle_ladder(
            config,
            questions,
            orchestrator=orchestrator,
            representations=representations,
        )
        print(
            json.dumps(
                {
                    "event": "cache_check",
                    "rows": len(repeat.rows),
                    "computed": repeat.computed,
                    "cache_hits": repeat.cache_hits,
                    "cache_rows": repeat.cache_rows,
                    "ok": repeat.computed == 0 and repeat.cache_hits == len(repeat.rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if repeat.computed != 0 or repeat.cache_hits != len(repeat.rows):
            return 1
    print(json.dumps({"event": "complete"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
