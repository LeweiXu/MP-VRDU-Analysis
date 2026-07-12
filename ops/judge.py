"""Judge phase: score a run's `predictions.jsonl` into `results.jsonl`.

Scores each ok cell with the chosen judge (stub / gemini / gpt-4o-mini) and writes
the full `ResultRow`s; failed cells pass through unscored, so `results.jsonl` stays a
strict superset of `predictions.jsonl`. Loads no reasoner; targets a run by `--spec`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from experiments.engine.paths import configure_logging, experiment_paths, log
from experiments.engine.paths import result_key as make_result_key
from schema import Score


def load_env_file(path: Path) -> None:
    """Load `KEY=VALUE` lines from a `.env` file into `os.environ` (real env wins).

    The judge needs the Gemini/OpenAI key, which lives in the repo `.env` rather than
    the shell. `setdefault` so an already-exported variable is never overridden.
    """

    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip("'\""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True,
                        help="YAML spec whose runs' predictions.jsonl should be scored")
    parser.add_argument("--judge-spec", default="stub", help="judge: stub (default), gemini, gpt-4o-mini")
    parser.add_argument("--verbose", action="store_true")
    return parser


def judge_run(config, task_name: str, questions: dict, judge) -> int:
    """Score one run's predictions.jsonl into results.jsonl; return rows written."""

    from pipeline.orchestrator import PredictionCache, ResultCache

    paths = experiment_paths(config, task_name)
    if not paths.predictions.exists():
        log.warning("judge %s: no predictions at %s (did generate run?)", task_name, paths.predictions)
        return 0
    predictions = PredictionCache(paths.predictions)
    results = ResultCache(paths.results)
    written = 0
    for record in predictions:
        question = questions.get(record.question_id)
        if question is None:
            raise KeyError(f"{task_name}: question {record.question_id!r} not found in dataset {config.dataset!r}")
        if record.status == "ok":
            score = judge.score(question, record.as_prediction())
        else:
            # A failed cell has nothing to score; carry it through with the judge
            # spec set so results.jsonl has one row per predictions.jsonl row.
            score = Score(value=0.0, correct=False, abstained=False, judge_spec=judge.spec)
        result_key = make_result_key(
            record.question_id, record.doc_id, record.condition, record.representation,
            record.model_spec, record.page_indices, judge.spec, record.visual_resolution,
        )
        if results.get(result_key) is not None:
            continue
        results.put(record.to_result_row(score, result_key))
        written += 1
    log.info("judge %s: scored %d prediction(s) (judge=%s) -> %s",
             task_name, written, judge.spec, paths.results)
    return written


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    from config import ROOT

    # The judge key lives in the repo .env, not the shell, so load it before the judge
    # (which reads GEMINI_API_KEY / GEMINI_API_KEY_SECONDARY / OPENAI_API_KEY).
    load_env_file(ROOT / ".env")

    from experiments.corpus.yaml_spec import config_from_spec, load_yaml_specs
    from ops.generate import load_corpus
    from pipeline.judge import get_judge

    judge = get_judge(args.judge_spec)
    corpus_cache: dict[str, list] = {}
    total = 0
    for spec in load_yaml_specs(args.spec):
        config = config_from_spec(spec)
        questions = {q.id: q for q in
                     load_corpus(config.dataset, config.paths.data_dir, require_complete=False, cache=corpus_cache)}
        total += judge_run(config, spec.task_name, questions, judge)
    log.info("judge %s: %d row(s) written across runs", args.spec, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
