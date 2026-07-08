"""Run Section-2 gate tooling from cached artifacts or pilot jobs.

Purpose:
    Provides operational commands for the first three full-run gates: F1 frontier
    divergence from Table 1, F2 judge-human agreement sheet creation/scoring, and
    F3 classifier pilot execution/scoring. The expensive model work remains in the
    experiment runner or classifier path; this CLI records the gate artifacts.

Pipeline role:
    Complements the experiment roles: after G1 has generated/judged full rows, this
    module evaluates the frontier gate and creates the agreement sample. It also
    runs or scores the classifier feasibility pilot before predicted routing is
    trusted.

CLI:
    `python -m gates <frontier|agreement-sample|agreement-score|classifier-pilot|classifier-score> ...`

Arguments:
    See `python -m gates --help` and each subcommand's help text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ROOT, ExperimentConfig
from covariates.classifier import QwenDocTypeClassifier
from experiments.corpus import load_questions
from gates.core import (
    agreement_sheet_rows,
    classifier_pilot_sample,
    frontier_divergence_gate,
    load_table1_frontiers,
    question_type_label,
    read_classifier_records,
    render_agreement_packet,
    run_classifier_pilot,
    score_agreement_sheet,
    score_classifier_records,
    stratified_question_sample,
    write_csv_records,
    write_gate_json,
)
from experiments.paths import experiment_paths
from reporting.tables import load_result_rows
from scripts.prestage import prepare_tool_cache_env


# Table 1 / F1 / F2 all source the primary oracle-ladder task.
GENERATION = "G1_sufficiency"
DEFAULT_GATE_DIR = ROOT / "results" / "gates"


def _paths_config(full: bool, run_tag: str | None) -> ExperimentConfig:
    """Config used only to resolve run-tag-aware cache/table paths (no HF env)."""

    return ExperimentConfig(smoke=not full, run_tag=run_tag)


def default_results_path(full: bool, run_tag: str | None) -> Path:
    """Resolve G1's judged results.jsonl under the current mode/run-tag."""

    return experiment_paths(_paths_config(full, run_tag), GENERATION).results


def default_table1_path(full: bool, run_tag: str | None) -> Path:
    """Resolve the Table-1 CSV under the current mode/run-tag."""

    return experiment_paths(_paths_config(full, run_tag), GENERATION).table_dir / "table1_headline.csv"


def build_parser() -> argparse.ArgumentParser:
    """Return the gate CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    frontier = sub.add_parser("frontier", help="evaluate F1 from a Table-1 CSV")
    frontier.add_argument("--table", type=Path, default=None, help="Table-1 CSV path (default: resolved from mode/run-tag)")
    frontier.add_argument("--smoke", action="store_true", help="resolve the smoke table instead of full")
    frontier.add_argument("--run-tag", help="run tag namespacing results/tables/<mode>-<tag>/")
    frontier.add_argument("--json-output", type=Path, help="optional gate JSON output path")

    agreement_sample = sub.add_parser("agreement-sample", help="write the F2 human-labelling sheet")
    agreement_sample.add_argument("--full", action="store_true", help="sample from the full corpus")
    agreement_sample.add_argument("--run-tag", help="run tag namespacing the cached results")
    agreement_sample.add_argument("--results", type=Path, default=None, help="judged ResultRow JSONL (default: resolved from mode/run-tag)")
    agreement_sample.add_argument("--output", type=Path, default=DEFAULT_GATE_DIR / "agreement_sample.csv")
    agreement_sample.add_argument("--n", type=int, default=200, help="number of question rows to sample")
    agreement_sample.add_argument("--seed", type=int, default=0)
    agreement_sample.add_argument("--condition", default="oracle", help="result-row condition filter")
    agreement_sample.add_argument("--representation", default="TLV", help="result-row representation filter")
    agreement_sample.add_argument("--model-spec", help="optional result-row model_spec filter")
    agreement_sample.add_argument(
        "--questions-only",
        action="store_true",
        help="write only the stratified question frame, without requiring result rows",
    )
    agreement_sample.add_argument(
        "--render",
        dest="render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also render the sampled documents into a viewing packet (default: on)",
    )
    agreement_sample.add_argument(
        "--render-dir",
        type=Path,
        default=DEFAULT_GATE_DIR / "agreement_view",
        help="where to write the viewing packet (page PNGs + agreement_view.md)",
    )

    agreement_score = sub.add_parser("agreement-score", help="score a completed F2 sheet")
    agreement_score.add_argument("--sheet", type=Path, required=True, help="completed agreement sheet CSV")
    agreement_score.add_argument("--threshold", type=float, default=0.75)
    agreement_score.add_argument("--json-output", type=Path, help="optional gate JSON output path")

    classifier_pilot = sub.add_parser("classifier-pilot", help="run the F3 classifier pilot")
    classifier_pilot.add_argument("--full", action="store_true", help="sample from the full corpus")
    classifier_pilot.add_argument("--n-docs", type=int, default=100)
    classifier_pilot.add_argument("--seed", type=int, default=0)
    classifier_pilot.add_argument("--output", type=Path, default=DEFAULT_GATE_DIR / "classifier_pilot.csv")
    classifier_pilot.add_argument("--json-output", type=Path, default=DEFAULT_GATE_DIR / "classifier_gate.json")
    classifier_pilot.add_argument(
        "--sample-only",
        action="store_true",
        help="write the 100-document pilot frame without running the classifier",
    )

    classifier_score = sub.add_parser("classifier-score", help="score existing F3 classifier records")
    classifier_score.add_argument("--predictions", type=Path, required=True, help="classifier CSV or JSONL")
    classifier_score.add_argument("--threshold", type=float, default=0.70)
    classifier_score.add_argument("--json-output", type=Path, help="optional gate JSON output path")
    return parser


def _config(full: bool, run_tag: str | None = None) -> ExperimentConfig:
    """Return the smoke/full experiment config for gate commands."""

    config = ExperimentConfig(smoke=not full, run_tag=run_tag)
    prepare_tool_cache_env(config.paths.hf_home)
    return config


def _emit_gate(result, json_output: Path | None) -> None:  # noqa: ANN001
    """Print a gate result and optionally write it to JSON."""

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if json_output is not None:
        write_gate_json(result, json_output)
        print(f"wrote: {json_output}")


def _question_frame(sample):  # noqa: ANN001
    """Return CSV rows for sampled questions without model predictions."""

    return [
        {
            "question_id": question.id,
            "doc_id": question.doc_id,
            "doc_type": question.doc_type,
            "question_type": question_type_label(question),
            "question": question.question,
            "gold_answer": question.gold_answer,
            "human_label": "",
            "notes": "",
        }
        for question in sample
    ]


def main(argv: list[str] | None = None) -> int:
    """Run one Section-2 gate command."""

    args = build_parser().parse_args(argv)

    if args.command == "frontier":
        table = args.table or default_table1_path(not args.smoke, args.run_tag)
        result = frontier_divergence_gate(load_table1_frontiers(table))
        _emit_gate(result, args.json_output)
        return 0

    if args.command == "agreement-sample":
        config = _config(args.full, args.run_tag)
        questions = load_questions(config)
        if args.questions_only:
            sample = stratified_question_sample(questions, n=args.n, seed=args.seed)
            rows = _question_frame(sample)
        else:
            results_path = args.results or default_results_path(args.full, args.run_tag)
            result_rows = load_result_rows(results_path)
            rows = agreement_sheet_rows(
                questions,
                result_rows,
                n=args.n,
                seed=args.seed,
                condition=args.condition or None,
                representation=args.representation or None,
                model_spec=args.model_spec,
            )
        write_csv_records(rows, args.output)
        print(f"wrote {len(rows)} rows: {args.output}")
        if args.render and not args.questions_only:
            packet = render_agreement_packet(config, rows, args.render_dir, task=GENERATION)
            print(f"wrote viewing packet: {packet}  (open in VSCode, label human_label in the CSV)")
        return 0

    if args.command == "agreement-score":
        result = score_agreement_sheet(args.sheet, threshold=args.threshold)
        _emit_gate(result, args.json_output)
        return 0

    if args.command == "classifier-pilot":
        config = _config(args.full)
        questions = load_questions(config)
        if args.sample_only:
            sample = classifier_pilot_sample(questions, n_docs=args.n_docs, seed=args.seed)
            rows = _question_frame(sample)
            write_csv_records(rows, args.output)
            print(f"wrote {len(rows)} pilot documents: {args.output}")
            return 0

        classifier = QwenDocTypeClassifier(
            data_dir=config.paths.data_dir,
            cache_dir=config.paths.cache_dir,
            dpi=config.dpi,
        )
        records = run_classifier_pilot(questions, classifier, n_docs=args.n_docs, seed=args.seed)
        write_csv_records(records, args.output)
        result = score_classifier_records(records)
        _emit_gate(result, args.json_output)
        print(f"wrote {len(records)} classifier records: {args.output}")
        return 0

    if args.command == "classifier-score":
        records = read_classifier_records(args.predictions)
        result = score_classifier_records(records, threshold=args.threshold)
        _emit_gate(result, args.json_output)
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
