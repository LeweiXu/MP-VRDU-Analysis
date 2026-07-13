"""Rebuild the retrieval memo for specific methods without touching retrieval.jsonl.

Runs the retrieval benchmark for the methods you name, writing the scored rows to a
separate `--filename` while the per-method rankings go to the shared memo
(`results/cache/<run_tag>/retrieval/`, keyed independently of the output file). Use it
to complete a method that OOM'd during a G2 run (e.g. Qwen3-Embedding-4B on a V100) on a
bigger GPU: the memo resumes from whatever it already has, and the main retrieval.jsonl
is left alone. Merge the new rows into retrieval.jsonl afterward if you want them there.

    python -m ops.scripts.complete_retrieval --spec ops/specs/kaya_g2_full.yaml \\
        --text-methods qwen3-embedding --vision-methods colqwen3 --joints matched
"""

# kaya: target=gpu
# kaya: env=true
# kaya: offline=true

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _split(value: str) -> tuple[str, ...]:
    return tuple(m.strip() for m in (value or "").split(",") if m.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in (value or "").replace(" ", "").split(",") if x)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="the run's spec (gives config + question pool + run_tag)")
    parser.add_argument("--text-methods", default="", help="comma list, e.g. qwen3-embedding")
    parser.add_argument("--vision-methods", default="", help="comma list, e.g. colqwen3")
    parser.add_argument("--joints", choices=("none", "matched"), default="none",
                        help="'matched' zips the text+vision method lists into joints")
    parser.add_argument("--single-ks", default="", help="override single-method k values (default: the spec's)")
    parser.add_argument("--joint-ks", default="", help="override joint k values (default: the spec's)")
    parser.add_argument("--parser-dpi", type=int, default=None,
                        help="override the spec's render DPI (for the visual-retrieval DPI sweep); "
                             "each DPI keys its own memo, so runs never collide")
    parser.add_argument("--filename", default="retrieval_extra.jsonl",
                        help="benchmark output file beside the run (NOT retrieval.jsonl)")
    parser.add_argument("--fresh", action="store_true",
                        help="delete each method's memo first and re-rank the whole rung, so no "
                             "rows are left over from an earlier run (e.g. mixing capped/uncapped)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from experiments.corpus.yaml_spec import config_from_spec, load_yaml_specs
    from experiments.engine.paths import configure_logging, experiment_paths, log
    from experiments.engine.side_artifacts import resolve_joints, write_retrieval_eval
    from experiments.registry import resolve
    from ops.generate import ensure_cache_env, load_corpus

    configure_logging(args.verbose)

    if args.filename == "retrieval.jsonl":
        raise SystemExit("refusing to write retrieval.jsonl; pick a different --filename")
    text_methods = _split(args.text_methods)
    vision_methods = _split(args.vision_methods)
    if not text_methods and not vision_methods:
        raise SystemExit("give --text-methods and/or --vision-methods")

    spec = load_yaml_specs(args.spec)[0]
    config = config_from_spec(spec)
    if args.parser_dpi is not None:
        from dataclasses import replace
        config = replace(config, dpi=int(args.parser_dpi))
    ensure_cache_env(config)

    single_ks = _ints(args.single_ks) or (tuple(config.k_values) or (1,))
    joint_ks = _ints(args.joint_ks) or (tuple(config.joint_k_values) or (1, 3, 5))
    joints = resolve_joints("matched", text_methods, vision_methods) if args.joints == "matched" else ()

    corpus = load_corpus(config.dataset, config.paths.data_dir, require_complete=False, cache={})
    task = resolve(spec.task_name)[0]
    pool = list(task.resolve_questions(config, corpus))
    paths = experiment_paths(config, task.name)

    log.info("complete-retrieval: %d questions | text=%s vision=%s joints=%s | single_ks=%s joint_ks=%s -> %s",
             len(pool), list(text_methods), list(vision_methods), list(joints), single_ks, joint_ks, args.filename)
    write_retrieval_eval(
        config, pool, paths.side_dir,
        single_ks=single_ks, joint_ks=joint_ks,
        text_methods=text_methods, vision_methods=vision_methods,
        joint_pairs=joints, filename=args.filename, fresh=args.fresh,
    )
    log.info("complete-retrieval: wrote %s; memo updated under %s",
             paths.side_dir / args.filename, config.paths.cache_dir / "retrieval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
