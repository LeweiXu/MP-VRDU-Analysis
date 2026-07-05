"""Generate (GPU phase) for one experiment or a group, on a Kaya compute node.

Purpose:
    The GPU half of the experiment pipeline: it runs a chosen experiment's cells
    (reasoner over conditioned pages) and any GPU side work (retrievers, the
    doc-type classifier) and caches the predictions per experiment. The judge and
    the table build run later, off the cluster (`cli.experiments --phase judge`
    after `kaya.kaya pull`). Nothing here needs the internet.

Pipeline role:
    Thin GPU entry point over `experiments.driver.run_generate`. Submitted with
    `kaya.kaya submit`; runs one experiment per job so a single table re-runs as
    its own small, fast-queueing job (or `--experiment all` in one job).

CLI:
    `python -m kaya.kaya submit kaya/generate.py -- --experiment SEL [--full]`

Arguments:
    --experiment SEL: experiment name (e.g. T1_headline) or group
        (all, section2, rq1, rq2, rq3, appendix). Default: all.
    --full: use the full config/corpus (8B). Default: smoke. A full mmlongbench
        run defaults to ~100 questions per Option-A bin (document-level subset).
    --questions N: global cap to the first N questions; overrides --per-bin-questions.
    --per-bin-questions N: full mmlongbench only. Keep ~N questions per bin by
        drawing whole documents (default 100). Pass 0 to run the whole corpus.
    --sample-seed N: pick which documents fill the per-bin subset (default 0);
        change it for a disjoint robustness subset.
    --quantization {4bit,8bit}: load the local reasoner quantized (bitsandbytes)
        so the 8B fits one 16GB V100. Appends a `-4bit`/`-8bit` suffix to the
        reasoner spec (its own cache rows). bf16 by default; mains stay bf16.
    --continue-on-error: for grouped runs, write a failure status for the
        failing experiment and continue to the next one.
"""

# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=generate

from __future__ import annotations

import argparse
import os

# Reduce CUDA fragmentation on the compute nodes. The allocator reads this when
# it first initializes (first CUDA alloc), so setting it before run_generate is
# enough; setdefault lets a submit script override it. This is the mitigation the
# OOM error message itself recommends; the real fix is the per-page pixel cap in
# ExperimentConfig.max_pixels.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from config import ExperimentConfig
from experiments.corpus import load_questions
from experiments.driver import configure_logging, run_generate
from kaya.prestage import prepare_tool_cache_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="all", help="experiment name or group (default: all)")
    parser.add_argument("--full", action="store_true", help="use the full config/corpus (default: smoke)")
    parser.add_argument("--questions", type=int, help="global cap: first N questions (overrides --per-bin-questions)")
    parser.add_argument("--per-bin-questions", type=int, help="full mmlongbench: ~N questions per Option-A bin by whole documents (default 100; 0 = whole corpus)")
    parser.add_argument("--sample-seed", type=int, help="which documents land in the per-bin subset (default 0; change for a robustness subset)")
    parser.add_argument("--quantization", choices=("4bit", "8bit"), help="load the local reasoner quantized so 8B fits one 16GB V100 (bf16 by default)")
    parser.add_argument("--continue-on-error", action="store_true", help="continue grouped runs after an experiment failure")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level per-cell/per-stage logging (smoke runs are verbose by default)")
    parser.add_argument("--quiet", action="store_true", help="force INFO-level logging even for smoke runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict = {"smoke": not args.full, "sample": args.questions}
    if args.per_bin_questions is not None:
        overrides["per_bin_sample"] = args.per_bin_questions or None
    if args.sample_seed is not None:
        overrides["sample_seed"] = args.sample_seed
    if args.quantization is not None:
        overrides["quantization"] = args.quantization
    config = ExperimentConfig(**overrides)
    # Smoke runs are verbose by default (they exist to surface failures); --quiet
    # opts out, --verbose forces DEBUG for a full run too.
    configure_logging(verbose=args.verbose or (config.smoke and not args.quiet))
    prepare_tool_cache_env(config.paths.hf_home)
    questions = load_questions(config, limit=args.questions)
    statuses = run_generate(config, args.experiment, questions, continue_on_error=args.continue_on_error)
    failed = [status for status in statuses if status.status != "success"]
    print(
        f"generated {args.experiment}: {len(questions)} questions "
        f"({'full' if args.full else 'smoke'}), {len(statuses) - len(failed)} succeeded, {len(failed)} failed"
    )
    for status in failed:
        print(f"failed {status.experiment}: {status.error_type}: {status.error} ({status.path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
