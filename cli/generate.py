# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=generate
"""Generate phase (GPU): cache predictions for one or more generation tasks.

Purpose:
    The thin GPU entry point a cluster submits. It parses args, builds the shared
    config, and hands off to `experiments.driver.run_generate`, which runs each
    selected `GenerationTask` (defined in `experiments/G*_*.py`). Nothing here
    needs the internet.

CLI:
    `python -m cli.generate [--generation SEL] [--full] [options]`
    Kaya: `kaya.kaya submit cli/generate.py -- --generation SEL ...`

Arguments:
    --generation SEL: a task (`G1_sufficiency`), a group (`all`, `reasoners`), or
        a comma list. --full uses the 8B/full corpus (smoke otherwise).
        --quantization / --visual-resolution / --run-tag tune the reasoner and
        cache namespace; --continue-on-error keeps a grouped run going after a
        task fails. See `build_parser`.
"""

from __future__ import annotations

import argparse
import os

# Reduce CUDA fragmentation on the compute nodes; the allocator reads this on the
# first CUDA alloc, so setting it before generation runs is enough. setdefault
# lets a submit script override it. (The real vision-token fix is
# --visual-resolution / the per-page pixel cap.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from experiments.corpus import load_questions
from experiments.driver import config_from_args, run_generate
from experiments.paths import configure_logging
from scripts.prestage import prepare_tool_cache_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", default="all", help="generation task or group (default: all)")
    parser.add_argument("--full", action="store_true", help="use the full config/corpus (default: smoke)")
    parser.add_argument("--questions", type=int, help="global cap: first N questions (overrides --per-bin-questions)")
    parser.add_argument("--per-bin-questions", type=int, help="full mmlongbench: ~N questions per Option-A bin by whole documents (default 100; 0 = whole corpus)")
    parser.add_argument("--sample-seed", type=int, help="which documents land in the per-bin subset (default 0)")
    parser.add_argument("--quantization", choices=("4bit", "8bit"), help="load the local reasoner quantized so 8B fits one 16GB V100 (bf16 by default)")
    parser.add_argument("--visual-resolution", choices=("full", "high", "med", "low", "min"), help="fix the per-page vision-token budget (overrides the size-aware default; lower = more downscaling)")
    parser.add_argument("--run-tag", help="namespace this run's cache tree (results/cache/<TAG>/) so parallel full runs don't share files; judge/build with the same tag")
    parser.add_argument("--continue-on-error", action="store_true", help="continue after a task failure and record its status")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level per-cell/per-stage logging (smoke runs are verbose by default)")
    parser.add_argument("--quiet", action="store_true", help="force INFO-level logging even for smoke runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    # Smoke runs are verbose by default (they exist to surface failures); --quiet
    # opts out, --verbose forces DEBUG for a full run too.
    configure_logging(verbose=args.verbose or (config.smoke and not args.quiet))
    prepare_tool_cache_env(config.paths.hf_home)
    questions = load_questions(config, limit=args.questions)
    statuses = run_generate(config, args.generation, questions, continue_on_error=args.continue_on_error)
    failed = [status for status in statuses if status.status != "success"]
    print(
        f"generated {args.generation}: {len(questions)} questions "
        f"({'full' if args.full else 'smoke'}), {len(statuses) - len(failed)} succeeded, {len(failed)} failed"
    )
    for status in failed:
        print(f"failed {status.experiment}: {status.error_type}: {status.error} ({status.path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
