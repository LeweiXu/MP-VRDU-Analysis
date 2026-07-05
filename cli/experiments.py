# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=generate
"""Run paper-table experiments: generate on a GPU, judge/build anywhere.

Purpose:
    The single entry point for running one experiment, an RQ group, or all of
    them, for the smoke or the full corpus. It drives the two phases:

    - `--phase generate` (needs a GPU): caches predictions per experiment. This
      is the half a cluster submits (`kaya.kaya submit cli/experiments.py --
      --phase generate ...`); it needs no internet.
    - `--phase judge` (needs internet + a judge key, no GPU): scores the cached
      predictions and writes the table CSVs.
    - `--phase all`: both, in one process (a machine with GPU + internet).

    The intended Kaya flow keeps the heavy half on the cluster and the light half
    local: `kaya.kaya submit cli/experiments.py -- --phase generate --experiment
    X`, then `kaya.kaya pull`, then `python -m cli.experiments --phase judge
    --experiment X` locally.

Pipeline role:
    Thin wrapper over `experiments.driver` + `experiments.registry`. Same code
    serves smoke and full; `--full` selects the full config/corpus. Submitted
    with `kaya.kaya submit` for the generate phase (one experiment per job so a
    single table re-runs as its own small, fast-queueing job, or
    `--experiment all` in one job).

CLI:
    `python -m cli.experiments [--experiment SEL] [--phase P] [--full] [options]`

Arguments:
    --experiment SEL: an experiment name (e.g. T1_headline) or a group
        (all, section2, rq1, rq2, rq3, appendix). Default: all.
    --phase {generate,judge,all}: which phase(s) to run. Default: all. A Kaya
        submit passes `--phase generate` (the GPU half; judge needs internet).
    --full: use the full config/corpus (8B). Default: smoke. A full mmlongbench
        run defaults to ~100 questions per Option-A bin (document-level subset).
    --judge SPEC: judge for the judge phase: gemini (default), gpt-4o-mini, stub.
    --questions N: global cap to the first N questions; overrides --per-bin-questions.
    --per-bin-questions N: full mmlongbench only. Keep ~N questions per bin by
        drawing whole documents (default 100). Pass 0 to run the whole corpus.
    --sample-seed N: pick which documents fill the per-bin subset (default 0);
        change it for a disjoint robustness subset.
    --quantization {4bit,8bit}: load the local reasoner quantized (bitsandbytes)
        so the 8B fits one 16GB V100. Appends a `-4bit`/`-8bit` suffix to the
        reasoner spec (its own cache rows). Must match between generate and judge
        phases so the judge reads the right predictions. bf16 by default.
    --visual-resolution {full,high,med,low,min}: fix the per-page vision-token
        budget for every reasoner, overriding the size-aware default. Lower =
        more downscaling = fewer vision tokens per page (fits a tighter GPU
        budget). `high` is the current 8B default; unset keeps the size-aware
        default. Not part of the cache key, so clear the cache when changing it
        for the same reasoner spec.
    --continue-on-error: generate/all grouped runs continue after an experiment
        failure and record its phase status; the judge phase skips cells that
        generate never produced (partial cache) so a partial table still builds.
    --run-tag TAG: namespace this run's whole cache tree under
        `results/cache/<TAG>/` (and tables under `results/tables/<mode>-<TAG>/`),
        so two runs sharing an experiment selection never write the same files.
        Use it when submitting two full runs that could run at once (the render
        cache is a non-atomic check-then-write). Judge with the same --run-tag.
"""

from __future__ import annotations

import argparse
import os

# Reduce CUDA fragmentation on the compute nodes. The allocator reads this when
# it first initializes (first CUDA alloc), so setting it before the generate
# phase is enough; setdefault lets a submit script override it. This is the
# mitigation the OOM error message itself recommends; the real fix is the
# per-page pixel cap in ExperimentConfig.max_pixels. Harmless for a judge-only
# run (no CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from config import ExperimentConfig
from experiments.corpus import load_questions
from experiments.driver import configure_logging, run_generate, run_judge
from scripts.prestage import prepare_tool_cache_env
from pipeline.judge import get_judge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="all", help="experiment name or group (default: all)")
    parser.add_argument("--phase", choices=("generate", "judge", "all"), default="all")
    parser.add_argument("--full", action="store_true", help="use the full config/corpus (default: smoke)")
    parser.add_argument("--judge", default="gemini", help="judge: gemini (default), gpt-4o-mini, or stub")
    parser.add_argument("--questions", type=int, help="global cap: first N questions (overrides --per-bin-questions)")
    parser.add_argument("--per-bin-questions", type=int, help="full mmlongbench: ~N questions per Option-A bin by whole documents (default 100; 0 = whole corpus)")
    parser.add_argument("--sample-seed", type=int, help="which documents land in the per-bin subset (default 0; change for a robustness subset)")
    parser.add_argument("--quantization", choices=("4bit", "8bit"), help="load the local reasoner quantized so 8B fits one 16GB V100 (bf16 by default)")
    parser.add_argument("--visual-resolution", choices=("full", "high", "med", "low", "min"), help="fix the per-page vision-token budget (overrides the size-aware default; lower = more downscaling)")
    parser.add_argument("--run-tag", help="namespace this run's cache tree (results/cache/<TAG>/) so parallel full runs don't share files; judge with the same tag")
    parser.add_argument("--continue-on-error", action="store_true", help="generate: continue after an experiment failure; judge: skip cells with no cached prediction (partial table)")
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
    if args.visual_resolution is not None:
        overrides["visual_resolution"] = args.visual_resolution
    if args.run_tag is not None:
        overrides["run_tag"] = args.run_tag
    config = ExperimentConfig(**overrides)
    # Smoke runs are verbose by default (they exist to surface failures); --quiet
    # opts out, --verbose forces DEBUG for a full run too.
    configure_logging(verbose=args.verbose or (config.smoke and not args.quiet))
    # Point tool/model caches at the root-relative staged weights for local runs.
    prepare_tool_cache_env(config.paths.hf_home)
    questions = load_questions(config, limit=args.questions)

    if args.phase in ("generate", "all"):
        statuses = run_generate(config, args.experiment, questions, continue_on_error=args.continue_on_error)
        failed = [status for status in statuses if status.status != "success"]
        print(
            f"generated {args.experiment}: {len(questions)} questions "
            f"({'full' if args.full else 'smoke'}), {len(statuses) - len(failed)} succeeded, {len(failed)} failed"
        )
        for status in failed:
            print(f"failed {status.experiment}: {status.error_type}: {status.error} ({status.path})")
    if args.phase in ("judge", "all"):
        written = run_judge(
            config,
            args.experiment,
            questions,
            judge_impl=get_judge(args.judge),
            continue_on_error=args.continue_on_error,
        )
        for key, path in sorted(written.items()):
            print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
