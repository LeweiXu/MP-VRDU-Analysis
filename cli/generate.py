# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=generate
"""Generate phase (GPU): cache predictions from a YAML generation spec.

Purpose:
    The thin GPU entry point a cluster submits. It loads a spec file, builds the
    dynamic generation tasks, writes manifests beside their caches, and hands them
    to the shared driver. Nothing here needs the internet. All run configuration
    (mode, corpus subset, run tag, resolution, quantization) lives in the YAML
    spec, so there are no config flags here.

CLI:
    `python -m cli.generate --spec specs/full_generation.yaml`
    Kaya: `kaya.kaya submit cli/generate.py -- --spec specs/full_generation.yaml`

Arguments:
    --spec PATH: YAML generation spec (required). See `build_parser` for the few
        run-behavior flags (`--continue-on-error`, `--verbose`, `--quiet`).
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
from experiments.driver import run_generate_tasks
from experiments.paths import configure_logging
from experiments.yaml_spec import load_yaml_experiment
from scripts.prestage import prepare_tool_cache_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="YAML generation spec (carries all run config)")
    parser.add_argument("--continue-on-error", action="store_true", help="continue after a task failure and record its status")
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level per-cell/per-stage logging (smoke runs are verbose by default)")
    parser.add_argument("--quiet", action="store_true", help="force INFO-level logging even for smoke runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_yaml_experiment(args.spec)
    config = spec.config
    # Smoke runs are verbose by default (they exist to surface failures); --quiet
    # opts out, --verbose forces DEBUG for a full run too.
    configure_logging(verbose=args.verbose or (config.smoke and not args.quiet))
    prepare_tool_cache_env(config.paths.hf_home)
    questions = load_questions(config)
    statuses = run_generate_tasks(
        config,
        spec.tasks,
        questions,
        continue_on_error=args.continue_on_error,
        before_task=spec.write_manifest,
    )
    failed = [status for status in statuses if status.status != "success"]
    print(
        f"generated {args.spec}: {len(spec.tasks)} run(s), "
        f"{len(statuses) - len(failed)} succeeded, {len(failed)} failed"
    )
    for status in failed:
        print(f"failed {status.experiment}: {status.error_type}: {status.error} ({status.path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
