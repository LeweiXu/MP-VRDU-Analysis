"""GPU generation entry point: runs a task over a small corpus and caches rows."""

from __future__ import annotations

import argparse
import logging
import os

from config import DEPLOYMENT_RESOLUTION, ExperimentConfig, hf_cache_environ
from data.binning import stamp_bins
from data.loader import load_longdocurl, load_mmlongbench
from experiments.engine.driver import generate
from experiments.engine.paths import configure_logging
from experiments.registry import resolve

log = logging.getLogger("mpvrdu.generate")

# Which loader stages each dataset the `dataset` axis can select.
DATASET_LOADERS = {
    "mmlongbench": load_mmlongbench,
    "longdocurl": load_longdocurl,
}


def load_corpus(dataset: str, data_dir, *, require_complete: bool, cache: dict) -> list:
    """Load (and bin) a dataset's questions once, memoized per dataset name."""

    if dataset not in cache:
        loader = DATASET_LOADERS.get(dataset)
        if loader is None:
            raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASET_LOADERS)}")
        cache[dataset] = stamp_bins(loader(data_dir), require_complete=require_complete)
    return cache[dataset]


def ensure_cache_env(config: ExperimentConfig) -> None:
    """Point HF and the parser subprocesses at the in-project cache if unset.

    Lets a direct run (e.g. on the H100 supervisor after prestage --local) find
    the staged weights with no manual exports. Uses setdefault so a Kaya job's own
    exports, or anything the operator set, always win.
    """

    for name, value in hf_cache_environ(config.paths.hf_home).items():
        os.environ.setdefault(name, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=None,
                        help="YAML spec file; provides task + config (CLI flags below override --limit only)")
    parser.add_argument("--task", default="all", help="task name or group (e.g. G1_oracle_ladder, reasoners, all)")
    parser.add_argument("--reasoner-spec", default="qwen3vl-2b-local")
    parser.add_argument("--quantization", choices=("4bit", "8bit"), default=None)
    parser.add_argument("--visual-resolution", default=DEPLOYMENT_RESOLUTION,
                        help="resolution preset (low/med/high)")
    parser.add_argument("--judge-spec", default="stub")
    parser.add_argument("--run-tag", default=None, help="per-run cache namespace (isolates a run's cells)")
    parser.add_argument("--limit", type=int, default=None, help="cap questions per task (smoke/debug)")
    parser.add_argument("--failed-only", action="store_true",
                        help="re-run only the cells that failed (status != ok) in a previous run, "
                             "upgrading them in place; ok cells and side artifacts are left alone")
    parser.add_argument("--require-complete-annotations", action="store_true",
                        help="stop if the annotation sheet exists but misses some corpus docs; "
                             "default is to allow blank bins (annotations are an optional enrichment)")
    parser.add_argument("--allow-unlabelled", action="store_true",
                        help="deprecated no-op: unlabelled docs are allowed by default now")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    if args.spec:
        from experiments.corpus.yaml_spec import config_from_spec, corpus_limit, load_yaml_specs

        runs = [
            (config_from_spec(spec), spec.task_name,
             args.limit if args.limit is not None else corpus_limit(spec))
            for spec in load_yaml_specs(args.spec)
        ]
    else:
        config = ExperimentConfig(
            reasoner_spec=args.reasoner_spec,
            quantization=args.quantization,
            visual_resolution=args.visual_resolution,
            judge_spec=args.judge_spec,
            run_tag=args.run_tag,
        )
        runs = [(config, args.task, args.limit)]

    # The HF cache location is the same across runs (run_tag only nests the results
    # cache), so set the cache env once. Each run's corpus is loaded per its dataset
    # and memoized, so a dataset sweep reloads each dataset at most once.
    ensure_cache_env(runs[0][0])
    corpus_cache: dict[str, list] = {}
    for config, selector, limit in runs:
        questions = load_corpus(config.dataset, config.paths.data_dir,
                                require_complete=args.require_complete_annotations, cache=corpus_cache)
        if config.run_tag:
            log.info("run_tag=%s | task=%s | dataset=%s", config.run_tag, selector, config.dataset)
        for task in resolve(selector):
            generate(config, task, questions, limit=limit, failed_only=args.failed_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
