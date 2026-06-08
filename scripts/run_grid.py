#!/usr/bin/env python3
"""Run a whole experiment grid (many configs) sequentially. RESUMABLE.

Expands a suite YAML into runs, then executes each as a normal pipeline run with
a deterministic output path (results/grid/<substudy>/<name>__<hash>.jsonl). A run
is skipped if its output already exists and is complete, so you can stop/restart
freely and chip away at a long local grid.

    # local 3B (as configured in the suite):
    python scripts/run_grid.py --suite experiments/grid_local_3b.yaml
    # preview what would run, without running:
    python scripts/run_grid.py --suite experiments/grid_local_3b.yaml --dry-run
    # quick machinery check: only a few questions per run
    python scripts/run_grid.py --suite experiments/grid_local_3b.yaml --max-questions 4
    # only one sub-study:
    python scripts/run_grid.py --suite experiments/grid_local_3b.yaml --only A_retrieval
    # later, override the generator for Kaya without editing the suite:
    python scripts/run_grid.py --suite experiments/grid_local_3b.yaml \
        --generator kaya_vlm --model-id Qwen/Qwen2.5-VL-7B-Instruct
"""

import argparse
import gc
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.analysis import aggregate_dir, to_markdown_table
from mpvrdu.data.load import load_dataset
from mpvrdu.experiment import load_suite
from mpvrdu.logging_utils import get_logger
from mpvrdu.pipeline import run
from mpvrdu.results import read_rows

log = get_logger("run_grid")


def _free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:        # ImportError, or a CUDA error after an OOM
        pass


def _apply_overrides(cfg, args):
    if args.slice:
        cfg.data.slice = args.slice
    if args.max_questions is not None:
        cfg.data.max_questions = args.max_questions
    if args.generator:
        cfg.generation.generator = args.generator
    if args.model_id:
        cfg.generation.model_id = args.model_id
    if args.load_in_4bit:
        cfg.generation.load_in_4bit = True
    return cfg.validate()


def _out_path(root: Path, substudy: str, cfg) -> Path:
    return root / substudy / f"{cfg.name}__{cfg.hash()}.jsonl"


def _is_complete(path: Path, expected_n: int) -> bool:
    if not path.exists():
        return False
    try:
        return len(read_rows(path)) >= expected_n
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--results-root", default="results/grid")
    ap.add_argument("--only", action="append", default=None,
                    help="run only these substudies (repeatable)")
    ap.add_argument("--slice", default=None, help="override data.slice for all runs")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--generator", default=None, help="override generator (e.g. kaya_vlm)")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="ignore existing results")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.results_root)

    if not args.aggregate_only:
        runs = load_suite(args.suite)
        if args.only:
            runs = [(s, c) for s, c in runs if s in set(args.only)]
        for _, cfg in runs:
            _apply_overrides(cfg, args)

        log.info("suite expands to %d runs across %d sub-studies",
                 len(runs), len({s for s, _ in runs}))

        # dataset cache keyed by data signature (avoid reloading 1091 q each run)
        ds_cache: dict = {}

        def get_ds(cfg):
            key = (cfg.data.name, cfg.data.slice, cfg.data.split, cfg.data.max_questions)
            if key not in ds_cache:
                ds_cache[key] = load_dataset(cfg.data)
            return ds_cache[key]

        if args.dry_run:
            for sub, cfg in runs:
                path = _out_path(root, sub, cfg)
                state = "done" if _is_complete(path, 1) else "TODO"
                print(f"[{state:4s}] {sub:16s} {cfg.name:45s} {cfg.hash()}")
            print(f"\n{len(runs)} runs total. (dry-run; nothing executed)")
            return

        done = failed = skipped = 0
        for idx, (sub, cfg) in enumerate(runs, 1):
            path = _out_path(root, sub, cfg)
            ds = get_ds(cfg)
            n = len(ds)
            if not args.rerun and _is_complete(path, n):
                log.info("[%d/%d] SKIP (done): %s", idx, len(runs), cfg.name)
                skipped += 1
                continue
            log.info("[%d/%d] RUN %s/%s (n=%d)", idx, len(runs), sub, cfg.name, n)
            try:
                run(cfg, dataset=ds, out_path=path)
                done += 1
            except Exception:
                failed += 1
                log.error("FAILED %s/%s:\n%s", sub, cfg.name, traceback.format_exc())
            finally:
                _free_cuda()
        log.info("grid finished: %d ran, %d skipped, %d failed", done, skipped, failed)

    # ---- aggregate everything under the results root ----
    summaries = aggregate_dir(root, pattern="**/*.jsonl")
    if summaries:
        table = to_markdown_table(summaries)
        out_md = root / "summary.md"
        out_md.write_text(table + "\n", encoding="utf-8")
        print("\n" + table)
        print(f"\n-> wrote {out_md}")
    else:
        print("no completed results to aggregate yet.")


if __name__ == "__main__":
    main()
