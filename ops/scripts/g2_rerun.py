"""One GPU job that completes the qwen3-embedding retrieval memo, then runs the G2
inference reusing it.

Stage 1 fills `results/cache/<run_tag>/retrieval/qwen3-embedding__dpi200.jsonl` in place
(resuming from whatever it has) and writes the qwen3-embedding + joint benchmark rows to a
separate file (`--complete-filename`, never `retrieval.jsonl`). Stage 2 runs
`ops.generate --skip-retrieval`, so it reuses the memo and never re-runs stage-1 or
rewrites `retrieval.jsonl`. The two stages share one allocation: the retriever frees the
GPU before the reasoner loads.

    kaya submit --gres gpu:v100:2 --time 48:00:00 ops/scripts/g2_rerun.py -- --spec ops/specs/kaya_g2_full.yaml
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="the G2 spec (reduced-k if you edited it)")
    parser.add_argument("--skip-complete", action="store_true",
                        help="skip stage 1 (the retrieval memo is already complete)")
    parser.add_argument("--skip-oom", action="store_true",
                        help="pass --skip-oom to stage 2: drop cells already recorded as oom so the "
                             "reduced-k inference does not re-attempt the high-k OOM cells")
    parser.add_argument("--complete-text-methods", default="qwen3-embedding",
                        help="text methods to complete in stage 1 (comma list)")
    parser.add_argument("--complete-vision-methods", default="colqwen3",
                        help="vision methods to complete in stage 1, for the joint (comma list)")
    parser.add_argument("--complete-filename", default="retrieval_qwen3.jsonl",
                        help="stage-1 benchmark output file (never retrieval.jsonl)")
    parser.add_argument("--fresh-complete", action="store_true",
                        help="pass --fresh to stage 1: wipe the completed methods' memos and re-rank "
                             "the whole rung (so no rows are left over from an earlier run)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from experiments.engine.paths import log
    from ops import generate
    from ops.scripts import complete_retrieval

    verbose = ["--verbose"] if args.verbose else []

    if not args.skip_complete:
        log.info("g2-rerun stage 1: completing retrieval memo for %s (joint with %s)",
                 args.complete_text_methods, args.complete_vision_methods)
        rc = complete_retrieval.main([
            "--spec", args.spec,
            "--text-methods", args.complete_text_methods,
            "--vision-methods", args.complete_vision_methods,
            "--joints", "matched",
            "--filename", args.complete_filename,
            *(["--fresh"] if args.fresh_complete else []),
            *verbose,
        ])
        if rc != 0:
            log.warning("g2-rerun: stage 1 returned %s; continuing to inference anyway", rc)

    log.info("g2-rerun stage 2: G2 inference (--skip-retrieval, memo reused, retrieval.jsonl untouched)")
    skip_oom = ["--skip-oom"] if args.skip_oom else []
    return generate.main(["--spec", args.spec, "--skip-retrieval", *skip_oom, *verbose])


if __name__ == "__main__":
    raise SystemExit(main())
