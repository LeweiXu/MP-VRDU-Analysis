#!/usr/bin/env python3
"""Download MMLongBench-Doc into a local cache (run where there IS internet).

On Kaya, run this on the LOGIN node into /group/<project>/ then run offline
(HF_HUB_OFFLINE=1). See context.md §12.

    python scripts/download_data.py --out data/mmlongbench

NAME-COLLISION GUARD (context.md §10): this only ever pulls
`yubo2333/MMLongBench-Doc` (the doc-VQA benchmark), never "MMLongBench".
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.data.dataset import HF_REPO_ID, load_from_hf
from mpvrdu.logging_utils import get_logger

log = get_logger("download")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mmlongbench",
                    help="cache dir to download into")
    ap.add_argument("--repo", default=HF_REPO_ID)
    args = ap.parse_args()

    ds = load_from_hf(args.out, repo_id=args.repo)
    log.info("downloaded %d questions, %d docs to %s",
             len(ds), len(ds.documents), args.out)
    log.info("type counts: %s", ds.type_counts())


if __name__ == "__main__":
    main()
