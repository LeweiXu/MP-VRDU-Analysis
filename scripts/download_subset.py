#!/usr/bin/env python3
"""Download a SMALL local subset of MMLongBench-Doc (no 662 MB full pull).

Grabs the questions parquet (small), selects a few documents — by default the
smallest ones that span all three question types — downloads only those PDFs,
and writes a local dataset dir (samples.json + documents/) that the ordinary
loader reads via `slice: <path>`.

    python scripts/download_subset.py --out data/mmlongbench_subset --docs 3

Then run, e.g.:
    python -m mpvrdu.pipeline --config configs/local_subset_oracle.yaml
"""

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpvrdu.data.dataset import HF_REPO_ID, samples_from_parquet
from mpvrdu.logging_utils import get_logger

log = get_logger("download_subset")


def _qtype(ev) -> str:
    try:
        pages = ast.literal_eval(ev) if ev else []
    except (ValueError, SyntaxError):
        pages = []
    if not pages:
        return "unanswerable"
    return "single" if len(pages) == 1 else "cross"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mmlongbench_subset")
    ap.add_argument("--docs", type=int, default=3,
                    help="number of documents to include")
    ap.add_argument("--doc-ids", nargs="*", default=None,
                    help="explicit doc_ids (overrides auto-selection)")
    ap.add_argument("--repo", default=HF_REPO_ID)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    sibs = api.repo_info(args.repo, repo_type="dataset", files_metadata=True).siblings
    parquet_remote = next(s.rfilename for s in sibs if s.rfilename.endswith(".parquet"))
    sizes = {s.rfilename.split("/", 1)[1]: (s.size or 0)
             for s in sibs if s.rfilename.startswith("documents/")}

    parquet_path = hf_hub_download(args.repo, parquet_remote, repo_type="dataset")
    samples = samples_from_parquet(parquet_path)

    # group questions by doc + the types each doc covers
    by_doc: dict[str, list] = {}
    types: dict[str, set] = {}
    for s in samples:
        d = s["doc_id"]
        by_doc.setdefault(d, []).append(s)
        types.setdefault(d, set()).add(_qtype(s.get("evidence_pages")))

    if args.doc_ids:
        chosen = args.doc_ids
    else:
        all3 = [d for d in by_doc if len(types[d]) == 3]
        all3.sort(key=lambda d: sizes.get(d, 0))   # smallest first
        chosen = all3[: args.docs]

    out = Path(args.out)
    pdf_dir = out / "documents"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    subset_samples = []
    for doc in chosen:
        local = hf_hub_download(args.repo, f"documents/{doc}", repo_type="dataset")
        shutil.copy2(local, pdf_dir / doc)
        for s in by_doc[doc]:
            subset_samples.append({k: (v if not hasattr(v, "item") else v.item())
                                   for k, v in s.items()})
        log.info("included %s (%.2f MB, %d qs, types=%s)",
                 doc, sizes.get(doc, 0) / 1e6, len(by_doc[doc]), sorted(types[doc]))

    (out / "samples.json").write_text(json.dumps(subset_samples, indent=2,
                                                 default=str), encoding="utf-8")
    log.info("wrote %d questions over %d docs to %s",
             len(subset_samples), len(chosen), out)


if __name__ == "__main__":
    main()
