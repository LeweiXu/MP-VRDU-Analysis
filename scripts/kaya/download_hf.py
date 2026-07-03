"""Download Hugging Face model or dataset snapshots into the configured cache."""

from __future__ import annotations

import argparse
import os


def _snapshot(repo_id: str, repo_type: str, revision: str | None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        cache_dir=os.environ.get("HF_HOME"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="model repo id, for example Qwen/Qwen3-VL-8B-Instruct")
    group.add_argument("--dataset", help="dataset repo id, for example yubo2333/MMLongBench-Doc")
    parser.add_argument("--revision", help="optional Hugging Face revision")
    args = parser.parse_args()

    if args.model:
        path = _snapshot(args.model, "model", args.revision)
        print(f"model {args.model} -> {path}")
    else:
        path = _snapshot(args.dataset, "dataset", args.revision)
        print(f"dataset {args.dataset} -> {path}")


if __name__ == "__main__":
    main()
