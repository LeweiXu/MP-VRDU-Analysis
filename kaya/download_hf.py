#!/usr/bin/env python
"""Download a Hugging Face model or dataset into $HF_HOME for offline use on Kaya.

    python download_hf.py Qwen/Qwen2.5-7B-Instruct
    python download_hf.py yubo2333/MMLongBench-Doc --dataset
"""
import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--dataset", action="store_true", help="download a dataset repo")
    args = parser.parse_args()

    repo_type = "dataset" if args.dataset else "model"
    path = snapshot_download(args.repo_id, repo_type=repo_type)
    print(path)


if __name__ == "__main__":
    main()
