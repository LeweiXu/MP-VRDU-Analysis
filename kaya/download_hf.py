"""Download and stage Hugging Face assets inside the Kaya mirror.

This module is called by `kaya/kaya.py prestage` on the Kaya login node. It can
download model snapshots into HF_HOME and stage MMLongBench-Doc into the
root-relative `.data/mmlongbench` layout expected by `cli.run_probe` and later
data-loading stages.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def configure_hf_transport(*, enable_xet: bool) -> None:
    """Prefer plain HTTP downloads for reproducible HPC prestaging."""

    if not enable_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def snapshot(
    repo_id: str,
    repo_type: str,
    revision: str | None,
    cache_dir: Path,
    force_download: bool = False,
    max_workers: int = 1,
) -> Path:
    """Download a Hugging Face snapshot and return its local path."""

    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "cache_dir": str(cache_dir),
        "max_workers": max_workers,
        "force_download": force_download,
        "resume_download": False,
    }
    try:
        return Path(snapshot_download(**kwargs))
    except Exception as exc:
        message = str(exc)
        retryable = (
            "Consistency check failed" in message
            or "Requested Range Not Satisfiable" in message
            or "416 Client Error" in message
            or "File size mismatch" in message
        )
        if force_download or not retryable:
            raise
        print("[download_hf] corrupt/resumable download failed; retrying with force_download=True")
        kwargs["force_download"] = True
        return Path(snapshot_download(**kwargs))


def replace_link_or_copy(source: Path, target: Path, copy: bool) -> None:
    """Replace `target` with a symlink to `source`, or a copy if requested."""

    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


def stage_mmlongbench(snapshot_dir: Path, data_dir: Path, copy: bool) -> Path:
    """Expose MMLongBench-Doc as `.data/mmlongbench/{data,documents}`."""

    target_root = data_dir / "mmlongbench"
    target_root.mkdir(parents=True, exist_ok=True)

    source_data = snapshot_dir / "data"
    source_documents = snapshot_dir / "documents"

    if source_data.is_dir():
        replace_link_or_copy(source_data, target_root / "data", copy)
    else:
        parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"no parquet files found in {snapshot_dir}")
        data_target = target_root / "data"
        data_target.mkdir(parents=True, exist_ok=True)
        for parquet in parquet_files:
            replace_link_or_copy(parquet, data_target / parquet.name, copy)

    if source_documents.is_dir():
        replace_link_or_copy(source_documents, target_root / "documents", copy)
    else:
        pdf_files = sorted(snapshot_dir.rglob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"no PDF documents found in {snapshot_dir}")
        docs_target = target_root / "documents"
        docs_target.mkdir(parents=True, exist_ok=True)
        for pdf in pdf_files:
            replace_link_or_copy(pdf, docs_target / pdf.name, copy)

    return target_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="model repo id, e.g. Qwen/Qwen3-VL-8B-Instruct")
    group.add_argument("--dataset", help="dataset repo id, e.g. yubo2333/MMLongBench-Doc")
    parser.add_argument("--revision", help="optional Hugging Face revision")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("HF_HOME", ".cache")))
    parser.add_argument("--data-dir", type=Path, default=Path(".data"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--enable-xet",
        action="store_true",
        help="allow Hugging Face Xet transport; disabled by default for Kaya prestaging",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="parallel Hugging Face downloads; default is serial for deterministic HPC staging",
    )
    parser.add_argument(
        "--stage-mmlongbench",
        action="store_true",
        help="after downloading a dataset, expose it under .data/mmlongbench",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy staged files instead of symlinking from the HF cache",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_hf_transport(enable_xet=args.enable_xet)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        path = snapshot(
            args.model,
            "model",
            args.revision,
            args.cache_dir,
            args.force_download,
            max_workers=args.max_workers,
        )
        print(f"model {args.model} -> {path}")
        return 0

    path = snapshot(
        args.dataset,
        "dataset",
        args.revision,
        args.cache_dir,
        args.force_download,
        max_workers=args.max_workers,
    )
    print(f"dataset {args.dataset} -> {path}")
    if args.stage_mmlongbench:
        staged = stage_mmlongbench(path, args.data_dir, args.copy)
        print(f"mmlongbench staged -> {staged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
