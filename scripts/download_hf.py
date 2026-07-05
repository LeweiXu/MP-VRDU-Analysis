"""Download and stage Hugging Face assets inside the Kaya mirror.

Purpose:
    Provides robust Hugging Face snapshot/file staging utilities for
    `scripts/prestage.py`, including local-cache probing, retry cleanup for partial
    Hub/Xet state, and a file-by-file MMLongBench staging path.

Pipeline role:
    Ensures reasoner/retriever weights land in root-relative `HF_HOME` and that
    MMLongBench appears under `.data/mmlongbench/{data,documents}` where the
    normal loader expects it. Compute jobs then run offline from those assets.

CLI:
    `python -m scripts.download_hf (--model ID | --dataset ID) [options]`

Arguments:
    --model ID: Hugging Face model repo to snapshot.
    --dataset ID: Hugging Face dataset repo to snapshot/stage.
    --revision REV: optional revision for downloads.
    --cache-dir PATH: Hugging Face cache root (default: `$HF_HOME` or `.cache`).
    --data-dir PATH: dataset staging root (default: `.data`).
    --force-download: force a Hub redownload.
    --max-workers N: parallel snapshot downloads.
    --stage-mmlongbench: expose a dataset repo in MMLongBench loader layout.
    --copy: copy staged files instead of symlinking from the Hub cache.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import tarfile
from pathlib import Path
from typing import Iterable


LONGDOCURL_ANNOTATION = "LongDocURL_public.jsonl"
LONGDOCURL_PDF_ARCHIVE = "pdf_files.tar.gz"


RETRYABLE_DOWNLOAD_ERRORS = (
    "Consistency check failed",
    "Requested Range Not Satisfiable",
    "416 Client Error",
    "File size mismatch",
)


def package_version(name: str) -> str | None:
    """Return an installed package version, or None if it is unavailable."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def is_retryable_download_error(exc: Exception) -> bool:
    """Return whether a Hub download error is likely caused by partial state."""

    message = str(exc)
    return any(fragment in message for fragment in RETRYABLE_DOWNLOAD_ERRORS)


def repo_cache_dir(cache_dir: Path, repo_id: str, repo_type: str) -> Path:
    """Return Hugging Face Hub's cache directory for one repo."""

    prefixes = {"model": "models", "dataset": "datasets", "space": "spaces"}
    prefix = prefixes.get(repo_type, repo_type)
    return cache_dir / f"{prefix}--{repo_id.replace('/', '--')}"


def xet_cache_dirs(cache_dir: Path) -> list[Path]:
    """Return likely Xet cache locations used by huggingface_hub/hf_xet."""

    candidates = [
        cache_dir / "xet",
        Path(os.environ.get("HF_XET_CACHE", cache_dir / "xet")),
        Path.home() / ".cache" / "huggingface" / "xet",
    ]
    out: list[Path] = []
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
    return out


def purge_partial_download_state(cache_dir: Path, repo_id: str, repo_type: str, *, purge_repo: bool) -> None:
    """Remove resumable/temporary Hub state that can poison a retry.

    For retry attempts we remove the repo cache too, because the observed Kaya
    failure includes completed blobs with the wrong size, not only `.incomplete`
    files.
    """

    removed = 0
    if purge_repo:
        path = repo_cache_dir(cache_dir, repo_id, repo_type)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    for pattern in ("*.incomplete", "*.lock", "*.tmp"):
        for path in cache_dir.rglob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed += 1
            except FileNotFoundError:
                pass
    for path in [*xet_cache_dirs(cache_dir), cache_dir / ".locks", cache_dir / "hub" / ".locks"]:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    if removed:
        print(f"[download_hf] removed {removed} partial/cache lock entries before retry")


def print_hf_diagnostics(cache_dir: Path) -> None:
    """Print one concise Hugging Face runtime report."""

    print(
        "[download_hf] runtime "
        f"huggingface_hub={package_version('huggingface_hub') or 'missing'} "
        f"hf_xet={package_version('hf_xet') or 'missing'} "
        f"HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'missing'} "
        f"HF_HOME={os.environ.get('HF_HOME', str(cache_dir))} "
        f"HF_XET_CACHE={os.environ.get('HF_XET_CACHE', str(cache_dir / 'xet'))} "
        f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET', '') or 'unset'}"
    )


def revision_label(revision: str | None) -> str:
    """Return a readable revision label for logs."""

    return revision or "default"


def snapshot(
    repo_id: str,
    repo_type: str,
    revision: str | None,
    cache_dir: Path,
    force_download: bool = False,
    max_workers: int = 8,
) -> Path:
    """Download a Hugging Face snapshot and return its local path.

    Skips the network entirely when the revision is already fully cached, by
    probing the Hub cache with `local_files_only=True` first. That call
    raises if any file is missing, so a successful return means the snapshot
    is complete and no HTTP request is needed.
    """

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    print_hf_diagnostics(cache_dir)
    from huggingface_hub import snapshot_download

    if not force_download:
        try:
            cached_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                )
            )
            print(f"[download_hf] {repo_id} already cached locally, skipping download -> {cached_path}")
            return cached_path
        except Exception:
            pass

    kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "revision": revision,
        "cache_dir": str(cache_dir),
        "max_workers": max_workers,
        "force_download": force_download,
        "token": os.environ.get("HF_TOKEN"),
    }
    attempts = [
        ("initial", dict(kwargs), False),
        ("fresh serial retry", {**kwargs, "force_download": True, "max_workers": 1}, True),
    ]
    last_error: Exception | None = None
    print(
        "[download_hf] snapshot "
        f"repo_type={repo_type} repo_id={repo_id} revision={revision_label(revision)} "
        f"cache_dir={cache_dir} max_workers={max_workers} force_download={force_download}"
    )
    for label, attempt_kwargs, purge_repo in attempts:
        try:
            if label != "initial":
                purge_partial_download_state(cache_dir, repo_id, repo_type, purge_repo=purge_repo)
                print(f"[download_hf] retrying {repo_id} with {label}")
            print(
                "[download_hf] attempt "
                f"label={label} max_workers={attempt_kwargs['max_workers']} "
                f"force_download={attempt_kwargs['force_download']} xet=enabled"
            )
            return Path(snapshot_download(**attempt_kwargs))
        except Exception as exc:
            last_error = exc
            if not is_retryable_download_error(exc):
                raise
            print(f"[download_hf] {label} failed with retryable Hub cache error: {exc}")
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"snapshot download failed unexpectedly for {repo_id}")


def list_repo_paths(repo_id: str, repo_type: str, revision: str | None) -> list[str]:
    """Return sorted repo file paths through the Hugging Face Hub client."""

    from huggingface_hub import list_repo_files

    return sorted(
        list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
        )
    )


def download_file(
    repo_id: str,
    repo_type: str,
    filename: str,
    revision: str | None,
    cache_dir: Path,
    force_download: bool,
) -> Path:
    """Download one repo file through the Hub cache and return its local path."""

    from huggingface_hub import hf_hub_download

    kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "filename": filename,
        "revision": revision,
        "cache_dir": str(cache_dir),
        "force_download": force_download,
        "token": os.environ.get("HF_TOKEN"),
    }
    attempts = [
        ("initial", dict(kwargs)),
        ("fresh retry", {**kwargs, "force_download": True}),
    ]
    last_error: Exception | None = None
    for label, attempt_kwargs in attempts:
        try:
            if label != "initial":
                purge_partial_download_state(cache_dir, repo_id, repo_type, purge_repo=False)
                print(f"[download_hf] retrying file {filename} with {label}")
            return Path(hf_hub_download(**attempt_kwargs))
        except Exception as exc:
            last_error = exc
            if not is_retryable_download_error(exc):
                raise
            print(f"[download_hf] file {filename} {label} failed with retryable Hub error: {exc}")
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"file download failed unexpectedly for {repo_id}:{filename}")


def hf_filesystem_path(repo_id: str, repo_type: str, filename: str, revision: str | None) -> str:
    """Return an HfFileSystem path for one file."""

    type_prefixes = {
        "dataset": "datasets",
        "model": "models",
        "space": "spaces",
    }
    repo = f"{repo_id}@{revision}" if revision else repo_id
    prefix = type_prefixes.get(repo_type)
    if prefix is None:
        return f"{repo}/{filename}"
    return f"{prefix}/{repo}/{filename}"


def validate_downloaded_file(path: Path, filename: str) -> None:
    """Do a small sanity check for files streamed outside the Hub cache."""

    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"{path} was not written or is empty")
    if filename.lower().endswith(".pdf"):
        with path.open("rb") as handle:
            head = handle.read(1024)
        if b"%PDF" not in head:
            raise OSError(f"{path} does not look like a PDF")


def stream_file_to_target(
    repo_id: str,
    repo_type: str,
    filename: str,
    revision: str | None,
    target: Path,
) -> Path:
    """Stream one file via Hugging Face's filesystem interface into `target`."""

    from huggingface_hub import HfFileSystem

    fs_path = hf_filesystem_path(repo_id, repo_type, filename, revision)
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    print(f"[download_hf] streaming via HfFileSystem {fs_path} -> {target}")
    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    with fs.open(fs_path, "rb") as source, tmp.open("wb") as dest:
        shutil.copyfileobj(source, dest, length=1024 * 1024 * 8)
    validate_downloaded_file(tmp, filename)
    tmp.replace(target)
    return target


def link_or_copy_or_stream(
    repo_id: str,
    repo_type: str,
    filename: str,
    revision: str | None,
    cache_dir: Path,
    target: Path,
    copy: bool,
    force_download: bool,
) -> Path:
    """Stage one repo file at `target`, falling back to streaming if needed."""

    try:
        source = download_file(repo_id, repo_type, filename, revision, cache_dir, force_download)
        replace_link_or_copy(source, target, copy)
        return target
    except Exception as exc:
        if not is_retryable_download_error(exc):
            raise
        print(
            f"[download_hf] Hub cache download failed for {filename}; "
            "falling back to HfFileSystem streaming"
        )
        return stream_file_to_target(repo_id, repo_type, filename, revision, target)


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


def target_is_staged(target: Path) -> bool:
    """Return whether a staged file already exists locally with content."""

    return target.is_file() and target.stat().st_size > 0


def target_name(filename: str, prefixes: Iterable[str]) -> str:
    """Return the flattened staging filename, dropping known repo subdirs."""

    path = Path(filename)
    parts = path.parts
    for prefix in prefixes:
        if parts and parts[0] == prefix and len(parts) > 1:
            return Path(*parts[1:]).name
    return path.name


def stage_mmlongbench_from_hub(
    repo_id: str,
    revision: str | None,
    cache_dir: Path,
    data_dir: Path,
    copy: bool,
    force_download: bool,
) -> Path:
    """Download and expose MMLongBench-Doc file-by-file.

    The dataset contains many PDFs, and on Kaya the monolithic snapshot download
    has repeatedly failed on a single corrupt/size-mismatched transfer. This path
    keeps the official Hub package interface while making progress and failures
    visible at file granularity.
    """

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    print_hf_diagnostics(cache_dir)
    print(
        "[download_hf] listing dataset "
        f"repo_id={repo_id} revision={revision_label(revision)} cache_dir={cache_dir}"
    )
    files = list_repo_paths(repo_id, "dataset", revision)
    parquet_files = [name for name in files if name.lower().endswith(".parquet")]
    pdf_files = [name for name in files if name.lower().endswith(".pdf")]
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found in {repo_id}")
    if not pdf_files:
        raise FileNotFoundError(f"no PDF documents found in {repo_id}")

    target_root = data_dir / "mmlongbench"
    data_target = target_root / "data"
    docs_target = target_root / "documents"
    data_target.mkdir(parents=True, exist_ok=True)
    docs_target.mkdir(parents=True, exist_ok=True)
    selected = [*parquet_files, *pdf_files]
    print(
        "[download_hf] mmlongbench files "
        f"parquet={len(parquet_files)} pdf={len(pdf_files)} total_selected={len(selected)} "
        f"repo_total={len(files)}"
    )

    total = len(selected)
    skipped = 0
    for index, filename in enumerate(parquet_files, start=1):
        target = data_target / target_name(filename, ("data",))
        if not force_download and target_is_staged(target):
            print(f"[download_hf] file {index}/{total} parquet {filename} already staged -> {target}, skipping")
            skipped += 1
            continue
        print(f"[download_hf] file {index}/{total} parquet {filename} -> {target}")
        link_or_copy_or_stream(
            repo_id,
            "dataset",
            filename,
            revision,
            cache_dir,
            target,
            copy,
            force_download,
        )
    for offset, filename in enumerate(pdf_files, start=1):
        index = len(parquet_files) + offset
        target = docs_target / target_name(filename, ("documents", "pdfs"))
        if not force_download and target_is_staged(target):
            print(f"[download_hf] file {index}/{total} pdf {filename} already staged -> {target}, skipping")
            skipped += 1
            continue
        print(f"[download_hf] file {index}/{total} pdf {filename} -> {target}")
        link_or_copy_or_stream(
            repo_id,
            "dataset",
            filename,
            revision,
            cache_dir,
            target,
            copy,
            force_download,
        )
    print(f"[download_hf] mmlongbench staging skipped {skipped}/{total} already-staged file(s)")

    return target_root


def stage_longdocurl_from_hub(
    repo_id: str,
    revision: str | None,
    cache_dir: Path,
    data_dir: Path,
    copy: bool,
    force_download: bool,
) -> Path:
    """Download LongDocURL and lay it out the way the loader expects.

    Grabs only the annotation JSONL and `pdf_files.tar.gz` (the ~5GB of PNG
    tarballs are unused), and exposes:

    - `.data/longdocurl/LongDocURL_public.jsonl`
    - `.data/longdocurl/documents/<doc_no>.pdf` (flattened out of the tarball)

    which is what `data.loader.load_longdocurl` / `resolve_longdocurl_pdf` read
    offline. Idempotent: staged files are skipped and a `.pdfs_staged` marker
    short-circuits re-extraction (use `--force-download` to redo it).
    """

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    target_root = data_dir / "longdocurl"
    docs_target = target_root / "documents"
    docs_target.mkdir(parents=True, exist_ok=True)

    annotation_target = target_root / LONGDOCURL_ANNOTATION
    if force_download or not target_is_staged(annotation_target):
        print(f"[download_hf] longdocurl annotations {LONGDOCURL_ANNOTATION} -> {annotation_target}")
        source = download_file(repo_id, "dataset", LONGDOCURL_ANNOTATION, revision, cache_dir, force_download)
        replace_link_or_copy(source, annotation_target, copy)
    else:
        print(f"[download_hf] longdocurl annotations already staged -> {annotation_target}, skipping")

    marker = target_root / ".pdfs_staged"
    if not force_download and marker.is_file():
        print(f"[download_hf] longdocurl PDFs already extracted (marker {marker}); skipping")
        return target_root

    print(f"[download_hf] downloading {LONGDOCURL_PDF_ARCHIVE} (~2.6GB) for {repo_id}")
    archive = download_file(repo_id, "dataset", LONGDOCURL_PDF_ARCHIVE, revision, cache_dir, force_download)
    print(f"[download_hf] extracting PDFs from {archive} -> {docs_target} (flattening)")
    extracted = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            # doc_no is the PDF filename, so flatten by basename (unique per doc).
            target = docs_target / Path(member.name).name
            if not force_download and target_is_staged(target):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            with handle, open(target, "wb") as out:
                shutil.copyfileobj(handle, out)
            extracted += 1
    marker.write_text("ok\n")
    staged_total = len(list(docs_target.glob("*.pdf")))
    print(f"[download_hf] longdocurl staged {extracted} new PDF(s); {staged_total} total -> {docs_target}")
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
        "--max-workers",
        type=int,
        default=8,
        help="parallel Hugging Face Hub downloads; hf_xet is used automatically when available",
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

    if args.stage_mmlongbench:
        staged = stage_mmlongbench_from_hub(
            args.dataset,
            args.revision,
            args.cache_dir,
            args.data_dir,
            args.copy,
            args.force_download,
        )
        print(f"mmlongbench staged -> {staged}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
