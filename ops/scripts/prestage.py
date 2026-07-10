"""Stage every configured model and MMLongBench asset used by offline generation.

Retriever adapter bases are discovered from their configs and staged recursively.
"""

# kaya: target=login
# kaya: env=true
# kaya: offline=false

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path

from ops.kaya.kaya import load_config
from ops.scripts.download_hf import snapshot, stage_mmlongbench_from_hub

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="alternate kaya JSON config listing the models/parsers/dataset to stage "
                             "(default: ops/kaya/config.json). Use a trimmed config to stage only what "
                             "a given spec needs, e.g. ops/kaya/h100_main.json")
    parser.add_argument("--local", action="store_true",
                        help="use this checkout's .cache/.data instead of Kaya remote paths")
    parser.add_argument("--smoke", action="store_true",
                        help="stage only the smallest reasoner, one retriever, and one parser")
    parser.add_argument("--skip-dataset", action="store_true", help="do not stage MMLongBench-Doc")
    parser.add_argument("--skip-models", action="store_true",
                        help="skip all reasoner and retrieval model snapshots")
    parser.add_argument("--skip-reasoner-models", action="store_true")
    parser.add_argument("--skip-retrieval-models", action="store_true")
    parser.add_argument("--skip-parsers", action="store_true", help="skip the three parser models")
    parser.add_argument("--model-id", action="append", help="reasoner repo id; repeatable")
    parser.add_argument("--retrieval-model-id", action="append", help="retrieval repo id; repeatable")
    parser.add_argument("--parser-id", action="append", help="parser repo id; repeatable")
    parser.add_argument("--revision", help="optional Hugging Face revision")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--copy", action="store_true",
                        help="copy staged dataset files instead of symlinking")
    parser.add_argument("--max-workers", type=int, help="override config hf.max_workers")
    return parser


def flatten_retrieval_models(raw: dict) -> list[str]:
    """Configured retrieval model ids in stable group order (text then vision)."""
    groups = raw.get("retrieval_models", {})
    if isinstance(groups, list):
        return [str(m) for m in groups]
    ids: list[str] = []
    for group in ("text", "vision"):
        for m in groups.get(group, []):
            if m not in ids:
                ids.append(str(m))
    for value in groups.values():
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            for m in value:
                if m not in ids:
                    ids.append(str(m))
    return ids


def parser_models(raw: dict) -> list[str]:
    """The parser model repo ids from config.parsers (dict or list)."""
    parsers = raw.get("parsers", {})
    if isinstance(parsers, dict):
        return [str(v) for v in parsers.values()]
    return [str(v) for v in parsers]


def parser_model_closure(raw: dict, selected: list[str]) -> list[str]:
    """Add configured auxiliary models for each selected parser."""

    models = list(dict.fromkeys(selected))
    parsers = raw.get("parsers", {})
    extras = raw.get("parser_aux_models", {})
    if not isinstance(parsers, dict) or not isinstance(extras, dict):
        return models
    for parser_name, model_id in parsers.items():
        if str(model_id) not in models:
            continue
        for extra_id in extras.get(parser_name, []):
            if str(extra_id) not in models:
                models.append(str(extra_id))
    return models


def smoke_reasoner_models(raw: dict) -> list[str]:
    models = [str(m) for m in raw.get("models", [])]
    for m in models:
        if "Qwen3-VL-2B" in m:
            return [m]
    return models[:1]


def smoke_retrieval_models(raw: dict) -> list[str]:
    ids = flatten_retrieval_models(raw)
    return ids[:1]


def prepare_hf_cache_env(cache_dir: Path) -> None:
    """Point every download cache into the project so nothing lands in $HOME.

    On Kaya these are already exported by the run wrapper; setting them here keeps
    the --local path (and any direct invocation) equally contained.
    """
    from config import hf_cache_environ

    (cache_dir / "xet").mkdir(parents=True, exist_ok=True)
    # MinerU pulls aux models from HF (contained); cache ModelScope in-project too.
    os.environ.update(hf_cache_environ(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    os.environ.pop("TRANSFORMERS_CACHE", None)


def adapter_base_model(snapshot_dir: Path) -> str | None:
    """Read a PEFT adapter's base model id, if the snapshot is an adapter."""

    config_path = snapshot_dir / "adapter_config.json"
    if not config_path.is_file():
        return None
    value = json.loads(config_path.read_text()).get("base_model_name_or_path")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def stage_all(
    model_ids: list[str],
    kind_label: str,
    revision,
    cache_dir,
    *,
    force,
    workers,
    include_adapter_bases: bool = False,
    dependency_map: dict[str, list[str]] | None = None,
) -> None:
    pending = list(dict.fromkeys(model_ids))
    scheduled = set(pending)
    print(f"[prestage] staging {len(pending)} {kind_label}: {', '.join(pending) or '(none)'}")
    for model_id in pending:
        path = snapshot(model_id, "model", revision, cache_dir,
                        force_download=force, max_workers=workers)
        print(f"[prestage] {kind_label[:-1] if kind_label.endswith('s') else kind_label} {model_id} -> {path}")
        dependencies = list((dependency_map or {}).get(model_id, []))
        if include_adapter_bases:
            base_id = adapter_base_model(Path(path))
            if base_id:
                dependencies.insert(0, base_id)
        for dependency_id in dependencies:
            if dependency_id not in scheduled:
                scheduled.add(dependency_id)
                pending.append(dependency_id)
                print(f"[prestage] discovered model dependency {model_id} -> {dependency_id}")


def snapshot_matches_directory(snapshot_dir: Path, dest: Path) -> bool:
    """Return whether every snapshot file exists at the destination with its size."""

    files = [path for path in snapshot_dir.rglob("*") if path.is_file()]
    return bool(files) and all(
        (dest / path.relative_to(snapshot_dir)).is_file()
        and (dest / path.relative_to(snapshot_dir)).stat().st_size == path.stat().st_size
        for path in files
    )


def stage_paddlex_model_cache(repo_id: str, cache_dir: Path, revision, workers: int) -> None:
    """Mirror a staged Paddle model snapshot into paddlex's official model cache.

    Paddlex does not read these models from the normal HF snapshot cache. It looks
    under `$PADDLE_PDX_CACHE_HOME/official_models/<name>` instead.
    """
    import shutil

    snap = Path(snapshot(repo_id, "model", revision, cache_dir,
                         force_download=False, max_workers=workers))
    dest = cache_dir / "paddlex" / "official_models" / repo_id.split("/")[-1]
    if snapshot_matches_directory(snap, dest):
        print(f"[prestage] paddlex model cache already staged -> {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    # symlinks=False dereferences the hub's blob symlinks so real files land here.
    shutil.copytree(snap, dest, symlinks=False, dirs_exist_ok=True)
    print(f"[prestage] paddlex model cache {repo_id} -> {dest}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config or ROOT / "ops" / "kaya" / "config.json")
    if args.local:
        cache_dir = ROOT / config.raw["paths"]["cache"]
        data_dir = ROOT / config.raw["paths"]["data"]
        root_label = str(ROOT)
    else:
        cache_dir = Path(config.remote_path("cache"))
        data_dir = Path(config.remote_path("data"))
        root_label = config.remote_root
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    workers = args.max_workers or int(config.raw.get("hf", {}).get("max_workers", 8))
    prepare_hf_cache_env(cache_dir)
    print(f"[prestage] start root={root_label} local={args.local} cache={cache_dir} "
          f"smoke={args.smoke} HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'missing'}")

    skip_reasoners = args.skip_models or args.skip_reasoner_models
    skip_retrievers = args.skip_models or args.skip_retrieval_models

    if not skip_reasoners:
        ids = args.model_id or (smoke_reasoner_models(config.raw) if args.smoke else list(config.raw["models"]))
        stage_all(ids, "reasoner models", args.revision, cache_dir, force=args.force_download, workers=workers)
    else:
        print("[prestage] skipping reasoner models")

    if not skip_retrievers:
        ids = args.retrieval_model_id or (
            smoke_retrieval_models(config.raw) if args.smoke else flatten_retrieval_models(config.raw))
        stage_all(
            ids,
            "retrieval models",
            args.revision,
            cache_dir,
            force=args.force_download,
            workers=workers,
            include_adapter_bases=True,
            dependency_map=config.raw.get("retrieval_model_dependencies", {}),
        )
    else:
        print("[prestage] skipping retrieval models")

    if not args.skip_parsers:
        ids = args.parser_id or parser_models(config.raw)
        if args.smoke:
            ids = ids[:1]
        ids = parser_model_closure(config.raw, ids)
        stage_all(ids, "parser models", args.revision, cache_dir, force=args.force_download, workers=workers)
        # PaddleOCR-VL and its layout detector use paddlex's separate model cache.
        parsers = config.raw.get("parsers", {})
        paddle_repo = parsers.get("paddleocrvl") if isinstance(parsers, dict) else None
        if paddle_repo and paddle_repo in ids:
            paddle_ids = parser_model_closure(config.raw, [paddle_repo])
            for repo_id in paddle_ids:
                stage_paddlex_model_cache(repo_id, cache_dir, args.revision, workers)
    else:
        print("[prestage] skipping parser models")

    if not args.skip_dataset:
        dataset_id = config.raw["datasets"]["mmlongbench"]
        staged = stage_mmlongbench_from_hub(dataset_id, args.revision, cache_dir, data_dir,
                                            args.copy, args.force_download)
        print(f"[prestage] mmlongbench -> {staged}")
    else:
        print("[prestage] skipping dataset")

    print("[prestage] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
