"""Download every model, parser, and dataset the offline compute jobs need.

Runs on the Kaya login node (internet available) and stages, into the shared HF
cache, the reasoner weights, retrieval weights, the three parser models, and the
MMLongBench-Doc / LongDocURL datasets. Compute-node jobs then read them offline.
Environment creation is not here; that is `setup_env.py`.

    python -m ops.kaya.kaya run ops/scripts/prestage.py -- [--smoke] [options]
"""

# kaya: target=login
# kaya: env=true
# kaya: offline=false

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

from ops.kaya.kaya import load_config
from ops.scripts.download_hf import snapshot, stage_longdocurl_from_hub, stage_mmlongbench_from_hub

ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true",
                        help="use this checkout's .cache/.data instead of Kaya remote paths")
    parser.add_argument("--smoke", action="store_true",
                        help="stage only the smallest reasoner, one retriever, and one parser")
    parser.add_argument("--skip-dataset", action="store_true", help="do not stage MMLongBench-Doc")
    parser.add_argument("--skip-longdocurl", action="store_true",
                        help="do not stage the LongDocURL replication set")
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
    (cache_dir / "xet").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")
    # MinerU pulls aux models from HF (contained); cache ModelScope in-project too.
    os.environ["MINERU_MODEL_SOURCE"] = "huggingface"
    os.environ["MODELSCOPE_CACHE"] = str(cache_dir / "modelscope")
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "huggingface"
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    os.environ.pop("TRANSFORMERS_CACHE", None)


def stage_all(model_ids: list[str], kind_label: str, revision, cache_dir, *, force, workers) -> None:
    print(f"[prestage] staging {len(model_ids)} {kind_label}: {', '.join(model_ids) or '(none)'}")
    for model_id in model_ids:
        path = snapshot(model_id, "model", revision, cache_dir,
                        force_download=force, max_workers=workers)
        print(f"[prestage] {kind_label[:-1] if kind_label.endswith('s') else kind_label} {model_id} -> {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(ROOT / "ops" / "kaya" / "config.json")
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
        stage_all(ids, "retrieval models", args.revision, cache_dir, force=args.force_download, workers=workers)
    else:
        print("[prestage] skipping retrieval models")

    if not args.skip_parsers:
        ids = args.parser_id or parser_models(config.raw)
        if args.smoke:
            ids = ids[:1]
        stage_all(ids, "parser models", args.revision, cache_dir, force=args.force_download, workers=workers)
        # MinerU's auxiliary layout/formula models and PaddleOCR-VL's pipeline
        # models download on first use inside their own env; that per-env warmup
        # runs in Phase 4 alongside tools/parser.py.
    else:
        print("[prestage] skipping parser models")

    if not args.skip_dataset:
        dataset_id = config.raw["datasets"]["mmlongbench"]
        staged = stage_mmlongbench_from_hub(dataset_id, args.revision, cache_dir, data_dir,
                                            args.copy, args.force_download)
        print(f"[prestage] mmlongbench -> {staged}")
    else:
        print("[prestage] skipping dataset")

    if args.skip_longdocurl or args.smoke:
        print("[prestage] skipping LongDocURL")
    else:
        longdocurl_id = config.raw.get("datasets", {}).get("longdocurl", "dengchao/LongDocURL")
        staged_ldu = stage_longdocurl_from_hub(longdocurl_id, args.revision, cache_dir, data_dir,
                                               args.copy, args.force_download)
        print(f"[prestage] longdocurl -> {staged_ldu}")

    print("[prestage] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
