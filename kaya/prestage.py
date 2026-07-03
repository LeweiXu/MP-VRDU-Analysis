"""Stage Hugging Face models and MMLongBench-Doc on Kaya's login node."""

# kaya: target=login
# kaya: env=true
# kaya: offline=false

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

from kaya.download_hf import snapshot, stage_mmlongbench_from_hub
from kaya.kaya import load_config


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-dataset", action="store_true", help="do not stage MMLongBench-Doc")
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="do not download any configured HF model weights",
    )
    parser.add_argument(
        "--skip-reasoner-models",
        action="store_true",
        help="do not download configured Qwen reasoner model weights",
    )
    parser.add_argument(
        "--skip-retrieval-models",
        action="store_true",
        help="do not download configured BGE/ColPali/ColQwen model weights",
    )
    parser.add_argument("--skip-tool-caches", action="store_true", help="do not warm PaddleOCR/Docling caches")
    parser.add_argument("--model-id", action="append", help="reasoner model repo id to stage; repeatable")
    parser.add_argument("--retrieval-model-id", action="append", help="retrieval model repo id to stage; repeatable")
    parser.add_argument("--revision", help="optional Hugging Face revision for all downloads")
    parser.add_argument("--force-download", action="store_true", help="force Hugging Face redownload")
    parser.add_argument("--copy", action="store_true", help="copy staged MMLongBench files instead of symlinking")
    parser.add_argument("--max-workers", type=int, help="override config hf.max_workers")
    return parser


def flatten_retrieval_models(raw: dict) -> list[str]:
    """Return configured retrieval model ids in stable group order."""

    groups = raw.get("retrieval_models", {})
    if isinstance(groups, list):
        return [str(model_id) for model_id in groups]

    model_ids: list[str] = []
    for group_name in ("text", "vision"):
        for model_id in groups.get(group_name, []):
            if model_id not in model_ids:
                model_ids.append(str(model_id))
    for value in groups.values():
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            for model_id in value:
                if model_id not in model_ids:
                    model_ids.append(str(model_id))
    return model_ids


def prepare_tool_cache_env(cache_dir: Path) -> None:
    """Point tool/model caches at the root-relative Kaya cache directory."""

    for path in [
        cache_dir,
        cache_dir / "paddle",
        cache_dir / "paddleocr",
        cache_dir / "paddlex",
        cache_dir / "docling",
        cache_dir / "matplotlib",
        cache_dir / "torch",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_HOME", str(cache_dir / "paddle"))
    os.environ.setdefault("PADDLEOCR_HOME", str(cache_dir / "paddleocr"))
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir / "paddlex"))
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    os.environ.setdefault("DOCLING_CACHE_DIR", str(cache_dir / "docling"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))


def warm_paddleocr_cache() -> None:
    """Initialise PaddleOCR so its detection/recognition models are downloaded."""

    from paddleocr import PaddleOCR

    PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
    )


def warm_docling_cache(cache_dir: Path, force: bool) -> None:
    """Download Docling layout/table models and initialise the converter."""

    from docling.document_converter import DocumentConverter
    from docling.utils.model_downloader import download_models

    download_models(
        output_dir=cache_dir / "docling" / "models",
        force=force,
        progress=True,
        with_layout=True,
        with_tableformer=True,
        with_code_formula=True,
        with_picture_classifier=True,
        with_smolvlm=False,
        with_smoldocling=False,
        with_smoldocling_mlx=False,
        with_granite_vision=False,
        with_easyocr=False,
    )
    DocumentConverter()


def warm_tool_caches(config_raw: dict, cache_dir: Path, force: bool) -> None:
    """Warm non-HF tool caches required by later stages."""

    tool_caches = config_raw.get("tool_caches", {})
    prepare_tool_cache_env(cache_dir)
    if tool_caches.get("paddleocr", False):
        print("[prestage] warming PaddleOCR cache")
        warm_paddleocr_cache()
        print("[prestage] PaddleOCR cache ready")
    if tool_caches.get("docling", False):
        print("[prestage] warming Docling cache")
        warm_docling_cache(cache_dir, force)
        print("[prestage] Docling cache ready")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(ROOT / "kaya/config.json")
    cache_dir = Path(config.remote_path("cache"))
    data_dir = Path(config.remote_path("data"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    max_workers = args.max_workers or int(config.raw.get("hf", {}).get("max_workers", 8))
    print(
        "[prestage] start "
        f"remote_root={config.remote_root} cache_dir={cache_dir} data_dir={data_dir} "
        f"skip_models={args.skip_models} skip_dataset={args.skip_dataset} "
        f"max_workers={max_workers} force_download={args.force_download} "
        f"HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'missing'}"
    )

    skip_reasoners = args.skip_models or args.skip_reasoner_models
    skip_retrievers = args.skip_models or args.skip_retrieval_models

    if not skip_reasoners:
        model_ids = args.model_id or list(config.raw["models"])
        print(f"[prestage] staging {len(model_ids)} reasoner model(s): {', '.join(model_ids)}")
        for model_id in model_ids:
            print(f"[prestage] downloading reasoner model {model_id}")
            path = snapshot(
                model_id,
                "model",
                args.revision,
                cache_dir,
                force_download=args.force_download,
                max_workers=max_workers,
            )
            print(f"model {model_id} -> {path}")
    else:
        print("[prestage] skipping reasoner model downloads")

    if not skip_retrievers:
        retrieval_model_ids = args.retrieval_model_id or flatten_retrieval_models(config.raw)
        print(
            "[prestage] staging "
            f"{len(retrieval_model_ids)} retrieval model(s): {', '.join(retrieval_model_ids)}"
        )
        for model_id in retrieval_model_ids:
            print(f"[prestage] downloading retrieval model {model_id}")
            path = snapshot(
                model_id,
                "model",
                args.revision,
                cache_dir,
                force_download=args.force_download,
                max_workers=max_workers,
            )
            print(f"retrieval model {model_id} -> {path}")
    else:
        print("[prestage] skipping retrieval model downloads")

    if not args.skip_dataset:
        dataset_id = config.raw["datasets"]["mmlongbench"]
        print(f"[prestage] downloading/staging dataset {dataset_id}")
        staged = stage_mmlongbench_from_hub(
            dataset_id,
            args.revision,
            cache_dir,
            data_dir,
            args.copy,
            args.force_download,
        )
        print(f"mmlongbench staged -> {staged}")
    else:
        print("[prestage] skipping dataset staging")

    if not args.skip_tool_caches:
        warm_tool_caches(config.raw, cache_dir, args.force_download)
    else:
        print("[prestage] skipping tool cache warmup")

    print("[prestage] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
