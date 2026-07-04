"""Stage Hugging Face models, tool caches, and MMLongBench-Doc on Kaya.

Purpose:
    Runs on the Kaya login node, where internet access is available, to prepare
    all assets that compute-node jobs must later read offline: reasoner weights,
    retrieval weights, MMLongBench files, and parser/OCR/layout tool caches.

Pipeline role:
    This is the setup barrier for MVP and full runs. `--smoke` stages the
    smallest inventory and verifies one tiny call through retrieval imports,
    PyMuPDF embedded text, PaddleOCR, Marker text/layout, and visual helpers.

CLI:
    `python -m kaya.kaya run kaya/prestage.py -- [options]`

Arguments:
    --smoke: stage only the MVP smoke subset and run fast tool smoke checks.
    --skip-dataset: do not stage MMLongBench-Doc.
    --skip-models: skip all configured model snapshots.
    --skip-reasoner-models: skip Qwen reasoner snapshots only.
    --skip-retrieval-models: skip BGE/ColPali/ColQwen snapshots only.
    --skip-tool-caches: skip Marker/PaddleOCR/Docling cache warmup.
    --model-id ID: repeatable reasoner model repo override.
    --retrieval-model-id ID: repeatable retrieval model repo override.
    --revision REV: optional Hugging Face revision for downloads.
    --force-download: force Hub redownload instead of cache probing.
    --copy: copy MMLongBench files instead of symlinking staged cache files.
    --max-workers N: override `hf.max_workers` from `kaya/config.json`.
"""

# kaya: target=login
# kaya: env=true
# kaya: offline=false

from __future__ import annotations

import argparse
import inspect
import os
import importlib.metadata
import json
from collections.abc import Iterable
from pathlib import Path

from data.loader import resolve_pdf
from data.render import render_pdf
from experiments.smoke import SMOKE_DOC_IDS
from kaya.download_hf import snapshot, stage_mmlongbench_from_hub
from kaya.kaya import load_config


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="stage only the MVP smoke subset and run fast tool smoke checks",
    )
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


def smoke_reasoner_models(raw: dict) -> list[str]:
    """Return the configured smallest Qwen3-VL reasoner for smoke runs."""

    models = [str(model_id) for model_id in raw.get("models", [])]
    for model_id in models:
        if "Qwen3-VL-2B" in model_id:
            return [model_id]
    return models[:1]


def smoke_retrieval_models(raw: dict) -> list[str]:
    """Return the text retriever and one ColQwen retriever for smoke runs."""

    groups = raw.get("retrieval_models", {})
    if isinstance(groups, list):
        candidates = [str(model_id) for model_id in groups]
        return candidates[:2]

    selected: list[str] = []
    text_models = [str(model_id) for model_id in groups.get("text", [])]
    if text_models:
        selected.append(text_models[0])
    vision_models = [str(model_id) for model_id in groups.get("vision", [])]
    colqwen = [model_id for model_id in vision_models if "colqwen" in model_id.casefold()]
    if colqwen:
        selected.append(colqwen[0])
    elif vision_models:
        selected.append(vision_models[0])
    return selected


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
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))


def warm_paddleocr_cache() -> None:
    """Initialise PaddleOCR so its detection/recognition models are downloaded."""

    from paddleocr import PaddleOCR
    from paddlex.inference import PaddlePredictorOption

    signature = inspect.signature(PaddlePredictorOption)
    supports_model_name = any(
        param.name == "model_name"
        and param.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        for param in signature.parameters.values()
    ) or any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in signature.parameters.values())
    if not supports_model_name:
        versions = {
            name: importlib.metadata.version(name)
            for name in ("paddleocr", "paddlex", "paddlepaddle")
        }
        raise RuntimeError(
            "PaddleOCR/PaddleX version mismatch: "
            f"PaddlePredictorOption signature is {signature}, versions={versions}. "
            "Run `envs/mpvrdu/bin/python -m kaya.kaya run kaya/setup_env.py` "
            "so requirements.txt pins `paddlex[ie,multimodal,ocr]>=3.1.0,<3.2.0`."
        )

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


def warm_marker_cache(cache_dir: Path, smoke_pdf: Path | None = None) -> None:
    """Initialise Marker and, when a PDF is available, run one page through it."""

    prepare_tool_cache_env(cache_dir)
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config_parser = ConfigParser(
        {
            "output_format": "json",
            "page_range": "0",
            "disable_image_extraction": True,
        }
    )
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    if smoke_pdf is not None:
        converter(str(smoke_pdf))


def smoke_pdf_path(data_dir: Path) -> Path | None:
    """Return a staged smoke PDF if the dataset is available."""

    for doc_id in SMOKE_DOC_IDS:
        try:
            return resolve_pdf(doc_id, data_dir=data_dir)
        except FileNotFoundError:
            continue
    return None


def warm_tool_caches(
    config_raw: dict,
    cache_dir: Path,
    force: bool,
    *,
    smoke: bool = False,
    data_dir: Path | None = None,
) -> None:
    """Warm non-HF tool caches required by later stages."""

    tool_caches = config_raw.get("tool_caches", {})
    prepare_tool_cache_env(cache_dir)
    staged_smoke_pdf = smoke_pdf_path(data_dir) if smoke and data_dir is not None else None
    if tool_caches.get("marker", False):
        print("[prestage] warming Marker cache")
        warm_marker_cache(cache_dir, staged_smoke_pdf)
        print("[prestage] Marker cache ready")
    if tool_caches.get("paddleocr", False):
        print("[prestage] warming PaddleOCR cache")
        warm_paddleocr_cache()
        print("[prestage] PaddleOCR cache ready")
    if tool_caches.get("docling", False):
        print("[prestage] warming Docling cache")
        warm_docling_cache(cache_dir, force)
        print("[prestage] Docling cache ready")


def verify_retrieval_smoke_imports() -> None:
    """Verify retrieval packages import and BM25 can run one tiny query."""

    from rank_bm25 import BM25Okapi
    import FlagEmbedding  # noqa: F401
    import colpali_engine  # noqa: F401

    scores = BM25Okapi([["smoke", "document"]]).get_scores(["smoke"])
    if not scores.size or float(scores[0]) < 0:
        raise RuntimeError("BM25 smoke query produced an invalid score")
    print("[prestage] retrieval imports ready: rank_bm25, FlagEmbedding, colpali_engine")


def verify_tool_smoke_calls(data_dir: Path, cache_dir: Path) -> None:
    """Run one tiny call through every M2 document tool on a smoke page."""

    pdf = smoke_pdf_path(data_dir)
    if pdf is None:
        raise FileNotFoundError(
            f"no smoke PDFs are staged under {data_dir}; run prestage without --skip-dataset first"
        )

    pages = render_pdf(pdf, page_indices=(0,), cache_dir=cache_dir, dpi=96)
    from tools.layout import marker_bbox_json, marker_text
    from tools.text import embedded, ocr
    from tools.visual import full_page, region_crop, resolution

    embedded_text = embedded(pages)
    if not embedded_text or not embedded_text[0].strip():
        raise RuntimeError("PyMuPDF embedded text smoke returned empty text")

    marker_page_text = marker_text(pages, allow_fallback=False)
    if not marker_page_text or not marker_page_text[0].strip():
        raise RuntimeError("Marker text smoke returned empty text")

    layout_json = marker_bbox_json(pages, allow_fallback=False)
    parsed = json.loads(layout_json[0])
    if parsed.get("source") != "marker" or not parsed.get("blocks"):
        raise RuntimeError(f"Marker bbox JSON smoke returned invalid payload: {parsed}")

    ocr_text = ocr(pages, allow_embedded_fallback=False)
    if not ocr_text or not ocr_text[0].strip():
        raise RuntimeError("PaddleOCR smoke returned empty text")

    full = full_page(pages)
    scaled = resolution(pages, 0.5)
    cropped = region_crop(pages, regions=[{"page": 0}])
    if not full or full[0].token_cost_estimate <= 0:
        raise RuntimeError("visual full_page smoke returned invalid token estimate")
    if not scaled or scaled[0].token_cost_estimate >= full[0].token_cost_estimate:
        raise RuntimeError("visual resolution smoke did not reduce token estimate")
    if cropped[0].source != "region_crop_page_fallback":
        raise RuntimeError("region_crop did not degrade to page-level fallback")
    print("[prestage] tool smoke calls ready: embedded, OCR, Marker text/layout, visual")


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
        f"smoke={args.smoke} max_workers={max_workers} force_download={args.force_download} "
        f"HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'missing'}"
    )

    skip_reasoners = args.skip_models or args.skip_reasoner_models
    skip_retrievers = args.skip_models or args.skip_retrieval_models

    if not skip_reasoners:
        model_ids = args.model_id or (
            smoke_reasoner_models(config.raw) if args.smoke else list(config.raw["models"])
        )
        print(f"[prestage] staging {len(model_ids)} reasoner model(s): {', '.join(model_ids)}")
        for model_id in model_ids:
            print(f"[prestage] checking reasoner model {model_id}")
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
        retrieval_model_ids = args.retrieval_model_id or (
            smoke_retrieval_models(config.raw) if args.smoke else flatten_retrieval_models(config.raw)
        )
        print(
            "[prestage] staging "
            f"{len(retrieval_model_ids)} retrieval model(s): {', '.join(retrieval_model_ids)}"
        )
        for model_id in retrieval_model_ids:
            print(f"[prestage] checking retrieval model {model_id}")
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
        warm_tool_caches(config.raw, cache_dir, args.force_download, smoke=args.smoke, data_dir=data_dir)
        if args.smoke:
            verify_retrieval_smoke_imports()
            verify_tool_smoke_calls(data_dir, cache_dir)
    else:
        print("[prestage] skipping tool cache warmup")

    print("[prestage] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
