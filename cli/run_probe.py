"""Stage 1 feasibility probes for MMLongBench-Doc and hardware readiness.

The probes are deliberately lightweight and independent. Local probes inspect
the already-staged MMLongBench-Doc files under `.data/`; hardware probes report
what can be checked without a GPU and leave the expensive load/generate checks
for Kaya compute jobs.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from profile_datasets import REGISTRY, class_breakdown, value_classes
except Exception:  # pragma: no cover - defensive fallback for broken script deps.
    REGISTRY = {
        "mmlongbench": {
            "repo_id": "yubo2333/MMLongBench-Doc",
            "repo_type": "dataset",
            "parquet_split": "train",
        }
    }

    def value_classes(value: Any) -> set[str]:
        parsed = parse_list(value)
        return {str(v) for v in parsed} if parsed else {str(value)}

    def class_breakdown(
        records: list[dict[str, Any]], field: str, doc_field: str | None
    ) -> tuple[list[tuple[str, int, int | None]], int, int | None]:
        counts: Counter[str] = Counter()
        docs: dict[str, set[str]] = defaultdict(set)
        all_docs: set[str] = set()
        total = 0
        for record in records:
            if field not in record:
                continue
            total += 1
            doc_id = str(record.get(doc_field)) if doc_field else None
            if doc_id is not None:
                all_docs.add(doc_id)
            for cls in value_classes(record[field]):
                counts[cls] += 1
                if doc_id is not None:
                    docs[cls].add(doc_id)
        rows = [
            (cls, count, len(docs[cls]) if doc_field else None)
            for cls, count in counts.most_common()
        ]
        return rows, total, len(all_docs) if doc_field else None


PASS = "pass"
PARTIAL = "partial"
FAIL = "fail"
NEEDS_HARDWARE = "needs_hardware"

QWEN3_VL_MODELS = (
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
)
BGE_PROBE_MODEL = "BAAI/bge-small-en-v1.5"
VISION_RETRIEVER_PROBE_MODEL = "vidore/colqwen2.5-v0.2"

BOX_FIELD_HINTS = (
    "bbox",
    "box",
    "bounding",
    "coordinate",
    "coordinates",
    "polygon",
    "quad",
    "points",
    "x0",
    "y0",
    "x1",
    "y1",
)

ABSTENTION_SURFACE_FORMS = (
    "not answerable",
    "cannot be answered",
    "can't be answered",
    "insufficient information",
    "not enough information",
    "no answer",
    "unknown from the document",
)


@dataclass(frozen=True)
class ProbeConfig:
    """Runtime settings shared by all Stage 1 probes."""

    root: Path = ROOT
    data_dir: Path = ROOT / ".data"
    sample: int = 64
    pdf_sample: int = 12
    max_pages_per_pdf: int = 3
    allow_network: bool = False
    run_heavy: bool = False
    model_ids: tuple[str, ...] = QWEN3_VL_MODELS
    bge_model_id: str = BGE_PROBE_MODEL
    vision_retriever_model_id: str = VISION_RETRIEVER_PROBE_MODEL
    heavy_timeout_seconds: int = 900


@dataclass(frozen=True)
class ProbeVerdict:
    """A structured probe result that can be printed or asserted in tests."""

    name: str
    status: str
    summary: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json_ready(asdict(self))


def json_ready(value: Any) -> Any:
    """Convert paths, counters, and nested containers to JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    return value


def parse_list(value: Any) -> list[Any]:
    """Parse real or stringified list fields used by MMLongBench-Doc."""

    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "nan", "null"}:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            for parser in (ast.literal_eval, json.loads):
                try:
                    parsed = parser(stripped)
                    if isinstance(parsed, (list, tuple, set)):
                        return list(parsed)
                    return [parsed]
                except Exception:
                    continue
        return [value]
    return [value]


def parse_int_list(value: Any) -> list[int]:
    """Parse a list field and retain values that can be read as integers."""

    out: list[int] = []
    for item in parse_list(value):
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def find_mmlongbench_root(data_dir: Path) -> Path:
    """Return the directory containing MMLongBench `data/` and `documents/`."""

    candidates = [data_dir / "mmlongbench", data_dir]
    for candidate in candidates:
        if (candidate / "data").is_dir() or (candidate / "documents").is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not find MMLongBench-Doc under {data_dir}; expected "
        "`mmlongbench/data/*.parquet` and `mmlongbench/documents/*.pdf`"
    )


def load_mmlongbench_records(
    config: ProbeConfig, limit: int | None = None
) -> tuple[list[dict[str, Any]], str]:
    """Load MMLongBench records locally, falling back to the profiler strategy."""

    try:
        dataset_root = find_mmlongbench_root(config.data_dir)
    except FileNotFoundError:
        dataset_root = None

    if dataset_root is not None:
        shards = sorted((dataset_root / "data").glob("*.parquet"))
        if shards:
            import pandas as pd

            frames = [pd.read_parquet(shard) for shard in shards]
            frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            if limit is not None:
                frame = frame.head(limit)
            return frame.to_dict(orient="records"), f"local parquet: {dataset_root / 'data'}"

    subset = config.data_dir / "mmlongbench_subset" / "samples.json"
    if subset.is_file():
        records = json.loads(subset.read_text())
        if limit is not None:
            records = records[:limit]
        return records, f"local subset json: {subset}"

    if config.allow_network:
        from profile_datasets import STRATEGIES

        entry = REGISTRY["mmlongbench"]
        max_records = limit if limit is not None else 10**9
        records, source, _ordered_keys, _image_fields = STRATEGIES[entry["strategy"]](
            entry, max_records
        )
        return records, source

    raise FileNotFoundError(
        f"no local MMLongBench records found under {config.data_dir}; "
        "stage the dataset or rerun with --allow-network"
    )


def resolve_pdf(config: ProbeConfig, doc_id: Any) -> Path | None:
    """Resolve a MMLongBench PDF id against the local `.data` layout."""

    name = str(doc_id)
    names = [name]
    if not name.lower().endswith(".pdf"):
        names.append(f"{name}.pdf")

    roots: list[Path] = []
    try:
        roots.append(find_mmlongbench_root(config.data_dir))
    except FileNotFoundError:
        pass
    roots.extend(
        [
            config.data_dir / "mmlongbench",
            config.data_dir / "mmlongbench_subset",
            config.data_dir,
        ]
    )

    for root in roots:
        for candidate_name in names:
            candidate = root / "documents" / candidate_name
            if candidate.is_file():
                return candidate
    return None


def unique_doc_ids(records: Iterable[dict[str, Any]]) -> list[str]:
    """Return document ids in first-seen order."""

    out: list[str] = []
    seen: set[str] = set()
    for record in records:
        doc_id = record.get("doc_id")
        if doc_id is None:
            continue
        key = str(doc_id)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def sample_pdfs(config: ProbeConfig, records: list[dict[str, Any]]) -> list[Path]:
    """Choose a deterministic sample of locally available PDFs."""

    paths: list[Path] = []
    seen: set[Path] = set()
    for doc_id in unique_doc_ids(records):
        path = resolve_pdf(config, doc_id)
        if path is not None and path not in seen:
            seen.add(path)
            paths.append(path)
        if len(paths) >= config.pdf_sample:
            return paths

    try:
        doc_dir = find_mmlongbench_root(config.data_dir) / "documents"
    except FileNotFoundError:
        return paths

    for path in sorted(doc_dir.glob("*.pdf")):
        if path not in seen:
            seen.add(path)
            paths.append(path)
        if len(paths) >= config.pdf_sample:
            break
    return paths


def probe_loader_smoke(config: ProbeConfig) -> ProbeVerdict:
    """Confirm fields parse and PDFs resolve for a small MMLongBench sample."""

    full_records, source = load_mmlongbench_records(config)
    sample_records = full_records[: config.sample]
    required = [
        "doc_id",
        "doc_type",
        "question",
        "answer",
        "evidence_pages",
        "evidence_sources",
        "answer_format",
    ]
    keys = set(full_records[0]) if full_records else set()
    missing = [field for field in required if field not in keys]

    parse_errors: list[dict[str, Any]] = []
    for index, record in enumerate(sample_records):
        pages = parse_int_list(record.get("evidence_pages"))
        sources = parse_list(record.get("evidence_sources"))
        if record.get("evidence_pages") not in (None, "", "[]") and not pages:
            parse_errors.append({"row": index, "field": "evidence_pages", "value": record.get("evidence_pages")})
        if record.get("evidence_sources") not in (None, "", "[]") and not sources:
            parse_errors.append(
                {"row": index, "field": "evidence_sources", "value": record.get("evidence_sources")}
            )

    sample_docs = unique_doc_ids(sample_records)
    resolved = {doc_id: resolve_pdf(config, doc_id) for doc_id in sample_docs}
    missing_pdfs = [doc_id for doc_id, path in resolved.items() if path is None]
    unanswerable_count = sum(
        1
        for record in full_records
        if str(record.get("answer", "")).strip().casefold() == "not answerable"
    )

    status = PASS
    if missing or parse_errors or missing_pdfs or unanswerable_count == 0:
        status = FAIL if missing or parse_errors else PARTIAL

    details = {
        "source": source,
        "records_total": len(full_records),
        "records_checked": len(sample_records),
        "required_fields_present": {field: field in keys for field in required},
        "parse_errors": parse_errors,
        "sample_documents": len(sample_docs),
        "pdfs_resolved": sum(1 for path in resolved.values() if path is not None),
        "missing_pdfs": missing_pdfs[:10],
        "unanswerable_count": unanswerable_count,
        "not_answerable_signal": "answer == 'Not answerable'",
    }
    summary = (
        f"loaded {len(full_records)} records from {source}; "
        f"resolved {details['pdfs_resolved']}/{len(sample_docs)} sample PDFs; "
        f"found {unanswerable_count} unanswerable questions"
    )
    return ProbeVerdict("loader", status, summary, details)


def pdf_text_lengths(path: Path, max_pages: int) -> tuple[int, list[int]]:
    """Return `(page_count, sampled text lengths)` for a PDF."""

    import fitz

    with fitz.open(path) as pdf:
        page_count = pdf.page_count
        lengths = [
            len(pdf.load_page(index).get_text("text").strip())
            for index in range(min(page_count, max_pages))
        ]
    return page_count, lengths


def probe_scanned_vs_born_digital(config: ProbeConfig) -> ProbeVerdict:
    """Estimate the fraction of sampled PDFs without an embedded text layer."""

    records, _source = load_mmlongbench_records(config, limit=max(config.sample, config.pdf_sample))
    paths = sample_pdfs(config, records)
    if not paths:
        return ProbeVerdict(
            "scanned",
            FAIL,
            "no local PDFs were available to inspect",
            {"pdf_sample_requested": config.pdf_sample, "pdfs_checked": 0},
        )

    checked: list[dict[str, Any]] = []
    scanned = 0
    born_digital = 0
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            page_count, lengths = pdf_text_lengths(path, config.max_pages_per_pdf)
        except Exception as exc:
            errors.append({"pdf": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        sampled_pages = len(lengths)
        text_pages = sum(1 for length in lengths if length >= 20)
        avg_chars = round(sum(lengths) / sampled_pages, 1) if sampled_pages else 0.0
        is_scanned_like = sampled_pages == 0 or text_pages == 0 or avg_chars < 20
        scanned += int(is_scanned_like)
        born_digital += int(not is_scanned_like)
        checked.append(
            {
                "pdf": path.name,
                "pages": page_count,
                "sampled_pages": sampled_pages,
                "text_pages": text_pages,
                "avg_chars_per_sampled_page": avg_chars,
                "classification": "scanned_like" if is_scanned_like else "born_digital_like",
            }
        )

    total = scanned + born_digital
    if total == 0:
        return ProbeVerdict(
            "scanned",
            FAIL,
            "PyMuPDF could not inspect any sampled PDFs",
            {"errors": errors, "pdfs_requested": len(paths)},
        )

    scanned_fraction = scanned / total
    if scanned and born_digital:
        status = PASS
        decision = "real embedded-text vs OCR slice exists in the local sample"
    elif scanned:
        status = PARTIAL
        decision = "sample is scanned-like only; find born-digital controls before RQ8"
    else:
        status = PARTIAL
        decision = "sample is born-digital only; synthetic degradation may be needed"

    details = {
        "pdfs_checked": total,
        "scanned_like": scanned,
        "born_digital_like": born_digital,
        "scanned_fraction": round(scanned_fraction, 4),
        "text_threshold_chars_per_page": 20,
        "max_pages_per_pdf": config.max_pages_per_pdf,
        "documents": checked,
        "errors": errors,
        "decision": decision,
    }
    return ProbeVerdict(
        "scanned",
        status,
        f"{scanned}/{total} sampled PDFs are scanned-like; {decision}",
        details,
    )


def probe_in_page_boxes(config: ProbeConfig) -> ProbeVerdict:
    """Check whether MMLongBench exposes in-page evidence coordinates."""

    records, source = load_mmlongbench_records(config, limit=config.sample)
    keys = sorted({key for record in records for key in record})
    candidate_fields = [
        key for key in keys if any(hint in key.lower() for hint in BOX_FIELD_HINTS)
    ]

    non_empty_candidates: dict[str, int] = {}
    for field in candidate_fields:
        non_empty_candidates[field] = sum(
            1 for record in records if parse_list(record.get(field))
        )

    if non_empty_candidates:
        status = PASS
        summary = "found candidate in-page coordinate fields"
        crop_decision = "region crops can be tested after field semantics are validated"
    else:
        status = PASS
        summary = "MMLongBench records expose page-level evidence only; no in-page boxes found"
        crop_decision = (
            "v1 region_crop must use page-level crops; LongDocURL remains the future "
            "source for true in-page boxes in optional Stage 10"
        )

    details = {
        "source": source,
        "records_checked": len(records),
        "fields": keys,
        "candidate_box_fields": non_empty_candidates,
        "crop_decision": crop_decision,
        "page_level_fields": ["evidence_pages", "evidence_sources"],
    }
    return ProbeVerdict("boxes", status, summary, details)


def run_child_python(label: str, code: str, args: list[str], timeout: int) -> dict[str, Any]:
    """Run an optional heavyweight probe in a child Python process."""

    try:
        result = subprocess.run(
            [sys.executable, "-c", code, *args],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "label": label,
            "returncode": result.returncode,
            "status": "pass" if result.returncode == 0 else "fail",
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "label": label,
            "status": "timeout",
            "timeout_seconds": timeout,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }
    except Exception as exc:
        return {"label": label, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def torch_cuda_available(details: dict[str, Any]) -> bool:
    """Read the CUDA availability flag from a probe details dictionary."""

    cuda = details.get("cuda")
    return isinstance(cuda, dict) and bool(cuda.get("available"))


VLLM_GENERATION_CODE = r"""
import json
import sys

from vllm import LLM, SamplingParams

model_id = sys.argv[1]
llm = LLM(
    model=model_id,
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.80,
)
outputs = llm.generate(
    ["Answer with exactly one word: OK"],
    SamplingParams(max_tokens=8, temperature=0.0),
)
print(json.dumps({"model_id": model_id, "text": outputs[0].outputs[0].text.strip()}))
"""


BGE_EMBEDDING_CODE = r"""
import json
import sys

import numpy as np
import torch
from FlagEmbedding import FlagModel

model_id = sys.argv[1]
model = FlagModel(model_id, use_fp16=bool(torch.cuda.is_available()))
embeddings = model.encode(["chart revenue", "plain text answer"], batch_size=2)
arr = np.asarray(embeddings)
print(json.dumps({"model_id": model_id, "shape": list(arr.shape)}))
"""


COLQWEN_INDEX_CODE = r"""
import json
import sys

import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor, ColQwen2_5, ColQwen2_5_Processor

model_id = sys.argv[1]
if "2.5" in model_id or "qwen2_5" in model_id.lower():
    model_cls = ColQwen2_5
    processor_cls = ColQwen2_5_Processor
else:
    model_cls = ColQwen2
    processor_cls = ColQwen2Processor

kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
}
if is_flash_attn_2_available():
    kwargs["attn_implementation"] = "flash_attention_2"

model = model_cls.from_pretrained(model_id, **kwargs).eval()
processor = processor_cls.from_pretrained(model_id)
images = [
    Image.new("RGB", (64, 64), color="white"),
    Image.new("RGB", (64, 64), color="black"),
]
queries = ["revenue chart", "plain text answer"]
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)
scores = processor.score_multi_vector(query_embeddings, image_embeddings)
print(json.dumps({"model_id": model_id, "scores_shape": list(scores.shape)}))
"""


def probe_model_family(config: ProbeConfig) -> ProbeVerdict:
    """Check model repository visibility and mark GPU generation work for Kaya."""

    repo_checks: dict[str, dict[str, Any]] = {}
    if config.allow_network:
        from huggingface_hub import HfApi

        api = HfApi()
        for model_id in config.model_ids:
            try:
                info = api.model_info(model_id)
                repo_checks[model_id] = {
                    "exists": True,
                    "private": bool(getattr(info, "private", False)),
                    "sha": getattr(info, "sha", None),
                }
            except Exception as exc:
                repo_checks[model_id] = {
                    "exists": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    else:
        repo_checks = {
            model_id: {"exists": None, "skipped": "rerun with --allow-network"}
            for model_id in config.model_ids
        }

    try:
        import transformers

        transformer_compat = {
            "version": transformers.__version__,
            "has_Qwen3VLForConditionalGeneration": hasattr(
                transformers, "Qwen3VLForConditionalGeneration"
            ),
            "has_AutoModelForMultimodalLM": hasattr(transformers, "AutoModelForMultimodalLM"),
            "has_AutoProcessor": hasattr(transformers, "AutoProcessor"),
        }
        if not (
            transformer_compat["has_Qwen3VLForConditionalGeneration"]
            or transformer_compat["has_AutoModelForMultimodalLM"]
        ):
            transformer_compat["action"] = (
                "Current transformers pin does not expose the Qwen3-VL classes shown "
                "on the model cards; Stage 6 must upgrade within the ColPali/vLLM "
                "compatibility window or use a vLLM path that supports Qwen3-VL."
            )
    except Exception as exc:
        transformer_compat = {"error": f"{type(exc).__name__}: {exc}"}

    local_interface_smoke = {
        "reasoner_contract": "answer(prompt: str) -> str",
        "local_stub_response": _EchoReasoner("local").answer("ping"),
        "api_stub_response": _EchoReasoner("api").answer("ping"),
        "note": "Stage 3 freezes the production Reasoner ABC; this smoke keeps the swap requirement explicit.",
    }

    heavy_details: dict[str, Any] = {
        "load_generate": "not run",
        "target": "Kaya compute node with pre-staged HF cache",
        "run_command": (
            "envs/mpvrdu/bin/python -m kaya.kaya run-probe model-family "
            "--target gpu --heavy --json"
        ),
    }
    if config.run_heavy:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            heavy_details.update(
                {
                    "torch_version": torch.__version__,
                    "cuda_available": cuda_available,
                    "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
                }
            )
            if cuda_available:
                heavy_details["load_generate"] = "running vLLM generation smoke per model id"
                heavy_details["vllm_generation"] = [
                    run_child_python(
                        f"vllm:{model_id}",
                        VLLM_GENERATION_CODE,
                        [model_id],
                        config.heavy_timeout_seconds,
                    )
                    for model_id in config.model_ids
                ]
            else:
                heavy_details["load_generate"] = "no CUDA device visible"
        except Exception as exc:
            heavy_details["load_generate"] = f"torch import/check failed: {type(exc).__name__}: {exc}"

    all_repos_known = all(check.get("exists") is True for check in repo_checks.values())
    any_repo_failed = any(check.get("exists") is False for check in repo_checks.values())
    generation_results = heavy_details.get("vllm_generation", [])
    any_generation_failed = any(
        isinstance(result, dict) and result.get("status") != "pass"
        for result in generation_results
    )
    if any_repo_failed or any_generation_failed:
        status = FAIL
    elif generation_results and all(
        isinstance(result, dict) and result.get("status") == "pass"
        for result in generation_results
    ):
        status = PASS if all_repos_known else PARTIAL
    elif all_repos_known and config.run_heavy:
        status = PARTIAL
    elif all_repos_known:
        status = PARTIAL
    else:
        status = NEEDS_HARDWARE

    details = {
        "model_ids": list(config.model_ids),
        "repo_checks": repo_checks,
        "transformers_compatibility": transformer_compat,
        "local_api_swap_smoke": local_interface_smoke,
        "heavy_checks": heavy_details,
        "qwen3_32b_scope_pending": (
            "Confirm on Kaya whether 32B supports full-doc, or must be scoped to "
            "oracle/retrieved conditions only."
        ),
    }
    summary = (
        "model repository checks "
        + ("completed" if config.allow_network else "skipped")
        + "; load/generate remains a Kaya compute probe"
    )
    return ProbeVerdict("model-family", status, summary, details)


class _EchoReasoner:
    """Tiny local/API stand-in used only to keep Stage 1's swap probe concrete."""

    def __init__(self, backend: str) -> None:
        self.backend = backend

    def answer(self, prompt: str) -> str:
        return f"{self.backend}: {prompt}"


def probe_vision_retrieval(config: ProbeConfig) -> ProbeVerdict:
    """Check text retrieval locally and leave BGE/ColPali memory tests for Kaya."""

    details: dict[str, Any] = {
        "bm25": {},
        "bge": {"status": "not run", "reason": "model load may download/cache weights"},
        "vision": {
            "status": "not run",
            "reason": "ColPali/ColQwen indexing requires GPU memory check on Kaya",
        },
        "run_command": (
            "envs/mpvrdu/bin/python -m kaya.kaya run-probe retrieval "
            "--target gpu --heavy --json"
        ),
    }

    try:
        from rank_bm25 import BM25Okapi

        corpus = [
            "chart revenue increased in the annual report",
            "plain text answer appears in the contract",
            "figure caption explains the workflow",
        ]
        tokenized = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores("revenue chart".split())
        best = int(max(range(len(scores)), key=lambda index: scores[index]))
        details["bm25"] = {
            "status": "pass",
            "top_index": best,
            "top_document": corpus[best],
            "scores": [float(score) for score in scores],
        }
    except Exception as exc:
        details["bm25"] = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}

    try:
        import FlagEmbedding  # noqa: F401

        details["bge"]["import"] = "pass"
    except Exception as exc:
        details["bge"]["import"] = f"fail: {type(exc).__name__}: {exc}"

    try:
        import colpali_engine  # noqa: F401

        details["vision"]["import"] = "pass"
    except Exception as exc:
        details["vision"]["import"] = f"fail: {type(exc).__name__}: {exc}"

    if config.run_heavy:
        try:
            import torch

            details["cuda"] = {
                "available": bool(torch.cuda.is_available()),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "torch_version": torch.__version__,
            }
            details["bge"]["status"] = "running" if torch.cuda.is_available() else "running_cpu"
            details["bge"]["model_id"] = config.bge_model_id
            details["bge"]["embedding_probe"] = run_child_python(
                "bge",
                BGE_EMBEDDING_CODE,
                [config.bge_model_id],
                config.heavy_timeout_seconds,
            )
            if torch.cuda.is_available():
                details["vision"]["status"] = "running"
                details["vision"]["model_id"] = config.vision_retriever_model_id
                details["vision"]["index_probe"] = run_child_python(
                    "colqwen",
                    COLQWEN_INDEX_CODE,
                    [config.vision_retriever_model_id],
                    config.heavy_timeout_seconds,
                )
            else:
                details["vision"]["status"] = "not run"
                details["vision"]["reason"] = "no CUDA device visible for ColQwen/ColPali memory probe"
        except Exception as exc:
            details["cuda"] = {"error": f"{type(exc).__name__}: {exc}"}

    bm25_ok = details["bm25"].get("status") == "pass"
    if not bm25_ok:
        status = FAIL
        summary = "BM25 text retrieval smoke failed"
    elif config.run_heavy:
        bge_result = details["bge"].get("embedding_probe")
        vision_result = details["vision"].get("index_probe")
        bge_ok = isinstance(bge_result, dict) and bge_result.get("status") == "pass"
        vision_ok = isinstance(vision_result, dict) and vision_result.get("status") == "pass"
        if bge_ok and vision_ok:
            status = PASS
            summary = "BM25, BGE, and ColQwen retrieval probes ran successfully"
        elif bge_ok and not torch_cuda_available(details):
            status = PARTIAL
            summary = "BM25 and BGE ran; vision retrieval still needs a CUDA Kaya job"
        else:
            status = FAIL
            summary = "one or more heavy retrieval probes failed"
    else:
        status = PARTIAL
        summary = "BM25 text retrieval works; BGE and ColPali/ColQwen memory probes remain for Kaya"
    return ProbeVerdict("retrieval", status, summary, details)


def probe_unanswerable_abstention(config: ProbeConfig) -> ProbeVerdict:
    """Count native unanswerables and draft the abstention rule."""

    records, source = load_mmlongbench_records(config)
    unanswerable = [
        record
        for record in records
        if str(record.get("answer", "")).strip().casefold() == "not answerable"
    ]
    total = len(records)
    rate = len(unanswerable) / total if total else 0.0
    examples = [
        {
            "doc_id": record.get("doc_id"),
            "question": record.get("question"),
            "evidence_pages": parse_int_list(record.get("evidence_pages")),
            "evidence_sources": parse_list(record.get("evidence_sources")),
        }
        for record in unanswerable[:3]
    ]
    definition = {
        "native_unanswerable": "gold answer string normalised with strip().casefold() == 'not answerable'",
        "abstains_if_prediction_contains": list(ABSTENTION_SURFACE_FORMS),
        "hallucinates_if": (
            "question is natively unanswerable, or retrieved condition has page recall 0, "
            "and the model gives a substantive non-abstaining answer"
        ),
        "generation_probe": (
            "not run locally; rerun model-family with --run-heavy after Qwen3-VL is staged on Kaya"
        ),
    }

    status = PASS if unanswerable else FAIL
    details = {
        "source": source,
        "records_total": total,
        "unanswerable_count": len(unanswerable),
        "unanswerable_rate": round(rate, 4),
        "examples": examples,
        "abstention_definition_proposal": definition,
    }
    return ProbeVerdict(
        "unanswerable",
        status,
        f"{len(unanswerable)}/{total} records are natively unanswerable ({rate:.1%})",
        details,
    )


def probe_doc_type_distribution(config: ProbeConfig) -> ProbeVerdict:
    """Report doc_type counts and propose the spectrum mapping for approval."""

    records, source = load_mmlongbench_records(config)
    rows, total_questions, total_documents = class_breakdown(records, "doc_type", "doc_id")

    evidence_by_doc_type: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        doc_type = str(record.get("doc_type"))
        for cls in value_classes(record.get("evidence_sources")):
            if cls and cls != "(none)":
                evidence_by_doc_type[doc_type][cls] += 1

    evidence_mix: dict[str, dict[str, Any]] = {}
    proposed_mapping: dict[str, str] = {}
    for doc_type, counter in sorted(evidence_by_doc_type.items()):
        total = sum(counter.values())
        visual = sum(counter[label] for label in ("Chart", "Figure"))
        text = sum(
            counter[label]
            for label in ("Pure-text (Plain-text)", "Generalized-text (Layout)")
        )
        table = counter["Table"]
        visual_fraction = visual / total if total else 0.0
        text_fraction = text / total if total else 0.0
        table_fraction = table / total if total else 0.0
        if visual_fraction >= 0.35:
            bucket = "visual-heavy"
        elif visual_fraction <= 0.2 and text_fraction >= 0.5:
            bucket = "text-heavy"
        else:
            bucket = "in-between"
        proposed_mapping[doc_type] = bucket
        evidence_mix[doc_type] = {
            "total_evidence_labels": total,
            "visual_fraction": round(visual_fraction, 4),
            "text_fraction": round(text_fraction, 4),
            "table_fraction": round(table_fraction, 4),
            "labels": dict(counter.most_common()),
        }

    question_counts = [
        {"doc_type": cls, "questions": questions, "documents": documents}
        for cls, questions, documents in rows
    ]
    details = {
        "source": source,
        "records_total": len(records),
        "total_questions_with_doc_type": total_questions,
        "total_documents_with_doc_type": total_documents,
        "question_counts": question_counts,
        "evidence_mix_by_doc_type": evidence_mix,
        "spectrum_mapping_proposal": proposed_mapping,
        "proposal_rule": (
            "visual-heavy if Chart/Figure >= 35%; text-heavy if Chart/Figure <= 20% "
            "and Pure-text/Layout >= 50%; otherwise in-between. Human approval required."
        ),
    }
    return ProbeVerdict(
        "doc-type",
        PASS,
        f"found {len(rows)} doc_type classes across {len(records)} records",
        details,
    )


ProbeFunc = Callable[[ProbeConfig], ProbeVerdict]

PROBES: dict[str, ProbeFunc] = {
    "loader": probe_loader_smoke,
    "scanned": probe_scanned_vs_born_digital,
    "boxes": probe_in_page_boxes,
    "model-family": probe_model_family,
    "retrieval": probe_vision_retrieval,
    "unanswerable": probe_unanswerable_abstention,
    "doc-type": probe_doc_type_distribution,
}

LOCAL_PROBES = ("loader", "scanned", "boxes", "unanswerable", "doc-type")
HEAVY_PROBES = ("model-family", "retrieval")


def run_selected(name: str, config: ProbeConfig) -> list[ProbeVerdict]:
    """Run one probe group."""

    if name == "list":
        return [
            ProbeVerdict(
                "list",
                PASS,
                "available probes",
                {"local": list(LOCAL_PROBES), "heavy": list(HEAVY_PROBES), "all": list(PROBES)},
            )
        ]
    if name == "local":
        return [PROBES[probe](config) for probe in LOCAL_PROBES]
    if name == "all":
        return [PROBES[probe](config) for probe in [*LOCAL_PROBES, *HEAVY_PROBES]]
    return [PROBES[name](config)]


def print_verdicts(verdicts: list[ProbeVerdict], as_json: bool) -> None:
    """Print verdicts either as machine-readable JSON or compact text."""

    if as_json:
        payload: Any = verdicts[0].to_dict() if len(verdicts) == 1 else [v.to_dict() for v in verdicts]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    for verdict in verdicts:
        print(f"[{verdict.status}] {verdict.name}: {verdict.summary}")
        if verdict.name == "doc-type":
            for row in verdict.details["question_counts"]:
                print(f"  {row['doc_type']}: {row['questions']} q / {row['documents']} docs")
            print("  proposed spectrum mapping:")
            for doc_type, bucket in verdict.details["spectrum_mapping_proposal"].items():
                print(f"    {doc_type}: {bucket}")
        elif verdict.name == "scanned":
            print(
                "  scanned_fraction="
                f"{verdict.details.get('scanned_fraction')} "
                f"({verdict.details.get('scanned_like')}/"
                f"{verdict.details.get('pdfs_checked')})"
            )
        elif verdict.name == "unanswerable":
            print(
                "  unanswerable_rate="
                f"{verdict.details.get('unanswerable_rate')} "
                f"count={verdict.details.get('unanswerable_count')}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "probe",
        choices=["list", "local", "all", *PROBES.keys()],
        help="probe or probe group to run",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--sample", type=int, default=64)
    parser.add_argument("--pdf-sample", type=int, default=12)
    parser.add_argument("--max-pages-per-pdf", type=int, default=3)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--run-heavy", action="store_true")
    parser.add_argument("--bge-model-id", default=BGE_PROBE_MODEL)
    parser.add_argument("--vision-retriever-model-id", default=VISION_RETRIEVER_PROBE_MODEL)
    parser.add_argument("--heavy-timeout-seconds", type=int, default=900)
    parser.add_argument("--json", action="store_true", help="print structured JSON")
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="override/append a model id to check; defaults to Qwen3-VL sizes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_ids = tuple(args.model_ids) if args.model_ids else QWEN3_VL_MODELS
    config = ProbeConfig(
        root=args.root.resolve(),
        data_dir=args.data_dir.resolve(),
        sample=args.sample,
        pdf_sample=args.pdf_sample,
        max_pages_per_pdf=args.max_pages_per_pdf,
        allow_network=args.allow_network,
        run_heavy=args.run_heavy,
        model_ids=model_ids,
        bge_model_id=args.bge_model_id,
        vision_retriever_model_id=args.vision_retriever_model_id,
        heavy_timeout_seconds=args.heavy_timeout_seconds,
    )
    try:
        verdicts = run_selected(args.probe, config)
    except Exception as exc:
        verdicts = [
            ProbeVerdict(
                args.probe,
                FAIL,
                f"probe raised {type(exc).__name__}: {exc}",
                {"error": str(exc), "cwd": os.getcwd()},
            )
        ]

    print_verdicts(verdicts, args.json)
    return 1 if any(verdict.status == FAIL for verdict in verdicts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
