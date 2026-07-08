"""Test Stage-M2 document tool artifacts and prestage smoke selection.

Purpose:
    Verifies that embedded text, OCR result parsing, Marker text/layout JSON,
    visual artifacts, resolution scaling, page-level crop fallback, and Kaya
    smoke prestage subset selection produce well-formed artifacts.

Test role:
    Uses injected fakes for heavy Marker/OCR paths where possible so local tests
    stay fast while Kaya `prestage --smoke` remains the real tool barrier.

Arguments:
    None. Run with `python -m pytest tests/test_tools_smoke.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from data.render import render_pdf
from tools import layout
from tools import text as text_tools
from tools.layout import marker_bbox_json, marker_text
from tools.text import embedded, ocr, text_channel
from tools.visual import full_page, region_crop, resolution


def write_pdf(path: Path, pages: list[str]) -> None:
    """Write a tiny PDF fixture."""

    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def rendered_page(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    write_pdf(pdf, ["This page has smoke text and a tiny table: A B C"])
    return render_pdf(pdf, page_indices=(0,), cache_dir=tmp_path / "results" / "cache", dpi=72)


def test_text_layout_and_visual_tools_return_well_formed_artifacts(tmp_path: Path, monkeypatch) -> None:
    pages = rendered_page(tmp_path)

    def fake_marker_render(page, output_format: str):
        return {
            "children": [
                {
                    "block_type": "Text",
                    "text": f"marker {output_format} page {page.index}",
                    "polygon": [[1, 2], [101, 2], [101, 22], [1, 22]],
                }
            ]
        }

    monkeypatch.setattr(layout, "_marker_render", fake_marker_render)
    monkeypatch.setattr(layout, "_marker_text_from_rendered", lambda rendered: "marker smoke text")

    assert embedded(pages)[0].strip()

    class FakeOCR:
        def predict(self, image_path: str):
            return [{"rec_texts": ["ocr smoke text"]}]

    assert ocr(pages, engine=FakeOCR(), allow_embedded_fallback=False) == ("ocr smoke text",)

    assert marker_text(pages, allow_fallback=False) == ("marker smoke text",)
    layout_payload = json.loads(marker_bbox_json(pages, allow_fallback=False)[0])
    assert layout_payload["source"] == "marker"
    assert layout_payload["page_index"] == 0
    assert layout_payload["blocks"][0]["bbox"] == [1.0, 2.0, 101.0, 22.0]

    full = full_page(pages)
    half = resolution(pages, 0.5)
    cropped = region_crop(pages, regions=[{"bbox": [0, 0, 10, 10]}])

    assert full[0].part.image_path and full[0].part.image_path.is_file()
    assert full[0].token_cost_estimate > 0
    assert half[0].width < full[0].width
    assert half[0].token_cost_estimate < full[0].token_cost_estimate
    assert cropped[0].source == "region_crop_page_fallback"
    assert cropped[0].metadata["regions_ignored"] is True


def test_marker_bbox_json_falls_back_to_pymupdf_layout(tmp_path: Path, monkeypatch) -> None:
    pages = rendered_page(tmp_path)
    monkeypatch.setattr(layout, "_marker_render", lambda page, output_format: (_ for _ in ()).throw(RuntimeError("no marker")))

    payload = json.loads(marker_bbox_json(pages)[0])

    assert payload["source"] == "pymupdf-fallback"
    assert payload["blocks"]
    assert payload["blocks"][0]["bbox"]


def write_annotation_sheet(path: Path, *, doc_id: str, label: str, human_label: str = "") -> None:
    """Write the minimum annotation sheet columns used by text routing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "doc_id,scan_label,auto_scan\n"
        f"{doc_id},{human_label},{label}\n"
    )


def test_text_channel_routes_scanned_docs_to_cached_ocr(tmp_path: Path, monkeypatch) -> None:
    pages = rendered_page(tmp_path)
    sheet = tmp_path / "annotations" / "doc_labels.csv"
    write_annotation_sheet(sheet, doc_id=pages[0].doc_id, label="scanned")
    monkeypatch.setattr(text_tools, "ANNOTATION_SHEET", sheet)
    text_tools._annotation_scan_labels.cache_clear()

    marker_calls = 0

    def fake_marker_text(_pages):
        nonlocal marker_calls
        marker_calls += 1
        return ("marker should not be used",)

    monkeypatch.setattr(layout, "marker_text", fake_marker_text)

    class FakeOCR:
        calls = 0

        def predict(self, image_path: str):
            self.calls += 1
            return [{"rec_texts": ["ocr routed text"]}]

    engine = FakeOCR()
    assert text_channel(pages, ocr_engine=engine) == ("ocr routed text",)
    assert text_channel(pages, ocr_engine=engine) == ("ocr routed text",)
    assert engine.calls == 1
    assert marker_calls == 0


def test_text_channel_routes_digital_docs_to_marker_and_human_label_wins(tmp_path: Path, monkeypatch) -> None:
    pages = rendered_page(tmp_path)
    sheet = tmp_path / "annotations" / "doc_labels.csv"
    write_annotation_sheet(sheet, doc_id=pages[0].doc_id, label="scanned", human_label="digital")
    monkeypatch.setattr(text_tools, "ANNOTATION_SHEET", sheet)
    text_tools._annotation_scan_labels.cache_clear()

    monkeypatch.setattr(layout, "marker_text", lambda _pages: ("marker routed text",))

    class FailOCR:
        def predict(self, image_path: str):
            raise AssertionError("OCR should not be called for a human-labelled digital doc")

    assert text_channel(pages, ocr_engine=FailOCR()) == ("marker routed text",)


def test_prestage_smoke_selects_small_subset_and_is_repeatable(tmp_path: Path, monkeypatch) -> None:
    import scripts.prestage as prestage

    class FakeConfig:
        remote_root = str(tmp_path)
        raw = {
            "models": [
                "Qwen/Qwen3-VL-2B-Instruct",
                "Qwen/Qwen3-VL-8B-Instruct",
            ],
            "retrieval_models": {
                "text": ["BAAI/bge-small-en-v1.5"],
                "vision": ["vidore/colpali-v1.3", "vidore/colqwen2.5-v0.2"],
            },
            "datasets": {"mmlongbench": "dataset/repo"},
            "tool_caches": {"marker": True, "paddleocr": True},
            "hf": {"max_workers": 4},
        }

        def remote_path(self, key: str) -> str:
            paths = {"cache": ".cache", "data": ".data"}
            return str(tmp_path / paths[key])

    monkeypatch.setattr(prestage, "load_config", lambda path: FakeConfig())
    calls: list[tuple[str, str, int]] = []

    def fake_snapshot(repo_id, repo_type, revision, cache_dir, *, force_download, max_workers):
        calls.append((repo_id, repo_type, max_workers))
        return Path(cache_dir) / repo_id.replace("/", "--")

    monkeypatch.setattr(prestage, "snapshot", fake_snapshot)

    argv = ["--smoke", "--skip-dataset", "--skip-tool-caches", "--max-workers", "1"]
    assert prestage.main(argv) == 0
    first_run = list(calls)
    assert prestage.main(argv) == 0
    second_run = calls[len(first_run) :]

    expected = [
        ("Qwen/Qwen3-VL-2B-Instruct", "model", 1),
        ("BAAI/bge-small-en-v1.5", "model", 1),
        ("vidore/colqwen2.5-v0.2", "model", 1),
    ]
    assert first_run == expected
    assert second_run == expected


def test_local_prestage_defaults_marker_surya_to_cpu(tmp_path: Path, monkeypatch) -> None:
    import scripts.prestage as prestage

    class FakeConfig:
        remote_root = str(tmp_path)
        raw = {
            "models": ["Qwen/Qwen3-VL-2B-Instruct"],
            "retrieval_models": {"text": [], "vision": []},
            "datasets": {"mmlongbench": "dataset/repo"},
            "tool_caches": {},
            "hf": {"max_workers": 4},
            "paths": {"cache": ".cache", "data": ".data"},
        }

        def remote_path(self, key: str) -> str:
            paths = {"cache": ".cache", "data": ".data"}
            return str(tmp_path / paths[key])

    monkeypatch.setattr(prestage, "ROOT", tmp_path)
    monkeypatch.setattr(prestage, "load_config", lambda path: FakeConfig())
    monkeypatch.delenv("TORCH_DEVICE", raising=False)

    argv = ["--local", "--skip-models", "--skip-dataset", "--skip-tool-caches"]
    assert prestage.main(argv) == 0

    assert os.environ["TORCH_DEVICE"] == "cpu"


def test_prestage_tool_device_override_wins(tmp_path: Path, monkeypatch) -> None:
    import scripts.prestage as prestage

    monkeypatch.setenv("TORCH_DEVICE", "cuda")
    prestage.prepare_tool_cache_env(tmp_path / ".cache", tool_device="cpu")

    assert os.environ["TORCH_DEVICE"] == "cpu"


def test_prestage_cache_env_overrides_home_defaults(tmp_path: Path, monkeypatch) -> None:
    import scripts.prestage as prestage

    cache_dir = tmp_path / ".cache"
    monkeypatch.setenv("MODEL_CACHE_DIR", "/home/lxu/.cache/datalab/models")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/home/lxu/.cache/huggingface/transformers")
    monkeypatch.setenv("HF_HUB_CACHE", "/home/lxu/.cache/huggingface/hub")

    prestage.prepare_tool_cache_env(cache_dir)

    assert os.environ["MODEL_CACHE_DIR"] == str(cache_dir / "datalab" / "models")
    assert "TRANSFORMERS_CACHE" not in os.environ
    assert os.environ["HF_HUB_CACHE"] == str(cache_dir)


def test_kaya_artifact_exports_align_repo_cache_roots(tmp_path: Path) -> None:
    from kaya.kaya import KayaConfig, artifact_exports

    config = KayaConfig({"remote_root": str(tmp_path), "paths": {"cache": ".cache"}}, Path("kaya/config.json"))

    lines = artifact_exports(config, offline=True)
    values = {
        line.removeprefix("export ").split("=", 1)[0]: line.removeprefix("export ").split("=", 1)[1]
        for line in lines
        if line.startswith("export ")
    }

    assert "unset TRANSFORMERS_CACHE" in lines
    assert values["HF_HOME"] == f"{tmp_path}/.cache"
    assert values["HF_HUB_CACHE"] == f"{tmp_path}/.cache"
    assert values["MODEL_CACHE_DIR"] == f"{tmp_path}/.cache/datalab/models"
