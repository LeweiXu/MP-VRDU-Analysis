# kaya: target=gpu
# kaya: env=true
# kaya: offline=true
# kaya: job-name=final_probe
"""One-GPU pre-flight probe: exercises every retriever, parser, tool, generation
task, and the resolution range, printing PASS/FAIL per component."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPVRDU_MACHINE", "kaya")
# Use the real PaddleOCR-VL model (not the det+rec floor) for the paddleocrvl parser.
os.environ.setdefault("MPVRDU_PADDLE_RICH", "1")

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, str]] = []


def free() -> None:
    """Drop cached GPU memory between heavy component loads."""

    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def check(name: str, fn) -> None:
    """Run one component test, recording PASS/FAIL and a short detail."""

    start = time.time()
    try:
        detail = fn() or ""
        RESULTS.append((name, "PASS", str(detail)))
        print(f"[PASS] {name:28s} ({time.time() - start:5.1f}s) {detail}", flush=True)
    except Exception as exc:  # noqa: BLE001 - the probe reports failures, never aborts on one
        RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {name:28s} ({time.time() - start:5.1f}s) {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
    finally:
        free()


def run_cmd(name: str, args: list[str], check_fn=None) -> None:
    """Run an ops entry point in a fresh process, then validate its output."""

    def _do():
        proc = subprocess.run([sys.executable, "-m", *args], cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
            raise RuntimeError(f"rc={proc.returncode}: {tail}")
        return check_fn() if check_fn else "ok"

    check(name, _do)


# -- shared sample: one answerable question with gold pages, plus a rendered page --

from config import VISUAL_RESOLUTION_PRESETS, ExperimentConfig  # noqa: E402
from data.binning import stamp_bins  # noqa: E402
from data.loader import load_mmlongbench, resolve_pdf, split_answerable  # noqa: E402
from data.render import pdf_page_count, render_pdf  # noqa: E402

CFG = ExperimentConfig(reasoner_spec="qwen3vl-2b-local", quantization="4bit", visual_resolution="med", judge_spec="stub")
# A probe is a plumbing test, so it does not require the annotation pass to be
# complete; real scored runs keep the strict binning gate.
_ans, _ = split_answerable(stamp_bins(load_mmlongbench(CFG.paths.data_dir), require_complete=False))
Q = next(q for q in _ans if q.evidence_pages)
_pdf = resolve_pdf(Q.doc_id, CFG.paths.data_dir)
PAGE_COUNT = min(pdf_page_count(_pdf), 6)  # bound rendering for the probe
PAGE = render_pdf(_pdf, Q.evidence_pages[:1], cache_dir=CFG.paths.cache_dir, dpi=CFG.dpi)[0]
print(f"probe sample: q={Q.id} doc={Q.doc_id} gold={Q.evidence_pages} pages_probed={PAGE_COUNT}", flush=True)


# -- environment ------------------------------------------------------------

def _gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    n = torch.cuda.device_count()
    note = "" if n == 1 else f" (WARNING: {n} GPUs visible, probe expects 1)"
    return f"{torch.cuda.get_device_name(0)} x{n}{note}"


check("gpu_available", _gpu)


# -- tools ------------------------------------------------------------------

def _embedded():
    from tools.text import embedded_text

    text = embedded_text([PAGE])
    assert text and text[0].strip(), "embedded text empty"
    return f"{len(text[0])} chars"


def _visual():
    from tools.visual import visual_channel

    parts = visual_channel([PAGE])
    assert parts, "no image part built"
    return f"{len(parts)} image part(s)"


def _resolution_presets():
    from tools.visual import tokens_for_pixel_cap

    order = ("min", "low", "med", "high", "full")
    toks = [tokens_for_pixel_cap(VISUAL_RESOLUTION_PRESETS[p]) for p in order]
    assert toks == sorted(toks) and toks[0] < toks[-1], f"presets not monotone: {toks}"
    return f"tokens/page min..full = {toks}"


check("tool.embedded_text", _embedded)
check("tool.visual_channel", _visual)
check("tool.resolution_presets", _resolution_presets)


# -- retrievers -------------------------------------------------------------

def _retrieve(build):
    r = build()
    try:
        pages = r.retrieve(Q, PAGE_COUNT, 3)
        assert isinstance(pages, tuple), f"expected tuple, got {type(pages).__name__}"
        assert len(pages) >= 1, "retriever returned no pages"
        return f"top-3={list(pages)}"
    finally:
        if hasattr(r, "unload"):
            r.unload()


def _bm25():
    from retrievers.text import Bm25Retriever

    return _retrieve(lambda: Bm25Retriever(data_dir=CFG.paths.data_dir, cache_dir=CFG.paths.cache_dir, dpi=CFG.dpi))


def _bge():
    from retrievers.text import BgeM3Retriever

    # No BM25 fallback: a load failure must surface, not silently degrade.
    return _retrieve(lambda: BgeM3Retriever(data_dir=CFG.paths.data_dir, cache_dir=CFG.paths.cache_dir,
                                            dpi=CFG.dpi, allow_bm25_fallback=False))


def _qwen_emb():
    from retrievers.text import Qwen3EmbeddingRetriever

    return _retrieve(lambda: Qwen3EmbeddingRetriever(data_dir=CFG.paths.data_dir, cache_dir=CFG.paths.cache_dir,
                                                     dpi=CFG.dpi, allow_bm25_fallback=False))


def _colmodern():
    from retrievers.vision import ColModernVbertRetriever

    return _retrieve(lambda: ColModernVbertRetriever(data_dir=CFG.paths.data_dir, cache_dir=CFG.paths.cache_dir, dpi=CFG.dpi))


def _colqwen25():
    from retrievers.vision import ColQwen25Retriever

    return _retrieve(lambda: ColQwen25Retriever(data_dir=CFG.paths.data_dir, cache_dir=CFG.paths.cache_dir, dpi=CFG.dpi))


def _colqwen3():
    from retrievers.vision import ColQwen3Retriever

    return _retrieve(lambda: ColQwen3Retriever(data_dir=CFG.paths.data_dir, cache_dir=CFG.paths.cache_dir, dpi=CFG.dpi))


def _joint():
    from retrievers.joint import union
    from retrievers.text import Bm25Retriever
    from retrievers.vision import ColQwen25Retriever

    text = Bm25Retriever(data_dir=CFG.paths.data_dir, cache_dir=CFG.paths.cache_dir, dpi=CFG.dpi)
    vis = ColQwen25Retriever(data_dir=CFG.paths.data_dir, cache_dir=CFG.paths.cache_dir, dpi=CFG.dpi)
    try:
        merged = union(text.retrieve(Q, PAGE_COUNT, 3), vis.retrieve(Q, PAGE_COUNT, 3), k=5)
        assert isinstance(merged, tuple) and len(merged) >= 1, "empty joint union"
        return f"union={list(merged)}"
    finally:
        vis.unload()


for _name, _fn in [
    ("retriever.bm25", _bm25),
    ("retriever.bge-m3", _bge),
    ("retriever.qwen3-embedding", _qwen_emb),
    ("retriever.colmodernvbert", _colmodern),
    ("retriever.colqwen2.5", _colqwen25),
    ("retriever.colqwen3", _colqwen3),
    ("retriever.joint", _joint),
]:
    check(_name, _fn)


# -- parsers (each in its isolated env) -------------------------------------

def _parser(tool):
    from tools.parser import _cache_file, cached_markdown, warm_parser_cache

    cache_file = _cache_file(PAGE, tool, CFG.dpi)
    if cache_file.exists():
        cache_file.unlink()  # force the parser to actually run this probe
    warm_parser_cache([PAGE], parser_tool=tool, dpi=CFG.dpi)
    text = cached_markdown(PAGE, tool, CFG.dpi)
    assert text is not None, "no markdown written (parser env/backend failed)"
    return f"{len(text)} chars markdown"


for _tool in ("paddleocrvl", "mineru", "unlimited"):
    check(f"parser.{_tool}", lambda t=_tool: _parser(t))


# -- generation tasks (fresh process; via the YAML spec) --------------------

def _tasks_output():
    base = CFG.paths.results_dir / "cache" / "v4" / "kaya-probe" / "full"
    status: list[str] = []
    for task in ("G1_oracle_ladder", "G2_retrieval", "G3_hallucination"):
        rows = [__import__("json").loads(l) for l in (base / task / "results.jsonl").read_text().splitlines() if l.strip()]
        counts = Counter(r["status"] for r in rows)
        status.append(f"{task}={len(rows)}rows{dict(counts)}")
    if not (base / "G4_classifier_pricing" / "classifier.jsonl").exists():
        raise RuntimeError("G4 classifier.jsonl missing")
    status.append("G4=classifier.jsonl ok")
    return " | ".join(status)


run_cmd("tasks.all(G1-G4)", ["ops.generate", "--spec", "ops/specs/kaya_probe.yaml", "--allow-unlabelled"], _tasks_output)


# -- resolution robustness (min vs full; distinct run_tags so they don't collide) --

def _res_run(preset, tag):
    return ["ops.generate", "--task", "G1_oracle_ladder", "--reasoner-spec", "qwen3vl-2b-local",
            "--quantization", "4bit", "--visual-resolution", preset, "--run-tag", tag, "--limit", "1",
            "--allow-unlabelled"]


def _vis_tokens(tag):
    import json

    path = CFG.paths.results_dir / "cache" / "v4" / tag / "full" / "G1_oracle_ladder" / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    v = [r for r in rows if r["representation"] == "V" and r["status"] == "ok"]
    assert v, "no ok V-rung row"
    return int(v[0]["total_visual_tokens"])


run_cmd("resolution.min", _res_run("min", "probe-res-min"), lambda: f"V tokens={_vis_tokens('probe-res-min')}")
run_cmd("resolution.full", _res_run("full", "probe-res-full"), lambda: f"V tokens={_vis_tokens('probe-res-full')}")


def _res_robust():
    lo, hi = _vis_tokens("probe-res-min"), _vis_tokens("probe-res-full")
    assert lo < hi, f"expected min<full vision tokens, got min={lo} full={hi}"
    return f"min={lo} < full={hi} vision tokens (resolution knob scales the input)"


check("resolution.robustness", _res_robust)


# -- table build ------------------------------------------------------------

def _tables_built():
    out = CFG.paths.results_dir / "tables"
    mds = list(out.rglob("all_tables.md"))
    assert mds, "no all_tables.md produced"
    return f"{len(mds)} table report(s)"


run_cmd("build.tables", ["ops.build", "--task", "all"], _tables_built)


# -- summary ----------------------------------------------------------------

passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
total = len(RESULTS)
print("\n" + "=" * 72, flush=True)
print(f"PROBE RESULT: {passed}/{total} checks passed", flush=True)
for name, status, detail in RESULTS:
    print(f"  {status:4s}  {name:28s}  {detail}", flush=True)
print("=" * 72, flush=True)
failed = [n for n, s, _ in RESULTS if s == "FAIL"]
if failed:
    print(f"FAILURES: {failed}", flush=True)
raise SystemExit(1 if failed else 0)
