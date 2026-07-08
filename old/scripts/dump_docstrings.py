"""Append a concise Python module map to docs/REPO_STRUCTURE.md.

Walks the repo (skipping virtualenvs, caches, data, results, and other gitignored
heavy dirs), reads each `.py` file's top-of-file `\"\"\"...\"\"\"` module docstring via
`ast`, and writes a filtered per-directory listing into the auto-generated section
of docs/REPO_STRUCTURE.md. Package markers and tests are omitted, and each retained
module is reduced to the first useful purpose sentence so the section stays usable
as a repo-rework map rather than a verbatim documentation dump.

Usage: python scripts/dump_docstrings.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "REPO_STRUCTURE.md"
MARKER = "## Per-file module docstrings (auto-generated)"

SUMMARY_OVERRIDES = {
    Path("config.py"): "Global run knobs and path layout; v4 rework should update representation order, parser/resolution/model sweep settings, answerable filtering, and telemetry defaults here.",
    Path("schema.py"): "Cross-pipeline dataclasses; extend cautiously for v4 manual labels and uniform telemetry without breaking cache/readers.",
    Path("cli/build.py"): "Local table-build entry point; should keep reading generated/judged artifacts while table routing changes for v4.",
    Path("cli/generate.py"): "GPU generation entry point for YAML specs; v4 should keep this YAML-first and collect all cheap telemetry every run.",
    Path("cli/judge.py"): "Local judging entry point; v4 answerable/unanswerable split and abstention scoring must stay consistent here.",
    Path("covariates/classifier.py"): "Routing classifier; v4 should predict manual `bin_label` rather than native `doc_type` or old Option-A bins.",
    Path("covariates/retriever.py"): "Page retrievers; v4 expands these into cost rungs for BM25/BGE/Qwen embeddings and ColModern/ColQwen variants.",
    Path("data/binning.py"): "Currently maps native `doc_type` to old bins; v4 should replace this with manual annotation lookup for `bin_label`.",
    Path("data/loader.py"): "Loads MMLongBench/replication rows into `Question`; v4 answerable filtering and manual document labels will join here or immediately downstream.",
    Path("data/render.py"): "PDF page rendering and embedded-text extraction substrate used by T, V, retrievers, parsers, and inspection tools.",
    Path("experiments/G1_sufficiency.py"): "Legacy headline oracle ladder task; v4 keeps oracle ladder but reframes it as cost-ordered and answerable-only.",
    Path("experiments/G2_family.py"): "Legacy model-family replication task; v4 treats model-family replication as a YAML run over the same core grid.",
    Path("experiments/G3_dataset.py"): "Legacy dataset replication task; v4 keeps replication but should avoid a bespoke task when a YAML variant is enough.",
    Path("experiments/G5_retrieval.py"): "Legacy retrieval task; v4 broadens retrieval into matched/cross, top-k, joint union, and page-F1 telemetry.",
    Path("experiments/G6_classifier.py"): "Legacy classifier side task; v4 routing should price classifier latency while targeting manual modality bins.",
    Path("experiments/artifacts.py"): "Artifact-driven judge/build helpers; important for v4 because generation settings should not have to be restated later.",
    Path("experiments/base.py"): "GenerationTask contract and shared cell factories; v4 should keep task count low and express variants through fields/specs.",
    Path("experiments/corpus.py"): "Question-set resolver; v4 needs clean answerable/unanswerable partitions and replication subsets here.",
    Path("experiments/driver.py"): "Generate/judge engine; v4 uniform telemetry and parser/retriever side artifacts should be emitted through this path.",
    Path("experiments/paths.py"): "Cache/table path layout; v4 cache keys must include changed parser, resolution, model, quantization, and prompt fields.",
    Path("experiments/registry.py"): "Legacy fixed-task registry; v4 should shrink reliance on new G* modules in favor of YAML variants.",
    Path("experiments/side_artifacts.py"): "Shared side-artifact writers; v4 retrieval benchmarks and classifier logs should use this common telemetry path.",
    Path("experiments/yaml_spec.py"): "YAML-to-task loader; central place for v4 parser, resolution, quantization, top-k, prompt, and model-size sweeps.",
    Path("metrics/abstention.py"): "Abstention detector for the v4 hallucination study over unanswerable questions.",
    Path("metrics/accuracy.py"): "Document-level accuracy summaries; v4 RQ1/RQ2 should run answerable-only through this metric.",
    Path("metrics/cost.py"): "Latency/token cost aggregation; v4 cost-frontier sweeps depend on this being complete and uniform.",
    Path("metrics/frontier.py"): "Sufficiency frontier rule; v4 still uses it, but over cost-ordered non-cumulative representations.",
    Path("metrics/retrieval.py"): "Page precision/recall/F1; v4 retrieval benchmark should report these per manual bin and method rung.",
    Path("models/__init__.py"): "Model spec parser/registry; v4 model family, size, and quantization variants should resolve through this boundary.",
    Path("models/internvl.py"): "InternVL backend for family replication; v4 should treat it as a model-family YAML variant.",
    Path("models/local_vlm.py"): "Qwen3-VL local backend; v4 size, quantization, resolution, and prompt variants must preserve the common `Prediction` contract.",
    Path("models/payload.py"): "Backend-neutral prompt/image container; v4 parser text and image-resolution changes should still enter through this boundary.",
    Path("pipeline/conditioner.py"): "Page-selection policies; v4 uses oracle pages for RQ1 and retrieved pages for RQ2/RQ3 hallucination.",
    Path("pipeline/judge.py"): "Scoring interface and API judges; v4 must preserve answerable accuracy and unanswerable abstention scoring semantics.",
    Path("pipeline/orchestrator.py"): "Cell runner and prediction/result caches; v4 cache keys and result rows must capture parser, resolution, prompt, and telemetry fields.",
    Path("pipeline/reasoner.py"): "Backend-agnostic reasoner ABC; v4 model swaps should stay behind this interface.",
    Path("pipeline/representation.py"): "T/TL/TLV/V composers; v4 makes the ladder cost-ordered, drops bbox JSON, and makes TL/TLV use parser text instead of embedded text.",
    Path("reporting/build.py"): "Routes judged rows and side artifacts into tables; v4 table set and source-task mapping will change here.",
    Path("reporting/tables/T1_headline.py"): "Headline RQ1 frontier table; v4 should use manual bins, answerable-only rows, and cost-ordered T/TL/TLV/V.",
    Path("reporting/tables/T2_analytical.py"): "Analytical slice table; v4 may demote or repurpose it after the RQ/table reshuffle.",
    Path("reporting/tables/T3_family.py"): "Family replication frontier table; v4 likely treats it as appendix/replication output from YAML variants.",
    Path("reporting/tables/T4_dataset.py"): "Dataset replication frontier table; v4 likely treats it as appendix/replication output from YAML variants.",
    Path("reporting/tables/T5_composition.py"): "Evidence-composition mediation table; v4 may be secondary to parser/resolution/retrieval core tables.",
    Path("reporting/tables/T6_retrieval.py"): "Retrieval table; v4 should cover matched/cross, top-k, joint union, and retrieval accuracy per manual bin.",
    Path("reporting/tables/T7_routing.py"): "Routing table; v4 policies should route over manual modality bins and include classifier latency.",
    Path("reporting/tables/T8_scale.py"): "Scale table; v4 expands this into model-size/quantization cost-frontier sweeps.",
    Path("reporting/tables/_common.py"): "Shared table helpers; v4 bin labels, representation order, telemetry columns, and answerable filters should be centralized here.",
    Path("reporting/tables/_markdown.py"): "Markdown report rendering for built CSVs; update only after v4 table schemas settle.",
    Path("scripts/annotate_docs.py"): "Manual document-label utility; v4 makes its `bin_label` output the source of truth for document modality bins.",
    Path("scripts/dump_docstrings.py"): "Regenerates this concise module map while skipping package markers and tests.",
    Path("scripts/inspect_results.py"): "Debug viewer for cached cells; useful after v4 changes to inspect pages, representations, predictions, and judged rows without mutating caches.",
    Path("scripts/prestage.py"): "Stages datasets, reasoners, retrievers, and parser caches; v4 parser stacks should be isolated and pre-warmed here.",
    Path("scripts/split_docs_by_type.py"): "Browsing helper for manual labeling; v4 annotation should not depend on old doc_type-derived bins.",
    Path("tools/layout.py"): "Current Marker/bbox layout path; v4 should remove bbox JSON and make this the parser-text provider for TL/TLV.",
    Path("tools/text.py"): "Text channel helpers; v4 T should be cheap PyMuPDF embedded text, not parser/OCR text.",
    Path("tools/visual.py"): "Image channel helpers and token estimates; v4 resolution sweeps should flow through this module.",
}

# Directory names pruned anywhere in the tree (heavy / vendored / generated).
SKIP_DIRS = {
    "envs", ".cache", ".data", "results", "logs", "__pycache__", ".git",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", ".agents", ".codex",
    ".claude", ".vscode", "inspect", "temp",
}


def iter_py_files():
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.name == "__init__.py" or rel.parts[0] == "tests":
            continue
        yield rel, path


def module_docstring(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return f"(could not parse: {exc})"
    return ast.get_docstring(tree, clean=True)


def group_key(rel: Path) -> str:
    return str(rel.parent) if str(rel.parent) != "." else "(root)"


def _purpose_text(doc: str) -> str:
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != "Purpose:":
            continue
        block: list[str] = []
        for raw in lines[i + 1:]:
            stripped = raw.strip()
            if re.fullmatch(r"[A-Z][A-Za-z0-9 /_-]*:", stripped):
                break
            if stripped:
                block.append(stripped)
        if block:
            return " ".join(block)
    return " ".join(line.strip() for line in lines if line.strip())


def short_summary(rel: Path, doc: str | None) -> str:
    if rel in SUMMARY_OVERRIDES:
        return SUMMARY_OVERRIDES[rel]
    if not doc:
        return "No module docstring."
    text = re.sub(r"\s+", " ", _purpose_text(doc)).strip()
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z`])", text)
    if match:
        text = text[: match.start()]
    words = text.split()
    if len(words) > 32:
        text = " ".join(words[:32]).rstrip(".,;:") + "."
    return text


def build_section() -> str:
    groups: dict[str, list[tuple[Path, str | None]]] = {}
    for rel, path in iter_py_files():
        groups.setdefault(group_key(rel), []).append((rel, module_docstring(path)))

    lines = [
        MARKER,
        "",
        "Generated by `scripts/dump_docstrings.py`, filtered for repo-rework use.",
        "Package markers and tests are omitted; each remaining entry is shortened",
        "to the first useful purpose sentence from the module docstring.",
        "",
    ]
    # (root) first, then the rest alphabetically.
    for key in sorted(groups, key=lambda k: (k != "(root)", k)):
        lines.append(f"### {key if key == '(root)' else key + '/'}")
        lines.append("")
        for rel, doc in groups[key]:
            lines.append(f"- **`{rel.as_posix()}`** — {short_summary(rel, doc)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    base = DOC.read_text(encoding="utf-8")
    head = base.split(MARKER, 1)[0].rstrip() + "\n\n"
    DOC.write_text(head + build_section(), encoding="utf-8")
    n = sum(1 for _ in iter_py_files())
    print(f"Wrote {n} module docstrings into {DOC.relative_to(REPO)}")


if __name__ == "__main__":
    main()
