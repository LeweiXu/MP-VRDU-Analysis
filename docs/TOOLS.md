# Document Tooling

This page records the non-reasoner tools used by the v3 pipeline. These tools
feed the frozen representation interfaces; model backends consume only the
resulting `Payload` / `ModelInput` objects.

## Primary Ladder Tools

| Tool | Code path | Input | Output | Serves |
|---|---|---|---|---|
| Marker text | `tools.layout.marker_text(pages)` | Rendered `Page` objects with `pdf_path` / page index | Per-page strings | `T`, `T+L`, `T+L+V`; Tables 1-5, 7-8 |
| Marker bbox JSON | `tools.layout.marker_bbox_json(pages)` | Rendered `Page` objects | Per-page serialized JSON with `source`, `doc_id`, `page_index`, and bbox-bearing `blocks` | `T+L`, `T+L+V`; Tables 1-5, 7-8 |
| Page image | `tools.visual.full_page(pages)` | Rendered `Page` objects with `image_path` | `VisualArtifact` with `ImagePart`, dimensions, provenance, and token estimate | `T+L+V`, `V`; Tables 1-8 |
| Resolution variant | `tools.visual.resolution(pages, scale)` | Rendered page images | Rescaled page images plus updated token estimate | Tool smoke and later cost/resolution sensitivity |

Marker is the primary parser for v3. The Python package is `marker-pdf==1.10.2`;
its code is GPL-3.0-or-later and its model weights use Datalab's modified AI
Pubs Open Rail-M license. This project uses Marker without LLM mode for the
main ladder.

## Appendix / Fallback Tools

| Tool | Code path | Input | Output | Purpose |
|---|---|---|---|---|
| PyMuPDF embedded text | `tools.text.embedded(pages)` | Rendered/extracted `Page` objects | Per-page embedded PDF text | Appendix parser swap and local fallback |
| PaddleOCR PP-OCRv5 | `tools.text.ocr(pages)` | Rendered page images | Per-page OCR text | Scanned/born-digital analysis and OCR fallback |
| Docling | `tools.layout.docling_available()` / cache warm in `kaya/prestage.py` | PDF | Parser-swap dependency check | Appendix parser swap |
| Page-level crop fallback | `tools.visual.region_crop(pages, regions=None)` | Rendered page images | Full-page `VisualArtifact` with fallback provenance | MMLongBench has no in-page evidence boxes |

The PyMuPDF fallback in `marker_bbox_json()` is intentionally not the primary
paper path. It exists so local unit tests and parser-swap probes remain runnable
before Marker is installed. Kaya `prestage --smoke` calls Marker with
`allow_fallback=False`, so the smoke barrier fails if the real Marker path is not
available.

## Prestaging

`kaya/prestage.py --smoke` stages the minimum non-model assets needed by the MVP:

- Qwen3-VL-2B reasoner weights.
- The configured BGE text retriever and one configured ColQwen vision retriever.
- MMLongBench-Doc source files.
- Marker, PaddleOCR, and configured appendix tool caches.

After staging, smoke mode runs one tiny call through PyMuPDF embedded text,
PaddleOCR, Marker text, Marker bbox JSON, `full_page`, `resolution`, and
`region_crop`, all on the frozen smoke corpus. Full prestage keeps the broader
configured inventory for later stages.
