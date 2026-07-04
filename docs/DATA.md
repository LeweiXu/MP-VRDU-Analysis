# Data Layer

Stage 2 normalises MMLongBench-Doc and the Section-F4 LongDocURL replication
into `schema.Question` objects and renders PDF pages through `data.render`.

## Paths

- Dataset root: `.data/mmlongbench`
- Parquet shards: `.data/mmlongbench/data/*.parquet`
- PDFs: `.data/mmlongbench/documents/*.pdf`
- LongDocURL annotations: `.data/longdocurl/LongDocURL_public.jsonl`
- LongDocURL PDFs: `.data/longdocurl/documents/<doc_no>.pdf`
- Render cache: `results/cache/renders/<pdf-stem>__dpi<N>/page_XXXX.png`

All paths are root-relative locally and on Kaya.

## Question Fields

`data.loader.load_mmlongbench()` reads the staged parquet files and returns
`Question` objects.

- `id`: stable row id, `mmlongbench:000000` style.
- `doc_id`: source PDF id from `doc_id`.
- `question`: source `question`.
- `gold_answer`: source `answer`.
- `answer_format`: source `answer_format`.
- `doc_type`: source `doc_type`.
- `evidence_pages`: source `evidence_pages`, normalised from one-based page
  numbers to zero-based internal indices.
- `evidence_sources`: source `evidence_sources`.
- `hop`: derived from evidence-page count: `none`, `single`, or `multi`.
- `is_unanswerable`: true when `gold_answer` normalises to `Not answerable`.
- `raw_fields`: original parquet row values for audit/debugging, plus
  `source_dataset="mmlongbench"`.

## LongDocURL

`data.loader.load_longdocurl()` reads `LongDocURL_public.jsonl` from staged data.
If that file is not present, it can read the annotation JSONL from a cached
`dengchao/LongDocURL` Hugging Face snapshot. The Hugging Face annotation cache
does not provide local PDFs, so the source PDFs must be staged separately under
`.data/longdocurl/documents/`.

Field mapping:

- `id`: `longdocurl:<question_id>`.
- `doc_id`: `longdocurl:<doc_no>`.
- `question`: source `question`.
- `gold_answer`: source `answer`, kept as text for scalars and JSON for lists.
- `answer_format`: source `answer_format`.
- `doc_type`: source `task_tag` (`Understanding`, `Locating`, or `Reasoning`),
  because LongDocURL does not provide the MMLongBench semantic document-domain
  labels.
- `evidence_pages`: source `page` or `evidence_pages`, treated as already
  zero-based internal page indices.
- `evidence_sources`: source `evidence_sources`.
- `raw_fields`: original JSONL row values for audit/debugging, plus
  `source_dataset="longdocurl"` and `doc_no`.

## Rendering

`data.render.render_question_pages()` resolves the question PDF and renders the
question's gold pages. Native unanswerable questions with no gold pages render
page 0 when no explicit page list is supplied, which gives cheap document
sanity coverage without inventing evidence.

Each rendered `Page` carries the zero-based page index, the PDF path, an
optional cached PNG path, and line-level text spans from PyMuPDF.
