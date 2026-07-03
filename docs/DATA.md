# Data Layer

Stage 2 normalises MMLongBench-Doc into `schema.Question` objects and renders
PDF pages through `data.render`.

## Paths

- Dataset root: `.data/mmlongbench`
- Parquet shards: `.data/mmlongbench/data/*.parquet`
- PDFs: `.data/mmlongbench/documents/*.pdf`
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
- `raw_fields`: original parquet row values for audit/debugging.

## Rendering

`data.render.render_question_pages()` resolves the question PDF and renders the
question's gold pages. Native unanswerable questions with no gold pages render
page 0 when no explicit page list is supplied, which gives cheap document
sanity coverage without inventing evidence.

Each rendered `Page` carries the zero-based page index, the PDF path, an
optional cached PNG path, and line-level text spans from PyMuPDF.
