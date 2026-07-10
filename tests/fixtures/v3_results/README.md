# v3 result fixtures (structural only)

These jsonl files are a snapshot of **v3** pipeline output, captured in Phase 0 of
the pivot-v4 do-over. They exist to test v4's **plumbing**, not its science.

**v3-shaped. Values are NOT comparable to v4.** v4 has a different ladder (parser
markdown, no bbox), different binning (manual annotation, not `doc_type`), no
input-token cap, and a reshaped task set. So these rows are the wrong shape in a
few fields and the wrong numbers everywhere. Do not assert any accuracy, frontier,
token count, or label value against them.

What they ARE good for:

- jsonl reader / parser shape tests (real rows, real field names)
- `--failed-only` selection logic (add synthetic `status` rows on top)
- build-step grouping / cardinality
- side-artifact readers (retrieval, classifier)

Reference commit for this snapshot is recorded in `docs/DECISIONS.md`.

## What's here

| Run dir | Tasks present | Notes |
|---|---|---|
| `bf16-lowres/` | G1_sufficiency, G3_dataset, G5_retrieval, G6_classifier | fullest set; has `results.jsonl` (judged) |
| `yaml-g1-g2-g5-rerun/` | G1_sufficiency, G2_family | covers the G2 shape |

Note the v3 task names (`G1_sufficiency`, `G2_family`, `G3_dataset`,
`G5_retrieval`, `G6_classifier`) are the **old** naming. The current tasks are
`G1_oracle_ladder` / `G2_retrieval` / `G3_hallucination` (the modality-bin
classifier is now a G3 side artifact, not its own task). The fixtures keep their
original v3 names on purpose (they are a v3 snapshot).

Per-file kinds:
- `predictions.jsonl` — per-cell prediction record (no judge)
- `results.jsonl` — judged rows (has `score`, `correct`, `judge_spec`)
- `generate_results.jsonl` — generation-pass summary rows
- `retrieval.jsonl` — retrieval side-artifact (page P/R/F1 per method×k)
- `classifier.jsonl` — classifier side-artifact
