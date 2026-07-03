#!/usr/bin/env python3
"""
dataset_stats.py
----------------
The "what's actually in the data" report for all five MP-VRDU datasets, as a
companion to `profile_datasets.py`. The profiler answers "which result tables can
this dataset fill"; this one just dumps the descriptive statistics you'd want for a
data section: how big the documents are, how many questions/documents there are, and
every categorical label field with its full class breakdown, whether or not it's
relevant to our study.

It reuses the fetch layer from `profile_datasets.py` (same REGISTRY, same loaders,
same heavy-value handling) so there's one place that knows how to pull each repo.
This file only adds the number-crunching and the CSV/Markdown output.

A note on "document length": the five datasets don't share a unit. Contract/filing
datasets (CUAD, DocFinQA) carry a text context, so length there is characters (and
words when the raw string is present). Page-image datasets (MMLongBench-Doc,
SlideVQA) have no text field, so length is pages — read from a real page field when
one exists, or by counting the local PDFs for MMLongBench. Whatever signal a dataset
exposes is reported and labelled; the rest show up as blank/"n/a" rather than a crash.
Lengths are computed per distinct document (deduped by the doc-id field), not per
question, so a 200-page doc with 30 questions counts once.

USAGE
  python scripts/dataset_stats.py
  python scripts/dataset_stats.py --only mmlongbench cuad --max-scan 200
  python scripts/dataset_stats.py --out-md docs/dataset_stats.md \
      --out-csv docs/dataset_stats.csv --out-dist-csv docs/dataset_label_distributions.csv

Outputs (defaults, all under docs/):
  - dataset_stats.md                  human-readable, per dataset + cross-dataset summary
  - dataset_stats.csv                 one row per dataset: scalar summary stats
  - dataset_label_distributions.csv   long form: one row per (dataset, field, class)
"""

import argparse
import csv
import json
import os
import statistics
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone

# profile_datasets.py sits next to this file; running the script puts scripts/ on
# sys.path[0], so a plain import works. Reuse its fetch + parsing layer wholesale.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_datasets import (  # noqa: E402
    REGISTRY,
    STRATEGIES,
    HINTS,
    Heavy,
    describe,
    find_keys,
    class_breakdown,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fields that, when present, tell us a document's size.
TEXT_FIELDS = ["context", "ocr", "text", "content"]
PAGE_FIELDS = ["num_pages", "n_pages", "page_count", "total_pages", "pages"]
# Where to look for local PDFs to count pages (MMLongBench-Doc), relative to ROOT.
PDF_DIRS = ["data/mmlongbench/documents", "data/mmlongbench_subset/documents"]


# ---------------------------------------------------------------------------
# Per-record length signals
# ---------------------------------------------------------------------------
def record_text_len(rec):
    """(chars, words) for a record's text field, or (None, None). Uses Heavy.length
    when the loader swapped a big string for a placeholder (CUAD/DocFinQA context);
    counts words only when the real string is present. Field match is case-insensitive
    (DocFinQA names its context `Context`)."""
    lower = {k.lower(): k for k in rec}
    for f in TEXT_FIELDS:
        if f not in lower:
            continue
        v = rec[lower[f]]
        if isinstance(v, Heavy) and v.kind in ("str", "bytes") and v.length is not None:
            return v.length, None
        if isinstance(v, str):
            return len(v), len(v.split())
    return None, None


def record_pages(rec):
    """A document's page count from a real page-count field, or None. (Evidence-page
    lists are NOT a total, so they're deliberately not used here.) Case-insensitive."""
    lower = {k.lower(): k for k in rec}
    for f in PAGE_FIELDS:
        if f not in lower or isinstance(rec[lower[f]], Heavy):
            continue
        v = rec[lower[f]]
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, (list, tuple)):
            return len(v)
        if isinstance(v, str) and v.strip().isdigit():
            return int(v)
    return None


def pdf_pages_for_docs(doc_ids):
    """Best-effort {doc_id: page_count} by opening local PDFs with PyMuPDF. Only
    considers doc-ids that are already `.pdf` filenames (which scopes this to
    MMLongBench, whose doc_id IS the pdf name, and avoids spurious collisions from
    appending `.pdf` to unrelated ids like SlideVQA deck names). Returns {} if
    PyMuPDF isn't installed or nothing matches — page-count then shows as n/a."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return {}
    dirs = [os.path.join(ROOT, d) for d in PDF_DIRS]
    out = {}
    for did in doc_ids:
        name = str(did)
        if not name.lower().endswith(".pdf"):
            continue
        for d in dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                try:
                    with fitz.open(p) as doc:
                        out[str(did)] = doc.page_count
                except Exception:
                    pass
                break
    return out


def per_doc(records, doc_field, extract):
    """Collapse per-question records to per-document values: for each distinct
    document id, the first non-None `extract(rec)`. Falls back to per-record (each
    record its own document) when there's no doc-id field."""
    seen = {}
    for i, rec in enumerate(records):
        did = str(rec.get(doc_field)) if doc_field else f"__row_{i}"
        if did in seen:
            continue
        val = extract(rec)
        if val is not None:
            seen[did] = val
    return list(seen.values())


def summarise(values):
    """avg/min/max/median/n over a list of numbers, rounded; None if empty."""
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return {
        "n": len(vals),
        "avg": round(statistics.mean(vals), 1),
        "min": min(vals),
        "max": max(vals),
        "median": round(statistics.median(vals), 1),
    }


# ---------------------------------------------------------------------------
# Label-field detection: split into document-labels vs question-labels.
# ---------------------------------------------------------------------------
def label_fields(keys):
    """Return (doc_label_fields, question_label_fields). A field claimed as a
    document label (doc_type/domain/...) is not double-counted as a question label,
    which is what disambiguates MMLongBench's `doc_type` (a document label whose name
    also trips the generic question_type '...type' hint)."""
    doc_fields = find_keys(keys, HINTS["domain"])
    q_fields = []
    for group in ("question_type", "answer_format"):
        for f in find_keys(keys, HINTS[group]):
            if f not in doc_fields and f not in q_fields:
                q_fields.append(f)
    for f in find_keys(keys, ["evidence_source"]):  # per-question modality label
        if f not in doc_fields and f not in q_fields:
            q_fields.append(f)
    return doc_fields, q_fields


def derived_hop(records, keys, doc_field):
    """single/multi/zero-page class breakdown from the evidence-page field, mirroring
    profile_datasets. Returns (field_name, rows) or (None, None)."""
    ep_fields = find_keys(keys, HINTS["evidence_page"])
    if not ep_fields:
        return None, None
    f = ep_fields[0]
    q = Counter()
    docs = {}
    for rec in records:
        v = rec.get(f)
        n = None
        if isinstance(v, (list, tuple)):
            n = len(v)
        elif isinstance(v, str) and v.strip().startswith("["):
            try:
                pv = json.loads(v)
                n = len(pv) if isinstance(pv, list) else None
            except Exception:
                n = None
        if n is None:
            continue
        key = "single (1 page)" if n == 1 else (f"multi ({n} pages)" if n > 1 else "zero (0 pages)")
        q[key] += 1
        if doc_field:
            docs.setdefault(key, set()).add(str(rec.get(doc_field)))
    if not q:
        return None, None
    rows = [(k, nq, (len(docs[k]) if doc_field else None)) for k, nq in q.most_common()]
    return f, rows


def unanswerable_count(records, keys):
    """(field, count) of unanswerable questions, however this dataset signals it:
    an is_impossible/answerable flag, or a literal 'Not answerable' answer."""
    ufields = find_keys(keys, HINTS["unanswerable"])
    for f in ufields:
        c = sum(1 for r in records if str(r.get(f)).lower() in ("true", "1", "yes"))
        if c:
            return f, c
    # MMLongBench signals it through the answer text, not a flag.
    for f in find_keys(keys, ["answer"]):
        c = sum(1 for r in records
                if isinstance(r.get(f), str) and r[f].strip().lower() == "not answerable")
        if c:
            return f"{f}=='Not answerable'", c
    return (ufields[0] if ufields else None), 0


# ---------------------------------------------------------------------------
# Profiling one dataset -> (markdown_section, summary_row, dist_rows)
# ---------------------------------------------------------------------------
def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def stat_cols(prefix, s):
    """Flatten a summarise() dict into CSV columns; blanks when absent."""
    keys = [f"{prefix}_avg", f"{prefix}_min", f"{prefix}_max", f"{prefix}_median", f"{prefix}_n"]
    if not s:
        return {k: "" for k in keys}
    return {f"{prefix}_avg": s["avg"], f"{prefix}_min": s["min"], f"{prefix}_max": s["max"],
            f"{prefix}_median": s["median"], f"{prefix}_n": s["n"]}


def length_table_md(name, s, unit):
    if not s:
        return f"- **{name}:** n/a"
    return (f"- **{name}** ({unit}, per document, n={s['n']}): "
            f"avg {s['avg']} · min {s['min']} · max {s['max']} · median {s['median']}")


def profile(key, entry, max_scan):
    pretty = entry["pretty"]
    md = [f"## {pretty}\n",
          f"- **Role in study:** {entry['role']}",
          f"- **Source:** `{entry.get('repo_id', entry.get('zip_url', ''))}`  "
          f"|  **Strategy:** `{entry['strategy']}`"]
    dist_rows = []
    row = {"dataset": pretty}

    try:
        records, source, ordered_keys, image_fields = STRATEGIES[entry["strategy"]](entry, max_scan)
    except Exception:
        md.append(f"\n> **LOAD FAILED**\n```\n{traceback.format_exc()}\n```\n")
        row["error"] = "load failed"
        return "\n".join(md) + "\n", row, dist_rows
    if not records:
        md.append("\n> **No records parsed.**\n")
        row["error"] = "no records"
        return "\n".join(md) + "\n", row, dist_rows

    keys = ordered_keys or list(records[0].keys())
    doc_field = next((f for f in find_keys(keys, HINTS["doc_id"])
                      if not isinstance(records[0].get(f), Heavy)), None)
    doc_ids = {str(r.get(doc_field)) for r in records} if doc_field else set()
    n_docs = len(doc_ids) if doc_field else len(records)

    md.append(f"- **Loaded from:** `{source}`")
    md.append(f"- **Questions:** {len(records)}  |  "
              f"**Distinct documents:** {n_docs}"
              f"{'' if doc_field else ' (no doc-id field; counted per record)'}"
              f"{f' (doc id = `{doc_field}`)' if doc_field else ''}")
    row.update(records_scanned=len(records), n_documents=n_docs, doc_id_field=doc_field or "")

    # ---- document length (per distinct document) ----
    chars = summarise(per_doc(records, doc_field, lambda r: record_text_len(r)[0]))
    words = summarise(per_doc(records, doc_field, lambda r: record_text_len(r)[1]))
    field_pages = summarise(per_doc(records, doc_field, record_pages))
    pdf_map = pdf_pages_for_docs(doc_ids) if doc_field else {}
    pdf_pages = summarise(list(pdf_map.values()))
    pages = field_pages or pdf_pages
    pages_src = "page field" if field_pages else ("local PDFs (PyMuPDF)" if pdf_pages else "")

    md.append("\n### Document length\n")
    md.append(length_table_md("Text length", chars, "characters"))
    md.append(length_table_md("Word count", words, "words"))
    md.append(length_table_md("Page count", pages, f"pages · via {pages_src}" if pages_src else "pages"))
    if not (chars or words or pages):
        md.append("- _(no length signal exposed by this dataset)_")
    row.update(stat_cols("chars", chars))
    row.update(stat_cols("words", words))
    row.update(stat_cols("pages", pages))
    row["pages_source"] = pages_src

    # ---- label fields (document + question), full class breakdowns ----
    doc_lab, q_lab = label_fields(keys)
    row["doc_label_fields"] = ";".join(doc_lab)
    row["question_label_fields"] = ";".join(q_lab)

    def emit_labels(title, fields, bucket):
        if not fields:
            md.append(f"\n### {title}\n_(none detected)_")
            return
        md.append(f"\n### {title}\n")
        for f in fields:
            rows, tot_q, tot_d = class_breakdown(records, f, doc_field)
            if not rows:
                continue
            md.append(f"\n**`{f}`** — {len(rows)} classes")
            if doc_field:
                md.append("\n| Class | Questions | Documents |\n|---|---:|---:|")
                for c, nq, nd in rows:
                    md.append(f"| {esc(c)} | {nq} | {nd} |")
                    dist_rows.append([pretty, bucket, f, c, nq, nd])
                md.append(f"| **Total** | {tot_q} | {tot_d} |")
            else:
                md.append("\n| Class | Questions |\n|---|---:|")
                for c, nq, _ in rows:
                    md.append(f"| {esc(c)} | {nq} |")
                    dist_rows.append([pretty, bucket, f, c, nq, ""])
                md.append(f"| **Total** | {tot_q} |")

    emit_labels("Document labels", doc_lab, "document")
    emit_labels("Question labels", q_lab, "question")

    # ---- derived hop + unanswerable ----
    hop_field, hop_rows = derived_hop(records, keys, doc_field)
    if hop_rows:
        md.append(f"\n### Derived hop — page count of `{hop_field}`\n")
        if doc_field:
            md.append("| Class | Questions | Documents |\n|---|---:|---:|")
            for c, nq, nd in hop_rows:
                md.append(f"| {c} | {nq} | {nd} |")
                dist_rows.append([pretty, "hop", hop_field, c, nq, nd])
        else:
            md.append("| Class | Questions |\n|---|---:|")
            for c, nq, _ in hop_rows:
                md.append(f"| {c} | {nq} |")
                dist_rows.append([pretty, "hop", hop_field, c, nq, ""])
        row["n_hop_classes"] = len(hop_rows)

    ufield, ucount = unanswerable_count(records, keys)
    md.append(f"\n### Unanswerable\n- **Signal:** `{ufield or 'none'}`  |  "
              f"**Count (in scan):** {ucount}")
    row.update(unanswerable_field=ufield or "", unanswerable_count=ucount)

    # ---- full field inventory ----
    md.append("\n### All fields (first record)\n| Field | Type / sample |\n|---|---|")
    first = records[0]
    for k in keys:
        md.append(f"| `{k}` | {esc(describe(first.get(k)))} |")

    md.append("")
    return "\n".join(md) + "\n", row, dist_rows


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------
SUMMARY_COLS = [
    "dataset", "records_scanned", "n_documents", "doc_id_field",
    "chars_avg", "chars_min", "chars_max", "chars_median", "chars_n",
    "words_avg", "words_min", "words_max", "words_median", "words_n",
    "pages_avg", "pages_min", "pages_max", "pages_median", "pages_n", "pages_source",
    "doc_label_fields", "question_label_fields", "n_hop_classes",
    "unanswerable_field", "unanswerable_count", "error",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of dataset keys: " + ", ".join(REGISTRY))
    ap.add_argument("--max-scan", type=int, default=None,
                    help="cap records scanned per dataset. Default: no cap — a FULL census "
                         "(every question, every document), which is the point of this report. "
                         "Pass e.g. 400 for a quick sample. Note: a full DocFinQA scan streams "
                         "its entire ~1.5MB/row context split, so it is slow and download-heavy.")
    ap.add_argument("--out-md", default="docs/dataset_stats.md")
    ap.add_argument("--out-csv", default="docs/dataset_stats.csv")
    ap.add_argument("--out-dist-csv", default="docs/dataset_label_distributions.csv")
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit("pip install -U datasets huggingface_hub pyarrow  (PyMuPDF for page counts)")

    # No cap by default: scan the whole dataset. (The loaders stop at max_recs; a huge
    # sentinel means "everything".)
    FULL = 10 ** 9
    max_scan = args.max_scan if args.max_scan is not None else FULL
    keys = args.only or list(REGISTRY)
    parts = ["# Dataset Statistics — MP-VRDU\n",
             f"_Generated: {datetime.now(timezone.utc).isoformat()}_\n",
             "Descriptive statistics for all five datasets: document-length distributions, "
             "question/document counts, and every categorical label field with its full class "
             "breakdown (relevant to the study or not). Companion to `dataset_profile.md`, which "
             "instead judges table-readiness. Lengths are per distinct document; the unit differs "
             "by dataset (characters/words for text corpora, pages for image corpora) and is "
             "labelled inline.\n",
             ("**Coverage:** full census — every question and document per dataset.\n"
              if args.max_scan is None
              else f"**Coverage:** sampled — first {args.max_scan} records per dataset "
                   f"(not the full dataset).\n")]
    summary, dist = [], []

    for k in keys:
        if k not in REGISTRY:
            parts.append(f"## {k}\n> not in registry\n")
            continue
        print(f"[stats] {k} …", file=sys.stderr)
        entry = dict(REGISTRY[k])
        if args.max_scan is None:
            # Don't let a dataset's own sampling cap (DocFinQA's stream_cap) shrink a
            # full census; --max-scan, when given, still governs.
            entry.pop("stream_cap", None)
        section, row, drows = profile(k, entry, max_scan)
        parts.append(section)
        summary.append(row)
        dist.extend(drows)

    # cross-dataset summary table in the md
    parts.append("## Cross-dataset summary\n")
    parts.append("| Dataset | Qs | Docs | Text len (avg chars) | Pages (avg) | "
                 "Doc labels | Question labels | Unanswerable |")
    parts.append("|---|---:|---:|---:|---:|---|---|---:|")
    for r in summary:
        parts.append(
            f"| {r.get('dataset')} | {r.get('records_scanned', '')} | {r.get('n_documents', '')} | "
            f"{r.get('chars_avg', '') or '—'} | {r.get('pages_avg', '') or '—'} | "
            f"{esc(r.get('doc_label_fields', '') or '—')} | "
            f"{esc(r.get('question_label_fields', '') or '—')} | {r.get('unanswerable_count', '')} |")
    parts.append("")

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(parts))

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for r in summary:
            w.writerow(r)

    with open(args.out_dist_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "label_bucket", "field", "class", "n_questions", "n_documents"])
        w.writerows(dist)

    print(f"[done] wrote {args.out_md}, {args.out_csv}, {args.out_dist_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
