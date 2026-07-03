#!/usr/bin/env python3
"""
profile_datasets.py
-------------------
One report for all five MP-VRDU datasets. For each one it loads actual question
records (not docs) and reports which result tables the dataset can populate:
question-type, multi-hop, domain, evidence-page localisation, unanswerable
(hallucination), vision, and text/layout.

Each dataset needs a slightly different fetch because the repos are laid out
differently. Rather than rely on `load_dataset` (which trips on loader scripts and
PDF-only default configs), every dataset declares a `strategy` and we load it the
way that actually works:

  - parquet_hf   : QA fields live inline in HF parquet shards. Read the light
                   columns, skip heavy image/byte columns.
                   -> MMLongBench-Doc, SlideVQA
  - annotation_hf: QA lives in a JSON/JSONL annotation file next to a PDF/PNG
                   tarball. Download just that file and parse it.
                   -> LongDocURL
  - squad_zip    : the HF repo is a loader SCRIPT only (datasets 5.x won't run
                   scripts). Grab the SQuAD-format zip the script would have
                   pulled and flatten it ourselves.
                   -> CUAD
  - hf_stream    : huge per-row context, so stream and cap; strip heavy fields
                   as we go to bound memory.
                   -> DocFinQA

Auth uses your `huggingface-cli login` token automatically.

USAGE
  python scripts/profile_datasets.py
  python scripts/profile_datasets.py --only slidevqa cuad
  python scripts/profile_datasets.py --max-scan 600 --out docs/dataset_profile.md

Outputs: docs/dataset_profile.md
"""

import argparse
import ast
import io
import json
import os
import sys
import traceback
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from itertools import islice

# ---------------------------------------------------------------------------
# Registry: one entry per dataset, declaring how to fetch its QA records.
# ---------------------------------------------------------------------------
REGISTRY = {
    "mmlongbench": {
        "pretty": "MMLongBench-Doc",
        "role": "general; primary for question-type / domain study",
        "strategy": "parquet_hf",
        "repo_id": "yubo2333/MMLongBench-Doc",
        "repo_type": "dataset",
        "parquet_split": "train",   # only split; benchmark is eval-only
        "expected": "gold evidence pages + evidence-source labels; unanswerable is "
                    "signalled by the ANSWER ('Not answerable'), not by empty evidence.",
    },
    "longdocurl": {
        "pretty": "LongDocURL",
        "role": "general; second / robustness",
        "strategy": "annotation_hf",
        "repo_id": "dengchao/LongDocURL",
        "repo_type": "dataset",
        "annotation_name_hints": ["longdocurl", "public", "qa", "anno", "data", "test", "val", "dev"],
        "annotation_ext": [".jsonl", ".json"],
        "record_container_keys": ["data", "questions", "examples", "annotations", "records"],
        "expected": "understanding/reasoning/locating task tags; evidence-page localisation.",
    },
    "cuad": {
        "pretty": "CUAD",
        "role": "text-heavy (contracts)",
        "strategy": "squad_zip",
        # repo is script-only; the script pulls this zip. We fetch it directly.
        "repo_id": "theatticusproject/cuad-qa",
        "repo_type": "dataset",
        "zip_url": "https://github.com/TheAtticusProject/cuad/raw/main/data.zip",
        "zip_member": "CUADv1.json",     # SQuAD-format full set (510 contracts)
        "squad_format": True,
        "expected": "SQuAD-style char-offset spans; 41 clause categories (id suffix); "
                    "is_impossible flag for clauses absent from a contract.",
    },
    "docfinqa": {
        "pretty": "DocFinQA",
        "role": "in-between (financial)",
        "strategy": "hf_stream",
        "repo_id": "kensho/DocFinQA",
        "repo_type": "dataset",
        "split": "test",
        # Context is ~1.5 MB/row, so cap the stream regardless of --max-scan.
        "stream_cap": 120,
        "expected": "golden context per question; numeric/program answers; long filings.",
    },
    "slidevqa": {
        "pretty": "SlideVQA",
        "role": "visual-heavy (slides)",
        "strategy": "parquet_hf",
        "repo_id": "NTT-hil-insight/SlideVQA",
        "repo_type": "dataset",
        "parquet_split": "test",
        "expected": "native evidence_pages list; arithmetic_expression marks numerical "
                    "questions; per-slide images in page_1..page_20.",
    },
}

# Field-name hints used to detect which study-relevant fields a record carries.
HINTS = {
    "question":      ["question", "query", "q_text"],
    "answer":        ["answer", "label", "target"],
    "question_type": ["question_type", "qtype", "reasoning_type", "task_tag", "task", "type", "category"],
    "hop":           ["hop", "multi_hop", "single_hop", "num_hops", "reasoning"],
    "domain":        ["domain", "doc_type", "document_type", "topic", "source_type"],
    "evidence_page": ["evidence_page", "evidence_pages", "answer_page", "page_id", "page_ids",
                      "gold_page", "evidence_index", "start_end_idx"],
    "evidence_other":["evidence", "supporting", "context", "answer_start", "span", "bbox",
                      "evidence_sources", "rationale", "arithmetic"],
    "unanswerable":  ["unanswerable", "answerable", "no_answer", "is_impossible"],
    "answer_format": ["answer_type", "answer_format", "answer_kind"],
    "doc_id":        ["doc_id", "docid", "document_id", "doc_no", "deck", "pdf", "file", "title"],
    "images":        ["image", "img", "page_image", "screenshot", "page_"],
    "pages":         ["pages", "num_pages", "n_pages", "page_count", "total_pages",
                      "evidence_pages", "start_end_idx"],
    "words":         ["context", "words", "n_words", "length"],
    "text":          ["context", "ocr", "text", "content"],
}


# ---------------------------------------------------------------------------
# Heavy-value placeholder. We never keep image bytes / multi-MB contexts in
# memory; loaders swap them for one of these so the field still shows up in the
# report (with its size) without bloating RAM or the JSON samples.
# ---------------------------------------------------------------------------
class Heavy:
    __slots__ = ("kind", "length")

    def __init__(self, kind, length=None):
        self.kind = kind
        self.length = length

    def __repr__(self):
        return "<heavy>"


def hf_list_files(repo_id, repo_type):
    from huggingface_hub import list_repo_files
    return list_repo_files(repo_id=repo_id, repo_type=repo_type)


# ---------------------------------------------------------------------------
# Loaders. Each returns: (records, source_str, ordered_keys, image_fields)
#   records      : list[dict] of QA records (heavy values are Heavy(...))
#   source_str   : human-readable note on where the records came from
#   ordered_keys : preferred column order for the field table (or None)
#   image_fields : list of column names that hold images (for the vision verdict)
# ---------------------------------------------------------------------------

def _parquet_field_is_image(name, arrow_type):
    """True if a parquet column holds image data (struct<bytes,path>, binary, or
    an obviously image-y name like page_1)."""
    import pyarrow as pa
    nl = name.lower()
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return True
    if pa.types.is_struct(arrow_type):
        child_names = {arrow_type.field(i).name.lower() for i in range(arrow_type.num_fields)}
        if "bytes" in child_names:
            return True
    if nl.startswith("page_") or any(h in nl for h in ["image", "img", "screenshot", "pixel"]):
        return True
    return False


def load_parquet_hf(entry, max_recs):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo_id, repo_type = entry["repo_id"], entry["repo_type"]
    split = entry.get("parquet_split")
    files = hf_list_files(repo_id, repo_type)
    shards = sorted(f for f in files
                    if f.endswith(".parquet") and (split is None or f"{split}-" in os.path.basename(f)))
    if not shards:
        raise RuntimeError(f"no parquet shards for split={split!r}")

    records, ordered_keys, image_fields, heavy_cols = [], None, [], None
    for shard in shards:
        path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=shard)
        pf = pq.ParquetFile(path)
        sch = pf.schema_arrow
        if ordered_keys is None:
            ordered_keys = list(sch.names)
            image_fields = [n for n in sch.names if _parquet_field_is_image(n, sch.field(n).type)]
            heavy_cols = set(image_fields)
        light = [n for n in ordered_keys if n not in heavy_cols]
        for batch in pf.iter_batches(batch_size=64, columns=light):
            for row in batch.to_pylist():
                for hc in heavy_cols:
                    row[hc] = Heavy("image", None)
                records.append(row)
                if len(records) >= max_recs:
                    src = f"{repo_id} :: {split} parquet ({len(shards)} shards)"
                    return records, src, ordered_keys, image_fields
    src = f"{repo_id} :: {split} parquet ({len(shards)} shards)"
    return records, src, ordered_keys, image_fields


def load_annotation_hf(entry, max_recs):
    from huggingface_hub import hf_hub_download
    repo_id, repo_type = entry["repo_id"], entry["repo_type"]
    files = hf_list_files(repo_id, repo_type)
    exts = tuple(entry.get("annotation_ext", [".jsonl", ".json"]))
    hints = entry.get("annotation_name_hints", [])

    def score(f):
        fl = f.lower()
        s = sum(2 for h in hints if h in fl)
        if "test" in fl: s += 3
        if "val" in fl or "dev" in fl: s += 2
        if "train" in fl: s += 1
        if any(x in fl for x in ["readme", "license", "config", "gitattributes"]): s -= 5
        return -s

    cands = sorted((f for f in files if f.lower().endswith(exts)), key=score)
    if not cands:
        raise RuntimeError("no JSON/JSONL annotation files in repo")

    errors = []
    for fname in cands[:6]:
        try:
            path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=fname)
            obj = _read_json_or_jsonl(path)
            recs = _container_to_records(obj, entry.get("record_container_keys", []), max_recs)
            if recs:
                return recs, f"{repo_id} :: {fname}", None, []
        except Exception as e:
            errors.append(f"{fname}: {type(e).__name__}")
    raise RuntimeError("no candidate yielded records (" + "; ".join(errors) + ")")


def load_squad_zip(entry, max_recs):
    url, member = entry["zip_url"], entry.get("zip_member")
    cache = os.path.join(os.path.expanduser("~/.cache/mpvrdu_profile"),
                         os.path.basename(url))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if not (os.path.exists(cache) and os.path.getsize(cache) > 0):
        urllib.request.urlretrieve(url, cache)
    with zipfile.ZipFile(cache) as z:
        names = z.namelist()
        pick = member if (member and member in names) else next(
            (n for n in names if n.lower().endswith(".json")), None)
        if pick is None:
            raise RuntimeError(f"no .json in zip; members={names}")
        obj = json.loads(z.read(pick))
    recs = flatten_squad(obj, max_recs)
    return recs, f"{url} :: {pick}", None, []


def load_hf_stream(entry, max_recs):
    from datasets import load_dataset
    cap = min(max_recs, entry.get("stream_cap", max_recs))
    ds = load_dataset(entry["repo_id"], split=entry.get("split", "test"), streaming=True)
    records = []
    for row in islice(ds, cap):
        rec = {}
        for k, v in dict(row).items():
            if isinstance(v, (bytes, bytearray)):
                rec[k] = Heavy("bytes", len(v))
            elif isinstance(v, str) and len(v) > 5000:
                rec[k] = Heavy("str", len(v))
            else:
                rec[k] = v
        records.append(rec)
    return records, f"{entry['repo_id']} :: {entry.get('split','test')} (streamed, capped {cap})", None, []


STRATEGIES = {
    "parquet_hf": load_parquet_hf,
    "annotation_hf": load_annotation_hf,
    "squad_zip": load_squad_zip,
    "hf_stream": load_hf_stream,
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _read_json_or_jsonl(path):
    if path.lower().endswith(".jsonl"):
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with open(path) as f:
        return json.load(f)


def _container_to_records(obj, container_keys, max_recs):
    if isinstance(obj, list):
        return obj[:max_recs]
    if isinstance(obj, dict):
        for k in container_keys:
            if k in obj and isinstance(obj[k], list):
                return obj[k][:max_recs]
        vals = list(obj.values())
        if vals and isinstance(vals[0], dict):
            return vals[:max_recs]
    return []


def flatten_squad(obj, max_recs):
    """Turn SQuAD-format CUAD into flat QA records."""
    out = []
    data = obj.get("data", obj if isinstance(obj, list) else [])
    for art in data:
        title = art.get("title")
        for para in art.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                qid = qa.get("id", "")
                out.append({
                    "title": title,
                    "question": qa.get("question"),
                    "id": qid,
                    "is_impossible": qa.get("is_impossible", False),
                    "answer_start": [a.get("answer_start") for a in qa.get("answers", [])],
                    "answer_text": [a.get("text") for a in qa.get("answers", [])],
                    "clause_category": (qid.split("__")[-1]
                                        if isinstance(qid, str) and "__" in qid else None),
                    "context": Heavy("str", len(context)),
                })
                if len(out) >= max_recs:
                    return out
    return out


def find_keys(keys, hint_list):
    res, seen = [], set()
    for k in keys:
        kl = str(k).lower()
        if any(h in kl for h in hint_list) and k not in seen:
            seen.add(k); res.append(k)
    return res


def value_classes(v):
    """Split a field value into its set of class labels. Real lists become their
    elements; stringified lists like "['Chart','Table']" (MMLongBench stores
    evidence_sources this way) are parsed into elements too; an empty list maps
    to '(none)'. Everything else is a single class."""
    if isinstance(v, str) and v.strip().startswith("[") and v.strip().endswith("]"):
        try:
            v = ast.literal_eval(v)
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return {str(x) for x in v} or {"(none)"}
    return {str(v)}


def class_breakdown(records, field, doc_field):
    """For every class/value of `field`, count distinct questions and distinct
    documents. Returns (rows, total_questions, total_docs) where rows is
    [(value, n_questions, n_docs), ...] sorted by question count, with EVERY
    class included (not capped). List-valued fields (e.g. evidence_sources) count
    a question once per distinct class it mentions. n_docs is None when the
    dataset has no document-id field."""
    q = Counter()
    docs = {}
    total_q = 0
    all_docs = set()
    for rec in records:
        if field not in rec or isinstance(rec[field], Heavy):
            continue
        classes = value_classes(rec[field])
        if not classes:
            continue
        total_q += 1
        did = str(rec.get(doc_field)) if doc_field else None
        if did is not None:
            all_docs.add(did)
        for c in classes:
            q[c] += 1
            if did is not None:
                docs.setdefault(c, set()).add(did)
    rows = [(c, n, (len(docs[c]) if doc_field else None)) for c, n in q.most_common()]
    return rows, total_q, (len(all_docs) if doc_field else None)


def describe(val):
    if isinstance(val, Heavy):
        ln = f", len={val.length}" if val.length is not None else ""
        return f"(heavy) `{val.kind}`{ln}"
    if isinstance(val, (bytes, bytearray)):
        return f"(heavy) `bytes` (len={len(val)})"
    if isinstance(val, str):
        s = val if len(val) <= 160 else val[:160] + "…"
        return f"`str` (len={len(val)}): {json.dumps(s)}"
    if isinstance(val, (list, tuple)):
        head = list(val)[:3]
        try: hs = json.dumps(head, default=str)
        except Exception: hs = str(head)
        if len(hs) > 200: hs = hs[:200] + "…"
        return f"`{type(val).__name__}` (len={len(val)}): {hs}"
    if isinstance(val, dict):
        return f"`dict` keys={list(val.keys())[:12]}"
    return f"`{type(val).__name__}`: {json.dumps(val, default=str)[:160]}"


def is_heavy(k, v):
    if isinstance(v, Heavy) or isinstance(v, (bytes, bytearray)):
        return True
    if isinstance(v, str) and len(v) > 2000:
        return True
    return False


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
def profile(name, entry, max_scan):
    pretty = entry["pretty"]
    md = [f"## {pretty}\n",
          f"- **Role in study:** {entry['role']}",
          f"- **Strategy:** `{entry['strategy']}`  |  **Source:** `{entry.get('repo_id', entry.get('zip_url',''))}`",
          f"- **Expected (to verify):** {entry['expected']}"]

    try:
        loader = STRATEGIES[entry["strategy"]]
        records, source, ordered_keys, image_fields = loader(entry, max_scan)
    except Exception:
        md.append(f"\n> **LOAD FAILED**\n```\n{traceback.format_exc()}\n```\n")
        md.append("> Edit this dataset's REGISTRY entry (repo / file / url) and re-run.\n")
        return "\n".join(md) + "\n", None

    if not records:
        md.append("\n> **No records parsed.** Inspect the registry entry.\n")
        return "\n".join(md) + "\n", None

    md.append(f"- **Loaded from:** `{source}`  |  **records parsed (capped):** {len(records)}")

    first = records[0]
    keys = ordered_keys or list(first.keys())

    md.append("\n### All fields (from first record)\n| Field | Description |\n|---|---|")
    for k in keys:
        v = first.get(k)
        md.append(f"| `{k}` | {describe(v)} |")

    detected = {}
    md.append("\n### Critical fields for the study\n| Need | Field(s) found | Present? |\n|---|---|---|")
    for need, hint in HINTS.items():
        found = find_keys(keys, hint)
        if need == "images" and image_fields:
            for f in image_fields:
                if f not in found:
                    found.append(f)
        detected[need] = found
        md.append(f"| {need} | {', '.join(f'`{x}`' for x in found) or '—'} | {'✅' if found else '❌'} |")

    def ok(n): return "yes" if detected.get(n) else "NO"
    md.append("\n### Table-readiness verdict\n")
    md.append(f"- **Question-type table:** {ok('question_type')}")
    md.append(f"- **Multi-hop slice:** {ok('hop')} (else derive from evidence-page count)")
    md.append(f"- **Domain study:** {ok('domain')}")
    md.append(f"- **Locate / decomposition (evidence pages):** {ok('evidence_page')}")
    md.append(f"- **Hallucination (unanswerable):** {ok('unanswerable')}")
    md.append(f"- **Vision condition (images):** {ok('images')}")
    md.append(f"- **Text/layout condition (text):** {ok('text')}")

    verdict_row = {n: ("✅" if detected.get(n) else "❌")
                   for n in ["question_type", "hop", "domain", "evidence_page",
                             "unanswerable", "images", "text"]}

    # ---- value distributions: every class of every relevant field, with the
    # number of questions AND the number of distinct documents per class ----
    # pick a per-record document id so we can count docs per class
    doc_field = next((f for f in detected.get("doc_id", [])
                      if not isinstance(first.get(f), Heavy)), None)
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")

    md.append("\n### Value distributions (parsed subset)\n")
    if doc_field:
        n_docs = len({str(r.get(doc_field)) for r in records})
        md.append(f"_Scanned {len(records)} questions across {n_docs} documents "
                  f"(document id = `{doc_field}`). Every class is listed; counts are "
                  f"over this subset._\n")
    else:
        md.append(f"_Scanned {len(records)} questions; no document-id field, so only "
                  f"question counts are shown._\n")

    detected["evidence_source"] = find_keys(keys, ["evidence_source"])
    dist_needs = ["question_type", "domain", "answer_format", "unanswerable", "evidence_source"]

    rendered, any_d = {}, False
    for need in dist_needs:
        for f in detected.get(need, []):
            if f in rendered:
                md.append(f"\n#### {need} — `{f}`\n(same field as **{rendered[f]}** above)")
                continue
            rows, tot_q, tot_d = class_breakdown(records, f, doc_field)
            if not rows:
                continue
            rendered[f], any_d = need, True
            md.append(f"\n#### {need} — `{f}`  ({len(rows)} classes)\n")
            if doc_field:
                md.append("| Class | Questions | Documents |\n|---|---:|---:|")
                for k, nq, nd in rows:
                    md.append(f"| {esc(k)} | {nq} | {nd} |")
                md.append(f"| **Total** | {tot_q} | {tot_d} |")
            else:
                md.append("| Class | Questions |\n|---|---:|")
                for k, nq, _ in rows:
                    md.append(f"| {esc(k)} | {nq} |")
                md.append(f"| **Total** | {tot_q} |")
    if not any_d:
        md.append("(no categorical label fields detected)")

    # derived hop (single vs multi page) from the evidence-page count
    ep_fields = detected.get("evidence_page", [])
    if ep_fields:
        f = ep_fields[0]
        hop_q, hop_docs = Counter(), {}
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
            hop_q[key] += 1
            if doc_field:
                hop_docs.setdefault(key, set()).add(str(rec.get(doc_field)))
        if hop_q:
            md.append(f"\n#### derived hop — page count of `{f}`  ({len(hop_q)} classes)\n")
            if doc_field:
                md.append("| Class | Questions | Documents |\n|---|---:|---:|")
                for k, nq in hop_q.most_common():
                    md.append(f"| {k} | {nq} | {len(hop_docs[k])} |")
            else:
                md.append("| Class | Questions |\n|---|---:|")
                for k, nq in hop_q.most_common():
                    md.append(f"| {k} | {nq} |")
            if not detected.get("hop"):
                verdict_row["hop"] = "derivable"

    md.append("\n### Sample records (heavy fields omitted)\n")
    for rec in records[:2]:
        safe = {k: ("<heavy>" if is_heavy(k, v) else v) for k, v in rec.items()}
        blob = json.dumps(safe, indent=2, default=str)
        if len(blob) > 2000:
            blob = blob[:2000] + "\n… (truncated)"
        md.append("```json\n" + blob + "\n```")

    md.append("")
    return "\n".join(md) + "\n", (pretty, verdict_row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of dataset keys: " + ", ".join(REGISTRY))
    ap.add_argument("--max-scan", type=int, default=400)
    ap.add_argument("--out", default="docs/dataset_profile.md")
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa
    except ImportError:
        sys.exit("pip install -U datasets huggingface_hub pyarrow")

    keys = args.only or list(REGISTRY)
    parts = ["# Dataset Profile Report — MP-VRDU Empirical Study\n",
             f"_Generated: {datetime.now(timezone.utc).isoformat()}_\n",
             "Profiles the five datasets by loading actual question records (not "
             "documentation) to confirm which result tables each can populate. Heavy "
             "fields (images / multi-MB context) are summarised, not dumped. Each "
             "dataset uses the fetch strategy that actually works for its repo layout "
             "(see the module docstring).\n",
             "## How to read this\n",
             "- **All fields**: every key on a question record, with type + sample.\n"
             "- **Critical fields**: whether study-relevant fields exist.\n"
             "- **Table-readiness verdict**: which skeleton tables this dataset can fill.\n"
             "- **Value distributions**: actual label values + counts.\n"]

    summary = []
    for k in keys:
        if k not in REGISTRY:
            parts.append(f"## {k}\n> not in registry\n")
            continue
        print(f"[profile] {k} …", file=sys.stderr)
        section, row = profile(k, REGISTRY[k], args.max_scan)
        parts.append(section)
        if row:
            summary.append(row)

    # cross-dataset matrix
    parts.append("## Cross-dataset summary\n")
    parts.append("| Dataset | qtype | hop | domain | evidence-pages | unanswerable | images | text |")
    parts.append("|---|---|---|---|---|---|---|---|")
    for pretty, row in summary:
        parts.append(f"| {pretty} | {row['question_type']} | {row['hop']} | {row['domain']} | "
                     f"{row['evidence_page']} | {row['unanswerable']} | {row['images']} | {row['text']} |")
    parts.append("\n_(✅ native field · ❌ absent · 'derivable' = compute from evidence-page count.)_\n")

    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print(f"[done] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
