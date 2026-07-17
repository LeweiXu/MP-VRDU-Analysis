"""Kaya GPU smoke for the Unlimited-OCR ParserCacheMiss fix.

Feeds a handful of the failing docs' cached render PNGs (plus one control page
from a doc that already parsed) through tools/parser_worker.py in the
parse-unlimited env, and reports markdown length + which infer mode each page
landed on. Confirms the crop-mode -> base-mode fallback recovers the pages that
came back empty, without regressing a page that already worked. Writes nothing to
the real parser cache (own scratch out_paths)."""

import json
import subprocess
from pathlib import Path

REND = Path("results/cache/g1-parser-full-unlimited/renders")
WORKER = "tools/parser_worker.py"
PYTHON = "envs/parse-unlimited/bin/python"
MODEL = "baidu/Unlimited-OCR"

samples: list[str] = []


def add(dirglob: str, pages: list[int]) -> None:
    d = next(iter(sorted(REND.glob(dirglob))), None)
    if d is None:
        print("MISSING render dir:", dirglob)
        return
    for p in pages:
        png = d / f"page_{p:04d}.png"
        if png.exists():
            samples.append(str(png))
        else:
            print("no render png:", png)


add("germanwings*", [0, 15])          # landscape slide deck (all pages failed)
add("owners-manual-2170416*", [0])    # near-square
add("0e94b4197b*", [1])               # widescreen
add("caltrain*", [0])                 # landscape
add("2303.05039*", [3])               # CONTROL: paper that already parsed fine

scratch = Path("scratchpad/smoke_md")
scratch.mkdir(parents=True, exist_ok=True)
jobs = []
for i, png in enumerate(samples):
    jobs.append({
        "pdf_path": "", "index": 0, "doc_id": Path(png).parent.name,
        "image_path": png, "out_path": str(scratch / f"s{i}.md"),
    })

payload = json.dumps({"parser_tool": "unlimited", "model_id": MODEL, "dpi": 200, "jobs": jobs})
print(f"running worker over {len(jobs)} pages via {PYTHON}")
proc = subprocess.run([PYTHON, WORKER], input=payload, text=True, capture_output=True)
print("=== worker stdout ===")
print(proc.stdout)
print("=== worker stderr ===")
print(proc.stderr)
print("=== per-page results ===")
for i, png in enumerate(samples):
    out = scratch / f"s{i}.md"
    n = len(out.read_text(encoding="utf-8").strip()) if out.exists() else -1
    tag = "OK  " if n > 0 else "FAIL"
    print(f"  {tag} chars={n:6d}  {png}")
print("worker rc:", proc.returncode)
