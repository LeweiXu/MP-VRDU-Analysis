# Annotating the MMLongBench documents

Thanks for helping label these. This is a one-time human check on an assumption
the paper makes. We group MMLongBench's seven native document types into three
"domains" (text-heavy / in-between / visual-heavy), and we want to know whether a
human, looking at the actual PDFs, would bin them the same way.

There are 135 documents total. The script opens each PDF, prompts for the labels,
and saves after every document so you can stop and resume.

## 1. Set up

Run this from the project root:

```bash
python -m pip install -r requirements-annotate.txt
```

Download the `.data` folder from Google Drive:

https://drive.google.com/drive/folders/16BNfcq_sJgHE2TncPn5ubBoPUmU03ADf?usp=sharing

The script needs the PDFs and the annotation sheet paths to line up. The easiest
setup is to put the downloaded `.data/` folder in the project root, next to
`scripts/` and `annotations/`.

You can download from the Drive web UI (right-click the folder, "Download",
which zips it), or with [`gdown`](https://github.com/wkentaro/gdown):

```bash
gdown --folder "https://drive.google.com/drive/folders/16BNfcq_sJgHE2TncPn5ubBoPUmU03ADf"
```

If you are preparing the folder yourself from the original MMLongBench files,
split the PDFs into doc-type folders first:

```bash
python scripts/split_docs_by_type.py
```

That creates:

```text
.data/mmlongbench_docs_split/
  Academic_paper/                 (26 docs)
  Administration-Industry_file/   (10)
  Brochure/                       (15)
  Financial_report/               (11)
  Guidebook/                      (22)
  Research_report_-_Introduction/ (34)
  Tutorial-Workshop/              (17)
```

## 2. Run the annotation script

If `annotations/doc_labels.csv` already exists, start here:

```bash
python -m scripts.annotate_docs annotate
```

The command opens each PDF in your system viewer and prompts for:

```text
bin_label
scan_label
dominant_visual
notes
```

Answers are written back to the CSV after every document. Re-run the same command
to resume. Use `q` at a prompt to save and exit cleanly.

Useful options:

```bash
python -m scripts.annotate_docs annotate --no-open
python -m scripts.annotate_docs annotate --open-cmd "evince"
python -m scripts.annotate_docs annotate --redo-all
python -m scripts.annotate_docs annotate --sheet annotations/my_doc_labels.csv
```

If you need to create a fresh blank sheet from the staged MMLongBench parquet and
PDFs, run:

```bash
python -m scripts.annotate_docs sheet --output annotations/doc_labels.csv
```

Do not use `--force` on a sheet that already contains hand labels unless you
intend to erase them.

After labeling, run:

```bash
python -m scripts.annotate_docs score
```

If you annotated a non-default sheet, score that same file:

```bash
python -m scripts.annotate_docs score --sheet annotations/my_doc_labels.csv
```

This prints agreement between your labels and the automatic doc-type grouping,
plus the scanned/digital and visual-element summaries.

## 3. What Each Field Means

Judge the document as a whole, not a single page. The doc-type folder it came
from is our guess, not the answer; feel free to disagree with it.

### `bin_label`

How text-heavy vs visual-heavy is the document?

Pick one:

- `text_heavy`: you would read this to get the content. Body text carries the
  meaning; any tables/figures are supporting.
- `visual_heavy`: you would look at this. Charts, figures, photos, or layout
  carry the meaning and the text is sparse or secondary.
- `in_between`: genuinely mixed. Neither text nor visuals clearly dominate.

### `scan_label`

Is the PDF scanned or digital-born?

Pick one:

- `scanned`: a scan/photo of paper. Text is part of the page image, you cannot
  select it, and edges may be skewed or speckled.
- `digital`: born digital, exported from Word/LaTeX/InDesign/etc. Text is crisp
  and selectable.

### `dominant_visual`

What is the main visual element?

Pick one or more:

- `tables`
- `charts`
- `figures`
- `photos`
- `none`

If several apply, select several in the script. In the CSV they are stored with
semicolons, for example `charts;photos`. Use `none` for a document that is
essentially just running text.

### `notes`

Optional free text for anything weird or any judgment call that was hard. This
helps us interpret disagreements later.

## 4. CSV Rules

The script writes a CSV with one row per document. The columns you fill are:

| column            | value                                               |
|-------------------|-----------------------------------------------------|
| `bin_label`       | `text_heavy` / `in_between` / `visual_heavy`        |
| `scan_label`      | `scanned` / `digital`                               |
| `dominant_visual` | one or more of `tables;charts;figures;photos;none`  |
| `notes`           | anything worth flagging (optional)                  |

Leave `doc_id`, `pdf_path`, `doc_type`, `auto_*`, `avg_chars_per_page`, and
`page_count` alone. Those columns identify the document and store the automatic
reference labels.

If you edit the CSV manually in Excel / Google Sheets / LibreOffice, keep the
same columns, do not reorder or rename them, and use the exact lowercase values
above. Use `;` between multiple dominant visuals.

If a document does not fit any option, put your best guess and explain it in
`notes`. Do not leave required fields blank.
