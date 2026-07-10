# Annotating the MMLongBench documents

Thanks for helping label these. These hand labels are the source of truth for the
document bins the whole paper is built on, so getting them right matters. You look
at each PDF and record which modality dominates it (text vs visual) and whether it
is scanned or digital-born.

There are 135 documents total. The script opens each PDF, prompts for the labels,
and saves after every document so you can stop and resume.

## 1. Set up

Run this from the project root:

```bash
python -m pip install -r docs/requirements/annotate.txt
```

Download the `.data` folder from Google Drive:

https://drive.google.com/drive/folders/16BNfcq_sJgHE2TncPn5ubBoPUmU03ADf?usp=sharing

The script needs the PDFs and the annotation sheet paths to line up. The easiest
setup is to put the downloaded `.data/` folder in the project root, next to
`ops/` and `annotations/`.

You can download from the Drive web UI (right-click the folder, "Download",
which zips it), or with [`gdown`](https://github.com/wkentaro/gdown):

```bash
gdown --folder "https://drive.google.com/drive/folders/16BNfcq_sJgHE2TncPn5ubBoPUmU03ADf"
```

If you are preparing the folder yourself from the original MMLongBench files,
split the PDFs into doc-type folders first (this only groups them by their native
type so they are easier to flip through, it does not label anything):

```bash
python -m ops.scripts.split_docs_by_type
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
python -m ops.scripts.annotate_docs annotate
```

The command opens each PDF in your system viewer and prompts for:

```text
bin_label
scan_label
dominant_visual   (optional, exploratory)
notes
```

Answers are written back to the CSV after every document. Re-run the same command
to resume. Use `q` at a prompt to save and exit cleanly.

Useful options:

```bash
python -m ops.scripts.annotate_docs annotate --no-open
python -m ops.scripts.annotate_docs annotate --open-cmd "evince"
python -m ops.scripts.annotate_docs annotate --redo-all
python -m ops.scripts.annotate_docs annotate --no-dominant-visual
python -m ops.scripts.annotate_docs annotate --sheet annotations/my_doc_labels.csv
```

`dominant_visual` is exploratory and never required. If you would rather not spend
time on it, pass `--no-dominant-visual` and the script skips that prompt entirely.

If you need to create a fresh blank sheet from the staged MMLongBench parquet and
PDFs, run:

```bash
python -m ops.scripts.annotate_docs sheet --output annotations/doc_labels.csv
```

Do not use `--force` on a sheet that already contains hand labels unless you
intend to erase them.

After labeling, run:

```bash
python -m ops.scripts.annotate_docs score
```

If you annotated a non-default sheet, score that same file:

```bash
python -m ops.scripts.annotate_docs score --sheet annotations/my_doc_labels.csv
```

This prints the distribution of your bin labels, how often your scan label matched
the automatic scanned/digital guess, and a tally of the dominant-visual values.

## 3. What each field means

Judge the document as a whole, not a single page. The doc-type folder it came
from is just MMLongBench's native type, not the answer; label what you actually
see.

### `bin_label`

Which modality dominates the document's information content? This is about where
the meaning lives, not page count or scan status.

Pick one:

- `text-dominant`: the information is linguistic, you would read it to get the
  content. A scanned, mostly-handwritten note is text-dominant even though it is
  an image on disk, because its information is words.
- `visual-dominant`: the information lives in the imagery, charts, or layout. A
  text-sparse magazine cover is visual-dominant, the meaning is in the design.
- `mixed-modality`: genuinely mixed, needs judgement. A text-dense academic paper
  with a few important figures and tables is the typical case, neither channel
  clearly dominates.

### `scan_label`

Is the PDF scanned or digital-born?

Pick one:

- `digital`: born digital, exported from Word/LaTeX/InDesign/etc. Text is crisp
  and selectable.
- `scanned`: a scan/photo of paper. Text is part of the page image, you cannot
  select it, and edges may be skewed or speckled.

Note that scan status is independent of the bin: a scanned document can still be
text-dominant (the handwriting case above).

### `dominant_visual` (optional, exploratory)

What is the main visual element? This one is exploratory, we collect it while we
are already looking but nothing in the pipeline depends on it, so skip it if you
prefer (see `--no-dominant-visual`).

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

## 4. CSV rules

The script writes a CSV with one row per document. The columns you fill are:

| column            | value                                                    |
|-------------------|----------------------------------------------------------|
| `bin_label`       | `text-dominant` / `mixed-modality` / `visual-dominant`   |
| `scan_label`      | `digital` / `scanned`                                    |
| `dominant_visual` | optional, one or more of `tables;charts;figures;photos;none` |
| `notes`           | anything worth flagging (optional)                       |

Leave `doc_id`, `pdf_path`, `doc_type`, `auto_scan`, `avg_chars_per_page`, and
`page_count` alone. Those columns identify the document and store the automatic
scan guess.

If you edit the CSV manually in Excel / Google Sheets / LibreOffice, keep the
same columns, do not reorder or rename them, and use the exact lowercase values
above (with the hyphens). Use `;` between multiple dominant visuals.

`bin_label` and `scan_label` are required; `dominant_visual` and `notes` are
optional. If a document does not fit any bin cleanly, put your best judgement and
explain it in `notes`.

## 5. Checking agreement between annotators (Cohen's kappa)

The bin axis carries the whole thesis, and `mixed-modality` in particular is a
judgement call, so we check inter-annotator agreement on a subset. The target is
Cohen's kappa >= 0.75 (the same bar we use for the automated judge).

The flow is: a second annotator labels a small subset **blind** (without seeing
the first annotator's answers), then we compare.

1. Build a blind subset sheet from the already-labelled primary sheet. This picks
   a random sample and blanks out the labels, so the second pass cannot peek:

   ```bash
   python -m ops.scripts.annotate_docs kappa-sheet --n 25 --seed 0
   ```

   Record the `--seed` you used so the subset is reproducible. It writes
   `annotations/kappa_subset.csv`.

2. The second annotator fills it like any other sheet:

   ```bash
   python -m ops.scripts.annotate_docs annotate --sheet annotations/kappa_subset.csv
   ```

3. Report the agreement:

   ```bash
   python -m ops.scripts.annotate_docs kappa
   ```

   This prints Cohen's kappa for `bin_label` and `scan_label` over the shared
   documents, the raw agreement percentage, and the specific documents where the
   two annotators disagreed, so you can see what drove any low score.
