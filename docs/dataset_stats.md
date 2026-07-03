# Dataset Statistics — MP-VRDU

_Generated: 2026-07-03T05:16:12.673306+00:00_

Descriptive statistics for all five datasets: document-length distributions, question/document counts, and every categorical label field with its full class breakdown (relevant to the study or not). Companion to `dataset_profile.md`, which instead judges table-readiness. Lengths are per distinct document; the unit differs by dataset (characters/words for text corpora, pages for image corpora) and is labelled inline.

**Coverage:** full census — every question and document per dataset.

## MMLongBench-Doc

- **Role in study:** general; primary for question-type / domain study
- **Source:** `yubo2333/MMLongBench-Doc`  |  **Strategy:** `parquet_hf`
- **Loaded from:** `yubo2333/MMLongBench-Doc :: train parquet (1 shards)`
- **Questions:** 1091  |  **Distinct documents:** 135 (doc id = `doc_id`)

### Document length

- **Text length:** n/a
- **Word count:** n/a
- **Page count** (pages · via local PDFs (PyMuPDF), per document, n=135): avg 48.3 · min 9 · max 468 · median 28

### Document labels


**`doc_type`** — 7 classes

| Class | Questions | Documents |
|---|---:|---:|
| Research report / Introduction | 293 | 34 |
| Academic paper | 204 | 26 |
| Guidebook | 156 | 22 |
| Tutorial/Workshop | 139 | 17 |
| Financial report | 117 | 11 |
| Brochure | 101 | 15 |
| Administration/Industry file | 81 | 10 |
| **Total** | 1091 | 135 |

### Question labels


**`answer_format`** — 5 classes

| Class | Questions | Documents |
|---|---:|---:|
| Int | 290 | 114 |
| Str | 250 | 113 |
| None | 244 | 112 |
| Float | 160 | 55 |
| List | 147 | 83 |
| **Total** | 1091 | 135 |

**`evidence_sources`** — 6 classes

| Class | Questions | Documents |
|---|---:|---:|
| Pure-text (Plain-text) | 298 | 100 |
| Figure | 293 | 96 |
| (none) | 254 | 112 |
| Table | 218 | 82 |
| Chart | 178 | 62 |
| Generalized-text (Layout) | 119 | 59 |
| **Total** | 1091 | 135 |

### Derived hop — page count of `evidence_pages`

| Class | Questions | Documents |
|---|---:|---:|
| single (1 page) | 485 | 132 |
| multi (2 pages) | 247 | 110 |
| zero (0 pages) | 246 | 112 |
| multi (3 pages) | 40 | 35 |
| multi (4 pages) | 22 | 20 |
| multi (5 pages) | 14 | 13 |
| multi (6 pages) | 12 | 11 |
| multi (10 pages) | 6 | 6 |
| multi (7 pages) | 5 | 5 |
| multi (8 pages) | 5 | 5 |
| multi (13 pages) | 2 | 2 |
| multi (9 pages) | 2 | 2 |
| multi (12 pages) | 2 | 2 |
| multi (21 pages) | 1 | 1 |
| multi (15 pages) | 1 | 1 |
| multi (24 pages) | 1 | 1 |

### Unanswerable
- **Signal:** `answer=='Not answerable'`  |  **Count (in scan):** 244

### All fields (first record)
| Field | Type / sample |
|---|---|
| `doc_id` | `str` (len=31): "PH_2016.06.08_Economy-Final.pdf" |
| `doc_type` | `str` (len=30): "Research report / Introduction" |
| `question` | `str` (len=98): "According to the report, how do 5% of the Latinos see economic upward mobility for their children?" |
| `answer` | `str` (len=13): "Less well-off" |
| `evidence_pages` | `str` (len=3): "[5]" |
| `evidence_sources` | `str` (len=9): "['Chart']" |
| `answer_format` | `str` (len=3): "Str" |


## LongDocURL

- **Role in study:** general; second / robustness
- **Source:** `dengchao/LongDocURL`  |  **Strategy:** `annotation_hf`
- **Loaded from:** `dengchao/LongDocURL :: LongDocURL_public.jsonl`
- **Questions:** 2325  |  **Distinct documents:** 396 (doc id = `doc_no`)

### Document length

- **Text length:** n/a
- **Word count:** n/a
- **Page count** (pages · via page field, per document, n=396): avg 85.6 · min 51 · max 149 · median 78.0

### Document labels
_(none detected)_

### Question labels


**`question_type`** — 9 classes

| Class | Questions | Documents |
|---|---:|---:|
| extract | 1243 | 316 |
| extract_fig2tab | 231 | 60 |
| topic2title | 201 | 34 |
| calculate | 145 | 69 |
| summary2title | 137 | 64 |
| summary2tab | 126 | 46 |
| count | 117 | 95 |
| compare | 112 | 87 |
| summarize | 13 | 13 |
| **Total** | 2325 | 396 |

**`task_tag`** — 3 classes

| Class | Questions | Documents |
|---|---:|---:|
| Understanding | 1243 | 316 |
| Locating | 695 | 177 |
| Reasoning | 387 | 195 |
| **Total** | 2325 | 396 |

**`answer_format`** — 5 classes

| Class | Questions | Documents |
|---|---:|---:|
| String | 941 | 312 |
| List | 757 | 241 |
| Integer | 431 | 197 |
| Float | 185 | 76 |
| None | 11 | 11 |
| **Total** | 2325 | 396 |

**`evidence_sources`** — 5 classes

| Class | Questions | Documents |
|---|---:|---:|
| Text | 994 | 305 |
| Table | 871 | 208 |
| Layout | 779 | 243 |
| Figure | 548 | 155 |
| Others | 8 | 5 |
| **Total** | 2325 | 396 |

### Derived hop — page count of `start_end_idx`

| Class | Questions | Documents |
|---|---:|---:|
| multi (2 pages) | 2325 | 396 |

### Unanswerable
- **Signal:** `answer=='Not answerable'`  |  **Count (in scan):** 8

### All fields (first record)
| Field | Type / sample |
|---|---|
| `question_id` | `str` (len=27): "free_gpt4o_4026369_60_70_12" |
| `doc_no` | `str` (len=7): "4026369" |
| `total_pages` | `int`: 70 |
| `start_end_idx` | `list` (len=2): [60, 70] |
| `question_type` | `str` (len=7): "extract" |
| `question` | `str` (len=63): "Which publications are stated as helpful in producing a manual?" |
| `answer` | `str` (len=43): "University/Advantsar Communications Project" |
| `detailed_evidences` | `str` (len=209): "The publications stated as helpful in producing a manual are 'University/Advantsar Communications Project' published in 1997.1 and other unspecified publication\u2026" |
| `evidence_pages` | `list` (len=1): [67] |
| `evidence_sources` | `list` (len=1): ["Layout"] |
| `answer_format` | `str` (len=6): "String" |
| `task_tag` | `str` (len=13): "Understanding" |
| `images` | `list` (len=30): ["/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_40.png", "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_41.png", "/data/oss_bucket_0/achao.d… |
| `pdf_path` | `str` (len=75): "/data/oss_bucket_0/achao.dc/public_datasets/ccpdf_zip/4000-4999/4026369.pdf" |


## CUAD

- **Role in study:** text-heavy (contracts)
- **Source:** `theatticusproject/cuad-qa`  |  **Strategy:** `squad_zip`
- **Loaded from:** `https://github.com/TheAtticusProject/cuad/raw/main/data.zip :: CUADv1.json`
- **Questions:** 20910  |  **Distinct documents:** 510 (doc id = `title`)

### Document length

- **Text length** (characters, per document, n=510): avg 52563.0 · min 645 · max 338211 · median 33143.0
- **Word count:** n/a
- **Page count:** n/a

### Document labels
_(none detected)_

### Question labels


**`clause_category`** — 41 classes

| Class | Questions | Documents |
|---|---:|---:|
| Document Name | 510 | 510 |
| Parties | 510 | 510 |
| Agreement Date | 510 | 510 |
| Effective Date | 510 | 510 |
| Expiration Date | 510 | 510 |
| Renewal Term | 510 | 510 |
| Notice Period To Terminate Renewal | 510 | 510 |
| Governing Law | 510 | 510 |
| Most Favored Nation | 510 | 510 |
| Non-Compete | 510 | 510 |
| Exclusivity | 510 | 510 |
| No-Solicit Of Customers | 510 | 510 |
| Competitive Restriction Exception | 510 | 510 |
| No-Solicit Of Employees | 510 | 510 |
| Non-Disparagement | 510 | 510 |
| Termination For Convenience | 510 | 510 |
| Rofr/Rofo/Rofn | 510 | 510 |
| Change Of Control | 510 | 510 |
| Anti-Assignment | 510 | 510 |
| Revenue/Profit Sharing | 510 | 510 |
| Price Restrictions | 510 | 510 |
| Minimum Commitment | 510 | 510 |
| Volume Restriction | 510 | 510 |
| Ip Ownership Assignment | 510 | 510 |
| Joint Ip Ownership | 510 | 510 |
| License Grant | 510 | 510 |
| Non-Transferable License | 510 | 510 |
| Affiliate License-Licensor | 510 | 510 |
| Affiliate License-Licensee | 510 | 510 |
| Unlimited/All-You-Can-Eat-License | 510 | 510 |
| Irrevocable Or Perpetual License | 510 | 510 |
| Source Code Escrow | 510 | 510 |
| Post-Termination Services | 510 | 510 |
| Audit Rights | 510 | 510 |
| Uncapped Liability | 510 | 510 |
| Cap On Liability | 510 | 510 |
| Liquidated Damages | 510 | 510 |
| Warranty Duration | 510 | 510 |
| Insurance | 510 | 510 |
| Covenant Not To Sue | 510 | 510 |
| Third Party Beneficiary | 510 | 510 |
| **Total** | 20910 | 510 |

### Unanswerable
- **Signal:** `is_impossible`  |  **Count (in scan):** 14208

### All fields (first record)
| Field | Type / sample |
|---|---|
| `title` | `str` (len=51): "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT" |
| `question` | `str` (len=143): "Highlight the parts (if any) of this contract related to \"Document Name\" that should be reviewed by a lawyer. Details: The name of the contract" |
| `id` | `str` (len=66): "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Document Name" |
| `is_impossible` | `bool`: false |
| `answer_start` | `list` (len=1): [44] |
| `answer_text` | `list` (len=1): ["DISTRIBUTOR AGREEMENT"] |
| `clause_category` | `str` (len=13): "Document Name" |
| `context` | (heavy) `str`, len=54290 |


## DocFinQA

- **Role in study:** in-between (financial)
- **Source:** `kensho/DocFinQA`  |  **Strategy:** `hf_stream`
- **Loaded from:** `kensho/DocFinQA :: test (streamed, capped 1000000000)`
- **Questions:** 922  |  **Distinct documents:** 922 (no doc-id field; counted per record)

### Document length

- **Text length** (characters, per document, n=922): avg 554533.9 · min 165236 · max 1595159 · median 433593.0
- **Word count:** n/a
- **Page count:** n/a

### Document labels
_(none detected)_

### Question labels
_(none detected)_

### Unanswerable
- **Signal:** `none`  |  **Count (in scan):** 0

### All fields (first record)
| Field | Type / sample |
|---|---|
| `Context` | (heavy) `str`, len=1595159 |
| `Question` | `str` (len=74): "what is the net change in net revenue during 2015 for entergy corporation?" |
| `Program` | `str` (len=138): "   net_revenue_2015 = 5829\n net_revenue_2014 = 5735\n net_revenue_change = net_revenue_2015 - net_revenue_2014\n answer = net_revenue_change" |
| `Answer` | `str` (len=2): "94" |


## SlideVQA

- **Role in study:** visual-heavy (slides)
- **Source:** `NTT-hil-insight/SlideVQA`  |  **Strategy:** `parquet_hf`
- **Loaded from:** `NTT-hil-insight/SlideVQA :: test parquet (12 shards)`
- **Questions:** 2215  |  **Distinct documents:** 400 (doc id = `deck_name`)

### Document length

- **Text length:** n/a
- **Word count:** n/a
- **Page count:** n/a
- _(no length signal exposed by this dataset)_

### Document labels
_(none detected)_

### Question labels
_(none detected)_

### Derived hop — page count of `evidence_pages`

| Class | Questions | Documents |
|---|---:|---:|
| single (1 page) | 1648 | 400 |
| multi (2 pages) | 562 | 250 |
| multi (3 pages) | 5 | 5 |

### Unanswerable
- **Signal:** `none`  |  **Count (in scan):** 0

### All fields (first record)
| Field | Type / sample |
|---|---|
| `deck_name` | `str` (len=47): "2012-02-20fy11roadshow-120221022442-phpapp02_95" |
| `deck_url` | `str` (len=59): "https://www.slideshare.net/Nestle_IR/20120220-fy-11roadshow" |
| `page_1` | (heavy) `image` |
| `page_2` | (heavy) `image` |
| `page_3` | (heavy) `image` |
| `page_4` | (heavy) `image` |
| `page_5` | (heavy) `image` |
| `page_6` | (heavy) `image` |
| `page_7` | (heavy) `image` |
| `page_8` | (heavy) `image` |
| `page_9` | (heavy) `image` |
| `page_10` | (heavy) `image` |
| `page_11` | (heavy) `image` |
| `page_12` | (heavy) `image` |
| `page_13` | (heavy) `image` |
| `page_14` | (heavy) `image` |
| `page_15` | (heavy) `image` |
| `page_16` | (heavy) `image` |
| `page_17` | (heavy) `image` |
| `page_18` | (heavy) `image` |
| `page_19` | (heavy) `image` |
| `page_20` | (heavy) `image` |
| `qa_id` | `int`: 0 |
| `question` | `str` (len=49): "How much is the Trading Operating Profit in 2011?" |
| `answer` | `str` (len=7): "12.5 bn" |
| `arithmetic_expression` | `str` (len=4): "None" |
| `evidence_pages` | `list` (len=1): [5] |


## Cross-dataset summary

| Dataset | Qs | Docs | Text len (avg chars) | Pages (avg) | Doc labels | Question labels | Unanswerable |
|---|---:|---:|---:|---:|---|---|---:|
| MMLongBench-Doc | 1091 | 135 | — | 48.3 | doc_type | answer_format;evidence_sources | 244 |
| LongDocURL | 2325 | 396 | — | 85.6 | — | question_type;task_tag;answer_format;evidence_sources | 8 |
| CUAD | 20910 | 510 | 52563.0 | — | — | clause_category | 14208 |
| DocFinQA | 922 | 922 | 554533.9 | — | — | — | 0 |
| SlideVQA | 2215 | 400 | — | — | — | — | 0 |
