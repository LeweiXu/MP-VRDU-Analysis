# Dataset Profile Report — MP-VRDU Empirical Study

_Generated: 2026-06-29T06:22:39.083254+00:00_

Profiles the five datasets by loading actual question records (not documentation) to confirm which result tables each can populate. Heavy fields (images / multi-MB context) are summarised, not dumped. Each dataset uses the fetch strategy that actually works for its repo layout (see the module docstring).

## How to read this

- **All fields**: every key on a question record, with type + sample.
- **Critical fields**: whether study-relevant fields exist.
- **Table-readiness verdict**: which skeleton tables this dataset can fill.
- **Value distributions**: actual label values + counts.

## MMLongBench-Doc

- **Role in study:** general; primary for question-type / domain study
- **Strategy:** `parquet_hf`  |  **Source:** `yubo2333/MMLongBench-Doc`
- **Expected (to verify):** gold evidence pages + evidence-source labels; unanswerable is signalled by the ANSWER ('Not answerable'), not by empty evidence.
- **Loaded from:** `yubo2333/MMLongBench-Doc :: train parquet (1 shards)`  |  **records parsed (capped):** 400

### All fields (from first record)
| Field | Description |
|---|---|
| `doc_id` | `str` (len=31): "PH_2016.06.08_Economy-Final.pdf" |
| `doc_type` | `str` (len=30): "Research report / Introduction" |
| `question` | `str` (len=98): "According to the report, how do 5% of the Latinos see economic upward mobility for their children?" |
| `answer` | `str` (len=13): "Less well-off" |
| `evidence_pages` | `str` (len=3): "[5]" |
| `evidence_sources` | `str` (len=9): "['Chart']" |
| `answer_format` | `str` (len=3): "Str" |

### Critical fields for the study
| Need | Field(s) found | Present? |
|---|---|---|
| question | `question` | ✅ |
| answer | `answer`, `answer_format` | ✅ |
| question_type | `doc_type` | ✅ |
| hop | — | ❌ |
| domain | `doc_type` | ✅ |
| evidence_page | `evidence_pages` | ✅ |
| evidence_other | `evidence_pages`, `evidence_sources` | ✅ |
| unanswerable | — | ❌ |
| answer_format | `answer_format` | ✅ |
| doc_id | `doc_id` | ✅ |
| images | — | ❌ |
| pages | `evidence_pages` | ✅ |
| words | — | ❌ |
| text | — | ❌ |

### Table-readiness verdict

- **Question-type table:** yes
- **Multi-hop slice:** NO (else derive from evidence-page count)
- **Domain study:** yes
- **Locate / decomposition (evidence pages):** yes
- **Hallucination (unanswerable):** NO
- **Vision condition (images):** NO
- **Text/layout condition (text):** NO

### Value distributions (parsed subset)

_Scanned 400 questions across 51 documents (document id = `doc_id`). Every class is listed; counts are over this subset._


#### question_type — `doc_type`  (7 classes)

| Class | Questions | Documents |
|---|---:|---:|
| Research report / Introduction | 164 | 19 |
| Academic paper | 65 | 9 |
| Guidebook | 52 | 8 |
| Tutorial/Workshop | 36 | 5 |
| Brochure | 35 | 5 |
| Administration/Industry file | 31 | 4 |
| Financial report | 17 | 1 |
| **Total** | 400 | 51 |

#### domain — `doc_type`
(same field as **question_type** above)

#### answer_format — `answer_format`  (5 classes)

| Class | Questions | Documents |
|---|---:|---:|
| None | 102 | 44 |
| Int | 101 | 45 |
| Str | 92 | 40 |
| List | 61 | 34 |
| Float | 44 | 22 |
| **Total** | 400 | 51 |

#### evidence_source — `evidence_sources`  (6 classes)

| Class | Questions | Documents |
|---|---:|---:|
| Pure-text (Plain-text) | 114 | 38 |
| (none) | 103 | 43 |
| Figure | 95 | 37 |
| Chart | 80 | 25 |
| Table | 73 | 37 |
| Generalized-text (Layout) | 69 | 32 |
| **Total** | 400 | 51 |

#### derived hop — page count of `evidence_pages`  (12 classes)

| Class | Questions | Documents |
|---|---:|---:|
| single (1 page) | 176 | 49 |
| zero (0 pages) | 101 | 45 |
| multi (2 pages) | 79 | 41 |
| multi (3 pages) | 17 | 16 |
| multi (4 pages) | 9 | 8 |
| multi (5 pages) | 5 | 5 |
| multi (6 pages) | 4 | 3 |
| multi (7 pages) | 2 | 2 |
| multi (10 pages) | 2 | 2 |
| multi (8 pages) | 2 | 2 |
| multi (9 pages) | 2 | 2 |
| multi (13 pages) | 1 | 1 |

### Sample records (heavy fields omitted)

```json
{
  "doc_id": "PH_2016.06.08_Economy-Final.pdf",
  "doc_type": "Research report / Introduction",
  "question": "According to the report, how do 5% of the Latinos see economic upward mobility for their children?",
  "answer": "Less well-off",
  "evidence_pages": "[5]",
  "evidence_sources": "['Chart']",
  "answer_format": "Str"
}
```
```json
{
  "doc_id": "PH_2016.06.08_Economy-Final.pdf",
  "doc_type": "Research report / Introduction",
  "question": "According to the report, which one is greater in population in the survey? Foreign born Latinos, or the Latinos interviewed by cellphone?",
  "answer": "Latinos interviewed by cellphone",
  "evidence_pages": "[19, 20]",
  "evidence_sources": "['Table']",
  "answer_format": "Str"
}
```


## LongDocURL

- **Role in study:** general; second / robustness
- **Strategy:** `annotation_hf`  |  **Source:** `dengchao/LongDocURL`
- **Expected (to verify):** understanding/reasoning/locating task tags; evidence-page localisation.
- **Loaded from:** `dengchao/LongDocURL :: LongDocURL_public.jsonl`  |  **records parsed (capped):** 400

### All fields (from first record)
| Field | Description |
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

### Critical fields for the study
| Need | Field(s) found | Present? |
|---|---|---|
| question | `question_id`, `question_type`, `question` | ✅ |
| answer | `answer`, `answer_format` | ✅ |
| question_type | `question_type`, `task_tag` | ✅ |
| hop | — | ❌ |
| domain | — | ❌ |
| evidence_page | `start_end_idx`, `evidence_pages` | ✅ |
| evidence_other | `detailed_evidences`, `evidence_pages`, `evidence_sources` | ✅ |
| unanswerable | — | ❌ |
| answer_format | `answer_format` | ✅ |
| doc_id | `doc_no`, `pdf_path` | ✅ |
| images | `images` | ✅ |
| pages | `total_pages`, `start_end_idx`, `evidence_pages` | ✅ |
| words | — | ❌ |
| text | — | ❌ |

### Table-readiness verdict

- **Question-type table:** yes
- **Multi-hop slice:** NO (else derive from evidence-page count)
- **Domain study:** NO
- **Locate / decomposition (evidence pages):** yes
- **Hallucination (unanswerable):** NO
- **Vision condition (images):** yes
- **Text/layout condition (text):** NO

### Value distributions (parsed subset)

_Scanned 400 questions across 203 documents (document id = `doc_no`). Every class is listed; counts are over this subset._


#### question_type — `question_type`  (9 classes)

| Class | Questions | Documents |
|---|---:|---:|
| extract | 214 | 125 |
| topic2title | 41 | 28 |
| extract_fig2tab | 41 | 23 |
| summary2title | 28 | 25 |
| calculate | 22 | 19 |
| compare | 21 | 19 |
| summary2tab | 19 | 13 |
| count | 10 | 10 |
| summarize | 4 | 4 |
| **Total** | 400 | 203 |

#### question_type — `task_tag`  (3 classes)

| Class | Questions | Documents |
|---|---:|---:|
| Understanding | 214 | 125 |
| Locating | 129 | 77 |
| Reasoning | 57 | 49 |
| **Total** | 400 | 203 |

#### answer_format — `answer_format`  (5 classes)

| Class | Questions | Documents |
|---|---:|---:|
| String | 181 | 124 |
| List | 132 | 86 |
| Integer | 59 | 50 |
| Float | 26 | 18 |
| None | 2 | 2 |
| **Total** | 400 | 203 |

#### evidence_source — `evidence_sources`  (5 classes)

| Class | Questions | Documents |
|---|---:|---:|
| Text | 166 | 112 |
| Layout | 149 | 100 |
| Table | 148 | 88 |
| Figure | 90 | 57 |
| Others | 2 | 2 |
| **Total** | 400 | 203 |

#### derived hop — page count of `start_end_idx`  (1 classes)

| Class | Questions | Documents |
|---|---:|---:|
| multi (2 pages) | 400 | 203 |

### Sample records (heavy fields omitted)

```json
{
  "question_id": "free_gpt4o_4026369_60_70_12",
  "doc_no": "4026369",
  "total_pages": 70,
  "start_end_idx": [
    60,
    70
  ],
  "question_type": "extract",
  "question": "Which publications are stated as helpful in producing a manual?",
  "answer": "University/Advantsar Communications Project",
  "detailed_evidences": "The publications stated as helpful in producing a manual are 'University/Advantsar Communications Project' published in 1997.1 and other unspecified publications (<box>(0.11, 0.41, 0.87, 0.61)</box>, page 67).",
  "evidence_pages": [
    67
  ],
  "evidence_sources": [
    "Layout"
  ],
  "answer_format": "String",
  "task_tag": "Understanding",
  "images": [
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_40.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_41.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_42.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_43.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_44.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_45.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_46.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_47.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_48.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_49.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_50.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_51.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_52.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_53.png",
    "/data/oss_bucket_0/achao.dc/public_datasets
… (truncated)
```
```json
{
  "question_id": "topic2title_4125651_5_9_3",
  "doc_no": "4125651",
  "total_pages": 104,
  "start_end_idx": [
    5,
    9
  ],
  "question_type": "topic2title",
  "question": "Which titles would provide insights into the importance of light efficiency and quality metrics in lighting products?\nSelect titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
  "answer": [
    "Uiterst hoge lichtefficientie (tot meer dan 100Lm/W)",
    "dimmable2.200KI 2.600K CRI95",
    "ambient-dimmable2.000K\u00b72.900KCRI95"
  ],
  "detailed_evidences": "",
  "evidence_pages": [
    5,
    9
  ],
  "evidence_sources": [
    "Layout"
  ],
  "answer_format": "List",
  "task_tag": "Locating",
  "images": [
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_0.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_1.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_2.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_3.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_4.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_5.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_6.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_7.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_8.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_9.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_10.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_11.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_12.png",
    "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4125/4125651_13.png",
    "
… (truncated)
```


## CUAD

- **Role in study:** text-heavy (contracts)
- **Strategy:** `squad_zip`  |  **Source:** `theatticusproject/cuad-qa`
- **Expected (to verify):** SQuAD-style char-offset spans; 41 clause categories (id suffix); is_impossible flag for clauses absent from a contract.
- **Loaded from:** `https://github.com/TheAtticusProject/cuad/raw/main/data.zip :: CUADv1.json`  |  **records parsed (capped):** 400

### All fields (from first record)
| Field | Description |
|---|---|
| `title` | `str` (len=51): "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT" |
| `question` | `str` (len=143): "Highlight the parts (if any) of this contract related to \"Document Name\" that should be reviewed by a lawyer. Details: The name of the contract" |
| `id` | `str` (len=66): "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Document Name" |
| `is_impossible` | `bool`: false |
| `answer_start` | `list` (len=1): [44] |
| `answer_text` | `list` (len=1): ["DISTRIBUTOR AGREEMENT"] |
| `clause_category` | `str` (len=13): "Document Name" |
| `context` | (heavy) `str`, len=54290 |

### Critical fields for the study
| Need | Field(s) found | Present? |
|---|---|---|
| question | `question` | ✅ |
| answer | `answer_start`, `answer_text` | ✅ |
| question_type | `clause_category` | ✅ |
| hop | — | ❌ |
| domain | — | ❌ |
| evidence_page | — | ❌ |
| evidence_other | `answer_start`, `context` | ✅ |
| unanswerable | `is_impossible` | ✅ |
| answer_format | — | ❌ |
| doc_id | `title` | ✅ |
| images | — | ❌ |
| pages | — | ❌ |
| words | `context` | ✅ |
| text | `answer_text`, `context` | ✅ |

### Table-readiness verdict

- **Question-type table:** yes
- **Multi-hop slice:** NO (else derive from evidence-page count)
- **Domain study:** NO
- **Locate / decomposition (evidence pages):** NO
- **Hallucination (unanswerable):** yes
- **Vision condition (images):** NO
- **Text/layout condition (text):** yes

### Value distributions (parsed subset)

_Scanned 400 questions across 10 documents (document id = `title`). Every class is listed; counts are over this subset._


#### question_type — `clause_category`  (41 classes)

| Class | Questions | Documents |
|---|---:|---:|
| Document Name | 10 | 10 |
| Parties | 10 | 10 |
| Agreement Date | 10 | 10 |
| Effective Date | 10 | 10 |
| Expiration Date | 10 | 10 |
| Renewal Term | 10 | 10 |
| Notice Period To Terminate Renewal | 10 | 10 |
| Governing Law | 10 | 10 |
| Most Favored Nation | 10 | 10 |
| Non-Compete | 10 | 10 |
| Exclusivity | 10 | 10 |
| No-Solicit Of Customers | 10 | 10 |
| Competitive Restriction Exception | 10 | 10 |
| No-Solicit Of Employees | 10 | 10 |
| Non-Disparagement | 10 | 10 |
| Termination For Convenience | 10 | 10 |
| Rofr/Rofo/Rofn | 10 | 10 |
| Change Of Control | 10 | 10 |
| Anti-Assignment | 10 | 10 |
| Revenue/Profit Sharing | 10 | 10 |
| Price Restrictions | 10 | 10 |
| Minimum Commitment | 10 | 10 |
| Volume Restriction | 10 | 10 |
| Ip Ownership Assignment | 10 | 10 |
| Joint Ip Ownership | 10 | 10 |
| License Grant | 10 | 10 |
| Non-Transferable License | 10 | 10 |
| Affiliate License-Licensor | 10 | 10 |
| Affiliate License-Licensee | 10 | 10 |
| Unlimited/All-You-Can-Eat-License | 10 | 10 |
| Irrevocable Or Perpetual License | 10 | 10 |
| Source Code Escrow | 9 | 9 |
| Post-Termination Services | 9 | 9 |
| Audit Rights | 9 | 9 |
| Uncapped Liability | 9 | 9 |
| Cap On Liability | 9 | 9 |
| Liquidated Damages | 9 | 9 |
| Warranty Duration | 9 | 9 |
| Insurance | 9 | 9 |
| Covenant Not To Sue | 9 | 9 |
| Third Party Beneficiary | 9 | 9 |
| **Total** | 400 | 10 |

#### unanswerable — `is_impossible`  (2 classes)

| Class | Questions | Documents |
|---|---:|---:|
| True | 273 | 10 |
| False | 127 | 10 |
| **Total** | 400 | 10 |

### Sample records (heavy fields omitted)

```json
{
  "title": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
  "question": "Highlight the parts (if any) of this contract related to \"Document Name\" that should be reviewed by a lawyer. Details: The name of the contract",
  "id": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Document Name",
  "is_impossible": false,
  "answer_start": [
    44
  ],
  "answer_text": [
    "DISTRIBUTOR AGREEMENT"
  ],
  "clause_category": "Document Name",
  "context": "<heavy>"
}
```
```json
{
  "title": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
  "question": "Highlight the parts (if any) of this contract related to \"Parties\" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract",
  "id": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Parties",
  "is_impossible": false,
  "answer_start": [
    244,
    148,
    49574,
    197,
    212
  ],
  "answer_text": [
    "Distributor",
    "Electric City Corp.",
    "Electric City of Illinois L.L.C.",
    "Company",
    "Electric City of Illinois LLC"
  ],
  "clause_category": "Parties",
  "context": "<heavy>"
}
```


## DocFinQA

- **Role in study:** in-between (financial)
- **Strategy:** `hf_stream`  |  **Source:** `kensho/DocFinQA`
- **Expected (to verify):** golden context per question; numeric/program answers; long filings.
- **Loaded from:** `kensho/DocFinQA :: test (streamed, capped 120)`  |  **records parsed (capped):** 120

### All fields (from first record)
| Field | Description |
|---|---|
| `Context` | (heavy) `str`, len=1595159 |
| `Question` | `str` (len=74): "what is the net change in net revenue during 2015 for entergy corporation?" |
| `Program` | `str` (len=138): "   net_revenue_2015 = 5829\n net_revenue_2014 = 5735\n net_revenue_change = net_revenue_2015 - net_revenue_2014\n answer = net_revenue_change" |
| `Answer` | `str` (len=2): "94" |

### Critical fields for the study
| Need | Field(s) found | Present? |
|---|---|---|
| question | `Question` | ✅ |
| answer | `Answer` | ✅ |
| question_type | — | ❌ |
| hop | — | ❌ |
| domain | — | ❌ |
| evidence_page | — | ❌ |
| evidence_other | `Context` | ✅ |
| unanswerable | — | ❌ |
| answer_format | — | ❌ |
| doc_id | — | ❌ |
| images | — | ❌ |
| pages | — | ❌ |
| words | `Context` | ✅ |
| text | `Context` | ✅ |

### Table-readiness verdict

- **Question-type table:** NO
- **Multi-hop slice:** NO (else derive from evidence-page count)
- **Domain study:** NO
- **Locate / decomposition (evidence pages):** NO
- **Hallucination (unanswerable):** NO
- **Vision condition (images):** NO
- **Text/layout condition (text):** yes

### Value distributions (parsed subset)

_Scanned 120 questions; no document-id field, so only question counts are shown._

(no categorical label fields detected)

### Sample records (heavy fields omitted)

```json
{
  "Context": "<heavy>",
  "Question": "what is the net change in net revenue during 2015 for entergy corporation?",
  "Program": "   net_revenue_2015 = 5829\n net_revenue_2014 = 5735\n net_revenue_change = net_revenue_2015 - net_revenue_2014\n answer = net_revenue_change",
  "Answer": "94"
}
```
```json
{
  "Context": "<heavy>",
  "Question": "what percentage of total facilities as measured in square feet are leased?",
  "Program": "",
  "Answer": "14%"
}
```


## SlideVQA

- **Role in study:** visual-heavy (slides)
- **Strategy:** `parquet_hf`  |  **Source:** `NTT-hil-insight/SlideVQA`
- **Expected (to verify):** native evidence_pages list; arithmetic_expression marks numerical questions; per-slide images in page_1..page_20.
- **Loaded from:** `NTT-hil-insight/SlideVQA :: test parquet (12 shards)`  |  **records parsed (capped):** 400

### All fields (from first record)
| Field | Description |
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

### Critical fields for the study
| Need | Field(s) found | Present? |
|---|---|---|
| question | `question` | ✅ |
| answer | `answer` | ✅ |
| question_type | — | ❌ |
| hop | — | ❌ |
| domain | — | ❌ |
| evidence_page | `evidence_pages` | ✅ |
| evidence_other | `arithmetic_expression`, `evidence_pages` | ✅ |
| unanswerable | — | ❌ |
| answer_format | — | ❌ |
| doc_id | `deck_name`, `deck_url` | ✅ |
| images | `page_1`, `page_2`, `page_3`, `page_4`, `page_5`, `page_6`, `page_7`, `page_8`, `page_9`, `page_10`, `page_11`, `page_12`, `page_13`, `page_14`, `page_15`, `page_16`, `page_17`, `page_18`, `page_19`, `page_20` | ✅ |
| pages | `evidence_pages` | ✅ |
| words | — | ❌ |
| text | — | ❌ |

### Table-readiness verdict

- **Question-type table:** NO
- **Multi-hop slice:** NO (else derive from evidence-page count)
- **Domain study:** NO
- **Locate / decomposition (evidence pages):** yes
- **Hallucination (unanswerable):** NO
- **Vision condition (images):** yes
- **Text/layout condition (text):** NO

### Value distributions (parsed subset)

_Scanned 400 questions across 90 documents (document id = `deck_name`). Every class is listed; counts are over this subset._

(no categorical label fields detected)

#### derived hop — page count of `evidence_pages`  (2 classes)

| Class | Questions | Documents |
|---|---:|---:|
| single (1 page) | 310 | 90 |
| multi (2 pages) | 90 | 56 |

### Sample records (heavy fields omitted)

```json
{
  "deck_name": "2012-02-20fy11roadshow-120221022442-phpapp02_95",
  "deck_url": "https://www.slideshare.net/Nestle_IR/20120220-fy-11roadshow",
  "qa_id": 0,
  "question": "How much is the Trading Operating Profit in 2011?",
  "answer": "12.5 bn",
  "arithmetic_expression": "None",
  "evidence_pages": [
    5
  ],
  "page_4": "<heavy>",
  "page_18": "<heavy>",
  "page_19": "<heavy>",
  "page_16": "<heavy>",
  "page_12": "<heavy>",
  "page_3": "<heavy>",
  "page_15": "<heavy>",
  "page_1": "<heavy>",
  "page_11": "<heavy>",
  "page_8": "<heavy>",
  "page_2": "<heavy>",
  "page_7": "<heavy>",
  "page_6": "<heavy>",
  "page_17": "<heavy>",
  "page_10": "<heavy>",
  "page_5": "<heavy>",
  "page_13": "<heavy>",
  "page_9": "<heavy>",
  "page_20": "<heavy>",
  "page_14": "<heavy>"
}
```
```json
{
  "deck_name": "2012-02-20fy11roadshow-120221022442-phpapp02_95",
  "deck_url": "https://www.slideshare.net/Nestle_IR/20120220-fy-11roadshow",
  "qa_id": 1,
  "question": "How much is the Trading Operating Profit in the year Nestl\u00e9 achieved the third largest Organic Growth in 10 years?",
  "answer": "12.5 bn",
  "arithmetic_expression": "None",
  "evidence_pages": [
    5,
    7
  ],
  "page_4": "<heavy>",
  "page_18": "<heavy>",
  "page_19": "<heavy>",
  "page_16": "<heavy>",
  "page_12": "<heavy>",
  "page_3": "<heavy>",
  "page_15": "<heavy>",
  "page_1": "<heavy>",
  "page_11": "<heavy>",
  "page_8": "<heavy>",
  "page_2": "<heavy>",
  "page_7": "<heavy>",
  "page_6": "<heavy>",
  "page_17": "<heavy>",
  "page_10": "<heavy>",
  "page_5": "<heavy>",
  "page_13": "<heavy>",
  "page_9": "<heavy>",
  "page_20": "<heavy>",
  "page_14": "<heavy>"
}
```


## Cross-dataset summary

| Dataset | qtype | hop | domain | evidence-pages | unanswerable | images | text |
|---|---|---|---|---|---|---|---|
| MMLongBench-Doc | ✅ | derivable | ✅ | ✅ | ❌ | ❌ | ❌ |
| LongDocURL | ✅ | derivable | ❌ | ✅ | ❌ | ✅ | ❌ |
| CUAD | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| DocFinQA | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| SlideVQA | ❌ | derivable | ❌ | ✅ | ❌ | ✅ | ❌ |

_(✅ native field · ❌ absent · 'derivable' = compute from evidence-page count.)_
