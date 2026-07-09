# Dataset Statistics — MP-VRDU

_Generated: 2026-07-09T06:18:13.572035+00:00_

Descriptive statistics for the selected dataset: document-length distributions, question/document counts, and every categorical label field with its full class breakdown (relevant to the study or not). Companion to `dataset_profile.md`, which instead judges table-readiness. Lengths are per distinct document; the unit differs by dataset (characters/words for text corpora, pages for image corpora) and is labelled inline.

**Coverage:** full census — every question and document per dataset.

## MMLongBench-Doc

- **Role in study:** general; primary for question-type / domain study
- **Source:** `yubo2333/MMLongBench-Doc`  |  **Strategy:** `parquet_hf`
- **Loaded from:** `local staged parquet: /home/lingwei/mpvrdu/.data/mmlongbench/data`
- **Questions:** 1091  |  **Distinct documents:** 135 (doc id = `doc_id`)

### Document length

- **Text length:** n/a
- **Word count:** n/a
- **Page count** (pages · via local PDFs (PyMuPDF), per document, n=135): avg 48.3 · min 9 · max 468 · median 28

### Document labels


**`doc_type`** — 7 classes

| Class | Questions | Documents | Digital docs | Scanned docs | Unknown scan docs |
|---|---:|---:|---:|---:|---:|
| Research report / Introduction | 293 | 34 | 23 | 11 | 0 |
| Academic paper | 204 | 26 | 26 | 0 | 0 |
| Guidebook | 156 | 22 | 20 | 2 | 0 |
| Tutorial/Workshop | 139 | 17 | 3 | 14 | 0 |
| Financial report | 117 | 11 | 11 | 0 | 0 |
| Brochure | 101 | 15 | 13 | 2 | 0 |
| Administration/Industry file | 81 | 10 | 8 | 2 | 0 |
| **Total** | 1091 | 135 | 104 | 31 | 0 |

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

| Class | Questions | Documents | Research report / Introduction | Academic paper | Guidebook | Tutorial/Workshop | Financial report | Brochure | Administration/Industry file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pure-text (Plain-text) | 298 | 100 | 78 | 53 | 44 | 32 | 25 | 23 | 43 |
| Figure | 293 | 96 | 39 | 63 | 62 | 70 | 4 | 41 | 14 |
| (none) | 254 | 112 | 80 | 54 | 39 | 29 | 12 | 24 | 16 |
| Table | 218 | 82 | 34 | 50 | 28 | 15 | 74 | 9 | 8 |
| Chart | 178 | 62 | 123 | 28 | 3 | 13 | 5 | 5 | 1 |
| Generalized-text (Layout) | 119 | 59 | 32 | 11 | 23 | 14 | 0 | 28 | 11 |
| **Total** | 1091 | 135 | — | — | — | — | — | — | — |

### Questions per document


#### Research report / Introduction

| Document | doc_type | Questions |
|---|---|---:|
| germanwingsdigitalcrisisanalysis-150403064828-conversion-gate01_95.pdf | Research report / Introduction | 16 |
| indonesiamobilemarketresearch-ag-150106055934-conversion-gate02_95.pdf | Research report / Introduction | 14 |
| 05-03-18-political-release.pdf | Research report / Introduction | 12 |
| PIP_Seniors-and-Tech-Use_040314.pdf | Research report / Introduction | 12 |
| earthlinkweb-150213112111-conversion-gate02_95.pdf | Research report / Introduction | 12 |
| ecommerceopportunityindia-141124010546-conversion-gate01_95.pdf | Research report / Introduction | 12 |
| PG_2020.05.21_International-Cooperation-COVID_FINAL.pdf | Research report / Introduction | 11 |
| PP_2019.01.17_Trump-economy_FINAL2.pdf | Research report / Introduction | 10 |
| dr-vorapptchapter1emissionsources-121120210508-phpapp02_95.pdf | Research report / Introduction | 10 |
| finalmediafindingspdf-141228031149-conversion-gate02_95.pdf | Research report / Introduction | 10 |
| reportq32015-151009093138-lva1-app6891_95.pdf | Research report / Introduction | 10 |
| 698bba535087fa9a7f9009e172a7f763.pdf | Research report / Introduction | 9 |
| 8e7c4cb542ad160f80fb3d795ada35d8.pdf | Research report / Introduction | 9 |
| earlybird-110722143746-phpapp02_95.pdf | Research report / Introduction | 9 |
| nielsen2015musicbizpresentation-final-150526143534-lva1-app6891_95.pdf | Research report / Introduction | 9 |
| 11-21-16-Updated-Post-Election-Release.pdf | Research report / Introduction | 8 |
| PI_2018.11.19_algorithms_FINAL.pdf | Research report / Introduction | 8 |
| PP_2020.08.06_COVID-19-Restrictions_FINAL-1.pdf | Research report / Introduction | 8 |
| PS_2018.01.09_STEM_FINAL.pdf | Research report / Introduction | 8 |
| asdaaburson-marstellerarabyouthsurvey2014-140407100615-phpapp01_95.pdf | Research report / Introduction | 8 |
| PG_2020.03.09_US-Germany_FINAL.pdf | Research report / Introduction | 7 |
| PG_2021.03.04_US-Views-on-China_FINAL.pdf | Research report / Introduction | 7 |
| PH_2016.06.08_Economy-Final.pdf | Research report / Introduction | 7 |
| PI_2017.10.04_Automation_FINAL.pdf | Research report / Introduction | 7 |
| caltraincapacitymountainview1-150701205750-lva1-app6891_95.pdf | Research report / Introduction | 7 |
| efd88e41c5f2606c57929cac6c1c0605.pdf | Research report / Introduction | 7 |
| fdac8d1e9ef56519371df7e6532df27d.pdf | Research report / Introduction | 7 |
| 12-15-15-ISIS-and-terrorism-release-final.pdf | Research report / Introduction | 6 |
| 379f44022bb27aa53efd5d322c7b57bf.pdf | Research report / Introduction | 6 |
| Independents-Report.pdf | Research report / Introduction | 6 |
| PG_20.07.30_U.S.-Views-China_final.pdf | Research report / Introduction | 6 |
| Pew-Research-Center_Hispanic-Identity-Report_12.20.2017.pdf | Research report / Introduction | 6 |
| PRE_2022.09.29_NSL-politics_REPORT.pdf | Research report / Introduction | 5 |
| PP_2021.04.22_voting-access_REPORT.pdf | Research report / Introduction | 4 |

#### Academic paper

| Document | doc_type | Questions |
|---|---|---:|
| 2311.16502v3.pdf | Academic paper | 16 |
| 2023.acl-long.386.pdf | Academic paper | 9 |
| 2210.02442v1.pdf | Academic paper | 9 |
| 2306.05425v1.pdf | Academic paper | 9 |
| RAR.pdf | Academic paper | 9 |
| 2023.findings-emnlp.248.pdf | Academic paper | 8 |
| 2303.05039v2.pdf | Academic paper | 8 |
| 2305.13186v3.pdf | Academic paper | 8 |
| 2307.09288v2.pdf | Academic paper | 8 |
| 2310.07609v1.pdf | Academic paper | 8 |
| 2312.09390v1.pdf | Academic paper | 8 |
| 2405.09818v1.pdf | Academic paper | 8 |
| STEPBACK.pdf | Academic paper | 8 |
| SnapNTell.pdf | Academic paper | 8 |
| f1f5242528411b262be447e61e2eb10f.pdf | Academic paper | 8 |
| tacl_a_00660.pdf | Academic paper | 8 |
| 2303.08559v2.pdf | Academic paper | 7 |
| 2310.05634v2.pdf | Academic paper | 7 |
| 2312.04350v3.pdf | Academic paper | 7 |
| 2312.10997v5.pdf | Academic paper | 7 |
| 2401.18059v1.pdf | Academic paper | 7 |
| 2005.12872v3.pdf | Academic paper | 6 |
| 2305.14160v4.pdf | Academic paper | 6 |
| 2309.17421v2.pdf | Academic paper | 6 |
| fd76bbefe469561966e5387aa709c482.pdf | Academic paper | 6 |
| 2310.09158v1.pdf | Academic paper | 5 |

#### Guidebook

| Document | doc_type | Questions |
|---|---|---:|
| 8dfc21ec151fb9d3578fc32d5c4e5df9.pdf | Guidebook | 12 |
| User_Manual_1500S_Classic_EN.pdf | Guidebook | 10 |
| mi_phone.pdf | Guidebook | 9 |
| san-francisco-11-contents.pdf | Guidebook | 9 |
| t480_ug_en.pdf | Guidebook | 9 |
| 91521110100M_4K_UHD_Display_User_Manual_V1.1.pdf | Guidebook | 8 |
| obs-productdesc-en.pdf | Guidebook | 8 |
| stereo_headset.pdf | Guidebook | 8 |
| Macbook_air.pdf | Guidebook | 7 |
| NUS-FASS-Graduate-Guidebook-2021-small.pdf | Guidebook | 7 |
| Sinopolis-Chengdu.pdf | Guidebook | 7 |
| bdf54dxa.pdf | Guidebook | 7 |
| DSA-278777.pdf | Guidebook | 6 |
| Guide-for-international-students-web.pdf | Guidebook | 6 |
| StudentSupport_Guidebook.pdf | Guidebook | 6 |
| guojixueshengshenghuozhinanyingwen9.1.pdf | Guidebook | 6 |
| honor_watch_gs_pro.pdf | Guidebook | 6 |
| SAO-StudentSupport_Guidebook-Content.pdf | Guidebook | 5 |
| mmdetection-readthedocs-io-en-v2.18.0.pdf | Guidebook | 5 |
| nova_y70.pdf | Guidebook | 5 |
| owners-manual-2170416.pdf | Guidebook | 5 |
| watch_d.pdf | Guidebook | 5 |

#### Tutorial/Workshop

| Document | doc_type | Questions |
|---|---|---:|
| catvsdogdlpycon15se-150512122612-lva1-app6891_95.pdf | Tutorial/Workshop | 12 |
| measuringsuccessonfacebooktwitterlinkedin-160317142140_95.pdf | Tutorial/Workshop | 12 |
| chapter8-geneticscompatibilitymode-141214140247-conversion-gate02_95.pdf | Tutorial/Workshop | 11 |
| c31e6580d0175ab3f9d99d1ff0bfa000.pdf | Tutorial/Workshop | 10 |
| ddoseattle-150627210357-lva1-app6891_95.pdf | Tutorial/Workshop | 10 |
| efis-140411041451-phpapp01_95.pdf | Tutorial/Workshop | 10 |
| bigdatatrends-120723191058-phpapp02_95.pdf | Tutorial/Workshop | 9 |
| digitalmeasurementframework22feb2011v6novideo-110221233835-phpapp01_95.pdf | Tutorial/Workshop | 9 |
| amb-siteaudits-ds15-150204174043-conversion-gate01_95.pdf | Tutorial/Workshop | 8 |
| bariumswallowpresentation-090810084400-phpapp01_95.pdf | Tutorial/Workshop | 8 |
| 0e94b4197b10096b1f4c699701570fbf.pdf | Tutorial/Workshop | 7 |
| disciplined-agile-business-analysis-160218012713_95.pdf | Tutorial/Workshop | 7 |
| formwork-150318073913-conversion-gate01_95.pdf | Tutorial/Workshop | 7 |
| 52b3137455e7ca4df65021a200aef724.pdf | Tutorial/Workshop | 6 |
| competitiveoutcomes-091006065143-phpapp01_95.pdf | Tutorial/Workshop | 5 |
| avalaunchpresentationsthatkickasteriskv3copy-150318114804-conversion-gate01_95.pdf | Tutorial/Workshop | 4 |
| b3m5kaeqm2w8n4bwcesw-140602121350-phpapp02_95.pdf | Tutorial/Workshop | 4 |

#### Financial report

| Document | doc_type | Questions |
|---|---|---:|
| BESTBUY_2023_10K.pdf | Financial report | 17 |
| afe620b9beac86c1027b96d31d396407.pdf | Financial report | 17 |
| COSTCO_2021_10K.pdf | Financial report | 16 |
| AMAZON_2017_10K.pdf | Financial report | 15 |
| NETFLIX_2015_10K.pdf | Financial report | 12 |
| NIKE_2021_10K.pdf | Financial report | 10 |
| f86d073b0d735ac873a65d906ba82758.pdf | Financial report | 9 |
| q1-2023-bilibili-inc-investor-presentation.pdf | Financial report | 8 |
| ACTIVISIONBLIZZARD_2019_10K.pdf | Financial report | 5 |
| 3M_2018_10K.pdf | Financial report | 4 |
| ADOBE_2015_10K.pdf | Financial report | 4 |

#### Brochure

| Document | doc_type | Questions |
|---|---|---:|
| Campaign_038_Introducing_AC_Whitepaper_v5e.pdf | Brochure | 8 |
| GPL-Graduate-Studies-Professional-Learning-Brochure-Jul-2021.pdf | Brochure | 8 |
| NYU_graduate.pdf | Brochure | 8 |
| camry_ebrochure.pdf | Brochure | 8 |
| finalpresentationdeck-whatwhyhowofcertificationsocial-160324220748_95.pdf | Brochure | 8 |
| PWC_opportunity_of_lifetime.pdf | Brochure | 7 |
| csewt7zsecmmbzjufbyx-signature-24d91a254426c21c3079384270e1f138dc43a271cfe15d6d520d68205855b2a3-poli-150306115347-conversion-gate01_95.pdf | Brochure | 7 |
| welcome-to-nus.pdf | Brochure | 7 |
| 2024.ug.eprospectus.pdf | Brochure | 6 |
| BRO-GL-MMONEY.pdf | Brochure | 6 |
| Bergen-Brochure-en-2022-23.pdf | Brochure | 6 |
| ISEP_student_handbook_2020.pdf | Brochure | 6 |
| NUS-Business-School-BBA-Brochure-2024.pdf | Brochure | 6 |
| transform-software-delivery-with-valueedge-brochure.pdf | Brochure | 6 |
| 2021-Apple-Catalog.pdf | Brochure | 4 |

#### Administration/Industry file

| Document | doc_type | Questions |
|---|---|---:|
| e79deb02a0c0e87511080836c5d4347b.pdf | Administration/Industry file | 18 |
| a5879805d70c854ea4361e43a84e3bb2.pdf | Administration/Industry file | 10 |
| f8d3a162ab9507e021d83dd109118b60.pdf | Administration/Industry file | 10 |
| 3276a5b991c49cf5f9a4af0f7d6fce67.pdf | Administration/Industry file | 8 |
| 0b85477387a9d0cc33fca0f4becaa0e5.pdf | Administration/Industry file | 7 |
| 936c0e2c2e6c8e0c07c51bfaf7fd0a83.pdf | Administration/Industry file | 7 |
| e639029d16094ea71d964e2fb953952b.pdf | Administration/Industry file | 7 |
| 7c3f6204b3241f142f0f8eb8e1fefe7a.pdf | Administration/Industry file | 6 |
| edb88a99670417f64a6b719646aed326.pdf | Administration/Industry file | 5 |
| a4f3ced0696009fec3179f493e4f28c4.pdf | Administration/Industry file | 3 |

### Derived hop — page count of `evidence_pages`

| Class | Questions | Documents | Research report / Introduction | Academic paper | Guidebook | Tutorial/Workshop | Financial report | Brochure | Administration/Industry file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single (1 page) | 485 | 132 | 109 | 85 | 73 | 62 | 63 | 49 | 44 |
| multi (2 pages) | 247 | 110 | 70 | 48 | 32 | 27 | 41 | 17 | 12 |
| zero (0 pages) | 246 | 112 | 79 | 51 | 36 | 30 | 10 | 24 | 16 |
| multi (3 pages) | 40 | 35 | 10 | 8 | 7 | 9 | 1 | 4 | 1 |
| multi (4 pages) | 22 | 20 | 4 | 4 | 3 | 3 | 2 | 4 | 2 |
| multi (5 pages) | 14 | 13 | 4 | 4 | 2 | 1 | 0 | 1 | 2 |
| multi (6 pages) | 12 | 11 | 5 | 2 | 1 | 1 | 0 | 2 | 1 |
| multi (10 pages) | 6 | 6 | 2 | 1 | 0 | 2 | 0 | 0 | 1 |
| multi (7 pages) | 5 | 5 | 4 | 0 | 0 | 1 | 0 | 0 | 0 |
| multi (8 pages) | 5 | 5 | 2 | 0 | 0 | 3 | 0 | 0 | 0 |
| multi (13 pages) | 2 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| multi (9 pages) | 2 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| multi (12 pages) | 2 | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| multi (21 pages) | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| multi (15 pages) | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| multi (24 pages) | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

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

