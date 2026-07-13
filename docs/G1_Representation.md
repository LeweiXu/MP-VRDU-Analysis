# Tables

### Headline: cost-ordered ladder accuracy by doc_type (oracle pages)

| doc_type | T | TL | TLV | V | frontier | n |
| --- | --- | --- | --- | --- | --- | --- |
| Academic paper | 39.2 [28.8-48.3] | 37.7 [28.6-47.1] | 42.0 [31.6-51.3] | 31.4 [24.5-38.4] | T | 550 |
| Administration/Industry file | 50.0 [33.3-69.1] | 57.9 [48.4-70.2] | 74.5 [64.7-88.1] | 50.8 [43.9-59.2] | TLV | 233 |
| Brochure | 32.3 [21.9-44.8] | 33.3 [20.9-48.3] | 40.3 [26.8-55.2] | 44.8 [31.1-58.6] | T | 260 |
| Financial report | 50.5 [40.4-59.8] | 64.3 [49.1-75.7] | 72.1 [66.0-79.4] | 36.1 [29.5-43.4] | TL | 353 |
| Guidebook | 38.2 [24.7-52.5] | 38.7 [25.4-52.0] | 53.4 [40.0-66.4] | 49.5 [37.5-61.1] | T | 428 |
| Research report / Introduction | 34.4 [25.0-43.8] | 32.8 [23.0-43.0] | 52.2 [40.5-63.6] | 43.4 [31.7-54.3] | TLV | 495 |
| Tutorial/Workshop | 34.8 [28.6-38.5] | 47.8 [28.6-57.7] | 72.7 [57.1-83.3] | 68.2 [57.1-75.0] | TLV | 90 |

### Parser comparison: TL/TLV accuracy by doc_type

_parser = paddleocrvl_

| doc_type | TL | TLV | n |
| --- | --- | --- | --- |
| Academic paper | 37.7 [28.6-47.1] | 42.0 [31.6-51.3] | 550 |
| Administration/Industry file | 57.9 [48.4-70.2] | 74.5 [64.7-88.1] | 233 |
| Brochure | 33.3 [20.9-48.3] | 40.3 [26.8-55.2] | 260 |
| Financial report | 64.3 [49.1-75.7] | 72.1 [66.0-79.4] | 353 |
| Guidebook | 38.7 [25.4-52.0] | 53.4 [40.0-66.4] | 428 |
| Research report / Introduction | 32.8 [23.0-43.0] | 52.2 [40.5-63.6] | 495 |
| Tutorial/Workshop | 47.8 [28.6-57.7] | 72.7 [57.1-83.3] | 90 |

### Resolution sweep: TLV/V accuracy by doc_type and preset

| doc_type | rung | med | n |
| --- | --- | --- | --- |
| Academic paper | TLV | 42.0 [31.6-51.3] | 119 |
| Academic paper | V | 31.4 [24.5-38.4] | 153 |
| Administration/Industry file | TLV | 74.5 [64.7-88.1] | 55 |
| Administration/Industry file | V | 50.8 [43.9-59.2] | 61 |
| Brochure | TLV | 40.3 [26.8-55.2] | 62 |
| Brochure | V | 44.8 [31.1-58.6] | 67 |
| Financial report | TLV | 72.1 [66.0-79.4] | 68 |
| Financial report | V | 36.1 [29.5-43.4] | 108 |
| Guidebook | TLV | 53.4 [40.0-66.4] | 103 |
| Guidebook | V | 49.5 [37.5-61.1] | 109 |
| Research report / Introduction | TLV | 52.2 [40.5-63.6] | 113 |
| Research report / Introduction | V | 43.4 [31.7-54.3] | 129 |
| Tutorial/Workshop | TLV | 72.7 [57.1-83.3] | 22 |
| Tutorial/Workshop | V | 68.2 [57.1-75.0] | 22 |

### Scale: accuracy vs VRAM/latency across reasoner specs

| model_spec | T | TL | TLV | V | peak_vram_mb | latency_ms | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3vl-8b-local | 40.1 [35.2-45.0] | 41.9 [36.5-47.0] | 54.4 [49.0-59.5] | 42.1 [37.8-46.7] | 14240 | 19752 | 2409 |

### Composition: accuracy by evidence source and rung (appendix)

| evidence_source | T | TL | TLV | V | n |
| --- | --- | --- | --- | --- | --- |
| (none) | 55.6 [33.3-75.0] | 68.8 [44.4-88.2] | 68.8 [41.2-93.3] | 44.4 [23.5-62.5] | 68 |
| Chart | 25.2 [18.1-32.6] | 24.4 [16.9-33.3] | 44.8 [34.8-54.5] | 42.2 [32.4-51.9] | 498 |
| Figure | 19.5 [12.4-26.4] | 17.9 [12.4-23.7] | 37.6 [29.7-45.3] | 33.2 [26.2-39.9] | 721 |
| Generalized-text (Layout) | 38.3 [26.5-50.5] | 43.4 [32.0-55.1] | 50.7 [39.4-62.5] | 47.6 [38.1-57.4] | 306 |
| Pure-text (Plain-text) | 48.9 [40.0-57.8] | 52.3 [43.3-59.9] | 59.0 [50.0-67.9] | 43.9 [36.3-51.9] | 867 |
| Table | 47.0 [39.6-53.7] | 57.0 [47.7-65.4] | 57.9 [49.1-65.6] | 32.8 [26.6-38.9] | 620 |

### Routing policies: accuracy vs latency

_assembled from G1 ladder rows + G3 classifier price_

| policy | accuracy | latency_ms | note |
| --- | --- | --- | --- |
| uniform_cheapest_T | 40.1 | 8336 |  |
| uniform_strongest_TLV | 54.4 | 29450 |  |
| oracle_routing | 48.4 | 15198 | per-doc_type frontier rung |
| predicted_routing | 48.4 | 15198 | oracle rung choice + classifier latency |
