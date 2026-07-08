# MP-VRDU results (paper tables)

Generated 2026-07-06 09:02 UTC. Source: `G1_sufficiency, G3_dataset, G5_retrieval`.

Cells are document-level accuracy (%) with a 95% bootstrap CI in [brackets]. Rungs: T = text, TL = text+layout, TLV = text+layout+vision, V = vision. The frontier (cheapest sufficient rung) is in **bold**.

## Table 1. Headline frontier (doc-type bins x representation ladder)

| Bin | n | T | TL | TLV | V | Frontier | Frontier lat (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text_heavy | 99 | **43.4 [35.0, 53.7]** | 44.4 [36.3, 54.3] | 45.5 [37.5, 55.4] | 41.4 [31.5, 50.0] | T | 1.61 |
| in_between | 108 | 41.7 [32.2, 51.6] | **43.5 [33.0, 54.3]** | 56.5 [42.2, 70.1] | 45.4 [31.0, 58.7] | TL | 4.08 |
| visual_heavy | 101 | 36.6 [29.0, 44.4] | **37.6 [28.4, 46.7]** | 44.6 [32.4, 56.5] | 47.5 [36.8, 58.5] | TL | 3.88 |

## Table 2. Analytical breakdown by question type

| Bin | Question type | n | T | TL | TLV | V |
| --- | --- | --- | --- | --- | --- | --- |
| text_heavy | single-hop text | 38 | 86.8 | 89.5 | 86.8 | 86.8 |
| text_heavy | table | 6 | 50.0 | 50.0 | 0.0 | 16.7 |
| text_heavy | chart-figure | 29 | 3.4 | 3.4 | 17.2 | 17.2 |
| text_heavy | multi-hop | 26 | 23.1 | 23.1 | 26.9 | 7.7 |
| in_between | single-hop text | 32 | 90.6 | 87.5 | 93.8 | 84.4 |
| in_between | table | 10 | 30.0 | 40.0 | 30.0 | 40.0 |
| in_between | chart-figure | 24 | 4.2 | 8.3 | 50.0 | 37.5 |
| in_between | multi-hop | 42 | 28.6 | 31.0 | 38.1 | 21.4 |
| visual_heavy | single-hop text | 30 | 83.3 | 83.3 | 83.3 | 83.3 |
| visual_heavy | table | 9 | 33.3 | 33.3 | 44.4 | 44.4 |
| visual_heavy | chart-figure | 34 | 5.9 | 8.8 | 20.6 | 32.4 |
| visual_heavy | multi-hop | 28 | 25.0 | 25.0 | 32.1 | 28.6 |

## Table 3. Family replication (reasoner families)

_Not built yet: its source experiments' generate/judge haven't all finished._

## Table 4. Dataset replication (held-out subset)

| Dataset | Bin | n | T | TL | TLV | V | Frontier | Frontier lat (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmlongbench_heldout | text_heavy | 104 | **41.3 [36.3, 47.3]** | 42.3 [36.1, 50.0] | 42.3 [38.4, 46.6] | 36.5 [30.7, 41.4] | T | 1.84 |
| mmlongbench_heldout | in_between | 105 | **43.8 [33.0, 57.4]** | 43.8 [33.3, 57.9] | 57.1 [42.9, 71.6] | 46.7 [33.1, 60.6] | T | 1.24 |
| mmlongbench_heldout | visual_heavy | 0 |  |  |  |  |  | 0.00 |

## Table 5. Composition and mediation by evidence modality

| Bin | Evidence | Share % | Modality frontier | Bin frontier | Predicted bin frontier | Match |
| --- | --- | --- | --- | --- | --- | --- |
| text_heavy | text | 51.3 | T | T | TLV | no |
| text_heavy | table | 4.5 | T | T | TLV | no |
| text_heavy | chart | 26.3 | TLV | T | TLV | no |
| text_heavy | figure | 11.4 | TLV | T | TLV | no |
| text_heavy | layout | 6.4 | T | T | TLV | no |
| in_between | text | 37.8 | T | TL | TLV | no |
| in_between | table | 21.3 | T | TL | TLV | no |
| in_between | chart | 5.3 | TLV | TL | TLV | no |
| in_between | figure | 30.0 | TLV | TL | TLV | no |
| in_between | layout | 5.6 | T | TL | TLV | no |
| visual_heavy | text | 40.6 | T | TL | TLV | no |
| visual_heavy | table | 6.4 | T | TL | TLV | no |
| visual_heavy | chart | 4.5 | TLV | TL | TLV | no |
| visual_heavy | figure | 31.7 | TLV | TL | TLV | no |
| visual_heavy | layout | 16.8 | T | TL | TLV | no |

## Table 6. Matched vs cross retrieval

| Bin | Pipeline | Retrieval | Accuracy [CI] | Δ vs matched (pts) | Retrieval F1 |
| --- | --- | --- | --- | --- | --- |
| in_between | matched_vision | vision | 26.9 [17.8, 36.3] | +0.0 | 17.5 |
| in_between | cross_text_to_vision | text | 29.6 [21.5, 38.8] | +2.8 | 21.0 |
| visual_heavy | matched_vision | vision | 26.7 [20.0, 33.3] | +0.0 | 16.9 |
| visual_heavy | cross_text_to_vision | text | 25.7 [18.4, 33.3] | -1.0 | 24.5 |

## Table 7. Routing policies

| Policy | Chosen rungs | n | Accuracy [CI] | Total latency (s) |
| --- | --- | --- | --- | --- |
| oracle_routing | T:99;TL:209 | 308 | 41.6 [35.8, 47.1] | 3.22 |
| predicted_routing | T:183;TL:125 | 308 | 40.6 [35.2, 45.8] | 10.16 |
| uniform_cheapest_T | T:308 | 308 | 40.6 [35.3, 45.8] | 1.43 |
| uniform_strongest_TLV | TLV:308 | 308 | 49.0 [42.1, 56.7] | 19.12 |

## Table 8. Scale sanity (2B/4B/8B/32B)

_Not built: scale task (G4) is not implemented._

