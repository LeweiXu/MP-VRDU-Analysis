# Tables

### Retrieval accuracy: page P/R/F1 by method (all doc_types)

| retriever | modality | k | P | R | F1 | n |
| --- | --- | --- | --- | --- | --- | --- |
| bge-m3 | text | 1 | 0.338 | 0.255 | 0.279 | 847 |
| bge-m3 | text | 3 | 0.187 | 0.388 | 0.239 | 847 |
| bge-m3 | text | 5 | 0.147 | 0.482 | 0.212 | 847 |
| bge-m3 | text | 7 | 0.120 | 0.532 | 0.184 | 847 |
| bge-m3 | text | 10 | 0.100 | 0.611 | 0.163 | 847 |
| bge-m3|colqwen2.5 | joint | 1 | 0.485 | 0.552 | 0.486 | 847 |
| bge-m3|colqwen2.5 | joint | 3 | 0.234 | 0.728 | 0.332 | 847 |
| bge-m3|colqwen2.5 | joint | 5 | 0.167 | 0.798 | 0.259 | 847 |
| bm25 | text | 1 | 0.295 | 0.222 | 0.242 | 847 |
| bm25 | text | 3 | 0.185 | 0.378 | 0.234 | 847 |
| bm25 | text | 5 | 0.140 | 0.460 | 0.201 | 847 |
| bm25 | text | 7 | 0.114 | 0.508 | 0.175 | 847 |
| bm25 | text | 10 | 0.095 | 0.585 | 0.154 | 847 |
| bm25|colmodernvbert | joint | 1 | 0.442 | 0.512 | 0.445 | 847 |
| bm25|colmodernvbert | joint | 3 | 0.223 | 0.698 | 0.316 | 847 |
| bm25|colmodernvbert | joint | 5 | 0.159 | 0.776 | 0.248 | 847 |
| colmodernvbert | vision | 1 | 0.588 | 0.460 | 0.496 | 847 |
| colmodernvbert | vision | 3 | 0.315 | 0.646 | 0.396 | 847 |
| colmodernvbert | vision | 5 | 0.227 | 0.725 | 0.322 | 847 |
| colmodernvbert | vision | 7 | 0.179 | 0.772 | 0.271 | 847 |
| colmodernvbert | vision | 10 | 0.140 | 0.817 | 0.222 | 847 |
| colqwen2.5 | vision | 1 | 0.633 | 0.488 | 0.528 | 847 |
| colqwen2.5 | vision | 3 | 0.338 | 0.683 | 0.425 | 847 |
| colqwen2.5 | vision | 5 | 0.236 | 0.749 | 0.335 | 847 |
| colqwen2.5 | vision | 7 | 0.185 | 0.792 | 0.279 | 847 |
| colqwen2.5 | vision | 10 | 0.144 | 0.838 | 0.229 | 847 |
| colqwen3 | vision | 1 | 0.702 | 0.541 | 0.586 | 847 |
| colqwen3 | vision | 3 | 0.368 | 0.739 | 0.460 | 847 |
| colqwen3 | vision | 5 | 0.255 | 0.806 | 0.361 | 847 |
| colqwen3 | vision | 7 | 0.199 | 0.839 | 0.299 | 847 |
| colqwen3 | vision | 10 | 0.152 | 0.874 | 0.241 | 847 |
| qwen3-embedding | text | 1 | 0.335 | 0.255 | 0.279 | 847 |
| qwen3-embedding | text | 3 | 0.199 | 0.413 | 0.254 | 847 |
| qwen3-embedding | text | 5 | 0.150 | 0.494 | 0.217 | 847 |
| qwen3-embedding|colqwen3 | joint | 1 | 0.519 | 0.587 | 0.517 | 847 |
| qwen3-embedding|colqwen3 | joint | 3 | 0.255 | 0.779 | 0.360 | 847 |


### Retrieval accuracy: page P/R/F1 by method and doc_type

| retriever | modality | k | doc_type | P | R | F1 | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bge-m3 | text | 1 | Academic paper | 0.364 | 0.276 | 0.301 | 154 |
| bge-m3 | text | 1 | Administration/Industry file | 0.422 | 0.290 | 0.319 | 64 |
| bge-m3 | text | 1 | Brochure | 0.390 | 0.299 | 0.324 | 77 |
| bge-m3 | text | 1 | Financial report | 0.343 | 0.257 | 0.285 | 108 |
| bge-m3 | text | 1 | Guidebook | 0.417 | 0.344 | 0.365 | 120 |
| bge-m3 | text | 1 | Research report / Introduction | 0.330 | 0.239 | 0.267 | 212 |
| bge-m3 | text | 1 | Tutorial/Workshop | 0.143 | 0.112 | 0.121 | 112 |
| bge-m3 | text | 3 | Academic paper | 0.195 | 0.428 | 0.258 | 154 |
| bge-m3 | text | 3 | Administration/Industry file | 0.271 | 0.458 | 0.299 | 64 |
| bge-m3 | text | 3 | Brochure | 0.216 | 0.439 | 0.273 | 77 |
| bge-m3 | text | 3 | Financial report | 0.179 | 0.396 | 0.240 | 108 |
| bge-m3 | text | 3 | Guidebook | 0.225 | 0.496 | 0.297 | 120 |
| bge-m3 | text | 3 | Research report / Introduction | 0.186 | 0.373 | 0.236 | 212 |
| bge-m3 | text | 3 | Tutorial/Workshop | 0.077 | 0.166 | 0.100 | 112 |
| bge-m3 | text | 5 | Academic paper | 0.162 | 0.545 | 0.240 | 154 |
| bge-m3 | text | 5 | Administration/Industry file | 0.216 | 0.579 | 0.273 | 64 |
| bge-m3 | text | 5 | Brochure | 0.164 | 0.558 | 0.240 | 77 |
| bge-m3 | text | 5 | Financial report | 0.137 | 0.468 | 0.206 | 108 |
| bge-m3 | text | 5 | Guidebook | 0.162 | 0.565 | 0.241 | 120 |
| bge-m3 | text | 5 | Research report / Introduction | 0.142 | 0.451 | 0.204 | 212 |
| bge-m3 | text | 5 | Tutorial/Workshop | 0.075 | 0.272 | 0.111 | 112 |
| bge-m3 | text | 7 | Academic paper | 0.130 | 0.591 | 0.204 | 154 |
| bge-m3 | text | 7 | Administration/Industry file | 0.190 | 0.683 | 0.256 | 64 |
| bge-m3 | text | 7 | Brochure | 0.130 | 0.596 | 0.204 | 77 |
| bge-m3 | text | 7 | Financial report | 0.107 | 0.503 | 0.173 | 108 |
| bge-m3 | text | 7 | Guidebook | 0.125 | 0.608 | 0.199 | 120 |
| bge-m3 | text | 7 | Research report / Introduction | 0.121 | 0.511 | 0.184 | 212 |
| bge-m3 | text | 7 | Tutorial/Workshop | 0.063 | 0.307 | 0.097 | 112 |
| bge-m3 | text | 10 | Academic paper | 0.114 | 0.694 | 0.188 | 154 |
| bge-m3 | text | 10 | Administration/Industry file | 0.161 | 0.776 | 0.231 | 64 |
| bge-m3 | text | 10 | Brochure | 0.105 | 0.672 | 0.174 | 77 |
| bge-m3 | text | 10 | Financial report | 0.083 | 0.566 | 0.143 | 108 |
| bge-m3 | text | 10 | Guidebook | 0.099 | 0.668 | 0.167 | 120 |
| bge-m3 | text | 10 | Research report / Introduction | 0.105 | 0.608 | 0.168 | 212 |
| bge-m3 | text | 10 | Tutorial/Workshop | 0.053 | 0.348 | 0.086 | 112 |
| bge-m3|colqwen2.5 | joint | 1 | Academic paper | 0.487 | 0.556 | 0.494 | 154 |
| bge-m3|colqwen2.5 | joint | 1 | Administration/Industry file | 0.539 | 0.554 | 0.500 | 64 |
| bge-m3|colqwen2.5 | joint | 1 | Brochure | 0.513 | 0.572 | 0.508 | 77 |
| bge-m3|colqwen2.5 | joint | 1 | Financial report | 0.380 | 0.465 | 0.397 | 108 |
| bge-m3|colqwen2.5 | joint | 1 | Guidebook | 0.512 | 0.523 | 0.494 | 120 |
| bge-m3|colqwen2.5 | joint | 1 | Research report / Introduction | 0.524 | 0.582 | 0.517 | 212 |
| bge-m3|colqwen2.5 | joint | 1 | Tutorial/Workshop | 0.433 | 0.590 | 0.472 | 112 |
| bge-m3|colqwen2.5 | joint | 3 | Academic paper | 0.244 | 0.760 | 0.355 | 154 |
| bge-m3|colqwen2.5 | joint | 3 | Administration/Industry file | 0.274 | 0.705 | 0.341 | 64 |
| bge-m3|colqwen2.5 | joint | 3 | Brochure | 0.234 | 0.731 | 0.333 | 77 |
| bge-m3|colqwen2.5 | joint | 3 | Financial report | 0.171 | 0.625 | 0.262 | 108 |
| bge-m3|colqwen2.5 | joint | 3 | Guidebook | 0.234 | 0.726 | 0.334 | 120 |
| bge-m3|colqwen2.5 | joint | 3 | Research report / Introduction | 0.248 | 0.732 | 0.348 | 212 |
| bge-m3|colqwen2.5 | joint | 3 | Tutorial/Workshop | 0.228 | 0.789 | 0.333 | 112 |
| bge-m3|colqwen2.5 | joint | 5 | Academic paper | 0.174 | 0.828 | 0.275 | 154 |
| bge-m3|colqwen2.5 | joint | 5 | Administration/Industry file | 0.208 | 0.818 | 0.286 | 64 |
| bge-m3|colqwen2.5 | joint | 5 | Brochure | 0.168 | 0.876 | 0.270 | 77 |
| bge-m3|colqwen2.5 | joint | 5 | Financial report | 0.119 | 0.684 | 0.199 | 108 |
| bge-m3|colqwen2.5 | joint | 5 | Guidebook | 0.161 | 0.780 | 0.252 | 120 |
| bge-m3|colqwen2.5 | joint | 5 | Research report / Introduction | 0.183 | 0.792 | 0.277 | 212 |
| bge-m3|colqwen2.5 | joint | 5 | Tutorial/Workshop | 0.157 | 0.835 | 0.249 | 112 |
| bm25 | text | 1 | Academic paper | 0.299 | 0.206 | 0.234 | 154 |
| bm25 | text | 1 | Administration/Industry file | 0.328 | 0.218 | 0.238 | 64 |
| bm25 | text | 1 | Brochure | 0.351 | 0.272 | 0.295 | 77 |
| bm25 | text | 1 | Financial report | 0.213 | 0.194 | 0.201 | 108 |
| bm25 | text | 1 | Guidebook | 0.417 | 0.332 | 0.357 | 120 |
| bm25 | text | 1 | Research report / Introduction | 0.344 | 0.248 | 0.275 | 212 |
| bm25 | text | 1 | Tutorial/Workshop | 0.089 | 0.073 | 0.077 | 112 |
| bm25 | text | 3 | Academic paper | 0.234 | 0.477 | 0.300 | 154 |
| bm25 | text | 3 | Administration/Industry file | 0.240 | 0.407 | 0.262 | 64 |
| bm25 | text | 3 | Brochure | 0.199 | 0.407 | 0.251 | 77 |
| bm25 | text | 3 | Financial report | 0.130 | 0.333 | 0.183 | 108 |
| bm25 | text | 3 | Guidebook | 0.211 | 0.453 | 0.275 | 120 |
| bm25 | text | 3 | Research report / Introduction | 0.203 | 0.388 | 0.248 | 212 |
| bm25 | text | 3 | Tutorial/Workshop | 0.068 | 0.152 | 0.090 | 112 |
| bm25 | text | 5 | Academic paper | 0.186 | 0.616 | 0.272 | 154 |
| bm25 | text | 5 | Administration/Industry file | 0.200 | 0.544 | 0.253 | 64 |
| bm25 | text | 5 | Brochure | 0.143 | 0.475 | 0.209 | 77 |
| bm25 | text | 5 | Financial report | 0.087 | 0.366 | 0.138 | 108 |
| bm25 | text | 5 | Guidebook | 0.145 | 0.500 | 0.214 | 120 |
| bm25 | text | 5 | Research report / Introduction | 0.152 | 0.459 | 0.211 | 212 |
| bm25 | text | 5 | Tutorial/Workshop | 0.061 | 0.235 | 0.093 | 112 |
| bm25 | text | 7 | Academic paper | 0.147 | 0.676 | 0.232 | 154 |
| bm25 | text | 7 | Administration/Industry file | 0.172 | 0.599 | 0.230 | 64 |
| bm25 | text | 7 | Brochure | 0.122 | 0.558 | 0.192 | 77 |
| bm25 | text | 7 | Financial report | 0.070 | 0.401 | 0.118 | 108 |
| bm25 | text | 7 | Guidebook | 0.111 | 0.528 | 0.175 | 120 |
| bm25 | text | 7 | Research report / Introduction | 0.128 | 0.510 | 0.189 | 212 |
| bm25 | text | 7 | Tutorial/Workshop | 0.050 | 0.264 | 0.081 | 112 |
| bm25 | text | 10 | Academic paper | 0.119 | 0.759 | 0.198 | 154 |
| bm25 | text | 10 | Administration/Industry file | 0.147 | 0.681 | 0.208 | 64 |
| bm25 | text | 10 | Brochure | 0.101 | 0.653 | 0.168 | 77 |
| bm25 | text | 10 | Financial report | 0.058 | 0.468 | 0.102 | 108 |
| bm25 | text | 10 | Guidebook | 0.082 | 0.555 | 0.137 | 120 |
| bm25 | text | 10 | Research report / Introduction | 0.111 | 0.614 | 0.174 | 212 |
| bm25 | text | 10 | Tutorial/Workshop | 0.049 | 0.334 | 0.082 | 112 |
| bm25|colmodernvbert | joint | 1 | Academic paper | 0.435 | 0.511 | 0.443 | 154 |
| bm25|colmodernvbert | joint | 1 | Administration/Industry file | 0.453 | 0.500 | 0.427 | 64 |
| bm25|colmodernvbert | joint | 1 | Brochure | 0.494 | 0.548 | 0.486 | 77 |
| bm25|colmodernvbert | joint | 1 | Financial report | 0.310 | 0.387 | 0.329 | 108 |
| bm25|colmodernvbert | joint | 1 | Guidebook | 0.508 | 0.530 | 0.491 | 120 |
| bm25|colmodernvbert | joint | 1 | Research report / Introduction | 0.491 | 0.556 | 0.488 | 212 |
| bm25|colmodernvbert | joint | 1 | Tutorial/Workshop | 0.371 | 0.513 | 0.408 | 112 |
| bm25|colmodernvbert | joint | 3 | Academic paper | 0.240 | 0.730 | 0.345 | 154 |
| bm25|colmodernvbert | joint | 3 | Administration/Industry file | 0.261 | 0.705 | 0.333 | 64 |
| bm25|colmodernvbert | joint | 3 | Brochure | 0.229 | 0.762 | 0.332 | 77 |
| bm25|colmodernvbert | joint | 3 | Financial report | 0.141 | 0.532 | 0.217 | 108 |
| bm25|colmodernvbert | joint | 3 | Guidebook | 0.216 | 0.680 | 0.309 | 120 |
| bm25|colmodernvbert | joint | 3 | Research report / Introduction | 0.256 | 0.737 | 0.353 | 212 |
| bm25|colmodernvbert | joint | 3 | Tutorial/Workshop | 0.195 | 0.709 | 0.289 | 112 |
| bm25|colmodernvbert | joint | 5 | Academic paper | 0.178 | 0.836 | 0.281 | 154 |
| bm25|colmodernvbert | joint | 5 | Administration/Industry file | 0.209 | 0.821 | 0.288 | 64 |
| bm25|colmodernvbert | joint | 5 | Brochure | 0.160 | 0.823 | 0.255 | 77 |
| bm25|colmodernvbert | joint | 5 | Financial report | 0.090 | 0.565 | 0.151 | 108 |
| bm25|colmodernvbert | joint | 5 | Guidebook | 0.147 | 0.754 | 0.232 | 120 |
| bm25|colmodernvbert | joint | 5 | Research report / Introduction | 0.183 | 0.819 | 0.279 | 212 |
| bm25|colmodernvbert | joint | 5 | Tutorial/Workshop | 0.138 | 0.779 | 0.223 | 112 |
| colmodernvbert | vision | 1 | Academic paper | 0.571 | 0.445 | 0.484 | 154 |
| colmodernvbert | vision | 1 | Administration/Industry file | 0.578 | 0.436 | 0.467 | 64 |
| colmodernvbert | vision | 1 | Brochure | 0.636 | 0.512 | 0.545 | 77 |
| colmodernvbert | vision | 1 | Financial report | 0.407 | 0.350 | 0.368 | 108 |
| colmodernvbert | vision | 1 | Guidebook | 0.600 | 0.475 | 0.511 | 120 |
| colmodernvbert | vision | 1 | Research report / Introduction | 0.637 | 0.490 | 0.531 | 212 |
| colmodernvbert | vision | 1 | Tutorial/Workshop | 0.652 | 0.493 | 0.535 | 112 |
| colmodernvbert | vision | 3 | Academic paper | 0.314 | 0.665 | 0.406 | 154 |
| colmodernvbert | vision | 3 | Administration/Industry file | 0.339 | 0.637 | 0.393 | 64 |
| colmodernvbert | vision | 3 | Brochure | 0.316 | 0.689 | 0.408 | 77 |
| colmodernvbert | vision | 3 | Financial report | 0.194 | 0.458 | 0.264 | 108 |
| colmodernvbert | vision | 3 | Guidebook | 0.308 | 0.646 | 0.394 | 120 |
| colmodernvbert | vision | 3 | Research report / Introduction | 0.355 | 0.692 | 0.438 | 212 |
| colmodernvbert | vision | 3 | Tutorial/Workshop | 0.348 | 0.687 | 0.430 | 112 |
| colmodernvbert | vision | 5 | Academic paper | 0.234 | 0.777 | 0.342 | 154 |
| colmodernvbert | vision | 5 | Administration/Industry file | 0.256 | 0.717 | 0.328 | 64 |
| colmodernvbert | vision | 5 | Brochure | 0.231 | 0.765 | 0.332 | 77 |
| colmodernvbert | vision | 5 | Financial report | 0.130 | 0.509 | 0.201 | 108 |
| colmodernvbert | vision | 5 | Guidebook | 0.215 | 0.729 | 0.312 | 120 |
| colmodernvbert | vision | 5 | Research report / Introduction | 0.258 | 0.771 | 0.357 | 212 |
| colmodernvbert | vision | 5 | Tutorial/Workshop | 0.243 | 0.751 | 0.341 | 112 |
| colmodernvbert | vision | 7 | Academic paper | 0.190 | 0.860 | 0.297 | 154 |
| colmodernvbert | vision | 7 | Administration/Industry file | 0.210 | 0.767 | 0.285 | 64 |
| colmodernvbert | vision | 7 | Brochure | 0.178 | 0.807 | 0.276 | 77 |
| colmodernvbert | vision | 7 | Financial report | 0.101 | 0.544 | 0.166 | 108 |
| colmodernvbert | vision | 7 | Guidebook | 0.163 | 0.759 | 0.253 | 120 |
| colmodernvbert | vision | 7 | Research report / Introduction | 0.207 | 0.819 | 0.304 | 212 |
| colmodernvbert | vision | 7 | Tutorial/Workshop | 0.187 | 0.775 | 0.280 | 112 |
| colmodernvbert | vision | 10 | Academic paper | 0.144 | 0.894 | 0.238 | 154 |
| colmodernvbert | vision | 10 | Administration/Industry file | 0.172 | 0.836 | 0.247 | 64 |
| colmodernvbert | vision | 10 | Brochure | 0.142 | 0.866 | 0.232 | 77 |
| colmodernvbert | vision | 10 | Financial report | 0.079 | 0.600 | 0.137 | 108 |
| colmodernvbert | vision | 10 | Guidebook | 0.127 | 0.802 | 0.206 | 120 |
| colmodernvbert | vision | 10 | Research report / Introduction | 0.160 | 0.855 | 0.249 | 212 |
| colmodernvbert | vision | 10 | Tutorial/Workshop | 0.146 | 0.819 | 0.230 | 112 |
| colqwen2.5 | vision | 1 | Academic paper | 0.610 | 0.459 | 0.505 | 154 |
| colqwen2.5 | vision | 1 | Administration/Industry file | 0.656 | 0.486 | 0.524 | 64 |
| colqwen2.5 | vision | 1 | Brochure | 0.636 | 0.515 | 0.548 | 77 |
| colqwen2.5 | vision | 1 | Financial report | 0.417 | 0.352 | 0.373 | 108 |
| colqwen2.5 | vision | 1 | Guidebook | 0.608 | 0.485 | 0.519 | 120 |
| colqwen2.5 | vision | 1 | Research report / Introduction | 0.717 | 0.540 | 0.589 | 212 |
| colqwen2.5 | vision | 1 | Tutorial/Workshop | 0.723 | 0.546 | 0.592 | 112 |
| colqwen2.5 | vision | 3 | Academic paper | 0.346 | 0.710 | 0.443 | 154 |
| colqwen2.5 | vision | 3 | Administration/Industry file | 0.344 | 0.637 | 0.397 | 64 |
| colqwen2.5 | vision | 3 | Brochure | 0.329 | 0.690 | 0.420 | 77 |
| colqwen2.5 | vision | 3 | Financial report | 0.225 | 0.535 | 0.309 | 108 |
| colqwen2.5 | vision | 3 | Guidebook | 0.328 | 0.695 | 0.421 | 120 |
| colqwen2.5 | vision | 3 | Research report / Introduction | 0.365 | 0.703 | 0.449 | 212 |
| colqwen2.5 | vision | 3 | Tutorial/Workshop | 0.399 | 0.763 | 0.486 | 112 |
| colqwen2.5 | vision | 5 | Academic paper | 0.240 | 0.791 | 0.351 | 154 |
| colqwen2.5 | vision | 5 | Administration/Industry file | 0.256 | 0.712 | 0.329 | 64 |
| colqwen2.5 | vision | 5 | Brochure | 0.234 | 0.809 | 0.345 | 77 |
| colqwen2.5 | vision | 5 | Financial report | 0.141 | 0.553 | 0.220 | 108 |
| colqwen2.5 | vision | 5 | Guidebook | 0.233 | 0.767 | 0.337 | 120 |
| colqwen2.5 | vision | 5 | Research report / Introduction | 0.258 | 0.767 | 0.357 | 212 |
| colqwen2.5 | vision | 5 | Tutorial/Workshop | 0.273 | 0.810 | 0.376 | 112 |
| colqwen2.5 | vision | 7 | Academic paper | 0.187 | 0.850 | 0.293 | 154 |
| colqwen2.5 | vision | 7 | Administration/Industry file | 0.221 | 0.801 | 0.301 | 64 |
| colqwen2.5 | vision | 7 | Brochure | 0.174 | 0.834 | 0.276 | 77 |
| colqwen2.5 | vision | 7 | Financial report | 0.112 | 0.604 | 0.186 | 108 |
| colqwen2.5 | vision | 7 | Guidebook | 0.179 | 0.805 | 0.275 | 120 |
| colqwen2.5 | vision | 7 | Research report / Introduction | 0.205 | 0.803 | 0.300 | 212 |
| colqwen2.5 | vision | 7 | Tutorial/Workshop | 0.208 | 0.827 | 0.306 | 112 |
| colqwen2.5 | vision | 10 | Academic paper | 0.145 | 0.899 | 0.239 | 154 |
| colqwen2.5 | vision | 10 | Administration/Industry file | 0.173 | 0.852 | 0.251 | 64 |
| colqwen2.5 | vision | 10 | Brochure | 0.131 | 0.858 | 0.218 | 77 |
| colqwen2.5 | vision | 10 | Financial report | 0.092 | 0.677 | 0.159 | 108 |
| colqwen2.5 | vision | 10 | Guidebook | 0.134 | 0.826 | 0.216 | 120 |
| colqwen2.5 | vision | 10 | Research report / Introduction | 0.164 | 0.856 | 0.253 | 212 |
| colqwen2.5 | vision | 10 | Tutorial/Workshop | 0.155 | 0.862 | 0.243 | 112 |
| colqwen3 | vision | 1 | Academic paper | 0.682 | 0.520 | 0.567 | 154 |
| colqwen3 | vision | 1 | Administration/Industry file | 0.734 | 0.585 | 0.618 | 64 |
| colqwen3 | vision | 1 | Brochure | 0.727 | 0.587 | 0.627 | 77 |
| colqwen3 | vision | 1 | Financial report | 0.593 | 0.472 | 0.510 | 108 |
| colqwen3 | vision | 1 | Guidebook | 0.683 | 0.555 | 0.593 | 120 |
| colqwen3 | vision | 1 | Research report / Introduction | 0.741 | 0.544 | 0.597 | 212 |
| colqwen3 | vision | 1 | Tutorial/Workshop | 0.750 | 0.558 | 0.609 | 112 |
| colqwen3 | vision | 3 | Academic paper | 0.387 | 0.792 | 0.495 | 154 |
| colqwen3 | vision | 3 | Administration/Industry file | 0.401 | 0.781 | 0.477 | 64 |
| colqwen3 | vision | 3 | Brochure | 0.346 | 0.743 | 0.447 | 77 |
| colqwen3 | vision | 3 | Financial report | 0.272 | 0.623 | 0.367 | 108 |
| colqwen3 | vision | 3 | Guidebook | 0.347 | 0.738 | 0.446 | 120 |
| colqwen3 | vision | 3 | Research report / Introduction | 0.392 | 0.730 | 0.472 | 212 |
| colqwen3 | vision | 3 | Tutorial/Workshop | 0.411 | 0.767 | 0.495 | 112 |
| colqwen3 | vision | 5 | Academic paper | 0.265 | 0.870 | 0.387 | 154 |
| colqwen3 | vision | 5 | Administration/Industry file | 0.287 | 0.832 | 0.375 | 64 |
| colqwen3 | vision | 5 | Brochure | 0.236 | 0.802 | 0.346 | 77 |
| colqwen3 | vision | 5 | Financial report | 0.198 | 0.730 | 0.303 | 108 |
| colqwen3 | vision | 5 | Guidebook | 0.227 | 0.769 | 0.330 | 120 |
| colqwen3 | vision | 5 | Research report / Introduction | 0.285 | 0.818 | 0.388 | 212 |
| colqwen3 | vision | 5 | Tutorial/Workshop | 0.268 | 0.793 | 0.370 | 112 |
| colqwen3 | vision | 7 | Academic paper | 0.201 | 0.897 | 0.314 | 154 |
| colqwen3 | vision | 7 | Administration/Industry file | 0.228 | 0.856 | 0.313 | 64 |
| colqwen3 | vision | 7 | Brochure | 0.197 | 0.877 | 0.304 | 77 |
| colqwen3 | vision | 7 | Financial report | 0.148 | 0.758 | 0.242 | 108 |
| colqwen3 | vision | 7 | Guidebook | 0.176 | 0.802 | 0.271 | 120 |
| colqwen3 | vision | 7 | Research report / Introduction | 0.225 | 0.853 | 0.326 | 212 |
| colqwen3 | vision | 7 | Tutorial/Workshop | 0.207 | 0.818 | 0.304 | 112 |
| colqwen3 | vision | 10 | Academic paper | 0.154 | 0.935 | 0.252 | 154 |
| colqwen3 | vision | 10 | Administration/Industry file | 0.178 | 0.905 | 0.259 | 64 |
| colqwen3 | vision | 10 | Brochure | 0.148 | 0.914 | 0.242 | 77 |
| colqwen3 | vision | 10 | Financial report | 0.113 | 0.809 | 0.195 | 108 |
| colqwen3 | vision | 10 | Guidebook | 0.133 | 0.826 | 0.215 | 120 |
| colqwen3 | vision | 10 | Research report / Introduction | 0.174 | 0.884 | 0.266 | 212 |
| colqwen3 | vision | 10 | Tutorial/Workshop | 0.155 | 0.843 | 0.242 | 112 |
| qwen3-embedding | text | 1 | Academic paper | 0.351 | 0.255 | 0.281 | 154 |
| qwen3-embedding | text | 1 | Administration/Industry file | 0.375 | 0.262 | 0.285 | 64 |
| qwen3-embedding | text | 1 | Brochure | 0.338 | 0.267 | 0.289 | 77 |
| qwen3-embedding | text | 1 | Financial report | 0.481 | 0.356 | 0.398 | 108 |
| qwen3-embedding | text | 1 | Guidebook | 0.400 | 0.327 | 0.349 | 120 |
| qwen3-embedding | text | 1 | Research report / Introduction | 0.325 | 0.247 | 0.272 | 212 |
| qwen3-embedding | text | 1 | Tutorial/Workshop | 0.098 | 0.083 | 0.088 | 112 |
| qwen3-embedding | text | 3 | Academic paper | 0.232 | 0.474 | 0.294 | 154 |
| qwen3-embedding | text | 3 | Administration/Industry file | 0.240 | 0.448 | 0.272 | 64 |
| qwen3-embedding | text | 3 | Brochure | 0.203 | 0.429 | 0.262 | 77 |
| qwen3-embedding | text | 3 | Financial report | 0.253 | 0.528 | 0.332 | 108 |
| qwen3-embedding | text | 3 | Guidebook | 0.225 | 0.482 | 0.290 | 120 |
| qwen3-embedding | text | 3 | Research report / Introduction | 0.192 | 0.397 | 0.247 | 212 |
| qwen3-embedding | text | 3 | Tutorial/Workshop | 0.062 | 0.146 | 0.084 | 112 |
| qwen3-embedding | text | 5 | Academic paper | 0.168 | 0.552 | 0.244 | 154 |
| qwen3-embedding | text | 5 | Administration/Industry file | 0.216 | 0.595 | 0.272 | 64 |
| qwen3-embedding | text | 5 | Brochure | 0.158 | 0.543 | 0.233 | 77 |
| qwen3-embedding | text | 5 | Financial report | 0.178 | 0.609 | 0.269 | 108 |
| qwen3-embedding | text | 5 | Guidebook | 0.167 | 0.557 | 0.242 | 120 |
| qwen3-embedding | text | 5 | Research report / Introduction | 0.141 | 0.457 | 0.203 | 212 |
| qwen3-embedding | text | 5 | Tutorial/Workshop | 0.057 | 0.218 | 0.087 | 112 |
| qwen3-embedding|colqwen3 | joint | 1 | Academic paper | 0.516 | 0.594 | 0.520 | 154 |
| qwen3-embedding|colqwen3 | joint | 1 | Administration/Industry file | 0.555 | 0.588 | 0.516 | 64 |
| qwen3-embedding|colqwen3 | joint | 1 | Brochure | 0.532 | 0.610 | 0.538 | 77 |
| qwen3-embedding|colqwen3 | joint | 1 | Financial report | 0.537 | 0.574 | 0.522 | 108 |
| qwen3-embedding|colqwen3 | joint | 1 | Guidebook | 0.542 | 0.588 | 0.538 | 120 |
| qwen3-embedding|colqwen3 | joint | 1 | Research report / Introduction | 0.533 | 0.584 | 0.521 | 212 |
| qwen3-embedding|colqwen3 | joint | 1 | Tutorial/Workshop | 0.424 | 0.579 | 0.461 | 112 |
| qwen3-embedding|colqwen3 | joint | 3 | Academic paper | 0.277 | 0.834 | 0.395 | 154 |
| qwen3-embedding|colqwen3 | joint | 3 | Administration/Industry file | 0.299 | 0.819 | 0.385 | 64 |
| qwen3-embedding|colqwen3 | joint | 3 | Brochure | 0.239 | 0.765 | 0.346 | 77 |
| qwen3-embedding|colqwen3 | joint | 3 | Financial report | 0.230 | 0.750 | 0.341 | 108 |
| qwen3-embedding|colqwen3 | joint | 3 | Guidebook | 0.241 | 0.747 | 0.343 | 120 |
| qwen3-embedding|colqwen3 | joint | 3 | Research report / Introduction | 0.267 | 0.765 | 0.371 | 212 |
| qwen3-embedding|colqwen3 | joint | 3 | Tutorial/Workshop | 0.225 | 0.776 | 0.327 | 112 |

