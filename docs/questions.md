	1. MMLongBench-Doc does not have enough labels for documents, and the doc_type label does necessarily reflect our text-heavy/in-between/visual-heavy domain. Will need to additionally annotate documents manually. Labels required are:
		- Digital-born vs scanned (text directly extractable vs OCR required)
		- text-heavy, in-between, visual heavy (our 3 domains via human judgement)
		- Any additional while we're at it?
	2. LongDocURL has no doc_type label (or corresponding domain label). Could possibly do manual annotation as well, but has too many documents (396 documents).  We can decide after mmlongbenchdoc annotation to see if we can do held-out replication.
	3. Current results are promising but not exactly surprising, they are just formalizing/confirming previous assumptions. Main headline (representation sufficiency) I feel is quite good, but want to add a second headline, possibly expanding on locating/retrieval as secondary headline.
		- RQ2 can have additional study for top k page vision vs text retrieval accuracy and vision + text retrieval accuracy
	4. Need to incorporate OCR, either as additional column in Table 1 (representation sufficiency), or incorporate into Text column, where we use direct text extraction for digital born, and OCR for scanned.
	5. Currently on 32gb VRAM gpu, using 16 bit quantization on 8B Qwen3-VL, need to use "low" resolution (1/4 original resolution) to work. See @README.md for more details. 
	6. Additional issue of non-answerable questions: should we exclude these from the main experiments, then do a proper hallucination/non-answerable experiment/table?
		- Want to rework RQ3 to be stronger: do proper hallucination study with non-answerable questions
		- Also prompt sensitivity: no prompt vs generic prompt vs hallucination targeted prompt.
	7. 8B model context window: text tokens generally not an issue, but when too many visual tokens/pages, need to truncate text to fit.
	8. Note to self: current generation tasks are somewhat tailored to the corresponding tables/studies. Can rework a bit so that each generation task is focused on collecting as much cheap data as possible (e.g. token count, vram usage, memory usage, latency etc.).
	9. Additional tables: 4 bit vs 8 bit vs 16 bit quantization, low vs medium vs high vs full resolution on vision modality.
