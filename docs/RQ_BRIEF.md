# RQ brief

Self-contained statement of the paper's three research questions, for a coding/analysis
agent that needs to know what each RQ means and why a table belongs to it. This is the
*what and why*, not the *how* — pipeline mechanics live in `CODEBASE_GUIDE.md`, exact
configs in its Part B. Deliberately excludes open-investigation status, prior-work
positioning, and any not-yet-confirmed findings: the agent reports what the data shows,
it does not adjudicate the science.

## Thesis (one line)

Standard MP-VRDU evaluation collapses a chain of distinct failures into a single
accuracy number; by holding a retrieve-then-generate pipeline fixed and swapping one
stage at a time, we decompose the *complete* loss surface of a multi-page **visual**
pipeline and locate where error is actually introduced.

## The three RQs

**RQ1 — Error attribution: where is the loss introduced?**
Across the retrieve-then-generate pipeline, how does answer error divide between
retrieval, representation, and reasoning? Answered by holding the reasoner fixed and
toggling one variable at a time on an oracle-page substrate, so any accuracy change
attributes to the evidence rather than to selection. Four named sub-parts:

- **Acquisition** — accuracy lost when a real parser transcribes the page (swap
  competing parsers on the text rung).
- **Fidelity** — whether selected evidence is still usable once encoded, compared
  across the T/TL/TLV/V representation ladder and read against the evidence source,
  since a parser can drop what a page image preserves. *(Lead sub-finding.)*
- **Integration** — whether the reasoner combines evidence spread across pages
  (accuracy conditioned on the number of gold evidence pages / hop count).
- **Faithfulness** — whether the reasoner abstains when the evidence does not contain
  the answer (measured on the unanswerable questions).

Each failure is charged to the earliest stage at which information was lost, so an
upstream loss is not double-counted as a reasoning error. RQ1 carries roughly half the
paper.

**RQ2 — Deployment feasibility: which representations can actually be run?**
Under a fixed on-premise memory budget, which representations fit, and how do
quantization, model size, and ingestion cost interact with the error profile?
Answered by recording peak memory, prefill latency, and out-of-memory status for every
cell, then reading the feasibility frontier against the accuracy each representation
buys. Reframes the lossy representations RQ1 charges with error as the ones that fit,
rather than a free design choice.

**RQ3 — Recoverable loss: which located losses can inference-time interventions fix?**
Which of the losses RQ1 locates can be repaired by a lever available without
retraining (parser, representation, retrieval modality, image resolution, prompt,
quantization), and which resist? Answered by sweeping each lever and asking which move
the matching loss and which leave it flat. Interventions that fail are reported next to
those that work: a lever that raises retrieval yet leaves accuracy unchanged, or a loss
no inference-time move repairs, points to error that lives in the model rather than the
evidence.

## Cross-cutting lens

One document-type lens — the seven native MMLongBench-Doc classes — runs through all
three RQs. It is a breakdown axis on the tables, not its own RQ.