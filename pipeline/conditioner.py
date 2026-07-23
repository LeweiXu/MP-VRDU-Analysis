"""Selects the pages fed to a cell: oracle, retrieved, similarity, or a
declared page_set rule over gold and non-gold pages."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pipeline.page_rules import PageSetRule, PageSetRuleError
from retrievers import Retriever
from schema import PageSet, Question


class InputConditioner(ABC):
    """Select the pages that reach the model for one question."""

    #: Stable short name used in cell keys and result rows.
    name: str = "conditioner"

    @abstractmethod
    def condition(self, question: Question, page_count: int) -> PageSet:
        """Return the `PageSet` fed to the representation for this question."""


class OracleConditioner(InputConditioner):
    """Feed exactly the gold evidence pages (the reasoning ceiling)."""

    name = "oracle"

    def condition(self, question: Question, page_count: int) -> PageSet:
        pages = tuple(p for p in question.evidence_pages if 0 <= p < page_count)
        if not pages:
            # Questions with no gold pages (native unanswerable) fall back to the
            # first page so the pipeline still has something to render.
            pages = (0,) if page_count else ()
        return PageSet(pages, "oracle")


class RetrievedTopK(InputConditioner):
    """Feed the top-k pages returned by a retriever."""

    def __init__(self, retriever: Retriever, k: int, name: str | None = None) -> None:
        self.retriever = retriever
        self.k = int(k)
        self.name = name or f"retrieved_{getattr(retriever, 'name', 'r')}_k{self.k}"

    def condition(self, question: Question, page_count: int) -> PageSet:
        ranked = self.retriever.retrieve(question, page_count, self.k)
        return PageSet(tuple(ranked)[: self.k], "retrieved", note=f"k={self.k}")


class JointTopK(InputConditioner):
    """Feed the free deduplicated union of two retrievers' top-k page sets.

    Joint retrieval is post-hoc and free (pivot 4.1): it unions two already-ranked
    page sets, no new retrieval and no score fusion (union, not RRF). Each retriever
    is asked for its own top-k, then `union` dedups them keeping first-seen order,
    so the result is at most 2k pages.
    """

    def __init__(self, text: Retriever, vision: Retriever, k: int, name: str | None = None) -> None:
        self.text = text
        self.vision = vision
        self.k = int(k)
        self.name = name or f"retrieved_joint_k{self.k}"

    def condition(self, question: Question, page_count: int) -> PageSet:
        from retrievers.joint import union

        text_pages = self.text.retrieve(question, page_count, self.k)
        vision_pages = self.vision.retrieve(question, page_count, self.k)
        merged = union(text_pages, vision_pages)
        return PageSet(tuple(merged), "retrieved", note=f"joint k={self.k}")


class SimilarityTopK(InputConditioner):
    """Feed a few similarity-retrieved pages for zero-gold-page questions.

    The hallucination study has no oracle arm (unanswerable questions have no
    gold pages), so the only coherent page selection is a small similarity-ranked
    set. Same mechanism as `RetrievedTopK` but tagged `similarity` provenance and
    a fixed small k.
    """

    def __init__(self, retriever: Retriever, k: int = 3, name: str | None = None) -> None:
        self.retriever = retriever
        self.k = int(k)
        self.name = name or f"similarity_{getattr(retriever, 'name', 'r')}_k{self.k}"

    def condition(self, question: Question, page_count: int) -> PageSet:
        ranked = self.retriever.retrieve(question, page_count, self.k)
        return PageSet(tuple(ranked)[: self.k], "similarity", note=f"k={self.k}")


class FullDoc(InputConditioner):
    """Feed every page (the feed-everything long-context baseline)."""

    name = "full"

    def condition(self, question: Question, page_count: int) -> PageSet:
        return PageSet.full(page_count)


class PageSetConditioner(InputConditioner):
    """Build the page set from a declared rule over gold and non-gold pages.

    The ranker's full k-independent ranking orders both pools; the rule keeps or
    drops gold pages by rank and adds top-ranked non-gold distractors. Pages are
    emitted in document order regardless of rank (PageSet sorts), so ordering
    never confounds a selection manipulation. Count-decidable degenerate cases
    are excluded at cell enumeration (`page_rules.enumeration_skip_reason`);
    anything only visible here (short non-gold pool under an exclude policy, an
    empty final set) raises `PageSetRuleError` so the cell records an error
    status row rather than a silently wrong page set.
    """

    def __init__(self, ranker: Retriever, rule: PageSetRule, name: str) -> None:
        self.ranker = ranker
        self.rule = rule
        self.name = name

    def condition(self, question: Question, page_count: int) -> PageSet:
        rule = self.rule
        ranking = list(self.ranker.rank(question, page_count))
        gold_set = {p for p in question.evidence_pages if 0 <= p < page_count}
        ranked_gold = [p for p in ranking if p in gold_set]
        ranked_nongold = [p for p in ranking if p not in gold_set]

        note_bits = [f"rank={rule.ranking_source}"]
        if not gold_set:
            if rule.on_no_gold != "distractors_only":
                raise PageSetRuleError(f"{self.name}: no gold pages and on_no_gold={rule.on_no_gold}")
            selected_gold: list[int] = []
            note_bits.append("gold=none (distractors_only)")
        else:
            selected_gold, gold_note = self._select_gold(ranked_gold)
            note_bits.append(gold_note)

        distractors = ranked_nongold[: rule.distractor_count]
        if len(distractors) < rule.distractor_count:
            if rule.on_insufficient_distractors == "exclude":
                raise PageSetRuleError(
                    f"{self.name}: non-gold pool {len(ranked_nongold)} < distractor count {rule.distractor_count}"
                )
            note_bits.append(f"d={rule.distractor_count} d_actual={len(distractors)} (padded)")
        else:
            note_bits.append(f"d={rule.distractor_count}")

        pages = tuple(selected_gold + distractors)
        if not pages:
            raise PageSetRuleError(f"{self.name}: rule produced an empty page set")
        return PageSet(pages, "constructed", note=" ".join(note_bits))

    def _select_gold(self, ranked_gold: list[int]) -> tuple[list[int], str]:
        """Apply the gold mode/count to the rank-ordered gold pages."""

        rule = self.rule
        mode, count = rule.gold_mode, rule.gold_count
        if mode == "all":
            return list(ranked_gold), f"gold=all({len(ranked_gold)})"
        satisfiable = len(ranked_gold) >= count if mode.startswith("keep_") else len(ranked_gold) > count
        if not satisfiable:
            # Enumeration excludes these under the default policy; reaching here
            # means keep_all (feed everything, noted) or a ranking/corpus drift.
            if rule.on_insufficient_gold == "keep_all":
                return list(ranked_gold), f"gold={mode}-{count} unsatisfiable, kept all {len(ranked_gold)}"
            raise PageSetRuleError(
                f"{self.name}: gold pool {len(ranked_gold)} cannot satisfy {mode}-{count}"
            )
        if mode == "keep_top":
            kept = ranked_gold[:count]
        elif mode == "keep_bottom":
            kept = ranked_gold[-count:]
        elif mode == "drop_top":
            kept = ranked_gold[count:]
        else:  # drop_bottom
            kept = ranked_gold[:-count]
        return list(kept), f"gold={mode}-{count} fed={len(kept)}/{len(ranked_gold)}"
