"""PageSetConditioner behaviour, page_set spec validation, the hop filter, and
the cell-enumeration branch."""

from types import SimpleNamespace

import pytest

from pipeline.conditioner import PageSetConditioner
from pipeline.page_rules import PageSetRule, PageSetRuleError
from schema import Question


class FakeRanker:
    """A retriever with a fixed full ranking (best first)."""

    name = "fake"

    def __init__(self, ranking):
        self.ranking = tuple(ranking)

    def rank(self, question, page_count):
        return self.ranking

    def retrieve(self, question, page_count, k):
        return self.ranking[:k]


def _question(evidence_pages, qid="q1"):
    return Question(id=qid, doc_id="d1", question="?", gold_answer="a",
                    answer_format="str", doc_type="t", evidence_pages=tuple(evidence_pages),
                    evidence_sources=(), hop="", is_unanswerable=False)


def _rule(**over):
    base = dict(ranking_source="fake", gold_mode="all", gold_count=0, distractor_count=0)
    base.update(over)
    return PageSetRule(**base)


# Ranking over 8 pages, best-to-worst. Gold pages {2, 5, 7} rank as 5 > 2 > 7.
RANKING = (5, 1, 2, 0, 7, 3, 6, 4)
GOLD = (2, 5, 7)


def _condition(rule, evidence=GOLD, page_count=8):
    cond = PageSetConditioner(FakeRanker(RANKING), rule, name="pageset_test__none")
    return cond.condition(_question(evidence), page_count)


def test_gold_modes_select_by_rank():
    # keep_top-1 keeps the best-ranked gold page (5).
    assert _condition(_rule(gold_mode="keep_top", gold_count=1)).page_indices == (5,)
    # keep_bottom-1 keeps the worst-ranked gold page (7).
    assert _condition(_rule(gold_mode="keep_bottom", gold_count=1)).page_indices == (7,)
    # drop_top-1 drops 5, keeps {2, 7} in document order.
    assert _condition(_rule(gold_mode="drop_top", gold_count=1)).page_indices == (2, 7)
    # drop_bottom-1 drops 7, keeps {2, 5}.
    assert _condition(_rule(gold_mode="drop_bottom", gold_count=1)).page_indices == (2, 5)
    # all keeps everything.
    assert _condition(_rule()).page_indices == (2, 5, 7)


def test_distractors_are_top_ranked_nongold_in_document_order():
    # Non-gold ranking: 1 > 0 > 3 > 6 > 4; top-2 = {1, 0} -> document order (0, 1).
    got = _condition(_rule(distractor_count=2))
    assert got.page_indices == (0, 1, 2, 5, 7)
    assert got.provenance == "constructed"


def test_pad_available_notes_actual_count():
    # Only 5 non-gold pages exist; asking for 6 pads and records d_actual.
    got = _condition(_rule(distractor_count=6))
    assert got.page_indices == (0, 1, 2, 3, 4, 5, 6, 7)
    assert "d_actual=5" in got.note


def test_exclude_on_short_distractor_pool_raises():
    rule = _rule(distractor_count=6, on_insufficient_distractors="exclude")
    with pytest.raises(PageSetRuleError):
        _condition(rule)


def test_distractors_only_feeds_gold_free_set():
    rule = _rule(distractor_count=3, on_no_gold="distractors_only")
    got = _condition(rule, evidence=())
    # No gold: the whole ranking is non-gold; top-3 = {5, 1, 2} -> (1, 2, 5).
    assert got.page_indices == (1, 2, 5)
    assert "distractors_only" in got.note


def test_keep_all_policy_feeds_all_gold_when_unsatisfiable():
    rule = _rule(gold_mode="drop_top", gold_count=1, on_insufficient_gold="keep_all")
    got = _condition(rule, evidence=(5,))
    assert got.page_indices == (5,)
    assert "kept all" in got.note


def test_unsatisfiable_exclude_raises_at_condition_time():
    rule = _rule(gold_mode="drop_top", gold_count=1)
    with pytest.raises(PageSetRuleError):
        _condition(rule, evidence=(5,))


def test_two_rules_same_pages_are_distinct_cells():
    # drop_top-1 and keep_bottom-2 both select {2, 7} here, but their condition
    # bases differ, so the prediction keys differ: the rule is the cell identity,
    # not just the pages.
    from experiments.engine.paths import prediction_key
    from pipeline.page_rules import encode_base

    a = _rule(gold_mode="drop_top", gold_count=1)
    b = _rule(gold_mode="keep_bottom", gold_count=2)
    assert _condition(a).page_indices == _condition(b).page_indices == (2, 7)
    keys = [prediction_key("q1", "d1", f"{encode_base(r)}__none", "TLV",
                           "qwen3vl-8b-local", (2, 7), "med") for r in (a, b)]
    assert keys[0] != keys[1]


def test_hop_filter():
    from experiments.corpus.resolve import filter_by_hop

    qs = [_question([1], "single"), _question([1, 2], "two"),
          _question([1, 2, 5], "three"), _question([], "none")]
    assert [q.id for q in filter_by_hop(qs, "any")] == ["single", "two", "three", "none"]
    assert [q.id for q in filter_by_hop(qs, "single")] == ["single"]
    assert [q.id for q in filter_by_hop(qs, "multi")] == ["two", "three"]
    # Exact counts: the +k design's blocking factor. yaml ints arrive as str.
    assert [q.id for q in filter_by_hop(qs, "1")] == ["single"]
    assert [q.id for q in filter_by_hop(qs, "2")] == ["two"]
    assert [q.id for q in filter_by_hop(qs, "3")] == ["three"]
    with pytest.raises(ValueError):
        filter_by_hop(qs, "both")
    with pytest.raises(ValueError):
        filter_by_hop(qs, "0")


def test_spec_page_set_validation():
    from experiments.corpus.yaml_spec import SpecError, parse_spec

    def spec(**over):
        raw = {"task_name": "G5_selection",
               "corpus": {"pool": "answerable", "sampling": "full", "hop": "multi"},
               "page_set": {"ranking_source": ["bm25"],
                            "gold": {"mode": "keep_top", "count": 1},
                            "distractor": {"count": [0, 1]}}}
        raw.update(over)
        return parse_spec(raw)

    parsed = spec()
    assert parsed.page_set["ranking_source"] == ("bm25",)
    assert parsed.page_set["distractor"]["count"] == (0, 1)
    # unknown corpus key now raises (the silent hole is closed)
    with pytest.raises(SpecError):
        spec(corpus={"pool": "answerable", "hopp": "multi"})
    # unknown page_set key
    with pytest.raises(SpecError):
        spec(page_set={"ranking_source": ["bm25"], "bogus": 1})
    # gold-removal rule without hop: multi
    with pytest.raises(SpecError):
        spec(corpus={"pool": "answerable", "hop": "any"})
    # distractors_only with max distractor count 0
    with pytest.raises(SpecError):
        spec(page_set={"ranking_source": ["bm25"], "gold": {"mode": "all"},
                       "distractor": {"count": 0}, "on_no_gold": "distractors_only"})
    # page_set: none reproduces today's behaviour
    assert spec(page_set="none").page_set is None


def test_g5_specs_parse_and_enumerate():
    from pathlib import Path

    from experiments.corpus.yaml_spec import config_from_spec, load_yaml_specs
    from experiments.tasks.base import Retrievers
    from experiments.tasks.task import Task

    root = Path(__file__).resolve().parents[1]
    specs = load_yaml_specs(root / "ops" / "specs" / "g2_sufficiency.yaml")
    assert [s.run_tag for s in specs] == ["g5a-drop-best", "g5a-drop-worst",
                                         "g5a-keep-best", "g5a-keep-worst"]
    config = config_from_spec(specs[0])
    assert config.hop_filter == "multi"

    retrievers = Retrievers(text=SimpleNamespace(), vision=SimpleNamespace(),
                            rankers={"colqwen3": FakeRanker(RANKING), "bm25": FakeRanker(RANKING)})
    questions = [_question([2, 5], "qa"), _question([4], "qb")]  # qb: single gold -> excluded
    cells = Task("G5_selection").generation_cells(config, questions, retrievers=retrievers)
    # qa x 2 rankers x 4 rungs x 1 mode = 8 cells; qb excluded under drop_top-1.
    assert len(cells) == 8
    assert all(c.conditioner.name.startswith("pageset:") for c in cells)
    assert {c.conditioner.rule.ranking_source for c in cells} == {"colqwen3", "bm25"}

    robustness = load_yaml_specs(root / "ops" / "specs" / "g2_robustness.yaml")
    assert [s.run_tag for s in robustness] == ["g5b-gold1", "g5b-gold2", "g5b-gold3"]
    config_b = config_from_spec(robustness[0])
    cells_b = Task("G5_selection").generation_cells(config_b, [questions[0]], retrievers=retrievers)
    # rankers x distractor counts x rungs for one multi-gold question, derived
    # from the spec so a scope change there does not break this test.
    expected = (len(config_b.page_set["ranking_source"])
                * len(config_b.page_set["distractor"]["count"])
                * len(config_b.representations))
    assert len(cells_b) == expected
