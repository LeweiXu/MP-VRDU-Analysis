"""The page_set condition grammar: codec round-trips, legacy-base passthrough,
and vocabulary rejection."""

import itertools

import pytest

from pipeline.page_rules import (
    DIST_POLICIES,
    GOLD_MODES,
    GOLD_POLICIES,
    NOGOLD_POLICIES,
    PageSetRule,
    encode_base,
    enumeration_skip_reason,
    parse_base,
)


def _rule(**over):
    base = dict(ranking_source="bm25", gold_mode="keep_top", gold_count=1,
                distractor_count=0)
    base.update(over)
    return PageSetRule(**base)


def test_round_trip_every_mode_and_policy():
    for mode, gp, dp, np in itertools.product(
            GOLD_MODES, GOLD_POLICIES.values(), DIST_POLICIES.values(), NOGOLD_POLICIES.values()):
        count = 0 if mode == "all" else 2
        d = 3 if np == "distractors_only" else 1
        rule = PageSetRule(ranking_source="colqwen2.5", gold_mode=mode, gold_count=count,
                           distractor_count=d, on_insufficient_gold=gp,
                           on_insufficient_distractors=dp, on_no_gold=np)
        assert parse_base(encode_base(rule)) == rule


def test_base_contains_no_double_underscore():
    # The condition splits at the first "__"; a pageset base must never contain one.
    for mode in GOLD_MODES:
        rule = _rule(gold_mode=mode, gold_count=0 if mode == "all" else 1,
                     ranking_source="bge-m3")
        assert "__" not in encode_base(rule)


def test_default_policies_omitted_from_base():
    assert encode_base(_rule()) == "pageset:r=bm25:g=keep_top-1:d=0"
    probe = _rule(gold_mode="all", gold_count=0, distractor_count=3, on_no_gold="distractors_only")
    assert encode_base(probe) == "pageset:r=bm25:g=all:d=3:p=xpo"


def test_parse_base_is_none_on_legacy_bases():
    for base in ("oracle", "retrieved_text_k3", "retrieved_joint_k1", "full", "similarity_bm25_k3"):
        assert parse_base(base) is None


def test_parse_base_rejects_malformed():
    for bad in ("pageset:", "pageset:r=bm25", "pageset:r=bm25:g=keep_top:d=1",
                "pageset:r=bm25:g=all:d=1:p=zzz", "pageset:r=bm25:g=all:d=1:x=9"):
        with pytest.raises(ValueError):
            parse_base(bad)


def test_rule_vocab_rejection():
    with pytest.raises(ValueError):
        _rule(gold_mode="keep_middle")
    with pytest.raises(ValueError):
        _rule(gold_mode="all", gold_count=1)  # count must be 0 for all
    with pytest.raises(ValueError):
        _rule(gold_count=0)  # keep_top needs count >= 1
    with pytest.raises(ValueError):
        _rule(distractor_count=-1)
    with pytest.raises(ValueError):
        _rule(on_no_gold="distractors_only", distractor_count=0)
    with pytest.raises(ValueError):
        _rule(ranking_source="has space")


def _question(evidence_pages):
    from schema import Question

    return Question(id="q", doc_id="d", question="?", gold_answer="a",
                    answer_format="str", doc_type="t", evidence_pages=tuple(evidence_pages),
                    evidence_sources=(), hop="", is_unanswerable=False)


def test_enumeration_skips_count_decidable_degenerates():
    # Single-gold question under a drop rule: excluded (top == bottom).
    assert enumeration_skip_reason(_rule(gold_mode="drop_top"), _question([4])) is not None
    # Two-gold under drop 1: fine.
    assert enumeration_skip_reason(_rule(gold_mode="drop_top"), _question([2, 7])) is None
    # keep_top-2 on a single-gold question: excluded.
    assert enumeration_skip_reason(_rule(gold_count=2), _question([4])) is not None
    # No gold pages + exclude: excluded; + distractors_only: runs.
    assert enumeration_skip_reason(_rule(gold_mode="all", gold_count=0), _question([])) is not None
    probe = _rule(gold_mode="all", gold_count=0, distractor_count=3, on_no_gold="distractors_only")
    assert enumeration_skip_reason(probe, _question([])) is None
    # keep_all policy defers to condition time rather than excluding.
    keep = _rule(gold_mode="drop_top", on_insufficient_gold="keep_all")
    assert enumeration_skip_reason(keep, _question([4])) is None
