"""The page_set condition grammar: a declarative rule over a question's gold and
non-gold pages, encoded into (and parsed back out of) the cell's condition base."""

from __future__ import annotations

from dataclasses import dataclass

from schema import Question

# Condition-base grammar (the base never contains "__", so the existing
# `<base>__<prompt_mode>` split keeps working):
#
#   base      ::= "pageset:r=" ranker ":g=" gold ":d=" count [":p=" policies]
#   gold      ::= "all" | mode "-" count
#   mode      ::= keep_top | keep_bottom | drop_top | drop_bottom
#   policies  ::= three chars, omitted when the default "xpx":
#                 gold-policy   x=exclude | k=keep_all
#                 dist-policy   p=pad_available | x=exclude
#                 nogold-policy x=exclude | o=distractors_only
#
# Examples: pageset:r=colqwen3:g=drop_top-1:d=0__none
#           pageset:r=bm25:g=all:d=3:p=xpo__abstain   (fabrication probe)
#
# The ranking source is part of the condition, so the same pages under two
# rankers are two cells: a shared row would be attributable to neither ranker,
# and the selection tables group on exactly these recorded fields.

PREFIX = "pageset:"
GOLD_MODES = ("all", "keep_top", "keep_bottom", "drop_top", "drop_bottom")
GOLD_POLICIES = {"x": "exclude", "k": "keep_all"}
DIST_POLICIES = {"p": "pad_available", "x": "exclude"}
NOGOLD_POLICIES = {"x": "exclude", "o": "distractors_only"}
_DEFAULT_POLICIES = "xpx"


class PageSetRuleError(RuntimeError):
    """A page_set rule could not produce a valid page set for this question.

    Raised at condition time; the driver's failure path records it as an error
    status row, so a rule that cannot be satisfied is data, never a silently
    wrong page set.
    """


@dataclass(frozen=True)
class PageSetRule:
    """One declared page-set construction rule (see the grammar above)."""

    ranking_source: str
    gold_mode: str = "all"
    gold_count: int = 0
    distractor_count: int = 0
    on_insufficient_gold: str = "exclude"
    on_insufficient_distractors: str = "pad_available"
    on_no_gold: str = "exclude"

    def __post_init__(self) -> None:
        if not self.ranking_source or not all(c.isalnum() or c in "-." for c in self.ranking_source):
            raise ValueError(f"ranking_source must be alphanumeric/dash/dot, got {self.ranking_source!r}")
        if self.gold_mode not in GOLD_MODES:
            raise ValueError(f"gold mode must be one of {GOLD_MODES}, got {self.gold_mode!r}")
        if self.gold_mode == "all" and self.gold_count:
            raise ValueError("gold_count must be 0 when gold_mode is 'all'")
        if self.gold_mode != "all" and self.gold_count < 1:
            raise ValueError(f"gold_mode {self.gold_mode!r} needs gold_count >= 1")
        if self.distractor_count < 0:
            raise ValueError(f"distractor_count must be >= 0, got {self.distractor_count}")
        if self.on_insufficient_gold not in GOLD_POLICIES.values():
            raise ValueError(f"on_insufficient_gold must be one of {sorted(GOLD_POLICIES.values())}")
        if self.on_insufficient_distractors not in DIST_POLICIES.values():
            raise ValueError(f"on_insufficient_distractors must be one of {sorted(DIST_POLICIES.values())}")
        if self.on_no_gold not in NOGOLD_POLICIES.values():
            raise ValueError(f"on_no_gold must be one of {sorted(NOGOLD_POLICIES.values())}")
        if self.on_no_gold == "distractors_only" and self.distractor_count < 1:
            raise ValueError("on_no_gold=distractors_only needs distractor_count >= 1")


def _policy_chars(rule: PageSetRule) -> str:
    inv_gold = {v: k for k, v in GOLD_POLICIES.items()}
    inv_dist = {v: k for k, v in DIST_POLICIES.items()}
    inv_nogold = {v: k for k, v in NOGOLD_POLICIES.items()}
    return inv_gold[rule.on_insufficient_gold] + inv_dist[rule.on_insufficient_distractors] + inv_nogold[rule.on_no_gold]


def encode_base(rule: PageSetRule) -> str:
    """The rule as a condition base (round-trips through `parse_base`)."""

    gold = rule.gold_mode if rule.gold_mode == "all" else f"{rule.gold_mode}-{rule.gold_count}"
    base = f"{PREFIX}r={rule.ranking_source}:g={gold}:d={rule.distractor_count}"
    policies = _policy_chars(rule)
    if policies != _DEFAULT_POLICIES:
        base += f":p={policies}"
    return base


def parse_base(base: str) -> PageSetRule | None:
    """The rule a condition base encodes, or None for a non-pageset base."""

    if not str(base).startswith(PREFIX):
        return None
    fields: dict[str, str] = {}
    for part in str(base)[len(PREFIX):].split(":"):
        key, sep, value = part.partition("=")
        if not sep or key in fields:
            raise ValueError(f"malformed pageset base {base!r}")
        fields[key] = value
    if set(fields) - {"r", "g", "d", "p"} or not {"r", "g", "d"} <= set(fields):
        raise ValueError(f"malformed pageset base {base!r}")
    gold = fields["g"]
    if gold == "all":
        gold_mode, gold_count = "all", 0
    else:
        gold_mode, sep, count_text = gold.rpartition("-")
        if not sep:
            raise ValueError(f"malformed gold field in {base!r}")
        gold_count = int(count_text)
    policies = fields.get("p", _DEFAULT_POLICIES)
    if len(policies) != 3 or policies[0] not in GOLD_POLICIES or policies[1] not in DIST_POLICIES \
            or policies[2] not in NOGOLD_POLICIES:
        raise ValueError(f"malformed policies field in {base!r}")
    return PageSetRule(
        ranking_source=fields["r"],
        gold_mode=gold_mode,
        gold_count=gold_count,
        distractor_count=int(fields["d"]),
        on_insufficient_gold=GOLD_POLICIES[policies[0]],
        on_insufficient_distractors=DIST_POLICIES[policies[1]],
        on_no_gold=NOGOLD_POLICIES[policies[2]],
    )


def enumeration_skip_reason(rule: PageSetRule, question: Question) -> str | None:
    """Why this (rule, question) pair should not become a cell, or None to run it.

    Only count-decidable exclusions live here (they need nothing but the gold
    count, so the cell is never enumerated and the skip is logged as policy).
    Ranking-dependent problems surface at condition time as `PageSetRuleError`
    status rows instead.
    """

    gold = len(question.evidence_pages)
    if gold == 0 and rule.on_no_gold == "exclude":
        return "no gold pages (on_no_gold=exclude)"
    if gold > 0 and rule.gold_mode != "all" and rule.on_insufficient_gold == "exclude":
        if rule.gold_mode.startswith("keep_") and gold < rule.gold_count:
            return f"gold pages {gold} < keep count {rule.gold_count}"
        if rule.gold_mode.startswith("drop_") and gold <= rule.gold_count:
            return f"gold pages {gold} <= drop count {rule.gold_count}"
    return None
