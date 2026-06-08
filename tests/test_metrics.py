"""Stage 2 — scoring is correct independent of model quality."""

from mpvrdu.eval.extract import compare, is_abstention, normalize_text
from mpvrdu.eval.metrics import aggregate, recall_at_k, score_answer


def test_str_compare():
    assert compare("42 million dollars", "42 million dollars", "Str")
    assert compare("The answer is Paris", "Paris", "Str")   # substring
    assert not compare("London", "Paris", "Str")


def test_int_float_compare():
    assert compare("5", "5", "Int")
    assert compare("the value is 5 units", "5", "Int")
    assert not compare("6", "5", "Int")
    assert compare("3.14", "3.141", "Float")               # within 1%
    assert not compare("3.5", "3.0", "Float")


def test_list_compare():
    assert compare("apple, banana", "banana, apple", "List")
    assert not compare("apple", "apple, banana", "List")


def test_percent_normalisation():
    assert normalize_text("88%") == normalize_text("88 percent")
    assert compare("88%", "88 percent", "Str")


def test_abstention_detection():
    assert is_abstention("Not answerable")
    assert is_abstention("This cannot be determined from the document")
    assert not is_abstention("42 dollars")


def test_score_answer_unanswerable():
    s = score_answer("Not answerable", "Not answerable", "None",
                     gold_is_unanswerable=True)
    assert s.correct and s.pred_abstained and not s.gold_answerable
    # answering an unanswerable question is wrong
    s2 = score_answer("42", "Not answerable", "None", gold_is_unanswerable=True)
    assert not s2.correct


def test_score_answer_abstain_on_answerable_is_wrong():
    s = score_answer("Not answerable", "42", "Int", gold_is_unanswerable=False)
    assert not s.correct


def test_aggregate_all_correct_vs_all_wrong():
    correct = [score_answer("42", "42", "Int") for _ in range(5)]
    wrong = [score_answer("0", "42", "Int") for _ in range(5)]
    assert aggregate(correct)["accuracy"] == 1.0
    assert aggregate(wrong)["accuracy"] == 0.0


def test_recall_at_k():
    assert recall_at_k([0, 1, 2], [1], k=4) == 1.0
    assert recall_at_k([0, 1, 2, 3], [4], k=4) == 0.0
    assert recall_at_k([0, 5], [0, 5], k=2) == 1.0
    assert recall_at_k([0], [0, 5], k=2) == 0.5
    assert recall_at_k([], [], k=4) == 1.0       # unanswerable: nothing to retrieve
    # top-k truncation matters
    assert recall_at_k([9, 9, 9, 1], [1], k=2) == 0.0
