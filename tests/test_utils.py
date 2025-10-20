"""Tests for utility helpers."""

from codebot.utils import jaccard_similarity, split_identifiers


def test_split_identifiers_extracts_tokens() -> None:
    text = "Outcome = logit(score_var) + Age_Group"
    tokens = split_identifiers(text)
    assert "Outcome" in tokens
    assert "score_var" in tokens
    assert "Age_Group" in tokens


def test_jaccard_similarity_case_insensitive() -> None:
    a = ["Age", "Sex"]
    b = ["age", "income"]
    assert jaccard_similarity(a, b) == 1 / 3
