"""Tests for repository mining."""

from codebot.ingest_repo import mine_code_ir


def test_mine_code_ir_detects_r_logistic() -> None:
    files = [
        {"path": "analysis.R", "content": "model <- glm(outcome ~ age, family = binomial)"},
    ]
    analyses = mine_code_ir(files)
    assert any(analysis.model_family == "logistic" for analysis in analyses)


def test_mine_code_ir_detects_python_cox() -> None:
    files = [
        {
            "path": "analysis.py",
            "content": "from lifelines import CoxPHFitter\nmodel = CoxPHFitter()\nmodel.fit(df, 'time', 'event')",
        }
    ]
    analyses = mine_code_ir(files)
    assert any(analysis.model_family == "cox" for analysis in analyses)


def test_mine_code_ir_detects_stata_ttest() -> None:
    files = [
        {
            "path": "analysis.do",
            "content": "ttest outcome, by(group)",
        }
    ]
    analyses = mine_code_ir(files)
    assert any(analysis.model_family == "t-test" for analysis in analyses)
