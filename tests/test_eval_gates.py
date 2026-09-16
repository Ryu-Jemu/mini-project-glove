"""평가 게이트 로직. 실호출 없이 결정적으로 검증한다(결과 파일도 건드리지 않는다)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

import run_eval  # noqa: E402

from baseball.answer_schema import ANSWER_KINDS  # noqa: E402


def _summary(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "cases": 17, "recall_at_6": 1.0, "recall_scored_cases": 11,
        "route_match": 17, "top1_ok": True, "with_llm": True,
        "schema_enabled": True,
        "status_match": 15, "status_total": 15,
        "must_contain_match": 12, "must_contain_total": 12,
        "refusal_exact": 2, "refusal_exact_total": 2,
        "sources_nonempty_ratio": 1.0,
        "answer_kind_match": 15, "answer_kind_total": 15,
        "schema_fallback": 0, "lint_warn": [],
    }
    base.update(kw)
    return base


def test_clean_summary_passes() -> None:
    assert run_eval.check_gates(_summary()) == []


def test_denominators_are_not_hardcoded_to_fifteen() -> None:
    """골든셋에 케이스를 더해도 게이트가 깨지지 않아야 한다."""
    assert run_eval.check_gates(_summary(cases=40, route_match=40)) == []
    failures = run_eval.check_gates(_summary(cases=40, route_match=39))
    assert failures and "39/40" in failures[0]


def test_route_must_be_perfect() -> None:
    assert run_eval.check_gates(_summary(route_match=16))


def test_status_tolerates_two_mismatches() -> None:
    assert run_eval.check_gates(_summary(status_match=13)) == []
    assert run_eval.check_gates(_summary(status_match=12))


def test_must_contain_is_now_a_gate() -> None:
    """예전에는 출력만 하고 게이트가 아니었다 — 포맷 회귀의 최저비용 탐지기."""
    failures = run_eval.check_gates(_summary(must_contain_match=11))
    assert any("must_contain 11/12" in f for f in failures)


def test_schema_fallback_fails_the_run() -> None:
    assert any("schema_fallback" in f for f in run_eval.check_gates(_summary(schema_fallback=1)))


@pytest.mark.parametrize("code", ["REFUSAL_INSIDE_ANSWER", "EMPTY_HEADLINE"])
def test_blocking_lint_fails_the_run(code: str) -> None:
    assert any("lint blocking" in f for f in run_eval.check_gates(_summary(lint_warn=[code])))


def test_warning_lint_does_not_fail_the_run() -> None:
    assert run_eval.check_gates(_summary(lint_warn=["HEDGE", "BULLET_COUNT"])) == []


def test_answer_kind_allows_one_mismatch() -> None:
    assert run_eval.check_gates(_summary(answer_kind_match=14)) == []
    assert run_eval.check_gates(_summary(answer_kind_match=13))


def test_legacy_run_skips_schema_gates() -> None:
    """ENABLE_ANSWER_SCHEMA=off 로 돌린 A/B 대조군이 스키마 게이트에 걸리면 안 된다."""
    legacy = _summary(schema_enabled=False, answer_kind_match=0, answer_kind_total=0,
                      schema_fallback=0, lint_warn=[])
    assert run_eval.check_gates(legacy) == []


def test_judge_gates_only_apply_when_judged() -> None:
    assert run_eval.check_gates(_summary()) == []            # --judge 없이 돌린 경우
    assert run_eval.check_gates(_summary(judge_grounded_avg=4.2,
                                         judge_schema_format_avg=4.1,
                                         judge_grounded_min=4)) == []


def test_judge_thresholds() -> None:
    assert any("judge_grounded_avg" in f
               for f in run_eval.check_gates(_summary(judge_grounded_avg=3.9)))
    assert any("judge_schema_format_avg" in f
               for f in run_eval.check_gates(_summary(judge_schema_format_avg=3.5)))
    assert any("최저" in f for f in run_eval.check_gates(_summary(judge_grounded_min=2)))


def test_readability_is_reported_not_gated() -> None:
    """주관적이고 심판 모델에 의존한다. 게이트로 두면 프롬프트 흔들기를 부른다."""
    assert run_eval.check_gates(_summary(judge_readability_avg=1.0,
                                         judge_conciseness_avg=1.0)) == []


def test_retrieval_only_run_ignores_llm_gates() -> None:
    assert run_eval.check_gates({"cases": 17, "recall_at_6": 1.0, "route_match": 17,
                                 "top1_ok": True, "with_llm": False}) == []


# --- 골든셋 정합성 --------------------------------------------------------------

def _cases() -> list[dict[str, Any]]:
    return yaml.safe_load(
        (ROOT / "evaluation" / "golden_v1.yaml").read_text(encoding="utf-8")
    )["cases"]


def test_every_answerable_case_declares_a_kind() -> None:
    missing = [c["id"] for c in _cases()
               if c["expect_route"] != "off_topic" and not c.get("expect_answer_kind")]
    assert missing == []


def test_declared_kinds_are_valid_and_cover_all_four() -> None:
    declared = {c.get("expect_answer_kind") for c in _cases()} - {None}
    assert declared <= set(ANSWER_KINDS)
    assert declared == set(ANSWER_KINDS), f"커버되지 않은 kind: {set(ANSWER_KINDS) - declared}"


def test_off_topic_cases_declare_no_kind() -> None:
    """게이트 경로는 LLM 을 부르지 않으므로 answer_kind 가 생기지 않는다."""
    assert all(not c.get("expect_answer_kind")
               for c in _cases() if c["expect_route"] == "off_topic")
