from __future__ import annotations

import pytest

from baseball.prompts import (
    NOT_IN_CONTEXT_REFUSAL,
    OFF_TOPIC_REFUSAL,
    detect_status,
    has_partial_refusal,
)

LONG = "인필드 플라이는 내야수가 평범한 수비로 잡을 수 있는 뜬공입니다. " * 5


@pytest.mark.parametrize(
    "answer, expected",
    [
        (OFF_TOPIC_REFUSAL, "out_of_scope"),
        (f'"{OFF_TOPIC_REFUSAL}"', "out_of_scope"),
        (NOT_IN_CONTEXT_REFUSAL[:-1], "not_in_rulebook"),
        (NOT_IN_CONTEXT_REFUSAL + "\n\n야구 규칙에 대해 질문해 주세요.", "not_in_rulebook"),
        (LONG + NOT_IN_CONTEXT_REFUSAL, "answered"),
        (LONG, "answered"),
        (NOT_IN_CONTEXT_REFUSAL.replace(".", "．"), "not_in_rulebook"),
        (NOT_IN_CONTEXT_REFUSAL.replace(" ", "\n"), "not_in_rulebook"),
        ("", "not_in_rulebook"),
    ],
)
def test_detect_status(answer: str, expected: str) -> None:
    assert detect_status(answer) == expected


def test_tail_budget_zero_flips_to_answered() -> None:
    text = NOT_IN_CONTEXT_REFUSAL + "\n\n야구 규칙에 대해 질문해 주세요."
    assert detect_status(text, tail_max_chars=0) == "answered"


def test_partial_refusal() -> None:
    assert has_partial_refusal(LONG + NOT_IN_CONTEXT_REFUSAL) is True
    assert has_partial_refusal(LONG) is False
    assert has_partial_refusal(NOT_IN_CONTEXT_REFUSAL) is False    # 전체 거부는 partial 아님
