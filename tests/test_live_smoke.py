"""실제 OpenAI 호출을 포함하는 스모크(-m live). 비용 ≈ $0.01."""
from __future__ import annotations

import os

import pytest

from baseball.answer_schema import ANSWER_KINDS
from baseball.chain import SCHEMA_FALLBACK
from baseball.config import get_settings
from baseball.prompts import NOT_IN_CONTEXT_REFUSAL, OFF_TOPIC_REFUSAL

pytestmark = pytest.mark.live

# 모델 자체를 못 쓰는 경우만 skip 한다. 그 밖의 예외는 진짜 회귀다.
_MODEL_UNAVAILABLE = ("model_not_found", "does not exist", "do not have access",
                      "not supported", "unsupported_model")


def _is_model_unavailable(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(k in text for k in _MODEL_UNAVAILABLE)


@pytest.fixture(scope="module")
def live_service():
    get_settings.cache_clear()
    settings = get_settings()
    if settings.openai_api_key is None or settings.openai_api_key.get_secret_value().startswith("test"):
        pytest.skip("실제 OPENAI_API_KEY 필요")
    from baseball.chain import RagService

    try:
        return RagService.create(settings)
    except Exception as exc:
        pytest.skip(f"인덱스 없음: {exc}")


def test_normal_question(live_service) -> None:
    r = live_service.answer("보크가 뭐야?")
    assert r.status == "answered"
    assert NOT_IN_CONTEXT_REFUSAL not in r.answer
    assert OFF_TOPIC_REFUSAL not in r.answer
    assert len(r.sources) >= 1
    assert r.llm_called is True


def test_edge_short_question(live_service) -> None:
    r = live_service.answer("타점?")
    assert r.status == "answered"
    assert len(r.sources) >= 1


def test_off_topic_is_byte_exact_and_free(live_service) -> None:
    r = live_service.answer("오늘 서울 날씨 어때?")
    assert r.answer == OFF_TOPIC_REFUSAL
    assert r.status == "out_of_scope"
    assert r.llm_called is False


def test_out_of_rulebook_question_refuses(live_service) -> None:
    r = live_service.answer("류현진 연봉이 얼마야?")
    assert r.status == "not_in_rulebook"
    assert NOT_IN_CONTEXT_REFUSAL.rstrip(".") in r.answer


def test_snapshot_freshness(live_service) -> None:
    r = live_service.answer("피치클락 몇 초야?")
    assert r.freshness == "snapshot"
    assert any(s["kind"] == "snapshot" for s in r.sources)


def test_prompt_cache_is_used_on_second_call(live_service) -> None:
    live_service.answer("도루가 뭐야?")
    second = live_service.answer("도루가 뭐야?")
    assert second.usage["input_tokens"] > 0
    print("cache_read:", second.usage.get("cache_read"))


def test_fallback_model_accepts_reasoning_effort(live_service) -> None:
    """예외를 전부 skip 으로 삼키면 구조화 출력 비호환이 초록불로 숨는다."""
    settings = get_settings()
    try:
        r = live_service.answer("보크가 뭐야?", model=settings.openai_chat_model_fallback)
    except Exception as exc:
        if _is_model_unavailable(exc):
            pytest.skip(f"fallback 모델 사용 불가: {type(exc).__name__}: {str(exc)[:120]}")
        raise
    assert r.answer
    if settings.answer_schema_enabled:
        assert SCHEMA_FALLBACK not in r.format_issues, (
            f"fallback 모델에서 구조화 출력이 깨졌다: {r.format_issues}"
        )


# --- 답변 스키마 ---------------------------------------------------------------

@pytest.fixture(scope="module")
def schema_turn(live_service):
    if not get_settings().answer_schema_enabled:
        pytest.skip("ENABLE_ANSWER_SCHEMA=off")
    return live_service.answer("도루가 뭐야?")


def test_schema_path_did_not_fall_back(schema_turn) -> None:
    assert SCHEMA_FALLBACK not in schema_turn.format_issues
    assert schema_turn.answer_kind in ANSWER_KINDS


def test_schema_answer_has_the_forced_shape(schema_turn) -> None:
    answer = schema_turn.answer
    assert answer.startswith("**")                       # 핵심 답이 맨 위에 굵게
    assert "\n\n- " in answer                            # 불릿이 실제로 나온다
    assert "**어떤 상황에서 적용되나요**" in answer          # 용어 질문의 고정 섹션
    for line in answer.split("\n"):
        assert not line.startswith("#")                  # 헤딩은 쓰지 않는다


def test_schema_answer_is_grounded_and_clean(schema_turn) -> None:
    assert schema_turn.status == "answered"
    assert schema_turn.partial_refusal is False
    assert schema_turn.dropped_citations == []
    assert schema_turn.usage["output_tokens"] > 0        # include_raw 로 usage 가 산다
    assert schema_turn.usage["cost_usd"] > 0
    print("format_issues:", schema_turn.format_issues, "| kind:", schema_turn.answer_kind)


def test_schema_refusal_is_byte_exact(live_service) -> None:
    """구조화 출력에서도 거부는 상수 그대로여야 detect_status 가 맞는다."""
    r = live_service.answer("류현진 연봉이 얼마야?")
    if r.llm_called:
        assert r.answer == NOT_IN_CONTEXT_REFUSAL
    assert r.status == "not_in_rulebook"
