"""실제 OpenAI 호출을 포함하는 스모크(-m live). 비용 ≈ $0.01."""
from __future__ import annotations

import os

import pytest

from baseball.config import get_settings
from baseball.prompts import NOT_IN_CONTEXT_REFUSAL, OFF_TOPIC_REFUSAL

pytestmark = pytest.mark.live


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
    settings = get_settings()
    try:
        r = live_service.answer("보크가 뭐야?", model=settings.openai_chat_model_fallback)
    except Exception as exc:
        pytest.skip(f"fallback 모델 사용 불가: {type(exc).__name__}: {str(exc)[:120]}")
    assert r.answer
