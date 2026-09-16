"""입력 즉시 범위 검증.

두 가지를 동시에 고정한다. 판정이 맞는지, 그리고 그 판정에 돈이 들지 않는지.
사전으로 끝나는 질문에서 LLM 을 부르면 질문마다 지연과 비용이 붙으므로 계약으로 못박는다.
"""
from __future__ import annotations

from typing import Any

import pytest

from baseball import scope


class RecordingLLM:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.calls.append(messages)
        return self.response


@pytest.mark.parametrize(
    "question",
    [
        "야구 몇 명이서 플레이해?",
        "한 팀에 선수가 몇 명이야?",
        "인필드 플라이가 뭐야?",
        "보크가 뭐야?",
        "5.09 알려줘",
        "LG 트윈스 홈구장이 어디야?",
        "KBO 순위 알려줘",
        "오늘 KT 경기 몇 시야?",
        "문현빈 어느 팀이야?",
        "야구 경기 목적이 뭐야?",
        "야구 영화 추천해줘",
        "야구랑 축구 중에 뭐가 재밌어?",
    ],
)
def test_baseball_questions_pass_for_free(question: str, settings) -> None:
    llm = RecordingLLM({"in_scope": False})
    r = scope.check(question, llm=llm, settings=settings)
    assert r.verdict == "in_scope"
    assert llm.calls == [], "사전으로 끝나는 질문에 LLM 을 부르면 안 된다"


@pytest.mark.parametrize(
    "question",
    [
        "오늘 서울 날씨 어때?",
        "파이썬 리스트 정렬하는 법 알려줘",
        "김치찌개 레시피 알려줘",
        "비트코인 시세 알려줘",
        "이 문장 영어로 해줘",
        "다이어트 어떻게 해?",
        "축구 경기 몇 명이서 해?",
        "리그오브레전드 롤드컵 언제야?",
    ],
)
def test_off_topic_is_blocked_for_free(question: str, settings) -> None:
    llm = RecordingLLM({"in_scope": True})
    r = scope.check(question, llm=llm, settings=settings)
    assert r.verdict == "out_of_scope" and r.blocked
    assert llm.calls == []


def test_other_sport_substring_does_not_count_as_baseball(settings) -> None:
    """'리그오브레전드' 안의 '리그' 가 야구 신호로 잡히면 안 된다."""
    r = scope.check("리그오브레전드 롤드컵 언제야?", llm=RecordingLLM({}), settings=settings)
    assert r.verdict == "out_of_scope"


def test_generic_words_alone_cannot_beat_another_sport(settings) -> None:
    """'몇 명'·'경기 방법' 은 어느 종목에나 있는 말이라 종목 신호를 이기지 못한다."""
    assert scope.check("축구 경기 몇 명이서 해?", settings=settings).verdict == "out_of_scope"
    # 야구가 함께 있으면 야구 신호가 이긴다.
    assert scope.check("야구랑 축구 중 뭐가 재밌어?", settings=settings).verdict == "in_scope"


def test_ambiguous_question_escalates_once(settings) -> None:
    llm = RecordingLLM({"in_scope": False})
    r = scope.check("그 사람 어제 잘했어?", llm=llm, settings=settings)
    assert (r.verdict, r.by) == ("out_of_scope", "llm")
    assert len(llm.calls) == 1


def test_escalation_failure_fails_open(settings) -> None:
    """막는 쪽으로 실패하면 멀쩡한 야구 질문이 조용히 사라진다. 통과 쪽으로 실패한다."""

    class Broken:
        def invoke(self, _messages: Any) -> Any:
            raise RuntimeError("timeout")

    r = scope.check("그 사람 어제 잘했어?", llm=Broken(), settings=settings)
    assert (r.verdict, r.by) == ("in_scope", "default")


def test_unparsable_escalation_fails_open(settings) -> None:
    r = scope.check("그 사람 어제 잘했어?", llm=RecordingLLM("쓰레기"), settings=settings)
    assert (r.verdict, r.by) == ("in_scope", "default")


def test_llm_tier_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from baseball.config import Settings

    s = Settings(_env_file=None, scope_gate_llm="off")
    r = scope.check("그 사람 어제 잘했어?", settings=s)
    assert (r.verdict, r.by) == ("in_scope", "default")


def test_gate_can_be_disabled_entirely() -> None:
    from baseball.config import Settings

    s = Settings(_env_file=None, enable_scope_gate="off")
    r = scope.check("오늘 서울 날씨 어때?", settings=s)
    assert (r.verdict, r.by) == ("in_scope", "disabled")


def test_empty_question_does_not_crash(settings) -> None:
    assert scope.check("", llm=RecordingLLM({"in_scope": False}), settings=settings).verdict


def test_lexicon_loads_real_data(settings) -> None:
    names = scope._names()
    assert len(names) > 500                      # 용어집 + 구단 + 선수
    assert "인필드 플라이" in names or "INFIELD FLY" in names


# --------------------------------------------------------------------------- 체인 연결

def test_chain_blocks_before_routing_or_retrieval(settings, fake_retriever) -> None:
    """차단은 라우터·검색·답변 LLM 어느 것도 건드리기 전에 끝나야 한다."""
    from baseball.chain import RagService
    from baseball.prompts import OFF_TOPIC_REFUSAL

    class Boom:
        def invoke(self, *_a: Any, **_k: Any) -> Any:
            raise AssertionError("차단된 질문에서 호출되면 안 된다")

    service = RagService(settings, fake_retriever, llm=Boom(), router_llm=Boom())
    out = service.answer("오늘 서울 날씨 어때?")

    assert out.answer == OFF_TOPIC_REFUSAL      # 거부 문장은 상수 그대로
    assert out.status == "out_of_scope"
    assert out.llm_called is False
    assert out.route == {"kind": "off_topic", "by": "scope"}
    assert out.sources == []


def test_chain_lets_baseball_questions_through(settings, fake_retriever) -> None:
    from baseball.chain import RagService

    from conftest import CANNED_ANSWER, fake_structured_model

    service = RagService(
        settings, fake_retriever,
        llm=fake_structured_model([CANNED_ANSWER]),
        router_llm=RecordingLLM('{"kind":"rule"}'),
    )
    out = service.answer("인필드 플라이가 뭐야?")
    assert out.status == "answered" and out.llm_called is True


# --------------------------------------------------------------------------- 맛집 허용

@pytest.mark.parametrize(
    "question",
    [
        "잠실 근처 맛집",
        "대구 맛집 추천해줘",
        "LG 홈구장 근처 식당 알려줘",
        "야구장 근처 밥집",
        "사직 경기 끝나고 먹을 곳",
        "고척스카이돔 주변 맛집",
    ],
)
def test_venue_anchored_food_questions_pass_for_free(question: str, settings) -> None:
    """경기가 열리는 구장 주변을 묻는 것은 야구를 보러 가는 일의 일부다."""
    llm = RecordingLLM({"in_scope": False})
    r = scope.check(question, llm=llm, settings=settings)
    assert r.verdict == "in_scope"
    assert llm.calls == [], "구장·연고지 이름이 있으면 LLM 없이 통과해야 한다"


@pytest.mark.parametrize(
    "question",
    ["김치찌개 레시피 알려줘", "파스타 만드는 법", "볶음밥 요리 순서", "축구 경기장 맛집"],
)
def test_cooking_and_other_sports_are_still_blocked(question: str, settings) -> None:
    """맛집만 풀었다. 조리법과 타 종목은 그대로 막는다."""
    llm = RecordingLLM({"in_scope": True})
    r = scope.check(question, llm=llm, settings=settings)
    assert r.verdict == "out_of_scope" and r.blocked
    assert llm.calls == []


def test_off_domain_regex_no_longer_lists_restaurants() -> None:
    """정규식이 되돌아가면 구장 맛집이 다시 우연에 기대게 된다."""
    pattern = scope.OFF_DOMAIN_RE.pattern
    for token in ("맛집", "식당", "밥집"):
        assert token not in pattern
    for token in ("요리", "레시피", "김치찌개"):
        assert token in pattern
