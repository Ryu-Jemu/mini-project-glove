from __future__ import annotations

from typing import Any

import pytest

from baseball.latest_info import match_snapshot, route as latest_route
from baseball.router import route


class RecordingLLM:
    """호출 여부를 기록하는 최소 스텁(FakeListChatModel 의 순환 인덱스 함정 회피)."""

    def __init__(self, response: dict[str, Any] | str) -> None:
        self.response = response
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.calls.append(messages)
        return self.response


@pytest.mark.parametrize(
    "question, kind",
    [
        ("피치클락 몇 초야?", "latest"),
        ("2026 KBO 포스트시즌 일정", "latest"),
        ("인필드 플라이가 뭐야", "rule"),
        ("피치클락 규칙 어떻게 되", "mixed"),
        ("타점이 뭐야?", "rule"),
    ],
)
def test_keyword_routes_never_call_llm(question: str, kind: str) -> None:
    llm = RecordingLLM({"kind": "off_topic"})
    r = route(question, llm=llm)
    assert (r.kind, r.by) == (kind, "keyword")
    assert llm.calls == []                     # 키워드 히트 시 LLM 미호출


def test_off_topic_comes_from_llm_only() -> None:
    llm = RecordingLLM({"kind": "off_topic"})
    r = route("오늘 서울 날씨 어때?", llm=llm)
    assert (r.kind, r.by) == ("off_topic", "llm")
    assert len(llm.calls) == 1


def test_unparsable_llm_output_falls_back_to_rule() -> None:
    llm = RecordingLLM("not json")
    r = route("음악 추천해줘", llm=llm)
    assert (r.kind, r.by) == ("rule", "llm")


def test_latest_info_snapshot_and_pending(settings) -> None:
    assert settings.web_search_enabled is False          # conftest 가 ENABLE_WEB_SEARCH=off 로 고정
    freshness, entries = latest_route("피치클락 몇 초야?", settings)
    assert freshness == "snapshot" and entries and entries[0].kind == "snapshot"

    freshness, entries = latest_route("오늘 KT 경기 몇 시야?", settings)
    assert (freshness, entries) == ("phase2_pending", [])

    assert match_snapshot("인필드 플라이가 뭐야") == []


@pytest.mark.parametrize(
    "question",
    [
        "야구 몇 명이서 플레이해?",
        "야구는 몇 명이 하나요?",
        "한 팀에 선수가 몇 명이야?",
        "수비 위치가 몇 개야?",
        "야구 경기 목적이 뭐야?",
        "야구 몇 이닝까지 해?",
    ],
)
def test_basic_rule_questions_go_to_the_rulebook(question: str) -> None:
    """경기의 기본 구조를 묻는 말은 규칙집 질문이다.

    예전에는 RULE_RE 에 이런 어휘가 없어 '야구'라는 도메인 단어 하나만 걸렸고,
    도메인 전용 폴백이 이를 최신정보 경로로 보냈다. 답은 규칙 1.00 에 있다.
    LLM 라우터를 부르지 않고 키워드 단계에서 끝나야 한다.
    """
    llm = RecordingLLM('{"kind":"off_topic"}')
    r = route(question, llm=llm)
    assert (r.kind, r.by) == ("rule", "keyword")
    assert llm.calls == []


@pytest.mark.parametrize(
    "question",
    ["LG 트윈스 홈구장이 어디야?", "오늘 KT 경기 몇 시야?", "두산 베어스 남은 경기 몇 개야?"],
)
def test_kbo_data_questions_still_route_to_latest(question: str) -> None:
    """구단 데이터 질문은 topics∧teams 분기가 먼저 걷어 간다(규칙집으로 새지 않는다)."""
    r = route(question, llm=RecordingLLM('{"kind":"rule"}'))
    assert r.kind == "latest"


def test_domain_only_question_prefers_the_rulebook() -> None:
    """도메인 단어만 있는 질문의 폴백이 latest → rule 로 바뀌었다.

    웹 검색이 규칙집보다 먼저 돌던 시절의 선택을 되돌린 것이다. 지금 rule 은
    '규칙집 → 없으면 웹 → 그래도 없으면 모델 지식' 을 뜻한다.
    """
    r = route("야구 재미있게 보는 법", llm=RecordingLLM('{"kind":"off_topic"}'))
    assert (r.kind, r.by, r.domain_hit) == ("rule", "keyword", True)
