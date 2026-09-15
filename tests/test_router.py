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
