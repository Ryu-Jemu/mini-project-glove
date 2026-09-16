"""KBO 주제·구단 태그. 기존 라우팅 계약을 건드리지 않는지 함께 본다."""
from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from baseball.router import kbo_topics, route


@pytest.mark.parametrize(
    "question,topics,teams",
    [
        ("LG 트윈스 순위 알려줘", ("standings",), ("LG",)),
        ("지금 KBO 1위가 어디야?", ("standings",), ()),
        ("LG 남은 경기 몇 개야?", ("schedule",), ("LG",)),
        ("두산 다음 경기 언제야?", ("schedule",), ("OB",)),
        ("기아 연고지가 어디야?", ("team_info",), ("HT",)),
        ("롯데 홈구장 어디야?", ("team_info",), ("LT",)),
        ("엘지 승률이랑 남은 일정", ("standings", "schedule"), ("LG",)),
        ("키움 선수 명단 알려줘", ("roster",), ("WO",)),
        ("인필드 플라이가 뭐야?", (), ()),
        ("보크가 뭐야?", (), ()),
    ],
)
def test_topic_and_team_extraction(question: str, topics: tuple, teams: tuple) -> None:
    got_topics, got_teams = kbo_topics(question)
    assert got_topics == topics
    assert got_teams == teams


@pytest.mark.parametrize(
    "alias,code",
    [("엘지", "LG"), ("LG트윈스", "LG"), ("기아", "HT"), ("KIA", "HT"),
     ("쓱", "SK"), ("케이티", "KT"), ("엔씨", "NC")],
)
def test_team_aliases(alias: str, code: str) -> None:
    _, teams = kbo_topics(f"{alias} 순위")
    assert teams == (code,)


@pytest.mark.parametrize(
    "question,kind,by",
    [
        ("인필드 플라이가 뭐야?", "rule", "keyword"),
        ("피치클락 몇 초야?", "latest", "keyword"),
        ("2026 KBO 포스트시즌 일정", "latest", "keyword"),
        ("피치클락 규칙 어떻게 되", "mixed", "keyword"),
    ],
)
def test_existing_routing_contract_unchanged(question: str, kind: str, by: str) -> None:
    """주제 태그를 더해도 kind 결정은 그대로여야 한다."""
    fake = FakeListChatModel(responses=['{"kind":"off_topic"}'])
    r = route(question, llm=fake)
    assert (r.kind, r.by) == (kind, by)


def test_route_to_dict_keeps_two_keys() -> None:
    """API 응답과 골든셋이 이 두 키에 고정되어 있다."""
    r = route("인필드 플라이가 뭐야?")
    assert r.to_dict() == {"kind": "rule", "by": "keyword"}


def test_off_topic_carries_no_kbo_tags() -> None:
    """거부 경로가 데이터 조회로 새지 않아야 한다."""
    fake = FakeListChatModel(responses=['{"kind":"off_topic"}'])
    r = route("오늘 서울 날씨 어때?", llm=fake)
    assert r.kind == "off_topic"
    assert r.topics == () and r.teams == ()
