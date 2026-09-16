"""도구 목록과 게이트. 무엇이 모델에게 붙는지를 고정한다."""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from baseball import tools
from baseball.chain import RagService
from baseball.config import Settings

from conftest import FakeRetriever


def _settings(**kw: Any) -> Settings:
    base = {"_env_file": None, "openai_api_key": SecretStr("test-key"),
            "enable_kbo_data": "on", "enable_web_search": "off"}
    return Settings(**{**base, **kw})


def _names(settings: Settings) -> list[str]:
    return [t.name for t in RagService(settings, FakeRetriever(abstain=True)).answer_tools()]


# --------------------------------------------------------------------------- 계약

def test_every_tool_has_a_docstring() -> None:
    """독스트링이 곧 모델에게 보이는 도구 설명이다. 비면 도구를 못 고른다."""
    for t in tools.TOOLS:
        assert (t.description or "").strip(), t.name


def test_answer_tool_args_are_a_single_query_string() -> None:
    """strict=True 가 도구 스키마로 전파돼 선택 인자가 required 로 바뀐다.

    인자를 query 하나로 통일하면 그 문제도, _call_key 의 키 붕괴도 함께 피한다.
    """
    optional = [t for _flag, t in tools.OPTIONAL_TOOLS] + list(tools.ANSWER_TOOLS)
    for t in optional:
        assert list(t.args) == ["query"], f"{t.name}: {list(t.args)}"


def test_tool_names_are_unique() -> None:
    names = [t.name for t in tools.TOOLS]
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------- 게이트

def test_nothing_is_bound_when_rounds_are_zero() -> None:
    assert _names(_settings(max_tool_rounds=0, enable_web_search="on",
                            tavily_api_key=SecretStr("t"), enable_schedule_tool="on")) == []


def test_schedule_tool_does_not_depend_on_tavily() -> None:
    """예전에는 전부 web_search_enabled 하나에 묶여 Tavily 를 끄면 일정까지 죽었다."""
    got = _names(_settings(enable_web_search="off", enable_schedule_tool="on"))
    assert got == ["kbo_schedule_lookup"]


def test_places_needs_web_search() -> None:
    assert _names(_settings(enable_places="on", enable_web_search="off")) == []
    got = _names(_settings(enable_places="on", enable_web_search="on",
                           tavily_api_key=SecretStr("t")))
    assert got == ["web_search", "find_restaurants"]


def test_highlights_need_both_tavily_and_a_youtube_key() -> None:
    base = dict(enable_highlights="on", enable_web_search="on",
                tavily_api_key=SecretStr("t"))
    assert _names(_settings(**base)) == ["web_search"]                     # 키 없음
    assert _names(_settings(**base, youtube_kbo_api_key=SecretStr("AIza"))) == \
        ["web_search", "find_game_highlight"]


def test_schedule_tool_needs_kbo_data() -> None:
    assert _names(_settings(enable_schedule_tool="on", enable_kbo_data="off")) == []


def test_all_flags_off_binds_nothing() -> None:
    assert _names(_settings()) == []


# --------------------------------------------------------------------------- 날짜 표기

def test_dates_are_written_so_they_are_not_read_as_rule_numbers() -> None:
    """citations.RULE_NO_RE 가 "9.13" 을 규칙 번호로 잡는다(실측)."""
    from datetime import date

    from baseball import citations as cite

    text = tools._when(date(2026, 9, 13))
    assert text == "9월 13일"
    assert cite.extract(text, [])[1] == []
