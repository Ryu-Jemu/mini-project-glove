"""턴의 근거 조달 순서를 고정한다.

  0단계 가진 데이터(KBO 실데이터·리그 규정 스냅샷)
  1단계 규칙집 PDF          ← 1차 근거
  2단계 Tavily 웹 검색       ← 규칙집이 빈손일 때만
  3단계 전부 빈손이면 거부

예전에는 웹이 검색보다 먼저 돌았고, abstain 이 곧바로 거부였다. 그래서 규칙집에 답이
있는 질문도 근거를 못 찾으면 LLM 을 부르지도 않고 거부 문장으로 끝났다.
"""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from baseball import latest_info
from baseball.chain import RagService
from baseball.config import Settings
from baseball.context import LatestEntry
from baseball.prompts import NOT_IN_CONTEXT_REFUSAL

from conftest import CANNED_ANSWER, FakeRetriever, fake_structured_model


def _settings(**kw: Any) -> Settings:
    base = {"_env_file": None, "openai_api_key": SecretStr("test-key"),
            "enable_kbo_data": "off", "enable_web_search": "off"}
    return Settings(**{**base, **kw})


def _web_on(**kw: Any) -> Settings:
    return _settings(enable_web_search="on", tavily_api_key=SecretStr("tvly-test"), **kw)


def _service(settings: Settings, *, abstain: bool = False) -> RagService:
    return RagService(
        settings, FakeRetriever(abstain=abstain),
        llm=fake_structured_model([CANNED_ANSWER]),
        router_llm=None,
    )


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch):
    """web_search 를 기록기로 갈아끼운다. 호출 여부와 순서를 본다."""

    class Recorder:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.results: list[LatestEntry] = []

        def __call__(self, query: str, settings=None, *, client=None) -> list[LatestEntry]:
            self.queries.append(query)
            return list(self.results)

    rec = Recorder()
    monkeypatch.setattr(latest_info, "web_search", rec)
    return rec


def _entry() -> LatestEntry:
    return LatestEntry(kind="web", label="웹 검색: 위키백과", text="야구는 9명이 한다.",
                       as_of="2026-09-16", source_url="https://example.com", confidence="likely")


# --------------------------------------------------------------------------- 순서

def test_rulebook_hit_never_reaches_the_web(web) -> None:
    """규칙집이 답을 주면 웹은 부르지 않는다. 불필요한 지연과 비용이다."""
    out = _service(_web_on()).answer("인필드 플라이가 뭐야?")
    assert out.status == "answered"
    assert web.queries == []


def test_web_runs_only_after_the_rulebook_comes_up_empty(web) -> None:
    web.results = [_entry()]
    out = _service(_web_on(), abstain=True).answer("인필드 플라이가 뭐야?")

    assert web.queries == ["인필드 플라이가 뭐야?"]
    assert out.freshness == "web"
    assert out.status == "answered"
    assert out.llm_called is True              # 예전에는 여기서 거부하고 끝났다
    assert [s["kind"] for s in out.sources] == ["web"]


def test_abstain_is_no_longer_an_immediate_refusal(web) -> None:
    """abstain 은 '근거가 약하다'지 '거부하라'가 아니다."""
    web.results = [_entry()]
    out = _service(_web_on(), abstain=True).answer("보크가 뭐야?")
    assert out.answer != NOT_IN_CONTEXT_REFUSAL


# --------------------------------------------------------------------------- 3단계 분기

def test_rule_question_with_nothing_anywhere_refuses(web) -> None:
    out = _service(_web_on(), abstain=True).answer("인필드 플라이가 뭐야?")
    assert web.queries                          # 웹까지 가 보긴 했다
    assert out.answer == NOT_IN_CONTEXT_REFUSAL
    assert out.status == "not_in_rulebook"
    assert out.llm_called is False


def test_live_question_without_web_reports_phase2(web) -> None:
    """실시간 정보가 필요한데 웹 경로가 닫혀 있으면 그 사실을 구분해 알린다."""
    out = _service(_settings(), abstain=True).answer("오늘 KT 경기 몇 시야?")

    assert web.queries == []                    # ENABLE_WEB_SEARCH=off 라 나가지 않는다
    assert out.status == "phase2_pending"
    assert out.needs_web is True
    assert out.llm_called is False


def test_web_disabled_rule_question_is_not_phase2(web) -> None:
    out = _service(_settings(), abstain=True).answer("보크가 뭐야?")
    assert out.status == "not_in_rulebook"
    assert out.needs_web is False


# --------------------------------------------------------------------------- 규칙집 혼입 방지

def test_rulebook_chunks_stay_out_of_pure_latest_turns(web) -> None:
    """대체 근거가 있는 순위·일정 질문에는 규칙집 조항을 섞지 않는다.

    섞이면 모델이 "자료에 없다" 며 답을 거부한다. 단, 대체 근거가 하나도 없으면
    규칙집이라도 주는 편이 낫기 때문에 조건이 (latest or kbo) 로 걸려 있다.
    """
    out = _service(_settings(), abstain=False).answer("피치클락 몇 초야?")

    assert out.freshness == "snapshot"                  # 0단계가 스냅샷을 채웠다
    kinds = {s["kind"] for s in out.sources}
    assert "static" not in kinds, "규칙집 근거가 순위·일정 답변에 섞였다"


def test_rulebook_is_kept_when_there_is_no_alternative(web) -> None:
    """대체 근거가 없으면 규칙집 조항이라도 넘긴다(빈손보다 낫다)."""
    out = _service(_settings(), abstain=False).answer("오늘 KT 경기 몇 시야?")
    assert {s["kind"] for s in out.sources} == {"static"}


def test_snapshot_is_preferred_over_the_web(web) -> None:
    """리그 규정 스냅샷은 공짜이고 확실하다. 웹보다 먼저다."""
    web.results = [_entry()]
    out = _service(_web_on(), abstain=True).answer("피치클락 몇 초야?")

    assert web.queries == []
    assert out.freshness == "snapshot"
