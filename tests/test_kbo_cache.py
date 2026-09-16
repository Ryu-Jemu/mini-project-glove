"""캐시·폴백. 어떤 실패도 사용자 경로로 올라가지 않아야 한다."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from baseball import kbo
from baseball.config import Settings
from baseball.kbo_models import (
    Game,
    KboUpstreamChanged,
    ScheduleSnapshot,
    StandingsSnapshot,
    TeamStanding,
)

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 9, 16)


@pytest.fixture(autouse=True)
def _clear_memo():
    kbo._MEM.clear()
    yield
    kbo._MEM.clear()


def _settings(**over) -> Settings:
    return Settings(enable_kbo_data="on", **over)


def _standings() -> StandingsSnapshot:
    r = json.loads((FIXTURES / "kbo_standings.json").read_text(encoding="utf-8"))["result"]
    return StandingsSnapshot(
        teams=[TeamStanding.model_validate(t) for t in r["seasonTeamStats"]],
        season=2026, as_of=date(2026, 9, 15), game_type=r["gameType"],
    )


def _schedule() -> ScheduleSnapshot:
    r = json.loads((FIXTURES / "kbo_schedule_remaining.json").read_text(encoding="utf-8"))["result"]
    return ScheduleSnapshot(
        games=[Game.model_validate(g) for g in r["games"]],
        as_of=date(2026, 9, 15), total=r["gameTotalCount"],
    )


class _Route:
    def __init__(self, topics=(), teams=()):
        self.topics, self.teams = topics, teams


def _no_db(monkeypatch) -> None:
    """DB 를 아예 못 쓰는 상황. 저장본 폴백이 없는 최악을 만든다."""
    def boom(*a, **k):
        raise RuntimeError("no database")

    monkeypatch.setattr("baseball.db.connect", boom)


def test_live_fetch_produces_live_entry(monkeypatch) -> None:
    _no_db(monkeypatch)
    monkeypatch.setattr("baseball.kbo_naver.fetch_standings", lambda *a, **k: _standings())
    entries = kbo.entries_for("LG 순위", _Route(("standings",), ("LG",)),
                              settings=_settings(), today=TODAY)
    assert len(entries) == 1
    assert entries[0].freshness == "live"
    assert "1위 KT" in entries[0].text


def test_disabled_returns_nothing() -> None:
    entries = kbo.entries_for("LG 순위", _Route(("standings",), ("LG",)),
                              settings=Settings(enable_kbo_data="off"), today=TODAY)
    assert entries == []


def test_no_topics_returns_nothing() -> None:
    assert kbo.entries_for("인필드 플라이", _Route(), settings=_settings(), today=TODAY) == []


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("network"), TimeoutError(), KboUpstreamChanged("shape"), KeyError("x"),
     ValueError("bad")],
)
def test_any_fetch_failure_degrades_quietly(monkeypatch, exc: Exception) -> None:
    """상류가 어떻게 실패하든 예외가 새어 나가면 안 된다."""
    _no_db(monkeypatch)

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr("baseball.kbo_naver.fetch_standings", boom)
    entries = kbo.entries_for("LG 순위", _Route(("standings",), ("LG",)),
                              settings=_settings(), today=TODAY)
    assert entries == []                       # 목록이지 예외가 아니다


def test_schedule_filtered_to_one_team(monkeypatch) -> None:
    _no_db(monkeypatch)
    monkeypatch.setattr("baseball.kbo_naver.fetch_schedule", lambda *a, **k: _schedule())
    entries = kbo.entries_for("LG 남은 경기", _Route(("schedule",), ("LG",)),
                              settings=_settings(), today=TODAY)
    assert entries[0].label == "LG 트윈스 남은 경기"
    assert "남은 경기 15경기" in entries[0].text


def test_schedule_league_wide_when_no_team(monkeypatch) -> None:
    _no_db(monkeypatch)
    monkeypatch.setattr("baseball.kbo_naver.fetch_schedule", lambda *a, **k: _schedule())
    entries = kbo.entries_for("남은 경기 알려줘", _Route(("schedule",), ()),
                              settings=_settings(), today=TODAY)
    assert "남은 경기 78경기" in entries[0].text


def test_team_info_is_static_and_carries_attribution(monkeypatch) -> None:
    _no_db(monkeypatch)
    monkeypatch.setattr("baseball.kbo_naver.fetch_standings", lambda *a, **k: _standings())
    entries = kbo.entries_for("두산 연고지", _Route(("team_info",), ("OB",)),
                              settings=_settings(), today=TODAY)
    e = entries[0]
    assert e.freshness == "static"
    assert "서울특별시" in e.text
    assert "CC BY-SA 4.0" in e.text            # 위키백과 라이선스 표기 의무
    assert e.source_url and "wikipedia" in e.source_url


def test_memo_avoids_second_fetch(monkeypatch) -> None:
    _no_db(monkeypatch)
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return _standings()

    monkeypatch.setattr("baseball.kbo_naver.fetch_standings", counting)
    r = _Route(("standings",), ("LG",))
    kbo.entries_for("LG 순위", r, settings=_settings(), today=TODAY)
    kbo.entries_for("LG 순위", r, settings=_settings(), today=TODAY)
    assert calls["n"] == 1


def test_entries_capped(monkeypatch) -> None:
    _no_db(monkeypatch)
    monkeypatch.setattr("baseball.kbo_naver.fetch_standings", lambda *a, **k: _standings())
    monkeypatch.setattr("baseball.kbo_naver.fetch_schedule", lambda *a, **k: _schedule())
    entries = kbo.entries_for("LG 순위랑 남은 일정이랑 연고지",
                              _Route(("standings", "schedule", "team_info"), ("LG",)),
                              settings=_settings(), today=TODAY)
    assert len(entries) <= kbo.MAX_ENTRIES


# --------------------------------------------------------------------------- 조회 창

def test_schedule_window_looks_back_for_past_games(monkeypatch) -> None:
    """하이라이트는 지난 경기다. start=today 면 과거 경기가 스냅샷에 아예 없다."""
    _no_db(monkeypatch)
    seen: dict[str, object] = {}

    def spy(**kw):
        seen.update(kw)
        return _schedule()

    monkeypatch.setattr("baseball.kbo_naver.fetch_schedule", spy)
    kbo.entries_for("LG 남은 경기", _Route(("schedule",), ("LG",)),
                    settings=_settings(), today=TODAY)

    assert seen["start"] == date(2026, 9, 2)        # TODAY - 14일
    assert seen["today"] == TODAY                    # 기준일은 그대로 오늘이다


def test_schedule_lookback_is_configurable(monkeypatch) -> None:
    _no_db(monkeypatch)
    seen: dict[str, object] = {}

    def spy(**kw):
        seen.update(kw)
        return _schedule()

    monkeypatch.setattr("baseball.kbo_naver.fetch_schedule", spy)
    kbo.entries_for("LG 남은 경기", _Route(("schedule",), ("LG",)),
                    settings=_settings(kbo_schedule_lookback_days=3), today=TODAY)

    assert seen["start"] == date(2026, 9, 13)


def test_schedule_cache_key_does_not_encode_the_window(monkeypatch) -> None:
    """창을 호출부마다 다르게 주면 안 되는 이유를 고정한다.

    _cached 의 키는 kind 문자열뿐이라, 창이 다른 두 조회는 같은 스냅샷 자리를
    서로 덮어쓴다. 그래서 창은 전역 설정 하나여야 한다.
    """
    assert kbo.SCHEDULE_KIND == "kbo:schedule"       # 창이 키에 들어가지 않는다
