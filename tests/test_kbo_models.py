"""상류 응답 모델의 불변식. 성립하는 것과 '깨질 때 실제로 터지는지'를 함께 본다."""
from __future__ import annotations

import copy
import json
from datetime import date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from baseball.kbo_models import (
    Game,
    KboUpstreamChanged,
    ScheduleSnapshot,
    StandingsSnapshot,
    TeamStanding,
)

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 9, 16)


def _standings_raw() -> dict:
    return json.loads((FIXTURES / "kbo_standings.json").read_text(encoding="utf-8"))["result"]


def _schedule_raw(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))["result"]


def _standings() -> StandingsSnapshot:
    r = _standings_raw()
    return StandingsSnapshot(
        teams=[TeamStanding.model_validate(t) for t in r["seasonTeamStats"]],
        season=2026, as_of=date(2026, 9, 15), game_type=r["gameType"],
    )


def _schedule(name: str) -> ScheduleSnapshot:
    r = _schedule_raw(name)
    return ScheduleSnapshot(
        games=[Game.model_validate(g) for g in r["games"]],
        as_of=date(2026, 9, 15), total=r.get("gameTotalCount"),
    )


def test_standings_fixture_satisfies_every_invariant() -> None:
    s = _standings()
    assert len(s.teams) == 10
    assert {t.ranking for t in s.teams} == set(range(1, 11))
    for t in s.teams:
        assert t.games == t.wins + t.draws + t.losses
        assert abs(t.wra - t.wins / (t.wins + t.losses)) <= 0.0015
    assert sum(t.wins for t in s.teams) == sum(t.losses for t in s.teams)


def test_schedule_fixtures_parse() -> None:
    rem = _schedule("kbo_schedule_remaining")
    assert len(rem.games) == 78
    assert len(rem.remaining(TODAY)) == 78
    edge = _schedule("kbo_schedule_edge")
    assert len(edge.games) == 111


def test_allstar_row_parses_but_is_excluded() -> None:
    """올스타전은 EA/WE 코드를 쓴다. 파싱은 되고 잔여 경기에서는 빠져야 한다."""
    edge = _schedule("kbo_schedule_edge")
    allstar = [g for g in edge.games if g.round_code == "kbo_as"]
    assert len(allstar) == 1
    assert {allstar[0].home_code, allstar[0].away_code} == {"EA", "WE"}
    assert allstar[0] not in edge.remaining(date(2026, 7, 1))


def test_cancelled_games_are_not_remaining() -> None:
    """취소된 우천 경기도 statusCode 는 BEFORE 로 남는다. cancel 을 함께 봐야 한다."""
    edge = _schedule("kbo_schedule_edge")
    cancelled = [g for g in edge.games if g.cancel]
    assert cancelled, "픽스처에 취소 경기가 있어야 의미가 있다"
    assert all(g.status_code == "BEFORE" for g in cancelled)
    assert not any(g in edge.remaining(date(2026, 7, 1)) for g in cancelled)


def test_remaining_per_team_sums_to_twice_game_count() -> None:
    rem = _schedule("kbo_schedule_remaining").remaining(TODAY)
    per_team: dict[str, int] = {}
    for g in rem:
        per_team[g.home_code] = per_team.get(g.home_code, 0) + 1
        per_team[g.away_code] = per_team.get(g.away_code, 0) + 1
    assert sum(per_team.values()) == len(rem) * 2
    assert len(per_team) == 10


@pytest.mark.parametrize(
    "field,value",
    [("teamId", "XX"), ("wra", 62.3), ("ranking", 11), ("lastFiveGames", "WWWWWW")],
)
def test_team_standing_rejects_broken_field(field: str, value: object) -> None:
    row = copy.deepcopy(_standings_raw()["seasonTeamStats"][0])
    row[field] = value
    with pytest.raises((ValidationError, KboUpstreamChanged)):
        TeamStanding.model_validate(row)


def test_team_standing_rejects_inconsistent_record() -> None:
    row = copy.deepcopy(_standings_raw()["seasonTeamStats"][0])
    row["winGameCount"] = row["winGameCount"] + 5          # 승패합이 경기수와 어긋난다
    with pytest.raises((ValidationError, KboUpstreamChanged)):
        TeamStanding.model_validate(row)


def test_standings_rejects_postseason_switch() -> None:
    r = _standings_raw()
    with pytest.raises((ValidationError, KboUpstreamChanged)):
        StandingsSnapshot(
            teams=[TeamStanding.model_validate(t) for t in r["seasonTeamStats"]],
            season=2026, as_of=date(2026, 9, 15), game_type="POSTSEASON",
        )


def test_standings_rejects_missing_team() -> None:
    r = _standings_raw()
    with pytest.raises((ValidationError, KboUpstreamChanged)):
        StandingsSnapshot(
            teams=[TeamStanding.model_validate(t) for t in r["seasonTeamStats"][:9]],
            season=2026, as_of=date(2026, 9, 15),
        )


def test_game_rejects_home_away_swap() -> None:
    """경기 ID 는 원정+홈 순서다. 뒤바뀌면 즉시 드러나야 한다."""
    row = copy.deepcopy(_schedule_raw("kbo_schedule_remaining")["games"][0])
    row["homeTeamCode"], row["awayTeamCode"] = row["awayTeamCode"], row["homeTeamCode"]
    with pytest.raises((ValidationError, KboUpstreamChanged)):
        Game.model_validate(row)


def test_game_rejects_score_before_start() -> None:
    row = copy.deepcopy(_schedule_raw("kbo_schedule_remaining")["games"][0])
    row["homeTeamScore"] = 3
    with pytest.raises((ValidationError, KboUpstreamChanged)):
        Game.model_validate(row)


def test_schedule_rejects_count_mismatch() -> None:
    r = _schedule_raw("kbo_schedule_remaining")
    with pytest.raises((ValidationError, KboUpstreamChanged)):
        ScheduleSnapshot(
            games=[Game.model_validate(g) for g in r["games"][:10]],
            as_of=date(2026, 9, 15), total=r["gameTotalCount"],
        )


# --------------------------------------------------------------------------- 시각 단위 선택자

def _g(day: int, hour: int, status: str = "BEFORE", *, away: str = "OB", home: str = "LG",
       cancel: bool = False) -> Game:
    return Game(
        game_id=f"2026{9:02d}{day:02d}{away}{home}00000",
        game_date=date(2026, 9, day),
        game_date_time=datetime(2026, 9, day, hour, 30),
        round_code="kbo_r", stadium="잠실",
        home_code=home, away_code=away, status_code=status, cancel=cancel,
    )


def _snap(games: list[Game]) -> ScheduleSnapshot:
    return ScheduleSnapshot(games=games, as_of=date(2026, 9, 16))


def test_upcoming_is_time_based_not_date_based() -> None:
    """remaining 은 날짜 단위라 세 시간 전에 시작한 오늘 경기도 '남은 경기' 다.

    upcoming 은 그 경기를 뺀다. "가장 가까운 경기" 에는 이 구분이 필요하다.
    """
    started = _g(16, 14)                       # 오늘 14:30 — 이미 시작
    later = _g(16, 18)                         # 오늘 18:30 — 아직
    snap = _snap([started, later])
    now = datetime(2026, 9, 16, 17, 0)         # naive KST

    assert snap.remaining(date(2026, 9, 16)) == [started, later]   # 둘 다 '남은 경기'
    assert snap.upcoming(now) == [later]                            # 시각으로 보면 하나뿐


def test_upcoming_is_sorted_by_start_time() -> None:
    snap = _snap([_g(18, 18), _g(16, 18), _g(17, 14)])
    out = snap.upcoming(datetime(2026, 9, 16, 9, 0))
    assert [g.game_date for g in out] == [date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18)]


def test_upcoming_skips_cancelled_games() -> None:
    snap = _snap([_g(17, 18, cancel=True), _g(18, 18)])
    assert [g.game_date for g in snap.upcoming(datetime(2026, 9, 16, 9, 0))] == [date(2026, 9, 18)]


def test_last_finished_picks_the_most_recent_played_game() -> None:
    """하이라이트가 가리킬 경기. 조회 창에 과거가 들어와야 비로소 의미가 있다."""
    snap = _snap([_g(12, 18, "RESULT"), _g(15, 18, "RESULT"), _g(18, 18, "BEFORE")])
    got = snap.last_finished(datetime(2026, 9, 16, 9, 0))
    assert got is not None and got.game_date == date(2026, 9, 15)


def test_last_finished_is_none_without_past_games() -> None:
    """조회 창이 start=today 면 이 경로가 항상 None 이라 하이라이트가 불가능하다."""
    assert _snap([_g(18, 18)]).last_finished(datetime(2026, 9, 16, 9, 0)) is None


def test_upcoming_rejects_aware_datetime() -> None:
    """game_date_time 이 naive KST 라 aware 를 넣으면 TypeError 가 난다. 계약을 고정한다."""
    from datetime import timezone

    snap = _snap([_g(18, 18)])
    with pytest.raises(TypeError):
        snap.upcoming(datetime(2026, 9, 16, 9, 0, tzinfo=timezone.utc))
