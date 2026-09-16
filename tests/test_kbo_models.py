"""상류 응답 모델의 불변식. 성립하는 것과 '깨질 때 실제로 터지는지'를 함께 본다."""
from __future__ import annotations

import copy
import json
from datetime import date
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
