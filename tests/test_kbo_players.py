"""선수 명단 데이터 파일과 그 파서."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from baseball import kbo_players

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "kbo_players.jsonl"
CODES = {"KT", "SS", "LG", "HT", "OB", "NC", "SK", "HH", "LT", "WO"}


@pytest.mark.parametrize(
    "raw,expected",
    [("155", 155.0), ("138 2/3", 138.667), ("148 1/3", 148.333),
     ("0", 0.0), (None, None), ("", None), ("이닝아님", None)],
)
def test_parse_innings(raw, expected) -> None:
    assert kbo_players.parse_innings(raw) == expected


def test_file_exists_and_has_meta_first() -> None:
    assert DATA.exists(), "python -m baseball.kbo_players build 로 생성한다"
    first = json.loads(DATA.read_text(encoding="utf-8").splitlines()[0])
    assert first.get("_meta") is True
    assert first["season"] >= 2026
    assert first["captured_at"]
    assert first["source"] == "네이버 스포츠"
    assert first["player_count"] > 0


def test_every_line_is_valid_json() -> None:
    for i, line in enumerate(DATA.read_text(encoding="utf-8").splitlines(), 1):
        json.loads(line)                    # 깨진 줄이 있으면 여기서 터진다


def test_player_count_matches_meta() -> None:
    meta, by_team = kbo_players.load()
    total = sum(len(v) for v in by_team.values())
    assert total == meta["player_count"]


def test_all_ten_clubs_present() -> None:
    _, by_team = kbo_players.load()
    assert set(by_team) == CODES
    for code, rows in by_team.items():
        assert rows, f"{code} 선수가 없다"


def test_required_fields_on_every_player() -> None:
    _, by_team = kbo_players.load()
    for rows in by_team.values():
        for r in rows:
            assert r["player_id"] and r["name"]
            assert r["team_code"] in CODES
            assert r["player_type"] in ("HITTER", "PITCHER")
            assert r["position"]
            assert isinstance(r["stats"], dict)


def test_both_player_types_for_every_club() -> None:
    _, by_team = kbo_players.load()
    for code, rows in by_team.items():
        types = {r["player_type"] for r in rows}
        assert types == {"HITTER", "PITCHER"}, f"{code}: {types}"


def test_pitcher_innings_are_parsed_for_sorting() -> None:
    rows = [r for r in kbo_players.roster("LG", "PITCHER") if r["stats"].get("innings")]
    assert rows
    for r in rows:
        assert isinstance(r["stats"]["innings_value"], float)


def test_derived_stats_are_excluded() -> None:
    """입문자용이므로 WAR·wOBA·wRC+ 같은 파생 지표는 담지 않는다."""
    _, by_team = kbo_players.load()
    sample = next(iter(by_team.values()))[0]
    assert not {"war", "woba", "wrc_plus", "babip", "wpa"} & set(sample["stats"])


def test_roster_filters_by_type() -> None:
    pitchers = kbo_players.roster("LG", "PITCHER")
    hitters = kbo_players.roster("LG", "HITTER")
    assert pitchers and hitters
    assert all(r["player_type"] == "PITCHER" for r in pitchers)
    assert all(r["player_type"] == "HITTER" for r in hitters)
    assert len(kbo_players.roster("LG")) == len(pitchers) + len(hitters)


def test_retired_players_excluded_by_default() -> None:
    all_rows = kbo_players.roster("LG", include_retired=True)
    active = kbo_players.roster("LG")
    assert len(active) <= len(all_rows)
    assert all(not r["is_retired"] for r in active)


def test_unknown_team_is_empty_not_an_error() -> None:
    assert kbo_players.roster("ZZ") == []


def test_normalize_extracts_position_from_profile() -> None:
    row = {
        "playerId": "1", "playerName": "홍길동", "teamId": "LG", "backNumber": 7,
        "profile": json.dumps({"position": "외야수"}, ensure_ascii=False),
        "hitterHra": 0.3333333, "hitterHr": 10,
    }
    out = kbo_players.normalize(row, "HITTER")
    assert out["position"] == "외야수"
    assert out["stats"]["avg"] == 0.333            # 3자리로 맞춘다
    assert out["stats"]["hr"] == 10


def test_normalize_falls_back_when_profile_missing() -> None:
    assert kbo_players.normalize({"playerId": "1", "teamId": "LG"}, "PITCHER")["position"] == "투수"
    assert kbo_players.normalize({"playerId": "1", "teamId": "LG"}, "HITTER")["position"] == "야수"
    assert kbo_players.normalize({"playerId": "1", "teamId": "LG", "profile": "{깨진"}, "PITCHER")["position"] == "투수"


def test_fetch_uses_safe_page_size() -> None:
    """pageSize 가 500 을 넘으면 상류가 조용히 0건을 준다."""
    seen: list[httpx.Request] = []
    payload = {"code": 200, "success": True,
               "result": {"seasonPlayerStats": [{"playerId": "1", "playerName": "가",
                                                 "teamId": "LG", "isRetire": "N"}]}}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        rows = kbo_players.fetch_all(2026, client=c)
    assert len(seen) == 2                          # HITTER + PITCHER
    for req in seen:
        assert int(dict(req.url.params)["pageSize"]) <= kbo_players.MAX_PAGE_SIZE
    assert {dict(r.url.params)["playerType"] for r in seen} == {"HITTER", "PITCHER"}
    assert len(rows) == 2


def test_empty_upstream_response_raises() -> None:
    from baseball.kbo_naver import KboSourceError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 200, "success": True,
                                         "result": {"seasonPlayerStats": []}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(KboSourceError):
            kbo_players.fetch_all(2026, client=c)
