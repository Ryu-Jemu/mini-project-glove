"""구단 정적 정보 파일. 라이선스 표기와 상류 대조까지 본다."""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _teams() -> dict:
    return json.loads((ROOT / "data" / "kbo_teams.json").read_text(encoding="utf-8"))


def test_ten_teams_with_expected_codes() -> None:
    d = _teams()
    codes = sorted(t["code"] for t in d["teams"])
    assert codes == ["HH", "HT", "KT", "LG", "LT", "NC", "OB", "SK", "SS", "WO"]


def test_codes_match_upstream_standings() -> None:
    upstream = {t["teamId"] for t in json.loads(
        (FIXTURES / "kbo_standings.json").read_text(encoding="utf-8"))["result"]["seasonTeamStats"]}
    assert {t["code"] for t in _teams()["teams"]} == upstream


def test_upstream_keyword_is_a_known_alias() -> None:
    """네이버는 'KIA'를 '기아 타이거즈'로 적는다. 정확 일치가 아니라 별칭 포함으로 본다."""
    by_code = {t["code"]: t for t in _teams()["teams"]}
    rows = json.loads((FIXTURES / "kbo_standings.json").read_text(encoding="utf-8"))["result"]["seasonTeamStats"]
    for row in rows:
        team = by_code[row["teamId"]]
        candidates = {a.replace(" ", "") for a in team["aliases"]} | {team["full"].replace(" ", "")}
        assert row["keyword"].replace(" ", "") in candidates, row["keyword"]


def test_stadium_short_matches_schedule_feed() -> None:
    by_code = {t["code"]: t for t in _teams()["teams"]}
    games = []
    for name in ("kbo_schedule_remaining", "kbo_schedule_edge"):
        games += json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))["result"]["games"]
    home = collections.defaultdict(collections.Counter)
    for g in games:
        if g.get("roundCode") == "kbo_r":
            home[g["homeTeamCode"]][g.get("stadium") or "?"] += 1
    for code, counter in home.items():
        assert by_code[code]["stadium_short"] == counter.most_common(1)[0][0]


def test_aliases_are_pairwise_disjoint() -> None:
    names = [a for t in _teams()["teams"] for a in t["aliases"]]
    assert len(names) == len(set(names))


def test_license_and_sources_present() -> None:
    d = _teams()
    assert d["license"] == "CC BY-SA 4.0"
    assert d["license_url"].startswith("https://creativecommons.org/")
    assert d["source"] == "ko.wikipedia.org"
    for t in d["teams"]:
        assert t["source_url"].startswith("https://ko.wikipedia.org/wiki/")
        assert t["hometown"] and t["stadium"]
