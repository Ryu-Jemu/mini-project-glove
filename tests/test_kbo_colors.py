"""구단 상징색: 출처, 대비, 정책 경계."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
if str(UI) not in sys.path:
    sys.path.insert(0, str(UI))

COLORS = ROOT / "data" / "kbo_team_colors.json"
CODES = {"KT", "SS", "LG", "HT", "OB", "NC", "SK", "HH", "LT", "WO"}
WHITE, NEAR_BLACK = "#FFFFFF", "#131313"
LIGHT_BG, DARK_BG = "#FFFFFF", "#131313"


def _lum(hx: str) -> float:
    r, g, b = (int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = f(r), f(g), f(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _doc() -> dict:
    return json.loads(COLORS.read_text(encoding="utf-8"))


def _teams() -> list[dict]:
    return _doc()["teams"]


def test_all_ten_clubs() -> None:
    assert {t["code"] for t in _teams()} == CODES


def test_every_color_is_a_valid_hex() -> None:
    import re

    pattern = re.compile(r"^#[0-9A-F]{6}$")
    for t in _teams():
        for key in ("primary", "on_primary", "dark_primary", "on_dark_primary"):
            assert pattern.match(t[key]), f"{t['code']} {key}={t[key]}"


def test_every_source_is_an_official_club_site() -> None:
    """색은 구단 공식 사이트에서만 가져온다. 금지 소스는 쓰지 않는다."""
    allowed = ("lgtwins.com", "doosanbears.com", "heroesbaseball.co.kr", "ssglanders.com",
               "ktwiz.co.kr", "hanwhaeagles.co.kr", "samsunglions.com", "giantsclub.com",
               "tigers.co.kr", "ncdinos.com")
    blocked = ("koreabaseball.com", "namu.wiki", "statiz")
    for t in _teams():
        url = t["source_url"]
        assert any(host in url for host in allowed), f"{t['code']}: {url}"
        assert not any(host in url for host in blocked), f"{t['code']}: {url}"
        assert t["occurrences"] >= 1


@pytest.mark.parametrize("code", sorted(CODES))
def test_text_on_primary_clears_aa(code: str) -> None:
    t = next(x for x in _teams() if x["code"] == code)
    assert _ratio(t["on_primary"], t["primary"]) >= 4.5


@pytest.mark.parametrize("code", sorted(CODES))
def test_text_on_dark_primary_clears_aa(code: str) -> None:
    t = next(x for x in _teams() if x["code"] == code)
    assert _ratio(t["on_dark_primary"], t["dark_primary"]) >= 4.5


@pytest.mark.parametrize("code", sorted(CODES))
def test_dark_variant_is_visible_on_dark_background(code: str) -> None:
    """짙은 남색 구단이 다크 배경에 묻히면 카드가 사라진다."""
    t = next(x for x in _teams() if x["code"] == code)
    assert _ratio(t["dark_primary"], DARK_BG) >= 3.0


def test_recorded_contrast_matches_computation() -> None:
    """기록해 둔 수치가 실제 계산과 어긋나면 데이터가 낡은 것이다."""
    for t in _teams():
        assert abs(t["contrast_on_primary"] - _ratio(t["on_primary"], t["primary"])) < 0.02
        assert abs(t["contrast_vs_dark_bg"] - _ratio(t["dark_primary"], DARK_BG)) < 0.02


def test_hanwha_is_the_only_dark_text_club() -> None:
    """한화 주황은 흰 글자가 2.88:1 이라 반드시 검은 글자여야 한다."""
    dark_text = {t["code"] for t in _teams() if t["on_primary"] == NEAR_BLACK}
    assert dark_text == {"HH"}


def test_codes_match_the_team_file() -> None:
    teams = json.loads((ROOT / "data" / "kbo_teams.json").read_text(encoding="utf-8"))
    assert {t["code"] for t in teams["teams"]} == {t["code"] for t in _teams()}


def test_css_defines_a_rule_for_every_club() -> None:
    import clubs

    css = clubs.club_css()
    for code in CODES:
        assert f".tw-club-strip--{code}" in css


def test_css_has_no_club_asset_references() -> None:
    """로고·워드마크·구단 서체를 끌어오지 않는다."""
    import clubs

    css = clubs.club_css().lower()
    for banned in ("logo", "emblem", "wordmark", "url(", "@font-face", "azurefd"):
        assert banned not in css
