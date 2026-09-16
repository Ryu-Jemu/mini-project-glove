"""구단 카드 화면.

각 구단의 상징색을 '채워진 헤더 띠'로만 쓴다. 글자색이나 얇은 선에는 쓰지 않는다.
채운 면은 자기 배경을 스스로 만들어 주므로 그 위 글자 대비만 지키면 되고,
카드 경계는 색과 무관하게 테마 보더가 항상 보장한다.

로고·워드마크·엠블럼·구단 서체는 쓰지 않는다. 구단 식별은 이름 텍스트로 한다.
적색 계열 구단이 일곱이라 색만으로는 구분이 되지 않기 때문이기도 하다.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

import theme

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

COLORS_PATH = ROOT / "data" / "kbo_team_colors.json"
TEAMS_PATH = ROOT / "data" / "kbo_teams.json"


@lru_cache(maxsize=1)
def _colors() -> dict[str, dict[str, Any]]:
    try:
        doc = json.loads(COLORS_PATH.read_text(encoding="utf-8"))
        return {t["code"]: t for t in doc.get("teams", [])}
    except Exception:                                   # noqa: BLE001
        return {}


@lru_cache(maxsize=1)
def _teams() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        doc = json.loads(TEAMS_PATH.read_text(encoding="utf-8"))
        return doc, {t["code"]: t for t in doc.get("teams", [])}
    except Exception:                                   # noqa: BLE001
        return {}, {}


def club_css() -> str:
    """구단마다 띠 색이 다르므로 코드별 규칙을 미리 만들어 둔다."""
    mode = theme.theme_mode()
    c = theme.BRAND[mode]
    rules = [
        f""".tw-club-strip {{
  border-radius: 10px 10px 0 0; padding: .7rem .9rem; margin: -1rem -1rem .85rem -1rem;
  display: flex; align-items: baseline; gap: .55rem; flex-wrap: wrap;
}}
.tw-club-name {{ font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em; }}
.tw-club-code {{ font-size: .75rem; opacity: .85; font-variant-numeric: tabular-nums; }}
.tw-club-meta {{ color: {c['muted']}; font-size: .82rem; margin: 0 0 .5rem 0; }}
.tw-club-meta b {{ color: {c['text']}; font-weight: 600; }}"""
    ]
    for code, t in _colors().items():
        bg = t["dark_primary"] if mode == "dark" else t["primary"]
        fg = t["on_dark_primary"] if mode == "dark" else t["on_primary"]
        rules.append(
            f'.tw-club-strip--{code} {{ background: {bg}; color: {fg}; }}'
        )
    return "<style>\n" + "\n".join(rules) + "\n</style>"


def _strip(code: str, name: str, subtitle: str) -> str:
    safe = st.markdown  # noqa: F841  (아래는 정적 문자열만 조립한다)
    return (
        f'<div class="tw-club-strip tw-club-strip--{code}">'
        f'<span class="tw-club-name">{name}</span>'
        f'<span class="tw-club-code">{subtitle}</span>'
        f"</div>"
    )


def _standing_for(code: str) -> dict[str, Any] | None:
    try:
        from datetime import date

        from baseball import kbo
        from baseball.config import get_settings

        snap, _fresh, _as_of = kbo._cached(
            get_settings(), kbo.STANDINGS_KIND,
            get_settings().kbo_standings_ttl_seconds,
            lambda: _fetch_standings(),
        )
        if snap is None:
            return None
        t = snap.by_code(code)
        if t is None:
            return None
        return {"rank": t.ranking, "wins": t.wins, "losses": t.losses, "draws": t.draws,
                "wra": t.wra, "gb": t.game_behind, "last5": t.last_five or "-"}
    except Exception:                                   # noqa: BLE001
        return None


def _fetch_standings() -> Any:
    from datetime import date

    from baseball import kbo, kbo_naver
    from baseball.config import get_settings

    settings = get_settings()
    today = date.today()
    return kbo_naver.fetch_standings(kbo.season_year(settings, today), today=today,
                                     timeout=settings.kbo_http_timeout_seconds)


HITTER_COLUMNS = [
    ("back_number", "등번호"), ("name", "이름"), ("position", "포지션"),
    ("games", "경기"), ("avg", "타율"), ("hits", "안타"), ("hr", "홈런"),
    ("rbi", "타점"), ("runs", "득점"), ("sb", "도루"), ("ops", "OPS"),
]
PITCHER_COLUMNS = [
    ("back_number", "등번호"), ("name", "이름"), ("position", "포지션"),
    ("games", "경기"), ("win", "승"), ("lose", "패"), ("save", "세이브"),
    ("hold", "홀드"), ("innings", "이닝"), ("so", "탈삼진"), ("era", "평균자책"),
]


def _rows(players: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for p in players:
        stats = p.get("stats", {})
        row = {}
        for key, label in columns:
            row[label] = p.get(key, stats.get(key))
        out.append(row)
    return out


def _roster_table(players: list[dict[str, Any]], kind: str) -> None:
    if not players:
        st.caption("표시할 선수가 없습니다.")
        return
    columns = PITCHER_COLUMNS if kind == "PITCHER" else HITTER_COLUMNS
    sort_key = "games"
    players = sorted(players, key=lambda p: -(p.get("stats", {}).get(sort_key) or 0))
    st.dataframe(
        _rows(players, columns), hide_index=True, width="stretch", height=330,
        column_config={
            "등번호": st.column_config.NumberColumn("등번호", format="%d", width="small"),
            "타율": st.column_config.NumberColumn("타율", format="%.3f"),
            "OPS": st.column_config.NumberColumn("OPS", format="%.3f"),
            "평균자책": st.column_config.NumberColumn("평균자책", format="%.2f"),
        },
    )


def render() -> None:
    codes = _colors()
    meta, teams = _teams()
    if not codes or not teams:
        st.info("구단 정보 파일이 없습니다.", icon=":material/info:")
        return

    st.html(club_css())
    order = sorted(teams, key=lambda c: teams[c]["full"])
    labels = {c: teams[c]["short"] for c in order}
    picked = st.segmented_control(
        "구단", order, format_func=lambda c: labels[c], default=order[0],
        key="club-pick", label_visibility="collapsed",
    ) or order[0]

    team, color = teams[picked], codes[picked]
    standing = _standing_for(picked)

    with st.container(border=True, key=f"club-card-{picked}"):
        rank = f"{standing['rank']}위" if standing else "순위 정보 없음"
        st.html(_strip(picked, team["full"], rank))

        st.html(
            f'<p class="tw-club-meta">연고지 <b>{team["hometown"]}</b> · '
            f'홈구장 <b>{team["stadium"]}</b></p>'
        )

        if standing:
            cols = st.columns(4)
            cols[0].metric("순위", f"{standing['rank']}위")
            cols[1].metric("승 · 패 · 무", f"{standing['wins']}-{standing['losses']}-{standing['draws']}")
            cols[2].metric("승률", f"{standing['wra']:.3f}")
            cols[3].metric("게임차", "-" if standing["gb"] == 0 else f"{standing['gb']:.1f}")
            st.caption(f"최근 5경기 {standing['last5']} · 순위 자료: 네이버 스포츠")
        else:
            st.warning("순위 자료를 불러오지 못했습니다. 선수 명단은 그대로 볼 수 있습니다.",
                       icon=":material/cloud_off:")

        from baseball import kbo_players

        pitchers = kbo_players.roster(picked, "PITCHER")
        hitters = kbo_players.roster(picked, "HITTER")
        tab_p, tab_h = st.tabs([f"투수 {len(pitchers)}명", f"타자 {len(hitters)}명"])
        with tab_p:
            _roster_table(pitchers, "PITCHER")
        with tab_h:
            _roster_table(hitters, "HITTER")

        st.caption(
            f"선수 기록 {kbo_players.captured_at()} 기준 · 출처 네이버 스포츠 · "
            f"연고지·홈구장 출처 위키백과(CC BY-SA 4.0) · "
            f"구단 상징색은 구단 공식 웹사이트에서 확인한 색상값만 사용했습니다."
        )
