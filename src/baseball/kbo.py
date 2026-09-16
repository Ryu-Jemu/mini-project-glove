"""KBO 데이터 조립: 캐시 → 엔트리 → 컨텍스트.

chain 이 import 하는 유일한 KBO 모듈이다. 어떤 실패도 사용자 경로로 올리지 않는다.
조회에 실패하면 저장본으로 물러나고, 저장본조차 없으면 빈 목록을 돌려준다.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from baseball import clock
from baseball.config import Settings, get_settings
from baseball.context import KboEntry
from baseball.kbo_models import KboUpstreamChanged, ScheduleSnapshot, StandingsSnapshot

log = logging.getLogger(__name__)

TEAMS_PATH = Path(__file__).resolve().parents[2] / "data" / "kbo_teams.json"
STANDINGS_KIND = "kbo:standings"
SCHEDULE_KIND = "kbo:schedule"
MAX_ENTRIES = 3

# 프로세스 내 TTL 캐시. Streamlit 은 상호작용마다 스크립트를 다시 돌리므로
# 이게 없으면 같은 데이터에 DB 왕복이 매번 붙는다.
_MEM: dict[str, tuple[float, Any, str]] = {}


@lru_cache(maxsize=1)
def load_teams() -> dict[str, Any]:
    try:
        return json.loads(TEAMS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001
        log.warning("구단 정보 파일을 읽지 못했다: %s", exc)
        return {"teams": [], "as_of": "", "source": "", "license": "", "license_url": ""}


@lru_cache(maxsize=1)
def teams_by_code() -> dict[str, dict[str, Any]]:
    return {t["code"]: t for t in load_teams().get("teams", [])}


def season_year(settings: Settings, today: date) -> int:
    return settings.kbo_season_year or today.year


@lru_cache(maxsize=1)
def venue_index() -> tuple[tuple[str, str], ...]:
    """구장·연고지 이름 -> 구단 코드. 긴 이름을 먼저 본다(부분 일치 오판 방지).

    잠실은 LG·두산이 공유하므로 한 이름이 여러 코드를 가리킬 수 있다. 그 경우
    질의에 구단명이 함께 없으면 어느 팀이든 그 구장에서 열리는 경기를 고른다.
    """
    pairs: list[tuple[str, str]] = []
    for team in load_teams().get("teams", []):
        for key in ("stadium_short", "stadium", "stadium_secondary_short", "hometown"):
            name = (team.get(key) or "").strip()
            if name:
                pairs.append((name, team["code"]))
    return tuple(sorted(set(pairs), key=lambda p: -len(p[0])))


def find_venue(question: str) -> tuple[str, tuple[str, ...]] | None:
    """질의에서 구장·연고지 이름을 찾는다. (이름, 그 구장을 쓰는 구단 코드들)."""
    text = unicodedata.normalize("NFKC", question or "")
    for name, code in venue_index():
        if name in text:
            codes = tuple(c for n, c in venue_index() if n == name)
            return name, codes
    return None


def find_team(question: str) -> dict[str, Any] | None:
    """질의에서 구단을 찾는다. 별칭이 먼저, 없으면 구장·연고지로."""
    text = unicodedata.normalize("NFKC", question or "")
    best: tuple[int, dict[str, Any]] | None = None
    for team in load_teams().get("teams", []):
        for alias in team.get("aliases", []):
            if alias and alias in text and (best is None or len(alias) > best[0]):
                best = (len(alias), team)
    if best is not None:
        return best[1]
    found = find_venue(question)
    if found and len(found[1]) == 1:
        return teams_by_code().get(found[1][0])
    return None


def schedule_snapshot(settings: Settings, *, today: date | None = None,
                      client: Any = None) -> tuple[Any | None, str, str]:
    """일정 스냅샷을 캐시 3단 폴백으로 가져온다. entries_for 와 같은 창·같은 키를 쓴다."""
    today = today or clock.today_kst()
    year = season_year(settings, today)

    from baseball import kbo_naver

    def _fetch() -> Any:
        start = today - timedelta(days=settings.kbo_schedule_lookback_days)
        return kbo_naver.fetch_schedule(
            start=start, end=date(year, 12, 31), today=today,
            timeout=settings.kbo_http_timeout_seconds, client=client)

    return _cached(settings, SCHEDULE_KIND, settings.kbo_schedule_ttl_seconds, _fetch)


def pick_game(snap: Any, *, now: datetime, codes: tuple[str, ...] = (),
              venue: str | None = None, prefer: str = "upcoming") -> tuple[Any | None, str]:
    """경기 하나를 고른다. (경기, 관계). 못 고르면 (None, "none").

    관계는 "upcoming" | "last_finished" 다. 답변이 어느 경기인지 밝힐 수 있게
    함께 돌려준다 — 사용자가 오인을 바로 알아채는 유일한 수단이다.
    """
    if snap is None:
        return None, "none"

    def _matches(game: Any) -> bool:
        if codes and not any(c in (game.home_code, game.away_code) for c in codes):
            return False
        if venue and venue not in (game.stadium or ""):
            # 구장 약칭이 경기 stadium 문자열에 없을 수 있어 구단 코드로도 본다
            return not codes
        return True

    if prefer == "upcoming":
        for game in snap.upcoming(now):
            if _matches(game):
                return game, "upcoming"
    finished = [g for g in snap.games
                if g.is_regular and not g.cancel and g.game_date_time <= now
                and g.status_code in {"RESULT", "STARTED"} and _matches(g)]
    if finished:
        return max(finished, key=lambda g: g.game_date_time), "last_finished"
    if prefer != "upcoming":
        for game in snap.upcoming(now):
            if _matches(game):
                return game, "upcoming"
    return None, "none"


def _cached(settings: Settings, kind: str, ttl: int,
            fetch: Callable[[], Any]) -> tuple[Any | None, str, str]:
    """(스냅샷, freshness, as_of). 예외를 올리지 않는다."""
    now = time.time()
    hit = _MEM.get(kind)
    if hit and now - hit[0] < ttl:
        return hit[1], "cached", hit[2]

    try:
        snap = fetch()
        as_of = str(getattr(snap, "as_of", ""))
        _MEM[kind] = (now, snap, as_of)
        _store(settings, kind, snap, as_of)
        return snap, "live", as_of
    except KboUpstreamChanged as exc:
        log.error("KBO 상류 스키마 변경 의심 kind=%s: %s", kind, exc)
    except Exception as exc:                       # noqa: BLE001
        log.warning("KBO 조회 실패 kind=%s: %s", kind, type(exc).__name__)

    return _from_snapshot(settings, kind, ttl)


def _store(settings: Settings, kind: str, snap: Any, as_of: str) -> None:
    try:
        from baseball import db

        with db.connect(settings, register=False) as conn:
            db.put_snapshot(conn, kind, snap.model_dump(mode="json"),
                            as_of=as_of or None, source_url=_source_url())
    except Exception as exc:                       # noqa: BLE001
        log.debug("스냅샷 저장 생략: %s", type(exc).__name__)


def _from_snapshot(settings: Settings, kind: str, ttl: int) -> tuple[Any | None, str, str]:
    try:
        from baseball import db

        with db.connect(settings, register=False) as conn:
            row = db.latest_snapshot(conn, kind)
    except Exception:                              # noqa: BLE001
        return None, "unavailable", ""
    if not row:
        return None, "unavailable", ""

    payload, as_of = row["payload"], str(row["as_of"] or "")
    fetched = row["fetched_at"]
    try:
        age = (time.time() - fetched.timestamp()) if fetched else ttl + 1
    except Exception:                              # noqa: BLE001
        age = ttl + 1
    model = StandingsSnapshot if kind == STANDINGS_KIND else ScheduleSnapshot
    try:
        snap = model.model_validate(payload)
    except Exception:                              # noqa: BLE001
        return None, "unavailable", ""
    return snap, ("cached" if age <= ttl else "stale"), as_of


def _source_url() -> str:
    from baseball.kbo_naver import SOURCE_URL

    return SOURCE_URL


# --- 블록 본문 ---------------------------------------------------------------

def _standings_text(snap: StandingsSnapshot) -> str:
    lines = []
    for t in sorted(snap.teams, key=lambda x: x.ranking):
        gb = "-" if t.game_behind == 0 else f"{t.game_behind:.1f}"
        streak = f" {t.streak}" if t.streak else ""
        lines.append(
            f"{t.ranking}위 {t.team_short} {t.wins}승 {t.losses}패 {t.draws}무 "
            f"승률 {t.wra:.3f} 게임차 {gb} 최근5경기 {t.last_five or '-'}{streak}"
        )
    return "\n".join(lines)


def _schedule_text(games: list[Any], total: int, limit: int, label_team: str | None) -> str:
    head = f"남은 경기 {total}경기"
    if len(games) > limit:
        head += f" (가까운 {limit}경기만 표시)"
    lines = [head]
    codes = teams_by_code()
    for g in games[:limit]:
        away = codes.get(g.away_code, {}).get("short", g.away_code)
        home = codes.get(g.home_code, {}).get("short", g.home_code)
        when = g.game_date_time.strftime("%Y-%m-%d(%a) %H:%M")
        where = f" {g.stadium}" if g.stadium else ""
        lines.append(f"{when} {away} vs {home}{where}")
    return "\n".join(lines)


def _team_info_text(team: dict[str, Any], standing: Any | None) -> str:
    lines = [
        f"구단: {team['full']}",
        f"연고지: {team['hometown']}",
        f"홈구장: {team['stadium']}",
    ]
    if standing is not None:
        lines.append(
            f"현재 순위: {standing.ranking}위 ({standing.wins}승 {standing.losses}패 "
            f"{standing.draws}무, 승률 {standing.wra:.3f})"
        )
    meta = load_teams()
    lines.append(f"출처: {meta.get('source', '')} ({meta.get('license', '')}) {team.get('source_url', '')}")
    return "\n".join(lines)



def _roster_text(rows: list[dict[str, Any]], limit: int = 14) -> str:
    """등번호·포지션·핵심 기록만. 입문자가 읽을 수 있는 만큼으로 줄인다."""
    lines: list[str] = []
    for row in rows[:limit]:
        st = row.get("stats", {})
        no = row.get("back_number")
        head = f"{('#' + str(no)) if no is not None else '  '} {row['name']} ({row.get('position') or '-'})"
        if row.get("player_type") == "PITCHER":
            era, w, l, sv = st.get("era"), st.get("win"), st.get("lose"), st.get("save")
            detail = f"{st.get('games') or 0}경기 {w or 0}승 {l or 0}패"
            if sv:
                detail += f" {sv}세이브"
            if era is not None:
                detail += f" 평균자책 {era}"
        else:
            avg, hr, rbi = st.get("avg"), st.get("hr"), st.get("rbi")
            detail = f"{st.get('games') or 0}경기"
            if avg is not None:
                detail += f" 타율 {avg:.3f}"
            detail += f" {hr or 0}홈런 {rbi or 0}타점"
        lines.append(f"{head} — {detail}")
    if len(rows) > limit:
        lines.append(f"(이 밖에 {len(rows) - limit}명)")
    return "\n".join(lines)


# --- 공개 진입점 -------------------------------------------------------------

def entries_for(question: str, route: Any, *, settings: Settings | None = None,
                today: date | None = None, client: Any = None) -> list[KboEntry]:
    """라우터가 붙인 주제·구단 태그로 KBO 블록을 만든다. 실패해도 예외를 올리지 않는다."""
    settings = settings or get_settings()
    if not settings.kbo_data_enabled:
        return []
    topics = tuple(getattr(route, "topics", ()) or ())
    if not topics:
        return []
    today = today or clock.today_kst()
    year = season_year(settings, today)
    codes = teams_by_code()
    teams = [c for c in (getattr(route, "teams", ()) or ()) if c in codes]

    from baseball import kbo_naver

    def _standings() -> Any:
        return kbo_naver.fetch_standings(
            year, today=today, timeout=settings.kbo_http_timeout_seconds, client=client)

    def _schedule() -> Any:
        # 지난 경기까지 함께 받는다. 하이라이트가 지난 경기를 가리키기 때문이다.
        # 창을 호출부마다 다르게 주면 안 된다. _cached 의 키가 kind 문자열뿐이라
        # 창이 다른 두 조회가 같은 스냅샷 자리를 서로 덮어쓴다.
        start = today - timedelta(days=settings.kbo_schedule_lookback_days)
        return kbo_naver.fetch_schedule(
            start=start, end=date(year, 12, 31), today=today,
            timeout=settings.kbo_http_timeout_seconds, client=client)

    entries: list[KboEntry] = []
    standings_snap: Any | None = None
    standings_fresh = "unavailable"
    standings_as_of = ""

    def _ensure_standings() -> None:
        nonlocal standings_snap, standings_fresh, standings_as_of
        if standings_snap is None and standings_fresh == "unavailable":
            standings_snap, standings_fresh, standings_as_of = _cached(
                settings, STANDINGS_KIND, settings.kbo_standings_ttl_seconds, _standings)

    if "standings" in topics:
        _ensure_standings()
        if standings_snap is not None:
            entries.append(KboEntry(
                kind="standings", label=f"KBO {year} 정규시즌 순위",
                text=_standings_text(standings_snap), as_of=standings_as_of,
                source_label=kbo_naver.SOURCE_LABEL, source_url=kbo_naver.SOURCE_URL,
                freshness=standings_fresh,
            ))

    if "schedule" in topics:
        snap, fresh, as_of = _cached(
            settings, SCHEDULE_KIND, settings.kbo_schedule_ttl_seconds, _schedule)
        if snap is not None:
            remaining = snap.remaining(today)
            label = f"KBO {year} 남은 경기"
            if len(teams) == 1:
                code = teams[0]
                remaining = [g for g in remaining if code in (g.home_code, g.away_code)]
                label = f"{codes[code]['full']} 남은 경기"
            entries.append(KboEntry(
                kind="schedule", label=label,
                text=_schedule_text(remaining, len(remaining),
                                    settings.kbo_schedule_max_games,
                                    teams[0] if len(teams) == 1 else None),
                as_of=as_of, source_label=kbo_naver.SOURCE_LABEL,
                source_url=kbo_naver.SOURCE_URL, freshness=fresh,
            ))

    if "roster" in topics and teams:
        from baseball import kbo_players

        code = teams[0]
        pitchers = kbo_players.roster(code, "PITCHER")
        hitters = kbo_players.roster(code, "HITTER")
        if pitchers or hitters:
            wants_pitcher = bool(re.search(r"투수|선발|불펜|마무리", question))
            wants_hitter = bool(re.search(r"타자|타선|야수|포수|내야|외야", question))
            if wants_pitcher and not wants_hitter:
                what, total = "투수", len(pitchers)
                body = _roster_text(pitchers)
            elif wants_hitter and not wants_pitcher:
                what, total = "타자", len(hitters)
                body = _roster_text(hitters)
            else:
                # 한쪽만 나오면 명단을 물은 뜻이 아니다. 양쪽을 나눠서 보여준다.
                what, total = "선수", len(pitchers) + len(hitters)
                body = (f"[투수 {len(pitchers)}명]\n{_roster_text(pitchers, 8)}\n\n"
                        f"[타자 {len(hitters)}명]\n{_roster_text(hitters, 8)}")
            entries.append(KboEntry(
                kind="roster",
                label=f"{codes[code]['full']} {what} 명단 ({total}명, 이번 시즌 출전 기준)",
                text=body, as_of=kbo_players.captured_at(),
                source_label="네이버 스포츠", source_url=kbo_naver.SOURCE_URL,
                freshness="static",
            ))

    if "team_info" in topics and teams:
        _ensure_standings()
        meta = load_teams()
        for code in teams[:2]:
            standing = standings_snap.by_code(code) if standings_snap is not None else None
            entries.append(KboEntry(
                kind="team_info", label=f"{codes[code]['full']} 구단 정보",
                text=_team_info_text(codes[code], standing),
                as_of=meta.get("as_of", ""), source_label="위키백과",
                source_url=codes[code].get("source_url"), freshness="static",
            ))

    return entries[:MAX_ENTRIES]


def main(argv: list[str] | None = None) -> int:
    import argparse

    from baseball.context import format_context
    from baseball.router import route as route_fn

    ap = argparse.ArgumentParser(prog="python -m baseball.kbo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="구단 파일 점검. --net 이면 상류까지 확인한다")
    c.add_argument("--net", action="store_true")
    r = sub.add_parser("render", help="질문에 대해 실제로 주입될 컨텍스트를 출력한다")
    r.add_argument("question")
    args = ap.parse_args(argv)

    settings = get_settings()
    today = clock.today_kst()

    if args.cmd == "check":
        meta = load_teams()
        codes = sorted(teams_by_code())
        print(f"KBO teams ok ({len(codes)}팀, as_of={meta.get('as_of')}) {codes}")
        if not args.net:
            return 0
        from baseball import kbo_naver

        year = season_year(settings, today)
        s = kbo_naver.fetch_standings(year, today=today,
                                      timeout=settings.kbo_http_timeout_seconds)
        print(f"standings ok — {len(s.teams)}팀, 1위 {s.teams[0].team_short} {s.teams[0].wra:.3f}")
        sc = kbo_naver.fetch_schedule(start=today, end=date(year, 12, 31), today=today,
                                      timeout=settings.kbo_http_timeout_seconds)
        print(f"schedule ok — 조회 {len(sc.games)}경기, 잔여 {len(sc.remaining(today))}경기")
        print("불변식 20개 전부 PASS (모델 검증 통과)")
        return 0

    rt = route_fn(args.question)
    entries = entries_for(args.question, rt, settings=settings, today=today)
    print(f"kind={rt.kind} topics={getattr(rt, 'topics', ())} teams={getattr(rt, 'teams', ())}")
    print(format_context([], [], entries, kbo_max_tokens=settings.kbo_context_max_tokens) or "(빈 컨텍스트)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
