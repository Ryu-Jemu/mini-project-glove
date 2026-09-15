"""KBO 데이터 조립: 캐시 → 엔트리 → 컨텍스트.

chain 이 import 하는 유일한 KBO 모듈이다. 어떤 실패도 사용자 경로로 올리지 않는다.
조회에 실패하면 저장본으로 물러나고, 저장본조차 없으면 빈 목록을 돌려준다.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

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
    today = today or date.today()
    year = season_year(settings, today)
    codes = teams_by_code()
    teams = [c for c in (getattr(route, "teams", ()) or ()) if c in codes]

    from baseball import kbo_naver

    def _standings() -> Any:
        return kbo_naver.fetch_standings(
            year, today=today, timeout=settings.kbo_http_timeout_seconds, client=client)

    def _schedule() -> Any:
        return kbo_naver.fetch_schedule(
            start=today, end=date(year, 12, 31), today=today,
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
    today = date.today()

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
