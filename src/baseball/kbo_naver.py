"""네이버 스포츠 공개 JSON 엔드포인트 어댑터.

키가 필요 없고 헤더도 필요 없다. 여기서만 네트워크를 만지며, 실패는 전부
KboSourceError 로 좁혀 올린다. 캐시 계층이 그것을 받아 저장본으로 물러난다.

금지 사항(프로젝트 규약):
  - m.sports.naver.com 은 robots.txt 가 Disallow: / 이므로 HTML 을 요청하지 않는다.
    이 모듈이 쓰는 api-gw.sports.naver.com 은 robots.txt 가 없다(404).
  - koreabaseball.com·statiz.co.kr·espn.com·namu.wiki 는 어떤 경우에도 요청하지 않는다.
  - youtube.com/feeds/videos.xml 은 robots.txt 가 Disallow 이므로 쓰지 않는다.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from baseball.kbo_models import (
    Game,
    KboUpstreamChanged,
    ScheduleSnapshot,
    StandingsSnapshot,
    TeamStanding,
)

BASE = "https://api-gw.sports.naver.com"
STANDINGS_PATH = "/statistics/categories/kbo/seasons/{year}/teams"
SCHEDULE_PATH = "/schedule/games"
SCHEDULE_FIELDS = "basic,stadium,roundCode,broadChannel,homeStarterName,awayStarterName"
SOURCE_LABEL = "네이버 스포츠"
SOURCE_URL = "https://m.sports.naver.com/kbaseball/record/index"


class KboSourceError(RuntimeError):
    """네트워크·HTTP·JSON 실패. 스키마 변경은 KboUpstreamChanged 로 따로 구분한다."""


def _get(url: str, params: dict[str, Any] | None, timeout: float,
         client: httpx.Client | None) -> dict[str, Any]:
    own = client is None
    c = client or httpx.Client(timeout=timeout, follow_redirects=False)
    try:
        r = c.get(url, params=params)
        if r.status_code != 200:
            raise KboSourceError(f"HTTP {r.status_code} {url}")
        body = r.json()
    except KboSourceError:
        raise
    except Exception as exc:                       # 연결 실패·타임아웃·JSON 파손
        raise KboSourceError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        if own:
            c.close()
    if body.get("success") is not True or body.get("code") != 200:
        raise KboSourceError(f"응답이 성공이 아님: code={body.get('code')}")
    result = body.get("result")
    if not isinstance(result, dict):
        raise KboUpstreamChanged("result 가 객체가 아님")
    return result


def fetch_standings(year: int, *, today: date, timeout: float = 4.0,
                    client: httpx.Client | None = None) -> StandingsSnapshot:
    result = _get(BASE + STANDINGS_PATH.format(year=year), None, timeout, client)
    rows = result.get("seasonTeamStats")
    if not isinstance(rows, list):
        raise KboUpstreamChanged("seasonTeamStats 가 목록이 아님")
    return StandingsSnapshot(
        teams=[TeamStanding.model_validate(r) for r in rows],
        season=year, as_of=today, game_type=result.get("gameType", "REGULAR_SEASON"),
    )


def fetch_schedule(*, start: date, end: date, today: date, timeout: float = 4.0,
                   client: httpx.Client | None = None) -> ScheduleSnapshot:
    params = {
        "fields": SCHEDULE_FIELDS,
        "upperCategoryId": "kbaseball",
        "categoryId": "kbo",
        "fromDate": start.isoformat(),
        "toDate": end.isoformat(),
        "size": 1000,
    }
    result = _get(BASE + SCHEDULE_PATH, params, timeout, client)
    rows = result.get("games")
    if not isinstance(rows, list):
        raise KboUpstreamChanged("games 가 목록이 아님")
    return ScheduleSnapshot(
        games=[Game.model_validate(g) for g in rows],
        as_of=today, total=result.get("gameTotalCount"),
    )
