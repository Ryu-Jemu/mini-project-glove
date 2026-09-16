"""경기 하이라이트: Tavily 로 찾고 YouTube Data API 로 검증한다.

두 단계로 나누는 이유
---------------------
Tavily 는 무엇이 있는지 알려 주지만 그 영상이 **임베드 가능한지** 는 모른다.
status.embeddable 이 false 인 영상을 iframe 으로 띄우면 화면에 빈 상자가 남는다.
그래서 발견은 Tavily(추가 쿼터 0), 검증은 videos.list(1 unit)로 나눈다.

없을 때가 정상이다
------------------
TVING 이 2024~2026 KBO 유무선 중계권을 독점하므로 전 경기 하이라이트가 YouTube 에
있다고 보장할 수 없다. 점수 미달이면 빈 결과를 돌려주고, 도구가 "찾지 못했습니다" 를
모델에게 말한다. 근접한 다른 경기 영상으로 대체하지 않는다.

날짜 표기 주의
--------------
citations.RULE_NO_RE 가 `[1-9]\\.\\d{2}` 를 규칙 번호로 잡는다. 실측으로 확인했다 —
"[9.13 vs 삼성]" 은 규칙 9.13 인용으로 오인된다. 그래서 **모델이 읽는 본문에는
"9월 13일" 로 적고**, 원래 제목은 media 로만 내보낸다.
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any

import httpx

from baseball import latest_info, youtube
from baseball.config import Settings, get_settings
from baseball.context import KboEntry

log = logging.getLogger(__name__)

HIGHLIGHT_RE = re.compile(r"하이라이트|highlight|풀\s*영상|명장면", re.I)
MIN_MATCH_SCORE = 2
PUBLISH_WINDOW_DAYS = 2          # 경기 다음 날 올라오는 경우가 흔하다


def build_query(game_date: date, home: str, away: str) -> str:
    """결정론적 질의. 사용자 문장을 그대로 넘기지 않는다."""
    return f"{away} {home} {game_date.month}월 {game_date.day}일 야구 하이라이트"


def _date_tokens(d: date) -> tuple[str, ...]:
    """제목에 흔한 날짜 표기들. 매칭 점수 계산에만 쓰고 본문에는 넣지 않는다."""
    return (f"{d.month}.{d.day}", f"{d.month:02d}.{d.day:02d}",
            f"{d.month}/{d.day}", f"{d.month}월 {d.day}일")


def score_match(video: dict[str, Any], *, game_date: date, opponents: tuple[str, ...]) -> int:
    """제목·발행일로 점수를 매긴다. 2점 미만이면 그 경기 영상으로 보지 않는다.

    구단 이름은 가산점이 아니라 **필수 조건**이다. 실측으로 확인했다 —
    "LG 경기 하이라이트" 에 "[KT위즈 vs 한화이글스] 9.16(수)" 가 걸렸다.
    같은 날 다른 경기 영상이 날짜 토큰만으로 임계값을 넘었기 때문이다.
    """
    title = video.get("title") or ""
    if not HIGHLIGHT_RE.search(title):
        return 0                                    # 하이라이트가 아니면 볼 것도 없다
    if opponents and not any(name and name in title for name in opponents):
        return 0                                    # 다른 경기 영상을 그 경기인 척 내보내지 않는다
    score = 1
    published = video.get("published_at") or ""
    try:
        pub = date.fromisoformat(published)
    except ValueError:
        pub = None
    if pub is not None and game_date <= pub <= game_date + timedelta(days=PUBLISH_WINDOW_DAYS):
        score += 1
    if any(tok in title for tok in _date_tokens(game_date)):
        score += 1
    if any(name and name in title for name in opponents):
        score += 1
    return score


def _name_variants(*names: str) -> tuple[str, ...]:
    """구단 정식명과 별칭을 모은다.

    네이버는 "LG 트윈스" 로 주는데 영상 제목은 "LG" · "엘지" · "트윈스" 로 쓴다.
    정식명만으로 찾으면 제대로 된 영상도 놓친다.
    """
    from baseball import kbo

    out: list[str] = []
    for team in kbo.load_teams().get("teams", []):
        if not any(n and (n == team.get("full") or n in team.get("aliases", [])) for n in names):
            continue
        out.append(team.get("full") or "")
        out.extend(team.get("aliases", []))
        out.append(team.get("short") or "")
    out.extend(n for n in names if n)
    return tuple(sorted({o for o in out if o}, key=len, reverse=True))


def find(
    *, game_date: date, home_name: str, away_name: str,
    settings: Settings | None = None,
    client: httpx.Client | None = None, tavily_client: httpx.Client | None = None,
) -> tuple[list[KboEntry], list[dict[str, Any]]]:
    """(컨텍스트 엔트리, 영상 미디어). 실패·비활성·미발견은 전부 빈 결과다."""
    settings = settings or get_settings()
    if not settings.highlights_enabled:
        return [], []

    query = build_query(game_date, home_name, away_name)
    results = latest_info.tavily_raw(
        query, settings, client=tavily_client,
        topic="general", use_domains=False,
        cache_prefix="yt:find:", ttl_seconds=settings.youtube_cache_ttl_seconds,
    )

    ids: list[str] = []
    for r in results:
        vid = youtube.extract_video_id(r.get("url") or "")
        if vid and vid not in ids:
            ids.append(vid)
    if not ids:
        return [], []

    key = settings.youtube_kbo_api_key
    try:
        details = youtube.fetch_video_details(
            ids, api_key=key.get_secret_value(),          # type: ignore[union-attr]
            timeout=settings.youtube_http_timeout_seconds, client=client,
        )
    except Exception as exc:                              # noqa: BLE001
        log.warning("YouTube 검증 실패: %s", exc)
        return [], []

    opponents = _name_variants(home_name, away_name)
    scored = [
        (score_match(v, game_date=game_date, opponents=opponents), v)
        for v in details.values()
    ]
    picked = [v for s, v in sorted(scored, key=lambda t: -t[0]) if s >= MIN_MATCH_SCORE]
    picked = picked[: settings.highlight_max_videos]
    if not picked:
        return [], []

    # 모델이 읽는 본문. 날짜를 "9.13" 이 아니라 "9월 13일" 로 적는다(모듈 독스트링 참조).
    when = f"{game_date.month}월 {game_date.day}일"
    lines = [f"{when} {away_name} 대 {home_name} 경기 하이라이트 영상 {len(picked)}건을 찾았다."]
    for v in picked:
        state = "재생 가능" if v["embeddable"] else "외부 재생 제한"
        lines.append(f"- 길이 {v['duration'] or '미상'} · {state} · 채널 {v['channel_title']}")
    lines.append("영상 제목과 주소는 화면에 그대로 표시되므로 본문에 옮겨 적지 않는다.")

    entry = KboEntry(
        kind="highlight",
        label=f"{when} {away_name} 대 {home_name} 하이라이트",
        text="\n".join(lines),
        as_of=game_date.isoformat(),
        source_label=youtube.SOURCE_LABEL,
        source_url=picked[0]["url"],
        freshness="live",
    )
    media = [{
        "kind": "video", "id": v["id"], "title": v["title"], "url": v["url"],
        "published_at": v["published_at"], "duration": v["duration"],
        "embeddable": v["embeddable"], "thumbnail_url": v["thumbnail_url"],
    } for v in picked]
    return [entry], media
