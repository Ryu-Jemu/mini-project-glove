"""연고지 맛집. Tavily 를 재사용하고, 좌표도 주소도 만들어 내지 않는다.

왜 이렇게 좁게 만드는가
------------------------
Tavily 가 주는 것은 장소 데이터가 아니라 **블로그 페이지**다. title 은 "잠실 맛집
베스트10 총정리" 같은 글 제목이지 상호명이 아니고, 좌표·주소·영업시간은 아예 없다.
본문에서 상호를 뽑아내려면 LLM 을 한 번 더 거쳐야 하는데 그게 정확히 할루시네이션이다.

그래서 다섯 겹으로 막는다.
  1. 점수 임계값을 올린다(PLACES_MIN_SCORE). Tavily 는 맞는 게 없어도 빈 배열 대신
     채움용 결과를 준다 — latest_info._keep_relevant 독스트링의 실측 기록이다.
  2. 질의를 사용자 문장이 아니라 경기·구장에서 결정론적으로 만든다.
  3. 이름은 검색 제공자가 준 제목 **그대로** 쓴다. 추출 단계가 설계에 없다.
  4. 주소·전화·영업시간 필드를 두지 않는다. 없는 필드는 틀릴 수 없다.
  5. url 이 없는 항목은 버린다. 출처로 확인할 수 없으면 카드로 만들지 않는다.

지도는 별개의 출처다. Google Maps Embed API 의 search 모드가 질의 문자열만으로
실제 가게 핀을 직접 그려 주므로, 좌표를 만들 필요도 상호를 추출할 필요도 없다.
Tavily 는 "왜 갈 만한가", 지도는 "실제로 어디에 뭐가 있는가" 를 맡는다.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

import httpx

from baseball import clock, latest_info
from baseball.config import Settings, get_settings
from baseball.context import LatestEntry

log = logging.getLogger(__name__)

# 한국어 600자(WEB_SNIPPET_MAX_CHARS)를 그대로 쓰면 4건 × 약 400토큰이 되어
# latest_context_max_tokens(800) 을 넘고, format_context 가 3번째부터 조용히 자른다.
PLACES_SNIPPET_MAX_CHARS = 220

NOTICE_LABEL = "맛집 정보 안내"
NOTICE_TEXT = (
    "아래 맛집 정보는 웹 검색 결과이며 영업 여부·위치·메뉴가 확인되지 않았습니다. "
    "각 항목의 출처 링크에서 직접 확인해야 합니다. "
    "상호는 검색 결과 제목에 적힌 그대로만 옮기고, 적혀 있지 않은 상호·주소·전화번호·"
    "영업시간·평점을 만들어 쓰지 마세요."
)


def build_query(stadium_short: str) -> str:
    """Tavily 질의. 블로그 제목에 흔한 구어체 이름을 쓴다."""
    return f"{stadium_short}야구장 근처 맛집 추천"


def map_query(stadium: str) -> str:
    """지도 질의. 구글에는 공식 구장명이 모호하지 않다."""
    return f"{stadium} 근처 맛집"


def _pid(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:10]


def search(
    *, stadium: str, stadium_short: str, settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> tuple[list[LatestEntry], list[dict[str, Any]], list[dict[str, Any]]]:
    """(컨텍스트 엔트리, 맛집 카드, 지도 미디어) 를 돌려준다.

    실패·비활성은 전부 빈 결과다. 예외를 사용자 경로로 올리지 않는다.
    """
    settings = settings or get_settings()
    if not settings.places_enabled:
        return [], [], []

    query = build_query(stadium_short)
    results = latest_info.tavily_raw(
        query, settings, client=client,
        # news 가 아니다. LIVE_WEB_RE 에 연고지·홈구장이 있어 자동 판정이 news 로 새는데
        # 맛집 블로그는 뉴스가 아니다. 게다가 news 와 include_domains 를 함께 주면
        # Tavily 가 0건을 돌려준다(실측 기록).
        topic="general",
        # 허용 도메인 목록은 KBO·언론사다. 맛집 블로그가 없으므로 제한을 걸면 0건이 확정이다.
        use_domains=False,
        min_score=settings.places_min_score,
        cache_prefix="places:",
        ttl_seconds=settings.places_cache_ttl_seconds,
    )

    today = clock.today_kst().isoformat()
    entries: list[LatestEntry] = []
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()

    for r in results[: settings.places_max_results]:
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        if not url or not title or url in seen:      # 출처 없는 항목은 만들지 않는다
            continue
        seen.add(url)
        score = r.get("score")
        confidence = "likely" if isinstance(score, (int, float)) and score >= 0.7 else "uncertain"
        content = (r.get("content") or "").strip()
        entries.append(LatestEntry(
            kind="place",
            label=f"맛집 정보: {title}",
            text=content[:PLACES_SNIPPET_MAX_CHARS],
            as_of=latest_info._as_of(r.get("published_date"), today),
            source_url=url,
            confidence=confidence,
        ))
        cards.append({
            "id": _pid(url), "name": title, "url": url,
            "snippet": content[:PLACES_SNIPPET_MAX_CHARS] or None,
            "confidence": confidence,
        })

    if not entries:
        return [], [], []

    # 안내를 맨 앞에 둔다. context.py 를 건드리지 않고 프롬프트에 지시를 얹는 방법이다.
    # MODEL_BLOCK_TEXT 가 모델 지식 경로에서 쓰는 것과 같은 수법이다.
    entries.insert(0, LatestEntry(
        kind="place", label=NOTICE_LABEL, text=NOTICE_TEXT, as_of=today,
        source_url=None, confidence="uncertain",
    ))

    media: list[dict[str, Any]] = []
    if settings.places_map_enabled:
        q = map_query(stadium)
        # 지도는 query 만 싣는다. Embed API 키는 클라이언트에 노출되므로 URL 조립은 UI 가 한다.
        media.append({"kind": "map", "id": _pid(q), "title": q, "query": q})

    return entries, cards, media
