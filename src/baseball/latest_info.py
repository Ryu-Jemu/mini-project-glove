"""최신정보 경로: YAML 스냅샷 + Tavily 웹 검색 → LatestEntry (모두 {context} 안으로 주입)."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence

import yaml

from baseball import db
from baseball.config import Settings, get_settings
from baseball.context import LatestEntry

log = logging.getLogger(__name__)

Freshness = Literal["static", "snapshot", "web", "live", "phase2_pending"]

LEAGUE_ONLY_RE = re.compile(
    r"ABS|자동\s*투구|피치\s*클락|체크\s*스윙|비디오\s*판독|엔트리|아시아\s*쿼터|연장|"
    r"포스트시즌|가을야구|3피트|승부치기|18인치",
    re.I,
)
LIVE_WEB_RE = re.compile(
    r"순위|승률|일정|남은\s*경기|잔여\s*경기|다음\s*경기|오늘.{0,8}경기|내일.{0,8}경기|"
    r"경기\s*결과|경기\s*시간|몇\s*시|하이라이트|선수\s*명단|로스터|엔트리\s*명단|뉴스"
    # 규칙집에 없는 인물·구단 정보(선수 소개, 연고지, 홈구장, 성적 등)
    r"|누구|어떤\s*선수|소속|연봉|프로필|몇\s*살|나이|출신|연고지|본거지|홈구장|창단|우승"
)

MAX_SNAPSHOT_ITEMS = 6
MAX_WEB_RESULTS = 5
WEB_CACHE_TTL_SECONDS = 6 * 3600


@lru_cache(maxsize=1)
def load_regulations(path: str | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    p = Path(path) if path else settings.base_dir / "data" / "league_regulations_2026.yaml"
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return list(data.get("items", []))


def match_snapshot(query: str, items: Sequence[dict[str, Any]] | None = None) -> list[LatestEntry]:
    items = items if items is not None else load_regulations()
    norm = unicodedata.normalize("NFKC", query or "")
    hits: list[LatestEntry] = []
    for item in items:
        keywords = [k for k in item.get("keywords", []) if k]
        if not any(re.search(re.escape(k).replace(r"\ ", r"\s*"), norm, re.I) for k in keywords):
            continue
        hits.append(LatestEntry(
            kind="snapshot",
            label=f"KBO 리그 규정 — {item['title']}",
            text=item["text"],
            as_of=str(item.get("as_of", "")),
            source_url=item.get("source_url"),
            confidence=item.get("confidence"),
        ))
        if len(hits) >= MAX_SNAPSHOT_ITEMS:
            break
    return hits


def _cache_get(settings: Settings, key: str) -> list[dict[str, Any]] | None:
    try:
        with db.connect(settings) as conn:
            row = conn.execute(
                "SELECT payload, fetched_at, ttl_seconds FROM assistant_cache WHERE key = %s", (key,)
            ).fetchone()
    except Exception as exc:                                  # DB 불가 시 캐시 미사용
        log.warning("assistant_cache 조회 실패: %s", exc)
        return None
    if not row:
        return None
    age = (datetime.now(timezone.utc) - row["fetched_at"]).total_seconds()
    if age > row["ttl_seconds"]:
        return None
    payload = row["payload"]
    return payload.get("results") if isinstance(payload, dict) else None


def _cache_put(settings: Settings, key: str, results: list[dict[str, Any]]) -> None:
    try:
        with db.connect(settings) as conn:
            conn.execute(
                """
                INSERT INTO assistant_cache (key, payload, ttl_seconds)
                VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET payload = EXCLUDED.payload,
                    fetched_at = now(), ttl_seconds = EXCLUDED.ttl_seconds
                """,
                (key, json.dumps({"results": results}, ensure_ascii=False), WEB_CACHE_TTL_SECONDS),
            )
            conn.commit()
    except Exception as exc:
        log.warning("assistant_cache 저장 실패: %s", exc)


def _extract_results(raw: Any) -> list[dict[str, Any]]:
    """langchain-tavily 는 결과가 없으면 dict 대신 안내 문자열을 반환한다."""
    if isinstance(raw, dict):
        return list(raw.get("results", []) or [])
    return []


def _tavily_search(query: str, settings: Settings) -> list[dict[str, Any]]:
    """1차: 허용 도메인 제한 검색 → 0건이면 2차: 제한 없이 재시도(출처는 항상 표기)."""
    from langchain_tavily import TavilySearch

    api_key = settings.tavily_api_key.get_secret_value()      # type: ignore[union-attr]
    domains = settings.tavily_include_domain_list
    attempts: list[dict[str, Any]] = []
    if domains:
        attempts.append({"include_domains": domains})
    attempts.append({})
    for extra in attempts:
        tool = TavilySearch(
            max_results=MAX_WEB_RESULTS, topic="news", search_depth="basic",
            tavily_api_key=api_key, **extra,
        )
        results = _extract_results(tool.invoke({"query": query}))
        if results:
            return results
    return []


def web_search(query: str, settings: Settings | None = None) -> list[LatestEntry]:
    """Tavily 1회 검색. 실패·비활성 시 빈 리스트(예외를 사용자 경로로 전파하지 않는다)."""
    settings = settings or get_settings()
    if not settings.web_search_enabled or settings.tavily_api_key is None:
        return []
    key = "tavily:" + hashlib.sha1(unicodedata.normalize("NFKC", query).encode()).hexdigest()
    cached = _cache_get(settings, key)
    results: list[dict[str, Any]]
    if cached is not None:
        results = cached
    else:
        try:
            results = _tavily_search(query, settings)
            if results:                      # 빈 결과는 캐시하지 않는다(일시적 실패 고착 방지)
                _cache_put(settings, key, results)
        except Exception as exc:
            log.warning("Tavily 검색 실패: %s", exc)
            return []

    today = date.today().isoformat()
    out: list[LatestEntry] = []
    for r in results[:MAX_WEB_RESULTS]:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        out.append(LatestEntry(
            kind="web",
            label=f"웹 검색: {r.get('title', '제목 없음')}",
            text=content[:600],
            as_of=today,
            source_url=r.get("url"),
            confidence="uncertain",
        ))
    return out


def route(query: str, settings: Settings | None = None) -> tuple[Freshness, list[LatestEntry]]:
    """chain 단계 1에서 route.kind 가 latest|mixed 일 때만 호출."""
    settings = settings or get_settings()
    norm = unicodedata.normalize("NFKC", query or "")

    if LIVE_WEB_RE.search(norm):
        entries = web_search(query, settings)
        if entries:
            return "web", entries
        return "phase2_pending", []

    if LEAGUE_ONLY_RE.search(norm):
        entries = match_snapshot(query)
        if entries:
            return "snapshot", entries
    return "static", []
