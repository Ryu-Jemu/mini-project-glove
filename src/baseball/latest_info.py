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

import httpx
import yaml

from baseball import clock, db
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

TAVILY_URL = "https://api.tavily.com/search"
# 실측 응답이 0.8~1.1초라 8초면 넉넉하다. 답변 경로를 오래 붙잡지 않는다.
TAVILY_TIMEOUT = httpx.Timeout(8.0, connect=3.0)
# kbo_naver.py 가 robots 정책상 직접 수집을 금지한 곳은 검색 결과에서도 받지 않는다.
TAVILY_EXCLUDE_DOMAINS = ("namu.wiki",)
WEB_SNIPPET_MAX_CHARS = 600
# 관련 결과와 채움용 결과를 가르는 선. 실측으로 쓸 만한 결과는 0.6 이상, 잡음은 0.16 이하였다.
WEB_MIN_SCORE = 0.5


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


def _cache_put(settings: Settings, key: str, results: list[dict[str, Any]],
               *, ttl_seconds: int = WEB_CACHE_TTL_SECONDS) -> None:
    try:
        with db.connect(settings) as conn:
            conn.execute(
                """
                INSERT INTO assistant_cache (key, payload, ttl_seconds)
                VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET payload = EXCLUDED.payload,
                    fetched_at = now(), ttl_seconds = EXCLUDED.ttl_seconds
                """,
                (key, json.dumps({"results": results}, ensure_ascii=False), ttl_seconds),
            )
            conn.commit()
    except Exception as exc:
        log.warning("assistant_cache 저장 실패: %s", exc)


def cache_get(settings: Settings, key: str) -> list[dict[str, Any]] | None:
    """assistant_cache 읽기. 미스는 None, 저장된 빈 결과는 [] 다.

    이 구분이 곧 부정 캐시다 — "아직 영상이 없다" 도 사실이라 저장할 값이 있고,
    저장된 빈 결과와 캐시 미스를 섞으면 매번 다시 조회하게 된다.
    """
    return _cache_get(settings, key)


def cache_put(settings: Settings, key: str, results: list[dict[str, Any]],
              *, ttl_seconds: int) -> None:
    """assistant_cache 쓰기. tavily_raw 와 달리 **빈 결과도 저장한다**.

    tavily_raw 가 빈 결과를 안 넣는 것은 일시적 웹 검색 실패를 고착시키지 않기
    위해서다. 하이라이트는 반대다 — 0건이 흔한 정상 결과라 짧은 TTL 로 저장해
    같은 질문이 매번 쿼터를 태우는 것을 막는다. 대신 TTL 을 호출자가 정한다.
    """
    _cache_put(settings, key, results, ttl_seconds=ttl_seconds)


class WebSearchError(RuntimeError):
    """Tavily 호출 실패. web_search 가 잡아서 빈 결과로 낮춘다."""


def _extract_results(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        return [r for r in (raw.get("results") or []) if isinstance(r, dict)]
    return []


def _post(body: dict[str, Any], api_key: str, client: httpx.Client | None) -> list[dict[str, Any]]:
    """키는 Authorization 헤더로만 보낸다. 본문 api_key 필드는 Tavily 가 폐기했고,
    본문은 캐시·로그에 남을 수 있어 비밀을 담기에 적절하지 않다."""
    own = client is None
    c = client or httpx.Client(timeout=TAVILY_TIMEOUT)
    try:
        r = c.post(TAVILY_URL, json=body,
                   headers={"Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"})
        if r.status_code != 200:
            raise WebSearchError(f"HTTP {r.status_code}")
        payload = r.json()
    except WebSearchError:
        raise
    except Exception as exc:                       # 연결 실패·타임아웃·JSON 파손
        raise WebSearchError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        if own:
            c.close()
    return _extract_results(payload)


def _tavily_search(
    query: str, settings: Settings, *, client: httpx.Client | None = None,
    topic: str | None = None, use_domains: bool = True, min_score: float = WEB_MIN_SCORE,
) -> list[dict[str, Any]]:
    """도메인 제한 → 무제한 → (news 였다면) general 순으로 최대 세 번 시도한다.

    예전에는 두 시도 모두 topic="news" 로 고정돼 있었다. 그런데 news 와 include_domains 를
    함께 주면 Tavily 가 0건을 돌려준다(실측). 그래서 1차가 늘 비고 매번 2차로 떨어졌다.
    시간에 민감한 질문에만 news 를 쓰고, 그 경우에만 general 재시도를 붙인다.
    news 결과에는 published_date 가 실려 as_of 를 실제 발행일로 채울 수 있다.
    """
    api_key = settings.tavily_api_key.get_secret_value()      # type: ignore[union-attr]
    domains = settings.tavily_include_domain_list
    # topic 을 명시하면 자동 판정을 우회한다. 맛집 질의에는 news 가 부적합한데
    # LIVE_WEB_RE 에 연고지·홈구장이 있어 그냥 두면 news 로 샌다.
    topic = topic or ("news" if LIVE_WEB_RE.search(unicodedata.normalize("NFKC", query))
                      else "general")

    def body(topic_: str, use_domains: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "query": query,
            "max_results": MAX_WEB_RESULTS,
            "search_depth": "basic",
            "topic": topic_,
            "exclude_domains": list(TAVILY_EXCLUDE_DOMAINS),
        }
        if topic_ == "news":
            out["include_published_date"] = True
        if use_domains and domains:
            out["include_domains"] = domains
        return out

    # use_domains=False 면 도메인 제한 시도를 아예 건너뛴다. 허용 도메인 목록은
    # KBO·언론사라 맛집 블로그가 없어, 제한을 걸면 0건이 확정이다.
    attempts = [body(topic, True)] if use_domains else []
    attempts.append(body(topic, False))
    if topic == "news":
        attempts.append(body("general", False))

    for attempt in attempts:
        results = _keep_relevant(_post(attempt, api_key, client), min_score)
        if results:
            return results
    return []


def _keep_relevant(results: list[dict[str, Any]],
                   min_score: float = WEB_MIN_SCORE) -> list[dict[str, Any]]:
    """점수 미달 결과를 버린다.

    Tavily 는 맞는 게 없어도 빈 배열 대신 채움용 결과를 준다. 실측 예로 "야구에서 인필드
    플라이 규칙" 을 허용 도메인 안에서 찾으면 0.153·0.025·0.024 짜리 영어 어휘 영상과
    보드게임이 5건 돌아온다. 이걸 성공으로 치면 사다리가 거기서 멈춰 다음 시도를 못 가고,
    잡음이 그대로 근거 자료로 들어간다. 같은 질문을 무제한으로 찾으면 0.919 가 나온다.
    """
    return [r for r in results if float(r.get("score") or 0.0) >= min_score]


def _as_of(raw: Any, today: str) -> str:
    """LatestEntry.as_of 는 YYYY-MM-DD 다. Tavily 는 RFC 822 로 준다.

    "Mon, 24 Aug 2026 00:00:00 GMT" 를 그냥 자르면 "Mon, 24 Au" 가 되어 날짜가 아니게 된다.
    """
    text = str(raw or "").strip()
    if not text:
        return today
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(text).date().isoformat()
    except Exception:                                         # noqa: BLE001
        return today


def tavily_raw(
    query: str, settings: Settings | None = None, *, client: httpx.Client | None = None,
    topic: str | None = None, use_domains: bool = True, min_score: float = WEB_MIN_SCORE,
    cache_prefix: str = "tavily:", ttl_seconds: int = WEB_CACHE_TTL_SECONDS,
) -> list[dict[str, Any]]:
    """캐시 → Tavily 사다리 → 점수 필터 → 캐시 저장. 결과 dict 를 그대로 돌려준다.

    맛집·영상 도구가 같은 HTTP·캐시 경로를 재사용하기 위한 공개 진입점이다.
    web_search 와 달리 LatestEntry 로 바꾸지 않는다. 용도마다 label·kind·스니펫
    상한이 다르기 때문이다.

    예외를 사용자 경로로 전파하지 않는다. 실패는 빈 리스트다.
    """
    settings = settings or get_settings()
    if not settings.web_search_enabled or settings.tavily_api_key is None:
        return []
    key = cache_prefix + hashlib.sha1(unicodedata.normalize("NFKC", query).encode()).hexdigest()
    cached = _cache_get(settings, key)
    if cached is not None:
        return cached
    try:
        results = _tavily_search(query, settings, client=client, topic=topic,
                                 use_domains=use_domains, min_score=min_score)
    except Exception as exc:
        log.warning("Tavily 검색 실패: %s", exc)
        return []
    if results:                          # 빈 결과는 캐시하지 않는다(일시적 실패 고착 방지)
        _cache_put(settings, key, results, ttl_seconds=ttl_seconds)
    return results


def web_search(
    query: str, settings: Settings | None = None, *, client: httpx.Client | None = None
) -> list[LatestEntry]:
    """Tavily 1회 검색. 실패·비활성 시 빈 리스트(예외를 사용자 경로로 전파하지 않는다)."""
    settings = settings or get_settings()
    results = tavily_raw(query, settings, client=client)

    today = clock.today_kst().isoformat()
    out: list[LatestEntry] = []
    for r in results[:MAX_WEB_RESULTS]:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        # 발행일이 있으면 그걸 쓴다. 오늘로 찍으면 오래된 기사가 최신인 척하게 된다.
        as_of = _as_of(r.get("published_date"), today)
        score = r.get("score")
        out.append(LatestEntry(
            kind="web",
            label=f"웹 검색: {r.get('title', '제목 없음')}",
            text=content[:WEB_SNIPPET_MAX_CHARS],
            as_of=as_of,
            source_url=r.get("url"),
            confidence="likely" if isinstance(score, (int, float)) and score >= 0.7 else "uncertain",
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
