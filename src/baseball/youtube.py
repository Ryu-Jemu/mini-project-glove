"""YouTube Data API v3 어댑터. 여기서만 YouTube 네트워크를 만진다.

금지 사항(프로젝트 규약)
------------------------
  - youtube.com/feeds/videos.xml 은 robots.txt 가 Disallow 이므로 쓰지 않는다.
    tests/test_kbo_source.py 의 AST 검사가 이 규약을 기계적으로 강제한다.
  - search.list 는 호출당 100 units 다. 기본 쿼터가 하루 10,000 units 이므로
    하루 100회면 바닥난다. 한 사용자 세션이 다 태울 수 있어 쓰지 않는다.
  - 저장한 API 데이터는 source_snapshots 에 넣지 않는다. 그 테이블은 TTL 이 없고
    kind 별 최근 몇 건만 유지하는 구조라, YouTube ToS 의 30일 갱신·삭제 의무를
    보장할 수 없다. assistant_cache 의 행별 ttl_seconds 를 쓴다.

쿼터
----
  videos.list  = 1 unit. id 를 최대 50개까지 콤마로 묶어 한 번에 받는다.
                 개수와 무관하게 1 unit 이므로 배치가 곧 절약이다.
  status.embeddable 을 반드시 함께 받는다. 이 값이 false 인 영상을 iframe 으로
  띄우면 화면에 빈 상자가 남는다. 플레이어 대신 링크로 격하해야 한다.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import httpx

from baseball import clock

log = logging.getLogger(__name__)

BASE = "https://www.googleapis.com/youtube/v3"
VIDEOS_PATH = "/videos"
VIDEOS_PARTS = "snippet,contentDetails,status"
MAX_IDS_PER_CALL = 50                     # 이 이상은 API 가 거부한다
VIDEOS_UNIT_COST = 1

SOURCE_LABEL = "YouTube (YouTube Data API v3)"
WATCH_URL = "https://www.youtube.com/watch?v={vid}"

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<d>\d+)D)?T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?$"
)

# 프로세스 내 일일 사용량. KST 날짜가 바뀌면 리셋한다.
_UNITS: dict[date, int] = {}


class YoutubeSourceError(RuntimeError):
    """호출 실패. 도구가 받아서 빈 결과로 낮춘다."""


class YoutubeQuotaExceeded(YoutubeSourceError):
    """일일 쿼터 소진. 재시도가 아니라 내일을 기다려야 한다."""


class YoutubeUpstreamChanged(RuntimeError):
    """응답 모양이 바뀌었다. 전송 실패와 구분한다."""


def spend(units: int, budget: int) -> None:
    """예산을 넘기면 요청을 내보내기 전에 막는다."""
    today = clock.today_kst()
    for day in [d for d in _UNITS if d != today]:
        del _UNITS[day]                                  # 날짜가 바뀌면 리셋
    used = _UNITS.get(today, 0)
    if used + units > budget:
        raise YoutubeQuotaExceeded(f"일일 예산 초과: {used}+{units} > {budget}")
    _UNITS[today] = used + units


def used_today() -> int:
    return _UNITS.get(clock.today_kst(), 0)


def extract_video_id(url: str) -> str | None:
    """watch?v=..., youtu.be/..., /shorts/... 에서 11자 id 를 꺼낸다."""
    try:
        parsed = urlparse(url or "")
    except Exception:                                    # noqa: BLE001
        return None
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        cand = parsed.path.lstrip("/").split("/")[0]
        return cand if _VIDEO_ID_RE.match(cand) else None
    if host not in {"youtube.com", "music.youtube.com"}:
        return None
    if parsed.path == "/watch":
        cand = (parse_qs(parsed.query).get("v") or [""])[0]
        return cand if _VIDEO_ID_RE.match(cand) else None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
        return parts[1] if _VIDEO_ID_RE.match(parts[1]) else None
    return None


def parse_duration(iso: str) -> int | None:
    """ISO8601 기간(PT12M41S)을 초로. 못 읽으면 None."""
    m = _ISO_DURATION_RE.match(iso or "")
    if not m:
        return None
    g = {k: int(v) for k, v in m.groupdict(default="0").items()}
    return g["d"] * 86400 + g["h"] * 3600 + g["m"] * 60 + g["s"]


def format_duration(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _get(path: str, params: dict[str, Any], timeout: float,
         client: httpx.Client | None) -> dict[str, Any]:
    own = client is None
    c = client or httpx.Client(timeout=timeout, follow_redirects=False)
    try:
        r = c.get(f"{BASE}{path}", params=params)
        if r.status_code != 200:
            body = ""
            try:
                body = str(r.json())
            except Exception:                            # noqa: BLE001
                body = r.text[:200]
            if r.status_code == 403 and "quota" in body.lower():
                raise YoutubeQuotaExceeded(f"HTTP 403 quotaExceeded: {body[:200]}")
            raise YoutubeSourceError(f"HTTP {r.status_code}: {body[:200]}")
        payload = r.json()
    except (YoutubeSourceError, YoutubeUpstreamChanged):
        raise
    except Exception as exc:                             # 연결 실패·타임아웃·JSON 파손
        raise YoutubeSourceError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        if own:
            c.close()
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise YoutubeUpstreamChanged("items 배열이 없다")
    return payload


def fetch_video_details(
    video_ids: Iterable[str], *, api_key: str, timeout: float = 4.0,
    client: httpx.Client | None = None, budget: int = 8000,
) -> dict[str, dict[str, Any]]:
    """videos.list 한 번(1 unit)으로 최대 50개를 검증한다. id -> 정규화 dict."""
    ids = [v for v in dict.fromkeys(video_ids) if _VIDEO_ID_RE.match(v or "")]
    if not ids:
        return {}
    ids = ids[:MAX_IDS_PER_CALL]
    spend(VIDEOS_UNIT_COST, budget)
    payload = _get(VIDEOS_PATH,
                   {"part": VIDEOS_PARTS, "id": ",".join(ids), "key": api_key},
                   timeout, client)

    out: dict[str, dict[str, Any]] = {}
    for item in payload["items"]:
        if not isinstance(item, dict):
            continue
        vid = item.get("id")
        if not isinstance(vid, str):
            continue
        snippet = item.get("snippet") or {}
        details = item.get("contentDetails") or {}
        status = item.get("status") or {}
        if "embeddable" not in status:
            raise YoutubeUpstreamChanged("status.embeddable 이 없다 — part 를 확인하라")
        seconds = parse_duration(str(details.get("duration") or ""))
        thumbs = (snippet.get("thumbnails") or {})
        thumb = (thumbs.get("medium") or thumbs.get("default") or {}).get("url")
        out[vid] = {
            "id": vid,
            "title": str(snippet.get("title") or ""),
            "channel_title": str(snippet.get("channelTitle") or ""),
            "published_at": str(snippet.get("publishedAt") or "")[:10],
            "duration_seconds": seconds,
            "duration": format_duration(seconds),
            "embeddable": bool(status.get("embeddable")),
            "thumbnail_url": thumb,
            "url": WATCH_URL.format(vid=vid),
        }
    return out
