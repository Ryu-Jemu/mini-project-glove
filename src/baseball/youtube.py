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
  playlistItems.list = 1 unit. 구단·리그 공식 채널의 **업로드 재생목록**을 읽는다.
                 search.list(100 units) 를 쓰지 않고도 그 채널의 최신 영상을 전부 본다.
                 설계서v1.md 12절이 정한 발견 경로가 이것이다.
  channels.list = 1 unit. 핸들 -> 채널 id 해석. 값이 불변이라 캐시에 오래 둔다.
  videos.list  = 1 unit. id 를 최대 50개까지 콤마로 묶어 한 번에 받는다.
                 개수와 무관하게 1 unit 이므로 배치가 곧 절약이다.
  status.embeddable 을 반드시 함께 받는다. 이 값이 false 인 영상을 iframe 으로
  띄우면 화면에 빈 상자가 남는다. 플레이어 대신 링크로 격하해야 한다.

시각은 전부 KST 로 바꿔서 내보낸다
----------------------------------
API 는 RFC3339(UTC) 로 준다. 앞 10자를 자르면 하루가 틀린다 — "2026-09-16T15:30:00Z"
는 KST 로 9월 17일이다. 경기 날짜(KST)와 맞대어 보는 값이므로 to_kst_date 로 옮긴다.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
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

CHANNELS_PATH = "/channels"
CHANNELS_PARTS = "snippet,contentDetails"
CHANNELS_UNIT_COST = 1

PLAYLIST_ITEMS_PATH = "/playlistItems"
PLAYLIST_ITEMS_PARTS = "snippet,contentDetails"
PLAYLIST_ITEMS_UNIT_COST = 1
MAX_PLAYLIST_RESULTS = 50                 # 이 이상은 API 가 거부한다

# 재생목록에 남는 껍데기 항목. 상류 변화가 아니라 정상 상태다.
SKIP_TITLES = frozenset({"Deleted video", "Private video"})

SOURCE_LABEL = "YouTube (YouTube Data API v3)"
WATCH_URL = "https://www.youtube.com/watch?v={vid}"

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")
_UPLOADS_ID_RE = re.compile(r"^UU[A-Za-z0-9_-]{22}$")
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


def to_kst_date(rfc3339: str) -> date | None:
    """RFC3339(UTC) 를 KST 날짜로. 못 읽으면 None.

    앞 10자를 자르면 안 된다. "2026-09-16T15:30:00Z" 는 KST 로 9월 17일이고,
    그 하루 차이가 경기 날짜 대조를 그대로 뒤집는다.
    """
    raw = (rfc3339 or "").strip()
    if not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:            # 상류가 오프셋을 뺀 적은 없지만 방어한다
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(clock.KST).date()


def uploads_playlist_id(channel_id: str) -> str | None:
    """채널 id 의 UC 를 UU 로 바꾸면 업로드 재생목록 id 다. 호출 0 units.

    YouTube 가 보장하는 규칙이라 channels.list 를 부르지 않고도 얻는다.
    """
    cid = (channel_id or "").strip()
    return "UU" + cid[2:] if _CHANNEL_ID_RE.match(cid) else None


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
         client: httpx.Client | None, *, require_items: bool = True) -> dict[str, Any]:
    """require_items=False 는 '0건이 정상' 인 조회용이다.

    channels.list 는 핸들이 없으면 items 키 자체를 빼고 200 을 준다. 그걸
    상류 변화로 보면 오타 하나가 YoutubeUpstreamChanged 로 올라온다.
    """
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
    if not isinstance(payload, dict):
        raise YoutubeUpstreamChanged("응답이 객체가 아니다")
    if not isinstance(payload.get("items"), list):
        if require_items:
            raise YoutubeUpstreamChanged("items 배열이 없다")
        payload = {**payload, "items": []}
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
        published_utc = str(snippet.get("publishedAt") or "")
        published_kst = to_kst_date(published_utc)
        thumbs = (snippet.get("thumbnails") or {})
        thumb = (thumbs.get("medium") or thumbs.get("default") or {}).get("url")
        out[vid] = {
            "id": vid,
            "title": str(snippet.get("title") or ""),
            "channel_title": str(snippet.get("channelTitle") or ""),
            "published_at": published_kst.isoformat() if published_kst else "",
            "published_at_utc": published_utc,
            "duration_seconds": seconds,
            "duration": format_duration(seconds),
            "embeddable": bool(status.get("embeddable")),
            "thumbnail_url": thumb,
            "url": WATCH_URL.format(vid=vid),
        }
    return out


def resolve_channel(
    *, api_key: str, handle: str | None = None, username: str | None = None,
    channel_id: str | None = None, timeout: float = 4.0,
    client: httpx.Client | None = None, budget: int = 8000,
) -> dict[str, str]:
    """channels.list 한 번(1 unit)으로 채널 id·업로드 재생목록·제목을 받는다.

    운영 경로에서는 거의 부르지 않는다. data/kbo_youtube_channels.json 에
    channel_id 가 적혀 있으면 uploads_playlist_id() 가 0 units 로 끝내고,
    비어 있을 때만 여기로 와서 결과를 캐시에 오래 남긴다.

    핸들은 구단이 리브랜딩하면 바뀌지만 채널 id 는 바뀌지 않는다. 그래서 한 번
    해석한 값을 JSON 에 박아 두는 것이 가장 안정적이다.
    """
    if channel_id:
        params = {"part": CHANNELS_PARTS, "id": channel_id, "key": api_key}
        asked = channel_id
    elif handle:
        # forHandle 은 @ 를 포함해도 되고 빼도 된다. 붙여서 보낸다.
        asked = handle if handle.startswith("@") else f"@{handle}"
        params = {"part": CHANNELS_PARTS, "forHandle": asked, "key": api_key}
    elif username:
        asked = username
        params = {"part": CHANNELS_PARTS, "forUsername": username, "key": api_key}
    else:
        raise ValueError("handle · username · channel_id 중 하나는 있어야 한다")

    spend(CHANNELS_UNIT_COST, budget)
    payload = _get(CHANNELS_PATH, params, timeout, client, require_items=False)
    items = [i for i in payload["items"] if isinstance(i, dict)]
    if not items:
        raise YoutubeSourceError(f"채널을 찾지 못했다: {asked}")

    item = items[0]
    cid = str(item.get("id") or "")
    related = ((item.get("contentDetails") or {}).get("relatedPlaylists") or {})
    uploads = str(related.get("uploads") or "") or (uploads_playlist_id(cid) or "")
    if not _UPLOADS_ID_RE.match(uploads):
        raise YoutubeUpstreamChanged(f"업로드 재생목록 id 를 읽지 못했다: {asked}")
    return {
        "channel_id": cid,
        "uploads_playlist_id": uploads,
        "title": str((item.get("snippet") or {}).get("title") or ""),
    }


def fetch_playlist_items(
    playlist_id: str, *, api_key: str, max_results: int = 15, timeout: float = 4.0,
    client: httpx.Client | None = None, budget: int = 8000,
) -> list[dict[str, Any]]:
    """업로드 재생목록의 최신 항목을 한 번(1 unit) 읽는다.

    업로드 재생목록은 최신순으로 정렬돼 오므로 pageToken 을 따라가지 않는다.
    한 페이지면 이틀치 업로드를 덮는다.

    업로드 시각은 contentDetails.videoPublishedAt 이다. snippet.publishedAt 은
    **재생목록에 추가된 시각**이라 다른 값이다 — 업로드 목록에서는 대개 같지만
    계약상 같다는 보장이 없다.

    embeddable 은 여기서 알 수 없다(part 에 status 가 없다). 그래서 발견은
    이 함수가, 검증은 fetch_video_details 가 맡는 2단 구조를 유지한다.
    """
    pid = (playlist_id or "").strip()
    if not _UPLOADS_ID_RE.match(pid):
        raise ValueError(f"업로드 재생목록 id 가 아니다: {pid!r}")

    spend(PLAYLIST_ITEMS_UNIT_COST, budget)
    payload = _get(
        PLAYLIST_ITEMS_PATH,
        {"part": PLAYLIST_ITEMS_PARTS, "playlistId": pid,
         "maxResults": min(max(int(max_results), 1), MAX_PLAYLIST_RESULTS),
         "key": api_key},
        timeout, client,
    )

    out: list[dict[str, Any]] = []
    for item in payload["items"]:
        if not isinstance(item, dict):
            continue
        snippet = item.get("snippet") or {}
        details = item.get("contentDetails") or {}
        resource = snippet.get("resourceId") or {}
        if "resourceId" not in snippet:
            raise YoutubeUpstreamChanged("snippet.resourceId 가 없다 — part 를 확인하라")
        vid = str(resource.get("videoId") or "")
        title = str(snippet.get("title") or "")
        if not _VIDEO_ID_RE.match(vid) or title in SKIP_TITLES:
            continue                                 # 삭제·비공개 영상은 정상 상태다
        published = to_kst_date(str(details.get("videoPublishedAt") or ""))
        if published is None:
            continue                                 # 아직 공개되지 않은 예약 업로드
        out.append({
            "id": vid,
            "title": title,
            "channel_title": str(snippet.get("videoOwnerChannelTitle")
                                 or snippet.get("channelTitle") or ""),
            "published_at": published.isoformat(),
            "published_at_utc": str(details.get("videoPublishedAt") or ""),
            "playlist_id": pid,
        })
    return out
