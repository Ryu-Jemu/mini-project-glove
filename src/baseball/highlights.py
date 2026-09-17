"""경기 하이라이트: 구단·리그 공식 채널의 업로드 재생목록에서 찾는다.

왜 재생목록인가
----------------
예전에는 Tavily 일반 웹 검색으로 영상을 "발견" 했다. 그런데 그 경로는 상위 5건에
youtube.com 주소가 섞이기를 기다리는 것이라, 한국어 질의에서는 네이버 스포츠·연합
기사가 자리를 다 채워 사실상 아무것도 못 찾았다. **유튜브를 검색하는 코드가 없었다.**

설계서v1.md 12절이 정한 원안이 업로드 재생목록이다. search.list 는 호출당 100 units
(하루 10,000 이면 100회)라 쓸 수 없지만, playlistItems.list 는 1 unit 이고 구단·리그가
자기 채널에 하이라이트를 직접 올리므로 회수가 가장 확실하다.

  KBO 공식  "[한화이글스 vs SSG랜더스] 8.27(목) 야구 하이라이트｜2026 KBO 리그"
  LG        "[2026 KBO 리그 H/L] LG vs NC (09.16)"
  KIA       "오늘 최고의 장면은? | 9월 13일 하이라이트 | KIA vs 한화"

KBO 공식 채널은 전 경기를 위 형식으로 올린다. 구단 핸들이 하나도 안 맞아도 이
채널 하나면 회수가 선다 — 그래서 조회 순서의 마지막 보루로 둔다.

쿼터
----
홈 구단 -> 원정 구단 -> KBO 공식 순으로 보되 **한 건이라도 통과하면 멈춘다.**
흔한 경우 재생목록 1회(1 unit) + 검증 1회(1 unit) = 2 units. 최악이 4 units 다.

없을 때가 정상이다
------------------
TVING 이 2024~2026 KBO 유무선 중계권을 독점하므로 전 경기 하이라이트가 YouTube 에
있다고 보장할 수 없다. 점수 미달이면 빈 결과를 돌려주고, 도구가 "찾지 못했습니다" 를
모델에게 말한다. 근접한 다른 경기 영상으로 대체하지 않는다.

다만 0건은 **짧게만** 기억한다. 하이라이트는 경기 종료 몇 시간 뒤에 올라오므로,
경기 직후의 0건을 7일 캐시에 넣으면 그 경기는 일주일 내내 찾을 수 없게 된다.

날짜 표기 주의
--------------
citations.RULE_NO_RE 가 `[1-9]\\.\\d{2}` 를 규칙 번호로 잡는다. 실측으로 확인했다 —
"[9.13 vs 삼성]" 은 규칙 9.13 인용으로 오인된다. 그래서 **모델이 읽는 본문에는
"9월 13일" 로 적고**, 원래 제목은 media 로만 내보낸다.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from baseball import latest_info, youtube
from baseball.config import Settings, get_settings
from baseball.context import KboEntry

log = logging.getLogger(__name__)

CHANNELS_PATH = Path(__file__).resolve().parents[2] / "data" / "kbo_youtube_channels.json"
LEAGUE_CODE = "KBO"

# 구단·리그 채널이 실제로 쓰는 표기. 실측으로 모았다.
#   "[한화이글스 vs SSG랜더스] 8.27(목) 야구 하이라이트｜2026 KBO 리그"
#   "[2026 KBO 리그 H/L] LG vs NC (09.16)"        ← 약어만 쓴다
#   "9.17 LG vs 삼성 주요장면"                     ← '하이라이트' 가 아예 없다
HIGHLIGHT_RE = re.compile(
    r"하이라이트|highlights?|H\s*[/.]\s*L|"
    r"풀\s*(?:영상|경기|하이라이트)|\d+\s*분\s*(?:하이라이트|요약)|"
    r"주요\s*장면|명장면|최고의\s*장면|"
    r"경기\s*(?:영상|요약|다시\s*보기)|다시\s*보기",
    re.I,
)
# 구단 채널이 같은 날 함께 올리는 '그 경기 영상이 아닌' 것들. 걸리면 즉시 탈락이다.
# 이게 없으면 "9.17 경기 예고" 가 날짜 토큰만으로 임계값을 넘는다.
EXCLUDE_RE = re.compile(
    r"예고|프리뷰|preview|미리\s*보기|중계\s*안내|"
    r"인터뷰|기자\s*회견|브이로그|vlog|"
    r"응원|치어|팬\s*미팅|시구|이벤트|당첨|"
    r"생중계|LIVE\s*중계",
    re.I,
)
MIN_MATCH_SCORE = 3
PUBLISH_WINDOW_DAYS = 2          # 경기 다음 날 올라오는 경우가 흔하다

# 빈 결과로 끝나는 모든 경로의 이름. 운영 로그에서 grep 'stage=' 로 단계를 특정한다.
STAGES = ("disabled", "no_channel", "playlist_empty", "no_candidate",
          "verify_failed", "score_below_min", "ok")


# --------------------------------------------------------------------------- 채널


@lru_cache(maxsize=1)
def load_channels() -> dict[str, Any]:
    try:
        return json.loads(CHANNELS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001
        log.warning("유튜브 채널 파일을 읽지 못했다: %s", exc)
        return {"channels": [], "as_of": "", "source": ""}


@lru_cache(maxsize=1)
def channels_by_code() -> dict[str, dict[str, Any]]:
    return {c["code"]: c for c in load_channels().get("channels", []) if c.get("code")}


def _uploads_playlist(
    channel: dict[str, Any], *, settings: Settings, api_key: str,
    client: httpx.Client | None = None,
) -> str | None:
    """채널 dict -> 업로드 재생목록 id. 가능한 한 호출 없이 끝낸다.

    1) JSON 에 channel_id 가 있으면 UC->UU 변환으로 0 units.
    2) 캐시에 해석 결과가 있으면 0 units.
    3) 그때만 channels.list 1 unit. 결과는 30일 캐시한다(값이 사실상 불변이다).
    """
    direct = youtube.uploads_playlist_id(channel.get("channel_id") or "")
    if direct:
        return direct

    code = channel.get("code") or "?"
    key = f"yt:channel:{code}"
    cached = latest_info.cache_get(settings, key)
    if cached:
        return str(cached[0].get("uploads_playlist_id") or "") or None

    try:
        found = youtube.resolve_channel(
            api_key=api_key, handle=channel.get("handle"),
            username=channel.get("username"),
            timeout=settings.youtube_http_timeout_seconds, client=client,
            budget=settings.youtube_daily_unit_budget,
        )
    except Exception as exc:                        # noqa: BLE001
        # 핸들 하나가 틀려도 다른 채널은 계속 본다. KBO 공식이 남아 있으면 회수가 선다.
        log.warning("채널 해석 실패 code=%s handle=%s: %s",
                    code, channel.get("handle"), exc)
        return None
    latest_info.cache_put(settings, key, [found],
                          ttl_seconds=settings.youtube_channel_ttl_seconds)
    log.info("채널 해석 code=%s -> %s (%s)", code, found["channel_id"], found["title"])
    return found["uploads_playlist_id"]


def channel_order(home_code: str, away_code: str) -> list[dict[str, Any]]:
    """홈 -> 원정 -> 리그 공식. 앞에서 찾으면 뒤는 부르지 않는다."""
    by_code = channels_by_code()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for code in (home_code, away_code, LEAGUE_CODE):
        chan = by_code.get(code or "")
        if chan is not None and chan["code"] not in seen:
            seen.add(chan["code"])
            out.append(chan)
    return out


# --------------------------------------------------------------------------- 채점


def build_query(game_date: date, home: str, away: str) -> str:
    """결정론적 질의. 사용자 문장을 그대로 넘기지 않는다.

    재생목록이 비었을 때의 Tavily 폴백에서만 쓴다.
    """
    return f"{away} {home} {game_date.month}월 {game_date.day}일 야구 하이라이트"


def _date_tokens(d: date) -> tuple[str, ...]:
    """제목에 흔한 날짜 표기들. 매칭 점수 계산에만 쓰고 본문에는 넣지 않는다."""
    return (f"{d.month}.{d.day}", f"{d.month:02d}.{d.day:02d}",
            f"{d.month}/{d.day}", f"{d.month:02d}/{d.day:02d}",
            f"{d.month}월 {d.day}일", f"{d.month}월{d.day}일")


def _variants(name: str) -> tuple[str, ...]:
    """한 구단의 정식명·별칭·약칭을 모은다.

    네이버는 "LG" 로 주는데 영상 제목은 "LG트윈스"·"엘지"·"트윈스" 로도 쓴다.
    두 구단을 한 튜플에 섞지 않는 것이 중요하다 — 섞으면 "둘 중 하나만 맞아도
    통과" 가 되어 같은 날 다른 경기 영상이 그 경기인 척 들어온다.
    """
    from baseball import kbo

    if not name:
        return ()
    out: list[str] = [name]
    for team in kbo.load_teams().get("teams", []):
        if name != team.get("full") and name not in team.get("aliases", []):
            continue
        out.append(team.get("full") or "")
        out.extend(team.get("aliases", []))
        out.append(team.get("short") or "")
    return tuple(sorted({o for o in out if o}, key=len, reverse=True))


def score_match(
    video: dict[str, Any], *, game_date: date,
    home_variants: tuple[str, ...] = (), away_variants: tuple[str, ...] = (),
    trusted: bool = False, require_both: bool = False,
) -> int:
    """제목·발행일로 점수를 매긴다. MIN_MATCH_SCORE 미만이면 그 경기 영상이 아니다.

    trusted 는 '구단 공식 채널의 업로드 목록에서 왔다' 는 뜻이다. 후보 풀이 이미
    믿을 만하므로 하이라이트 키워드를 필수에서 가산점으로 낮춘다. 실측 근거 —
    "9.17 LG vs 삼성 주요장면" 은 예전 키워드 게이트에 걸려 0점이었다. 대신 날짜를
    필수로 올린다. 한 구단은 하루에 한 경기만 하므로 채널 + 날짜면 경기가 정해진다.

    require_both 는 '채널만으로는 어느 경기인지 모른다' 는 뜻이다. KBO 공식 채널은
    전 경기를 올리고 Tavily 폴백은 아무 채널이나 주므로, 그때는 두 구단 이름이
    **모두** 있어야 한다. 한쪽만 요구하면 같은 날 다른 경기가 통과한다 — 실측으로
    확인했다. "LG 경기 하이라이트" 에 "[KT위즈 vs 한화이글스] 9.16(수)" 가 걸렸다.
    """
    title = html.unescape(video.get("title") or "")
    if EXCLUDE_RE.search(title):
        return 0                                    # 예고·인터뷰는 그 경기 영상이 아니다

    has_keyword = bool(HIGHLIGHT_RE.search(title))
    has_token = any(tok in title for tok in _date_tokens(game_date))
    try:
        pub: date | None = date.fromisoformat(video.get("published_at") or "")
    except ValueError:
        pub = None
    in_window = (pub is not None
                 and game_date <= pub <= game_date + timedelta(days=PUBLISH_WINDOW_DAYS))

    if not (has_token or in_window):
        return 0                                    # 날짜가 안 맞으면 볼 것도 없다
    if not (has_token or has_keyword):
        return 0                                    # 그 채널의 그냥 다른 영상이다

    home_hit = any(v in title for v in home_variants)
    away_hit = any(v in title for v in away_variants)
    if require_both and not (home_hit and away_hit):
        return 0
    if not trusted:
        if not has_keyword:
            return 0                                # 믿을 수 없는 풀에서는 키워드가 필수다
        if not (home_hit or away_hit):
            return 0

    score = 1 if trusted else 0                     # 공식 채널이라는 사실 자체가 1점이다
    if has_keyword:
        score += 1
    if has_token:
        score += 1
    if in_window:
        score += 1
    if home_hit and away_hit:
        score += 2
    elif home_hit or away_hit:
        score += 1
    return score


# --------------------------------------------------------------------------- 탐색


def _game_key(game_date: date, away_code: str, home_code: str) -> str:
    """경기 하나를 가리키는 결정론적 캐시 키.

    질의 문장 해시가 아니라 경기 자체를 가리킨다. 짧고 읽을 수 있어 운영 중
    `SELECT key FROM assistant_cache WHERE key LIKE 'yt:hl:%'` 로 눈으로 확인된다.
    """
    return f"yt:hl:{game_date.isoformat()}:{away_code or '?'}:{home_code or '?'}"


def _stop(stage: str, trace: list[dict[str, Any]] | None,
          **fields: Any) -> tuple[list[KboEntry], list[dict[str, Any]]]:
    """빈 결과로 끝나는 모든 경로는 여기를 지난다. 단계가 로그에 반드시 남는다.

    예전에는 두 군데가 말없이 빈 리스트를 돌려줘서, 배포본에서 '발견 0건' 인지
    '검증 실패' 인지 '점수 미달' 인지 구분할 방법이 아예 없었다.
    """
    log.warning("하이라이트 0건 stage=%s %s", stage,
                " ".join(f"{k}={v}" for k, v in fields.items()))
    if trace is not None:
        trace.append({"stage": stage, **fields})
    return [], []


def _note(trace: list[dict[str, Any]] | None, stage: str, **fields: Any) -> None:
    if trace is not None:
        trace.append({"stage": stage, **fields})


def _candidates_from_channel(
    channel: dict[str, Any], *, game_date: date, settings: Settings, api_key: str,
    home_variants: tuple[str, ...], away_variants: tuple[str, ...],
    client: httpx.Client | None, trace: list[dict[str, Any]] | None,
) -> list[tuple[int, dict[str, Any]]]:
    """채널 하나에서 후보를 뽑아 점수를 매긴다. 실패는 빈 목록이다."""
    code = channel.get("code") or "?"
    playlist = _uploads_playlist(channel, settings=settings, api_key=api_key, client=client)
    if not playlist:
        _note(trace, "no_channel", code=code, handle=channel.get("handle"))
        return []

    n = settings.youtube_uploads_max_results
    key = f"yt:uploads:{playlist}:{n}"
    items = latest_info.cache_get(settings, key)
    cached = items is not None
    if not cached:
        try:
            items = youtube.fetch_playlist_items(
                playlist, api_key=api_key, max_results=n,
                timeout=settings.youtube_http_timeout_seconds, client=client,
                budget=settings.youtube_daily_unit_budget,
            )
        except Exception as exc:                     # noqa: BLE001
            log.warning("재생목록 조회 실패 code=%s playlist=%s: %s", code, playlist, exc)
            _note(trace, "playlist_empty", code=code, error=f"{type(exc).__name__}: {exc}")
            return []
        latest_info.cache_put(settings, key, items,
                              ttl_seconds=settings.youtube_uploads_ttl_seconds)

    items = items or []
    require_both = code == LEAGUE_CODE
    scored = [
        (score_match(v, game_date=game_date, home_variants=home_variants,
                     away_variants=away_variants, trusted=True, require_both=require_both), v)
        for v in items
    ]
    passed = [(s, v) for s, v in scored if s >= MIN_MATCH_SCORE]
    top = max((s for s, _ in scored), default=0)
    log.info("하이라이트 채점 code=%s 캐시=%s 후보=%d 통과=%d 최고점=%d units=%d",
             code, "hit" if cached else "miss", len(items), len(passed), top,
             youtube.used_today())
    if not passed:
        log.debug("하이라이트 탈락 code=%s %s", code,
                  [(v.get("title", "")[:60], s)
                   for s, v in sorted(scored, key=lambda t: -t[0])[:5]])
    _note(trace, "channel", code=code, cached=cached, received=len(items),
          passed=len(passed), top=top)
    return passed


def _tavily_fallback(
    *, game_date: date, home_name: str, away_name: str, settings: Settings,
    tavily_client: httpx.Client | None, trace: list[dict[str, Any]] | None,
) -> list[tuple[int, dict[str, Any]]]:
    """재생목록이 전부 비었을 때만 도는 폴백. 추가 YouTube 쿼터를 쓰지 않는다.

    예전의 유일한 발견 경로였다. 회수가 낮아 주 경로에서 물러났지만, 등록되지
    않은 채널이 올린 영상을 주워 올 수는 있어 남겨 둔다. 캐시 TTL 은 짧다 —
    7일을 쓰면 유튜브 주소가 없는 기사 5건이 그 경기를 일주일 동안 막는다.

    여기서는 제목을 모르므로 채점하지 않는다. videos.list 검증 뒤에 매긴다.
    """
    if not settings.web_search_enabled:
        return []
    query = build_query(game_date, home_name, away_name)
    results = latest_info.tavily_raw(
        query, settings, client=tavily_client, topic="general", use_domains=False,
        cache_prefix="yt:find:", ttl_seconds=settings.youtube_uploads_ttl_seconds,
    )
    ids: list[str] = []
    for r in results:
        vid = youtube.extract_video_id(r.get("url") or "")
        if vid and vid not in ids:
            ids.append(vid)
    log.info("하이라이트 폴백 stage=fallback tavily=%d 유튜브id=%d", len(results), len(ids))
    _note(trace, "fallback", results=len(results), ids=len(ids))
    return [(MIN_MATCH_SCORE, {"id": vid, "title": "", "published_at": ""}) for vid in ids]


def find(
    *, game_date: date, home_name: str, away_name: str,
    home_code: str = "", away_code: str = "",
    settings: Settings | None = None,
    client: httpx.Client | None = None, tavily_client: httpx.Client | None = None,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[list[KboEntry], list[dict[str, Any]]]:
    """(컨텍스트 엔트리, 영상 미디어). 실패·비활성·미발견은 전부 빈 결과다."""
    settings = settings or get_settings()
    if not settings.highlights_enabled:
        return _stop("disabled", trace)

    key = settings.youtube_kbo_api_key
    api_key = key.get_secret_value()                  # type: ignore[union-attr]

    home_variants, away_variants = _variants(home_name), _variants(away_name)
    cache_key = _game_key(game_date, away_code, home_code)
    cached_media = latest_info.cache_get(settings, cache_key)
    if cached_media is not None:
        log.info("하이라이트 캐시 hit key=%s 건수=%d", cache_key, len(cached_media))
        _note(trace, "cache", key=cache_key, hit=len(cached_media))
        if not cached_media:
            return [], []
        return [_entry(game_date, home_name, away_name, cached_media)], list(cached_media)

    channels = channel_order(home_code, away_code)
    log.info("하이라이트 탐색 date=%s %s(%s) 대 %s(%s) 채널=%s",
             game_date, away_name, away_code, home_name, home_code,
             [c.get("code") for c in channels])
    if not channels:
        latest_info.cache_put(settings, cache_key, [],
                              ttl_seconds=settings.youtube_negative_ttl_seconds)
        return _stop("no_channel", trace, home=home_code, away=away_code)

    passed: list[tuple[int, dict[str, Any]]] = []
    for channel in channels:
        passed = _candidates_from_channel(
            channel, game_date=game_date, settings=settings, api_key=api_key,
            home_variants=home_variants, away_variants=away_variants,
            client=client, trace=trace)
        if passed:
            break                                     # 앞에서 찾으면 뒤 채널은 안 부른다

    unscored = False
    if not passed:
        passed = _tavily_fallback(
            game_date=game_date, home_name=home_name, away_name=away_name,
            settings=settings, tavily_client=tavily_client, trace=trace)
        unscored = bool(passed)

    if not passed:
        latest_info.cache_put(settings, cache_key, [],
                              ttl_seconds=settings.youtube_negative_ttl_seconds)
        return _stop("no_candidate", trace, date=game_date.isoformat(),
                     channels=len(channels))

    ids = [v["id"] for _s, v in sorted(passed, key=lambda t: -t[0])][:youtube.MAX_IDS_PER_CALL]
    try:
        details = youtube.fetch_video_details(
            ids, api_key=api_key, timeout=settings.youtube_http_timeout_seconds,
            client=client, budget=settings.youtube_daily_unit_budget,
        )
    except Exception as exc:                          # noqa: BLE001
        log.warning("YouTube 검증 실패 stage=verify: %s", exc)
        # 검증 실패는 '영상이 없다' 가 아니다. 캐시에 남기지 않는다.
        return _stop("verify_failed", trace, ids=len(ids),
                     error=f"{type(exc).__name__}: {exc}")

    if unscored:
        # 폴백 후보는 제목을 이제 알았으므로 여기서 채점한다. 신뢰할 수 없는 풀이라
        # 키워드를 필수로 두고 두 구단 이름을 모두 요구한다.
        rescored = [
            (score_match(v, game_date=game_date, home_variants=home_variants,
                         away_variants=away_variants, trusted=False, require_both=True), v)
            for v in details.values()
        ]
    else:
        rescored = [(s, details[v["id"]]) for s, v in passed if v["id"] in details]

    picked = [v for s, v in sorted(rescored, key=lambda t: -t[0]) if s >= MIN_MATCH_SCORE]
    picked = picked[: settings.highlight_max_videos]
    log.info("하이라이트 검증 stage=verify ids=%d 수신=%d 통과=%d",
             len(ids), len(details), len(picked))
    if not picked:
        latest_info.cache_put(settings, cache_key, [],
                              ttl_seconds=settings.youtube_negative_ttl_seconds)
        return _stop("score_below_min", trace, verified=len(details),
                     top=max((s for s, _ in rescored), default=0))

    media = [{
        "kind": "video", "id": v["id"], "title": v["title"], "url": v["url"],
        "published_at": v["published_at"], "duration": v["duration"],
        "embeddable": v["embeddable"], "thumbnail_url": v["thumbnail_url"],
    } for v in picked]
    latest_info.cache_put(settings, cache_key, media,
                          ttl_seconds=settings.youtube_cache_ttl_seconds)
    log.info("하이라이트 완료 stage=ok 선택=%d units=%d", len(picked), youtube.used_today())
    _note(trace, "ok", picked=len(picked), units=youtube.used_today())
    return [_entry(game_date, home_name, away_name, media)], media


def _entry(game_date: date, home_name: str, away_name: str,
           media: list[dict[str, Any]]) -> KboEntry:
    """모델이 읽는 본문. 날짜를 "9.13" 이 아니라 "9월 13일" 로 적는다(모듈 독스트링 참조)."""
    when = f"{game_date.month}월 {game_date.day}일"
    lines = [f"{when} {away_name} 대 {home_name} 경기 하이라이트 영상 {len(media)}건을 찾았다."]
    for v in media:
        state = "재생 가능" if v.get("embeddable", True) else "외부 재생 제한"
        lines.append(f"- 길이 {v.get('duration') or '미상'} · {state}")
    lines.append("영상 제목과 주소는 화면에 그대로 표시되므로 본문에 옮겨 적지 않는다.")
    return KboEntry(
        kind="highlight",
        label=f"{when} {away_name} 대 {home_name} 하이라이트",
        text="\n".join(lines),
        as_of=game_date.isoformat(),
        source_label=youtube.SOURCE_LABEL,
        source_url=media[0]["url"],
        freshness="live",
    )


# --------------------------------------------------------------------------- 진단 CLI


def _cmd_channels() -> int:
    """네트워크를 타지 않는다. 채널 파일의 정합만 본다."""
    from baseball import kbo

    data = load_channels()
    chans = data.get("channels", [])
    codes = {c.get("code") for c in chans}
    want = set(kbo.teams_by_code()) | {LEAGUE_CODE}
    print(f"채널 파일 as_of={data.get('as_of', '?')} 항목={len(chans)}")
    missing = want - codes
    extra = codes - want
    for chan in chans:
        cid = chan.get("channel_id") or ""
        state = "id 있음(호출 0)" if youtube.uploads_playlist_id(cid) else "미해석(첫 조회 때 1 unit)"
        print(f"  {chan.get('code'):<4} {chan.get('handle', ''):<26} {state}")
    if missing:
        print(f"누락된 코드: {sorted(missing)}")
    if extra:
        print(f"알 수 없는 코드: {sorted(extra)}")
    return 1 if (missing or extra) else 0


def _cmd_resolve(codes: list[str]) -> int:
    """channels.list 로 채널 id 를 받아 출력만 한다. 파일은 사람이 고친다."""
    settings = get_settings()
    if settings.youtube_kbo_api_key is None:
        print("YOUTUBE_KBO_API_KEY 가 없다.")
        return 1
    api_key = settings.youtube_kbo_api_key.get_secret_value()
    by_code = channels_by_code()
    targets = [by_code[c] for c in (codes or list(by_code)) if c in by_code]
    out: list[dict[str, Any]] = []
    failed = 0
    for chan in targets:
        try:
            found = youtube.resolve_channel(
                api_key=api_key, handle=chan.get("handle"), username=chan.get("username"),
                timeout=settings.youtube_http_timeout_seconds,
                budget=settings.youtube_daily_unit_budget)
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"  {chan['code']:<4} 실패 handle={chan.get('handle')} — {exc}")
            continue
        out.append({**chan, "channel_id": found["channel_id"],
                    "verified_at": str(found["title"])})
        print(f"  {chan['code']:<4} {found['channel_id']}  {found['title']}")
    print(f"\n해석 {len(out)}건 / 실패 {failed}건 / units={youtube.used_today()}")
    if out:
        print("\n--- data/kbo_youtube_channels.json 의 channel_id 에 붙여 넣는다 ---")
        print(json.dumps({c["code"]: c["channel_id"] for c in out},
                         ensure_ascii=False, indent=2))
    return 1 if failed else 0


def _cmd_uploads(code: str, limit: int) -> int:
    settings = get_settings()
    if settings.youtube_kbo_api_key is None:
        print("YOUTUBE_KBO_API_KEY 가 없다.")
        return 1
    chan = channels_by_code().get(code)
    if chan is None:
        print(f"알 수 없는 코드: {code}. 가능한 값: {sorted(channels_by_code())}")
        return 1
    api_key = settings.youtube_kbo_api_key.get_secret_value()
    playlist = _uploads_playlist(chan, settings=settings, api_key=api_key)
    if not playlist:
        print(f"{code}: 업로드 재생목록을 해석하지 못했다(핸들 확인 필요).")
        return 1
    items = youtube.fetch_playlist_items(
        playlist, api_key=api_key, max_results=limit,
        timeout=settings.youtube_http_timeout_seconds,
        budget=settings.youtube_daily_unit_budget)
    print(f"{code} {playlist} — {len(items)}건 (units={youtube.used_today()})")
    for item in items:
        print(f"  {item['published_at']}  {item['id']}  {item['title'][:70]}")
    return 0


def _cmd_find(query: str, when: str | None) -> int:
    """경기 선택 -> 채널 -> 재생목록 -> 채점 -> 검증을 단계별로 보여 준다."""
    from baseball import clock, kbo, tools

    settings = get_settings()
    print(f"0 게이트     highlights_enabled={settings.highlights_enabled} "
          f"web_search={settings.web_search_enabled}")
    if not settings.highlights_enabled:
        print("  ENABLE_HIGHLIGHTS=on 과 YOUTUBE_KBO_API_KEY 가 있어야 한다.")
        return 1

    game, _team, relation, _venue = tools._resolve(query, settings, prefer="last_finished")
    if game is None:
        print("1 경기       찾지 못했다(비시즌이거나 일정 조회 실패).")
        return 1
    game_date = (date.fromisoformat(when) if when else game.game_date)
    print(f"1 경기       {game_date} {game.away_name}({game.away_code}) 대 "
          f"{game.home_name}({game.home_code})  relation={relation}")

    chans = channel_order(game.home_code, game.away_code)
    print(f"2 채널       {[c['code'] for c in chans]}")

    trace: list[dict[str, Any]] = []
    entries, media = find(
        game_date=game_date, home_name=game.home_name, away_name=game.away_name,
        home_code=game.home_code, away_code=game.away_code,
        settings=settings, trace=trace)
    for step in trace:
        stage = step.pop("stage")
        detail = "  ".join(f"{k}={v}" for k, v in step.items())
        print(f"3 {stage:<16} {detail}")

    print(f"4 결과       {len(media)}건  units_today={youtube.used_today()}")
    for m in media:
        print(f"  {m['published_at']}  {m['duration'] or '미상'}  {m['url']}")
        print(f"      {m['title'][:78]}")
    if not entries:
        print("  STOP — 위 stage 줄이 어느 단계에서 0이 됐는지 말한다.")
        return 1
    return 0


def _cmd_purge() -> int:
    """만료된 yt:% 캐시 행을 지운다. YouTube ToS 의 30일 갱신·삭제 의무."""
    from baseball import db

    settings = get_settings()
    try:
        with db.connect(settings) as conn:
            cur = conn.execute(
                "DELETE FROM assistant_cache "
                "WHERE key LIKE 'yt:%' AND fetched_at < now() - interval '30 days'")
            conn.commit()
            print(f"삭제 {cur.rowcount}행")
    except Exception as exc:                          # noqa: BLE001
        print(f"캐시 정리 실패: {exc}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    import logging as _logging

    ap = argparse.ArgumentParser(prog="python -m baseball.highlights")
    ap.add_argument("-v", "--verbose", action="store_true", help="단계 로그를 함께 본다")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("channels", help="채널 파일 정합 점검(네트워크 없음)")
    p_res = sub.add_parser("resolve", help="핸들 -> 채널 id 해석(코드당 1 unit)")
    p_res.add_argument("codes", nargs="*", help="비우면 전체")
    p_up = sub.add_parser("uploads", help="채널의 최근 업로드를 본다(1 unit)")
    p_up.add_argument("code")
    p_up.add_argument("--limit", type=int, default=15)
    p_find = sub.add_parser("find", help="탐색 전 구간을 단계별로 본다")
    p_find.add_argument("query", nargs="?", default="어제 경기 하이라이트")
    p_find.add_argument("--date", dest="when", default=None, help="YYYY-MM-DD 로 경기일 고정")
    sub.add_parser("purge", help="30일 지난 하이라이트 캐시 삭제(YouTube ToS)")
    args = ap.parse_args(argv)

    _logging.basicConfig(level=_logging.DEBUG if args.verbose else _logging.WARNING,
                         format="%(levelname)s %(name)s %(message)s")
    if args.command == "channels":
        return _cmd_channels()
    if args.command == "resolve":
        return _cmd_resolve(args.codes)
    if args.command == "uploads":
        return _cmd_uploads(args.code, args.limit)
    if args.command == "find":
        return _cmd_find(args.query, args.when)
    return _cmd_purge()


if __name__ == "__main__":
    sys.exit(main())
