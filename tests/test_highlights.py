"""경기 하이라이트. 발견은 업로드 재생목록이고, 없을 때가 정상 경로다.

예전 이 파일의 _tavily() 헬퍼는 **언제나** 유튜브 URL 을 score 0.9 로 돌려줬다.
프로덕션에서 실제로 깨지는 단계를 mock 이 통째로 대신해서, 테스트는 전부 초록인데
서비스는 0건이었다. 이제 재생목록 응답을 라우팅해 그 단계를 실제로 지나게 한다.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from baseball import citations as cite
from baseball import highlights, latest_info, youtube
from baseball.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"
GAME_DATE = date(2026, 9, 13)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(latest_info, "_cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(latest_info, "_cache_put", lambda *_a, **_k: None)
    youtube._UNITS.clear()
    highlights.load_channels.cache_clear()
    highlights.channels_by_code.cache_clear()
    yield
    youtube._UNITS.clear()


def _settings(**kw: Any) -> Settings:
    base = {"_env_file": None, "enable_web_search": "off", "enable_highlights": "on",
            "youtube_kbo_api_key": SecretStr("AIza-test")}
    return Settings(**{**base, **kw})


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _api(
    *, playlist: dict[str, Any] | None = None, videos: dict[str, Any] | None = None,
    channels: dict[str, Any] | None = None, seen: list[httpx.Request] | None = None,
    status: int = 200,
) -> httpx.Client:
    """경로별로 응답을 갈라 주는 MockTransport.

    하나의 client 로 channels.list -> playlistItems.list -> videos.list 를 전부
    받아야 find() 가 실제 코드 경로를 그대로 지난다.
    """
    bodies = {
        "/youtube/v3/channels": channels if channels is not None else _fixture("youtube_channels_list.json"),
        "/youtube/v3/playlistItems": playlist if playlist is not None else _fixture("youtube_playlist_items.json"),
        "/youtube/v3/videos": videos if videos is not None else _fixture("youtube_videos_list.json"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=bodies.get(request.url.path, {"items": []}))

    return httpx.Client(transport=httpx.MockTransport(handler))


def _find(*, client: httpx.Client | None = None, tavily: httpx.Client | None = None,
          home_code: str = "LG", away_code: str = "SS", **skw: Any):
    return highlights.find(
        game_date=GAME_DATE, home_name="LG 트윈스", away_name="삼성 라이온즈",
        home_code=home_code, away_code=away_code,
        settings=_settings(**skw), client=client or _api(), tavily_client=tavily,
    )


def _tavily(urls: list[str]) -> httpx.Client:
    body = {"results": [{"title": f"t{i}", "url": u, "content": "c", "score": 0.9}
                        for i, u in enumerate(urls)]}
    return httpx.Client(transport=httpx.MockTransport(
        lambda _r: httpx.Response(200, json=body)))


# --------------------------------------------------------------------------- 채점

LG = ("LG 트윈스", "LG")
SS = ("삼성 라이온즈", "삼성")


def _score(title: str, published: str = "2026-09-13", *, trusted: bool = True,
           require_both: bool = False, game_date: date = GAME_DATE) -> int:
    return highlights.score_match(
        {"title": title, "published_at": published}, game_date=game_date,
        home_variants=highlights._variants("LG 트윈스"),
        away_variants=highlights._variants("삼성 라이온즈"),
        trusted=trusted, require_both=require_both)


def test_exact_game_scores_high() -> None:
    assert _score("[9.13 vs 삼성] LG 트윈스 하이라이트") >= highlights.MIN_MATCH_SCORE


def test_other_game_falls_below_the_threshold() -> None:
    assert _score("[8.02 vs 두산] 하이라이트", "2026-08-02") < highlights.MIN_MATCH_SCORE


def test_bare_title_without_keyword_matches_on_a_club_channel() -> None:
    """실측 회귀: "LG vs 삼성 9.13" 은 예전 키워드 게이트에서 0점이었다.

    구단 공식 채널의 업로드 목록은 이미 믿을 만한 풀이라, 키워드를 필수에서
    가산점으로 낮추고 날짜를 필수로 올렸다.
    """
    assert _score("LG vs 삼성 9.13") >= highlights.MIN_MATCH_SCORE


@pytest.mark.parametrize("title", [
    "9.13 삼성전 주요장면",          # '하이라이트' 가 없다
    "[풀영상] 9.13 LG vs 삼성",
    "9.13 LG vs 삼성 경기 요약",
    "5분 하이라이트 9.13 LG vs 삼성",
])
def test_widened_keywords_match(title: str) -> None:
    assert _score(title) >= highlights.MIN_MATCH_SCORE


@pytest.mark.parametrize("title", [
    "[LG트윈스] 9.13 경기 예고",
    "오지환 인터뷰 9.13",
    "9.13 응원 브이로그",
    "9.13 중계 안내",
])
def test_non_game_uploads_are_excluded(title: str) -> None:
    """구단 채널은 같은 날 예고·인터뷰도 올린다. 날짜만으로 통과하면 안 된다."""
    assert _score(title) == 0


def test_talk_video_on_the_club_channel_is_rejected() -> None:
    """날짜 토큰도 키워드도 없으면 그 채널의 그냥 다른 영상이다."""
    assert _score("또 4실점 패전, LG트윈스 치리노스의 문제점은 뭘까?", "2026-09-14") == 0


def test_kbo_official_requires_both_team_names() -> None:
    """실측 회귀: "LG 경기 하이라이트" 에 "[KT위즈 vs 한화이글스] 9.16(수)" 가 걸렸다.

    KBO 공식 채널은 전 경기를 올리므로 채널만으로는 어느 경기인지 모른다. 예전
    코드는 두 구단 중 **하나만** 맞아도 통과시켜 같은 날 다른 경기가 들어왔다.
    """
    assert _score("[KT위즈 vs 한화이글스] 9.13(일) 야구 하이라이트｜2026 KBO 리그",
                  require_both=True) == 0
    assert _score("[삼성라이온즈 vs LG트윈스] 9.13(일) 야구 하이라이트｜2026 KBO 리그",
                  require_both=True) >= highlights.MIN_MATCH_SCORE


def test_untrusted_pool_still_requires_the_keyword() -> None:
    """Tavily 폴백은 아무 채널이나 준다. 거기서는 키워드가 필수로 남는다."""
    assert _score("LG 트윈스 삼성 라이온즈 9.13", trusted=False, require_both=True) == 0


def test_html_escaped_titles_are_unescaped_before_scoring() -> None:
    assert _score("LG트윈스 &amp; 삼성라이온즈 9.13 하이라이트",
                  require_both=True) >= highlights.MIN_MATCH_SCORE


@pytest.mark.parametrize("title", [
    "[9.13 vs 삼성] LG 트윈스 하이라이트", "[LG vs 삼성] 9.13 하이라이트",
    "엘지 트윈스 9월 13일 하이라이트", "LG 하이라이트 9.13",
])
def test_short_forms_still_match(title: str) -> None:
    """네이버는 "LG 트윈스" 로 주는데 영상 제목은 "LG"·"엘지"·"트윈스" 로 쓴다."""
    assert _score(title) >= highlights.MIN_MATCH_SCORE


def test_variants_are_per_team_not_merged() -> None:
    """두 구단을 한 튜플에 섞으면 '둘 중 하나만 맞아도 통과' 가 된다."""
    lg = set(highlights._variants("LG 트윈스"))
    ss = set(highlights._variants("삼성 라이온즈"))
    assert {"LG", "엘지", "트윈스"} <= lg
    assert {"삼성", "라이온즈"} <= ss
    assert not (lg & ss)


def test_zero_padded_date_in_title_counts() -> None:
    """구단 채널은 "(09.13)" 처럼 0 을 채워 쓴다."""
    tokens = highlights._date_tokens(GAME_DATE)
    assert {"09.13", "9.13", "9월 13일", "9월13일"} <= set(tokens)


# --------------------------------------------------------------------------- 실측 제목

REAL_TITLES = [
    ("[2026 KBO 리그 H/L] LG vs 삼성 (09.13)", True, "LG 구단 채널 — 약어만 쓴다"),
    ("오늘 최고의 장면은? | 9월 13일 하이라이트 | LG vs 삼성", True, "KIA 형식"),
    ("9.13 LG vs 삼성 주요장면", True, "'하이라이트' 가 아예 없다"),
    ("또 4실점 패전, LG트윈스 치리노스의 문제점은 뭘까?", False, "토크 영상"),
    ("[LG트윈스] 9.13 경기 예고", False, "예고"),
]


@pytest.mark.parametrize("title,expected,label", REAL_TITLES)
def test_real_channel_titles(title: str, expected: bool, label: str) -> None:
    score = _score(title)
    assert (score >= highlights.MIN_MATCH_SCORE) is expected, f"{label}: {score}점"


# --------------------------------------------------------------------------- 채널 선택

def test_channel_order_is_home_away_league() -> None:
    assert [c["code"] for c in highlights.channel_order("LG", "SS")] == ["LG", "SS", "KBO"]


def test_home_channel_alone_answers_the_common_case() -> None:
    """앞 채널에서 찾으면 뒤 채널은 부르지 않는다. 흔한 경우 2 units 다."""
    seen: list[httpx.Request] = []
    entries, media = _find(client=_api(seen=seen))
    assert entries and media
    playlists = [r for r in seen if r.url.path == "/youtube/v3/playlistItems"]
    assert len(playlists) == 1              # 원정·KBO 채널은 안 불렀다
    assert youtube.used_today() <= 4


def test_falls_through_to_the_next_channel_when_the_first_is_empty() -> None:
    seen: list[httpx.Request] = []
    entries, _media = _find(client=_api(playlist={"items": []}, seen=seen))
    assert entries == []
    playlists = [r for r in seen if r.url.path == "/youtube/v3/playlistItems"]
    assert len(playlists) == 3              # 홈 -> 원정 -> KBO 공식까지 다 봤다


# --------------------------------------------------------------------------- 없을 때가 정상

def test_empty_playlist_returns_nothing() -> None:
    assert _find(client=_api(playlist={"items": []})) == ([], [])


def test_no_candidate_scores_high_enough_returns_nothing() -> None:
    body = _fixture("youtube_playlist_items.json")
    for i, title in enumerate(["야구 규칙 설명 영상", "훈련 브이로그"]):
        body["items"][i]["snippet"]["title"] = title
    body["items"] = body["items"][:2]
    assert _find(client=_api(playlist=body)) == ([], [])


def test_youtube_failure_degrades_quietly() -> None:
    assert _find(client=_api(status=500)) == ([], [])


def test_disabled_returns_nothing() -> None:
    assert _find(enable_highlights="off") == ([], [])


def test_missing_youtube_key_disables_the_feature() -> None:
    assert _find(youtube_kbo_api_key=None) == ([], [])


def test_unknown_team_code_still_tries_the_league_channel() -> None:
    """구단 코드를 못 알아내도 KBO 공식 채널이 남는다."""
    assert [c["code"] for c in highlights.channel_order("", "")] == ["KBO"]


# --------------------------------------------------------------------------- 캐시

def _capture_puts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list, int]]:
    puts: list[tuple[str, list, int]] = []
    monkeypatch.setattr(latest_info, "cache_put",
                        lambda _s, k, v, *, ttl_seconds: puts.append((k, v, ttl_seconds)))
    return puts


def test_zero_result_uses_the_short_negative_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """회귀 고정: 예전에는 0건이 7일 캐시에 박혀 그 경기를 일주일 막았다.

    하이라이트는 경기 종료 몇 시간 뒤에 올라온다. 경기 직후의 0건을 오래 기억하면
    영상이 올라온 뒤에도 계속 "없다" 고 답한다.
    """
    puts = _capture_puts(monkeypatch)
    _find(client=_api(playlist={"items": []}))
    game_puts = [p for p in puts if p[0].startswith("yt:hl:")]
    assert game_puts and game_puts[-1][1] == []
    assert game_puts[-1][2] == _settings().youtube_negative_ttl_seconds == 900


def test_found_result_uses_the_seven_day_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    puts = _capture_puts(monkeypatch)
    entries, _media = _find()
    assert entries
    game_puts = [p for p in puts if p[0].startswith("yt:hl:")]
    assert game_puts[-1][2] == _settings().youtube_cache_ttl_seconds == 604800


def test_verify_failure_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """검증 실패는 '영상이 없다' 가 아니다. 캐시에 남기면 장애가 고착된다."""
    puts = _capture_puts(monkeypatch)
    videos_500 = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(500 if r.url.path.endswith("/videos") else 200,
                                 json=_fixture("youtube_playlist_items.json")
                                 if r.url.path.endswith("playlistItems")
                                 else _fixture("youtube_channels_list.json"))))
    assert _find(client=videos_500) == ([], [])
    assert [p for p in puts if p[0].startswith("yt:hl:")] == []


def test_game_cache_key_is_readable_and_deterministic() -> None:
    """질의 문장 해시가 아니라 경기를 가리킨다. 운영 중 눈으로 확인된다."""
    assert highlights._game_key(GAME_DATE, "SS", "LG") == "yt:hl:2026-09-13:SS:LG"


def test_cached_game_is_served_without_any_http(monkeypatch: pytest.MonkeyPatch) -> None:
    media = [{"kind": "video", "id": "aAAAAAAAAA1", "title": "t",
              "url": "https://www.youtube.com/watch?v=aAAAAAAAAA1",
              "published_at": "2026-09-13", "duration": "12:41",
              "embeddable": True, "thumbnail_url": None}]
    monkeypatch.setattr(latest_info, "cache_get",
                        lambda _s, k: media if k.startswith("yt:hl:") else None)
    seen: list[httpx.Request] = []
    entries, got = _find(client=_api(seen=seen))
    assert len(entries) == 1 and got == media
    assert seen == []


def test_cached_zero_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """저장된 빈 결과는 캐시 미스와 다르다. 부정 캐시가 여기서 작동한다."""
    monkeypatch.setattr(latest_info, "cache_get",
                        lambda _s, k: [] if k.startswith("yt:hl:") else None)
    seen: list[httpx.Request] = []
    assert _find(client=_api(seen=seen)) == ([], [])
    assert seen == []


# --------------------------------------------------------------------------- 찾았을 때

def test_found_videos_become_media_not_prose() -> None:
    entries, media = _find()
    assert len(entries) == 1 and entries[0].kind == "highlight"
    assert [m["id"] for m in media] == ["aAAAAAAAAA1", "bBBBBBBBBB2"]
    assert all(m["url"] for m in media)


def test_non_embeddable_is_flagged_for_the_ui() -> None:
    _, media = _find()
    assert {m["id"]: m["embeddable"] for m in media} == {
        "aAAAAAAAAA1": True, "bBBBBBBBBB2": False}


def test_body_never_produces_a_fake_rule_citation() -> None:
    """citations.RULE_NO_RE 가 '9.13' 을 규칙 번호로 잡는다(실측).

    그래서 모델이 읽는 본문에는 '9월 13일' 로 적고 원래 제목은 media 로만 보낸다.
    """
    entries, media = _find()
    text = entries[0].text + entries[0].label
    assert "9.13" not in text
    assert "9월 13일" in text
    assert cite.extract(text, [])[1] == []
    assert media[0]["title"] == "[9.13 vs 삼성] LG 트윈스 하이라이트"


def test_result_count_is_capped() -> None:
    _, media = _find(highlight_max_videos=1)
    assert len(media) == 1


# --------------------------------------------------------------------------- 폴백

def test_query_is_deterministic() -> None:
    assert highlights.build_query(GAME_DATE, "LG 트윈스", "삼성 라이온즈") == \
        "삼성 라이온즈 LG 트윈스 9월 13일 야구 하이라이트"


def test_tavily_fallback_is_skipped_when_web_search_is_off() -> None:
    """발견이 더는 Tavily 가 아니다. 꺼져 있어도 하이라이트는 산다."""
    assert _settings(enable_web_search="off").highlights_enabled is True
    assert _find(client=_api(playlist={"items": []}), enable_web_search="off") == ([], [])


def test_tavily_fallback_runs_only_when_playlists_are_empty() -> None:
    entries, media = _find(
        client=_api(playlist={"items": []}),
        tavily=_tavily(["https://youtu.be/aAAAAAAAAA1"]),
        enable_web_search="on", tavily_api_key=SecretStr("tvly-test"))
    assert [m["id"] for m in media] == ["aAAAAAAAAA1"]
    assert entries


def test_tavily_fallback_result_is_scored_strictly() -> None:
    """신뢰할 수 없는 풀이라 두 구단 이름이 모두 있어야 한다."""
    body = _fixture("youtube_videos_list.json")
    body["items"][0]["snippet"]["title"] = "9.13 LG 하이라이트"       # 삼성이 없다
    body["items"] = body["items"][:1]
    assert _find(client=_api(playlist={"items": []}, videos=body),
                 tavily=_tavily(["https://youtu.be/aAAAAAAAAA1"]),
                 enable_web_search="on", tavily_api_key=SecretStr("tvly-test")) == ([], [])


# --------------------------------------------------------------------------- 관측성

@pytest.mark.parametrize("stage,kw", [
    ("disabled", {"enable_highlights": "off"}),
    ("no_candidate", {}),
])
def test_every_empty_path_records_a_stage(stage: str, kw: Any,
                                          caplog: pytest.LogCaptureFixture) -> None:
    """어느 단계에서 0이 됐는지 로그로 특정된다.

    예전에는 두 군데가 말없이 빈 리스트를 돌려줘서, 배포본에서 '발견 0건' 인지
    '검증 실패' 인지 '점수 미달' 인지 구분할 방법이 아예 없었다.
    """
    trace: list[dict[str, Any]] = []
    with caplog.at_level("WARNING"):
        highlights.find(
            game_date=GAME_DATE, home_name="LG 트윈스", away_name="삼성 라이온즈",
            home_code="LG", away_code="SS", settings=_settings(**kw),
            client=_api(playlist={"items": []}), trace=trace)
    assert f"stage={stage}" in caplog.text
    assert trace[-1]["stage"] == stage


def test_trace_records_each_channel_attempt() -> None:
    trace: list[dict[str, Any]] = []
    highlights.find(
        game_date=GAME_DATE, home_name="LG 트윈스", away_name="삼성 라이온즈",
        home_code="LG", away_code="SS", settings=_settings(),
        client=_api(playlist={"items": []}), trace=trace)
    codes = [t.get("code") for t in trace if t["stage"] == "channel"]
    assert codes == ["LG", "SS", "KBO"]
    assert trace[-1]["stage"] == "no_candidate"


# --------------------------------------------------------------------------- 진단 CLI

def test_cli_help_renders() -> None:
    """argparse 는 help 문자열의 % 를 포맷 지시자로 읽는다.

    실측 회귀: "30일 지난 yt:% 캐시 삭제" 가 --help 를 ValueError 로 죽였다.
    진단 도구가 열리지 않으면 진단을 못 한다.
    """
    with pytest.raises(SystemExit) as exc:
        highlights.main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("argv", [
    ["channels"], ["resolve"], ["resolve", "LG"], ["uploads", "LG"],
    ["uploads", "LG", "--limit", "5"], ["find"], ["find", "어제 LG 경기 하이라이트"],
    ["find", "x", "--date", "2026-09-13"], ["purge"], ["-v", "channels"],
])
def test_cli_subcommands_parse(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    for name in ("_cmd_channels", "_cmd_resolve", "_cmd_uploads", "_cmd_find", "_cmd_purge"):
        monkeypatch.setattr(highlights, name,
                            lambda *_a, _n=name, **_k: ran.append(_n) or 0)
    assert highlights.main(argv) == 0
    assert len(ran) == 1


def test_cli_channels_reports_the_file_is_consistent() -> None:
    """네트워크 없이 도는 점검이다. 코드 누락이 있으면 0 이 아니어야 한다."""
    assert highlights._cmd_channels() == 0
