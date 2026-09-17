"""YouTube Data API 어댑터. 네트워크 없이 요청·응답 계약과 쿼터 방어를 고정한다."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from baseball import youtube

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_units():
    youtube._UNITS.clear()
    yield
    youtube._UNITS.clear()


def _payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "youtube_videos_list.json").read_text(encoding="utf-8"))


def _client(body: dict[str, Any], seen: list[httpx.Request] | None = None,
            status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- 요청 계약

def test_videos_request_shape() -> None:
    seen: list[httpx.Request] = []
    with _client(_payload(), seen) as c:
        youtube.fetch_video_details(["aAAAAAAAAA1"], api_key="AIza-test", client=c)
    q = dict(seen[0].url.params)
    assert seen[0].url.host == "www.googleapis.com"
    assert seen[0].url.path == "/youtube/v3/videos"
    assert "status" in q["part"]           # embeddable 이 여기 들어 있다
    assert "contentDetails" in q["part"]   # duration
    assert q["key"] == "AIza-test"


def test_fifty_ids_cost_one_call() -> None:
    """videos.list 는 개수와 무관하게 1 unit 이다. 배치가 곧 절약이다."""
    seen: list[httpx.Request] = []
    ids = [f"{i:011d}".replace("0", "a") for i in range(50)]
    with _client({"items": []}, seen) as c:
        youtube.fetch_video_details(ids, api_key="k", client=c)
    assert len(seen) == 1
    assert len(dict(seen[0].url.params)["id"].split(",")) == 50
    assert youtube.used_today() == 1


def test_no_browser_disguise_headers() -> None:
    seen: list[httpx.Request] = []
    with _client({"items": []}, seen) as c:
        youtube.fetch_video_details(["aAAAAAAAAA1"], api_key="k", client=c)
    assert "Referer" not in seen[0].headers


def test_empty_id_list_makes_no_request() -> None:
    seen: list[httpx.Request] = []
    with _client({"items": []}, seen) as c:
        assert youtube.fetch_video_details([], api_key="k", client=c) == {}
    assert seen == [] and youtube.used_today() == 0


# --------------------------------------------------------------------------- 응답 해석

def test_details_are_normalised() -> None:
    with _client(_payload()) as c:
        out = youtube.fetch_video_details(["aAAAAAAAAA1", "bBBBBBBBBB2"], api_key="k", client=c)
    first = out["aAAAAAAAAA1"]
    assert first["title"] == "[9.13 vs 삼성] LG 트윈스 하이라이트"
    assert first["published_at"] == "2026-09-13"
    assert first["duration"] == "12:41"
    assert first["embeddable"] is True
    assert first["url"] == "https://www.youtube.com/watch?v=aAAAAAAAAA1"
    assert first["thumbnail_url"].endswith("mqdefault.jpg")


def test_non_embeddable_video_is_flagged_not_dropped() -> None:
    """외부 재생 제한 영상은 버리지 않는다. 플레이어 대신 링크로 격하한다."""
    with _client(_payload()) as c:
        out = youtube.fetch_video_details(["bBBBBBBBBB2"], api_key="k", client=c)
    assert out["bBBBBBBBBB2"]["embeddable"] is False


# --------------------------------------------------------------------------- 실패 구분

@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_http_error_becomes_source_error(status: int) -> None:
    with _client({"error": {"message": "nope"}}, status=status) as c:
        with pytest.raises(youtube.YoutubeSourceError):
            youtube.fetch_video_details(["aAAAAAAAAA1"], api_key="k", client=c)


def test_quota_exceeded_is_a_distinct_error() -> None:
    body = {"error": {"errors": [{"reason": "quotaExceeded"}], "message": "quota exceeded"}}
    with _client(body, status=403) as c:
        with pytest.raises(youtube.YoutubeQuotaExceeded):
            youtube.fetch_video_details(["aAAAAAAAAA1"], api_key="k", client=c)


def test_connection_failure_becomes_source_error() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(youtube.YoutubeSourceError):
            youtube.fetch_video_details(["aAAAAAAAAA1"], api_key="k", client=c)


def test_shape_change_is_distinct_from_transport_failure() -> None:
    with _client({"items": "nope"}) as c:
        with pytest.raises(youtube.YoutubeUpstreamChanged):
            youtube.fetch_video_details(["aAAAAAAAAA1"], api_key="k", client=c)


def test_missing_embeddable_field_is_upstream_change() -> None:
    """part 에서 status 가 빠지면 조용히 재생 가능으로 오판하게 된다. 그걸 막는다."""
    body = _payload()
    del body["items"][0]["status"]["embeddable"]
    with _client(body) as c:
        with pytest.raises(youtube.YoutubeUpstreamChanged):
            youtube.fetch_video_details(["aAAAAAAAAA1"], api_key="k", client=c)


# --------------------------------------------------------------------------- 쿼터 방어

def test_budget_stops_before_the_request_goes_out() -> None:
    seen: list[httpx.Request] = []
    youtube._UNITS[__import__("baseball.clock", fromlist=["x"]).today_kst()] = 8000
    with _client(_payload(), seen) as c:
        with pytest.raises(youtube.YoutubeQuotaExceeded):
            youtube.fetch_video_details(["aAAAAAAAAA1"], api_key="k", client=c, budget=8000)
    assert seen == []                      # 요청이 나가지 않았다


def test_counter_resets_on_a_new_kst_day() -> None:
    from datetime import timedelta

    from baseball import clock

    youtube._UNITS[clock.today_kst() - timedelta(days=1)] = 9999
    with _client({"items": []}) as c:
        youtube.fetch_video_details(["aAAAAAAAAA1"], api_key="k", client=c)
    assert youtube.used_today() == 1
    assert len(youtube._UNITS) == 1        # 어제 기록은 사라졌다


# --------------------------------------------------------------------------- 규약

def test_module_never_uses_forbidden_endpoints() -> None:
    """feeds/videos.xml 은 robots Disallow, search 는 100 units 다. 둘 다 쓰지 않는다."""
    import ast

    src = Path("src/baseball/youtube.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
    literals = "\n".join(n.value for n in ast.walk(tree)
                         if isinstance(n, ast.Constant) and isinstance(n.value, str)
                         and n.value not in docstrings)
    assert "feeds/videos.xml" not in literals
    assert "/search" not in literals
    for host in ("koreabaseball.com", "statiz", "espn.com", "namu.wiki"):
        assert host not in literals


# --------------------------------------------------------------------------- 업로드 재생목록

def _playlist() -> dict[str, Any]:
    return json.loads((FIXTURES / "youtube_playlist_items.json").read_text(encoding="utf-8"))


def _channels() -> dict[str, Any]:
    return json.loads((FIXTURES / "youtube_channels_list.json").read_text(encoding="utf-8"))


def test_playlist_items_request_shape() -> None:
    seen: list[httpx.Request] = []
    with _client(_playlist(), seen) as c:
        youtube.fetch_playlist_items("UUaaaaaaaaaaaaaaaaaaaaaa", api_key="AIza-test",
                                     max_results=15, client=c)
    q = dict(seen[0].url.params)
    assert seen[0].url.path == "/youtube/v3/playlistItems"
    assert "snippet" in q["part"] and "contentDetails" in q["part"]
    assert q["playlistId"] == "UUaaaaaaaaaaaaaaaaaaaaaa"
    assert q["maxResults"] == "15"
    assert q["key"] == "AIza-test"


def test_playlist_items_cost_one_unit() -> None:
    """search.list 는 100 units 다. 재생목록은 몇 건을 받든 1 unit 이다."""
    with _client(_playlist()) as c:
        got = youtube.fetch_playlist_items("UUaaaaaaaaaaaaaaaaaaaaaa", api_key="k",
                                           max_results=50, client=c)
    assert youtube.used_today() == youtube.PLAYLIST_ITEMS_UNIT_COST == 1
    assert len(got) >= 4


def test_max_results_is_clamped_to_the_api_limit() -> None:
    seen: list[httpx.Request] = []
    with _client(_playlist(), seen) as c:
        youtube.fetch_playlist_items("UUaaaaaaaaaaaaaaaaaaaaaa", api_key="k",
                                     max_results=500, client=c)
    assert dict(seen[0].url.params)["maxResults"] == str(youtube.MAX_PLAYLIST_RESULTS)


def test_private_and_deleted_items_are_skipped_not_fatal() -> None:
    """재생목록에 남는 껍데기다. 상류 변화가 아니라 정상 상태다."""
    with _client(_playlist()) as c:
        got = youtube.fetch_playlist_items("UUaaaaaaaaaaaaaaaaaaaaaa", api_key="k", client=c)
    assert "fFFFFFFFFF6" not in {g["id"] for g in got}


def test_missing_resource_id_is_an_upstream_change() -> None:
    body = _playlist()
    del body["items"][0]["snippet"]["resourceId"]
    with _client(body) as c:
        with pytest.raises(youtube.YoutubeUpstreamChanged):
            youtube.fetch_playlist_items("UUaaaaaaaaaaaaaaaaaaaaaa", api_key="k", client=c)


def test_playlist_id_must_be_an_uploads_playlist() -> None:
    with _client(_playlist()) as c:
        with pytest.raises(ValueError):
            youtube.fetch_playlist_items("PLnotanuploadsplaylist", api_key="k", client=c)


def test_upload_time_comes_from_content_details() -> None:
    """snippet.publishedAt 은 '재생목록에 추가된 시각' 이라 다른 값이다."""
    body = _playlist()
    body["items"][0]["snippet"]["publishedAt"] = "2020-01-01T00:00:00Z"
    with _client(body) as c:
        got = youtube.fetch_playlist_items("UUaaaaaaaaaaaaaaaaaaaaaa", api_key="k", client=c)
    assert got[0]["published_at"] == "2026-09-13"


# --------------------------------------------------------------------------- KST 변환

def test_utc_timestamp_becomes_a_kst_date() -> None:
    """회귀 고정: [:10] 으로 자르면 하루가 틀린다.

    "2026-09-16T15:30:00Z" 는 KST 로 9월 17일이다. 경기 날짜(KST)와 맞대어 보는
    값이라 이 하루가 그대로 오탈락이 된다.
    """
    from datetime import date as _date

    assert youtube.to_kst_date("2026-09-16T15:30:00Z") == _date(2026, 9, 17)
    assert youtube.to_kst_date("2026-09-16T14:30:00Z") == _date(2026, 9, 16)
    assert youtube.to_kst_date("") is None
    assert youtube.to_kst_date("엉뚱한 값") is None


def test_video_details_publish_date_is_kst() -> None:
    body = _payload()
    body["items"][0]["snippet"]["publishedAt"] = "2026-09-13T15:30:00Z"
    with _client(body) as c:
        got = youtube.fetch_video_details(["aAAAAAAAAA1"], api_key="k", client=c)
    assert got["aAAAAAAAAA1"]["published_at"] == "2026-09-14"
    assert got["aAAAAAAAAA1"]["published_at_utc"] == "2026-09-13T15:30:00Z"


# --------------------------------------------------------------------------- 채널 해석

def test_uploads_id_is_the_channel_id_with_uu() -> None:
    """YouTube 가 보장하는 규칙이라 channels.list 없이 얻는다(0 units)."""
    assert youtube.uploads_playlist_id("UCKp8knO8a6tSI1oaLjfd9XA") == "UUKp8knO8a6tSI1oaLjfd9XA"
    assert youtube.uploads_playlist_id("") is None
    assert youtube.uploads_playlist_id("UUalreadyuploads1234567") is None


def test_resolve_channel_reads_related_playlists() -> None:
    seen: list[httpx.Request] = []
    with _client(_channels(), seen) as c:
        got = youtube.resolve_channel(api_key="AIza-test", handle="@lgtwinstv", client=c)
    q = dict(seen[0].url.params)
    assert seen[0].url.path == "/youtube/v3/channels"
    assert q["forHandle"] == "@lgtwinstv"
    assert got["uploads_playlist_id"] == "UUaaaaaaaaaaaaaaaaaaaaaa"
    assert youtube.used_today() == youtube.CHANNELS_UNIT_COST == 1


def test_resolve_channel_adds_the_at_sign() -> None:
    seen: list[httpx.Request] = []
    with _client(_channels(), seen) as c:
        youtube.resolve_channel(api_key="k", handle="lgtwinstv", client=c)
    assert dict(seen[0].url.params)["forHandle"] == "@lgtwinstv"


def test_unknown_handle_is_a_source_error_not_an_upstream_change() -> None:
    """핸들 오타 하나가 '상류가 바뀌었다' 로 올라오면 안 된다."""
    with _client({"kind": "youtube#channelListResponse", "pageInfo": {"totalResults": 0}}) as c:
        with pytest.raises(youtube.YoutubeSourceError):
            youtube.resolve_channel(api_key="k", handle="@nope", client=c)


def test_resolve_channel_needs_an_identifier() -> None:
    with pytest.raises(ValueError):
        youtube.resolve_channel(api_key="k")


def test_budget_blocks_before_the_request_goes_out() -> None:
    seen: list[httpx.Request] = []
    with _client(_playlist(), seen) as c:
        with pytest.raises(youtube.YoutubeQuotaExceeded):
            youtube.fetch_playlist_items("UUaaaaaaaaaaaaaaaaaaaaaa", api_key="k",
                                         client=c, budget=0)
    assert seen == []
