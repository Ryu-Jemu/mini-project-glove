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
