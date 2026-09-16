"""Tavily 웹 검색 티어. 네트워크 없이 요청·응답 계약을 고정한다.

langchain-tavily 대신 httpx 로 직접 부른다. 그 패키지가 langchain·langgraph 를 끌어와
배포본 메모리를 크게 늘리는데, 여유가 690MB 중 70MB 뿐이다. httpx 는 이미 배포본에 있다.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from baseball import latest_info
from baseball.config import Settings


def _settings(**kw: Any) -> Settings:
    base = {
        "_env_file": None,
        "enable_web_search": "on",
        "tavily_api_key": SecretStr("tvly-test-key"),
    }
    return Settings(**{**base, **kw})


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """캐시는 실제 Postgres 를 타므로 검사에서 떼어 둔다."""
    monkeypatch.setattr(latest_info, "_cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(latest_info, "_cache_put", lambda *_a, **_k: None)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _result(**kw: Any) -> dict[str, Any]:
    base = {"title": "제목", "url": "https://example.com/a", "content": "본문", "score": 0.5}
    return {**base, **kw}


# --------------------------------------------------------------------------- 요청 계약

def test_key_travels_in_the_header_never_in_the_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": [_result()]})

    latest_info.web_search("보크가 뭐야?", _settings(), client=_client(handler))

    req = seen[0]
    assert str(req.url) == latest_info.TAVILY_URL
    assert req.method == "POST"
    assert req.headers["Authorization"] == "Bearer tvly-test-key"
    body = json.loads(req.content)
    assert "api_key" not in body            # Tavily 가 폐기했고, 본문은 로그·캐시에 남는다
    assert "tvly-test-key" not in req.content.decode()


def test_time_sensitive_question_uses_the_news_topic() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [_result()]})

    latest_info.web_search("오늘 KT 경기 몇 시야?", _settings(), client=_client(handler))
    assert seen[0]["topic"] == "news"
    assert seen[0]["include_published_date"] is True


def test_plain_question_uses_the_general_topic() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [_result()]})

    latest_info.web_search("인필드 플라이가 뭐야?", _settings(), client=_client(handler))
    assert seen[0]["topic"] == "general"


def test_banned_domain_is_excluded() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [_result()]})

    latest_info.web_search("보크가 뭐야?", _settings(), client=_client(handler))
    assert "namu.wiki" in seen[0]["exclude_domains"]


# --------------------------------------------------------------------------- 시도 사다리

def test_first_attempt_restricts_domains_second_does_not() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        empty_first = len(seen) == 1
        return httpx.Response(200, json={"results": [] if empty_first else [_result()]})

    out = latest_info.web_search("인필드 플라이가 뭐야?", _settings(), client=_client(handler))

    assert len(seen) == 2
    assert seen[0]["include_domains"]            # 1차는 허용 도메인 제한
    assert "include_domains" not in seen[1]      # 2차는 무제한
    assert len(out) == 1


def test_successful_first_attempt_does_not_retry() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": [_result()]})

    latest_info.web_search("인필드 플라이가 뭐야?", _settings(), client=_client(handler))
    assert len(seen) == 1


def test_news_falls_back_to_general_as_a_third_attempt() -> None:
    """news 와 include_domains 를 함께 주면 Tavily 가 0건을 준다(실측). 그래서 폴백이 필요하다."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [] if len(seen) < 3 else [_result()]})

    out = latest_info.web_search("오늘 KT 경기 몇 시야?", _settings(), client=_client(handler))

    assert [b["topic"] for b in seen] == ["news", "news", "general"]
    assert len(out) == 1


# --------------------------------------------------------------------------- 실패 격리

@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_failures_degrade_to_empty(status: int) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "nope"})

    assert latest_info.web_search("보크", _settings(), client=_client(handler)) == []


def test_transport_failure_degrades_to_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    assert latest_info.web_search("보크", _settings(), client=_client(handler)) == []


def test_malformed_payload_degrades_to_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="이건 JSON 이 아니다")

    assert latest_info.web_search("보크", _settings(), client=_client(handler)) == []


def test_disabled_or_keyless_never_calls_out() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("호출되면 안 된다")

    c = _client(handler)
    assert latest_info.web_search("보크", _settings(enable_web_search="off"), client=c) == []
    assert latest_info.web_search("보크", Settings(_env_file=None), client=c) == []


# --------------------------------------------------------------------------- 응답 매핑

def test_results_map_onto_latest_entries() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [
            _result(title="기사", url="https://a.example/1", content="가" * 900,
                    score=0.91, published_date="2026-09-01T09:00:00Z"),
            _result(title="보통 점수", score=0.55),
            _result(content="   "),                      # 본문이 비면 버린다
        ]})

    out = latest_info.web_search("오늘 KT 경기 몇 시야?", _settings(), client=_client(handler))

    assert len(out) == 2
    first, second = out
    assert first.kind == "web"
    assert first.label == "웹 검색: 기사"
    assert first.source_url == "https://a.example/1"
    assert len(first.text) == latest_info.WEB_SNIPPET_MAX_CHARS
    assert first.as_of == "2026-09-01"                   # 발행일을 쓴다(오늘로 찍지 않는다)
    assert first.confidence == "likely"                  # score >= 0.7
    assert second.confidence == "uncertain"


def test_low_score_filler_is_dropped() -> None:
    """Tavily 는 맞는 게 없어도 빈 배열 대신 채움용 결과를 준다.

    실측: "야구에서 인필드 플라이 규칙" 을 허용 도메인 안에서 찾으면 0.153·0.025·0.024 짜리
    영어 어휘 영상과 보드게임이 5건 온다. 이걸 성공으로 치면 사다리가 멈추고 잡음이 근거가 된다.
    """
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(200, json={"results": [
                _result(title="영어 어휘", score=0.153),
                _result(title="보드게임", score=0.024),
            ]})
        return httpx.Response(200, json={"results": [_result(title="위키백과", score=0.919)]})

    out = latest_info.web_search("인필드 플라이 규칙", _settings(), client=_client(handler))

    assert len(seen) == 2, "잡음만 온 시도는 실패로 보고 다음 시도로 넘어가야 한다"
    assert [e.label for e in out] == ["웹 검색: 위키백과"]


def test_all_low_score_yields_nothing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_result(score=0.1)]})

    assert latest_info.web_search("인필드 플라이", _settings(), client=_client(handler)) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Mon, 24 Aug 2026 00:00:00 GMT", "2026-08-24"),   # Tavily 가 실제로 주는 형식
        ("2026-09-01T09:00:00Z", "2026-09-01"),
        ("", None),
        (None, None),
        ("쓰레기", None),
    ],
)
def test_published_date_is_normalised(raw: Any, expected: str | None) -> None:
    today = "2026-09-16"
    assert latest_info._as_of(raw, today) == (expected or today)


def test_missing_published_date_falls_back_to_today() -> None:
    from datetime import date

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_result()]})

    out = latest_info.web_search("인필드 플라이", _settings(), client=_client(handler))
    assert out[0].as_of == date.today().isoformat()


def test_empty_results_are_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """일시적 실패를 6시간 동안 고착시키지 않는다."""
    puts: list[Any] = []
    monkeypatch.setattr(latest_info, "_cache_put", lambda *a, **k: puts.append(a))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    assert latest_info.web_search("보크", _settings(), client=_client(handler)) == []
    assert puts == []
