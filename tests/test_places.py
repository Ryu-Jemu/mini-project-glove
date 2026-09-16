"""연고지 맛집. Tavily 요청 형태와 할루시네이션 방어 다섯 겹을 고정한다."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from baseball import latest_info, places
from baseball.config import Settings


def _settings(**kw: Any) -> Settings:
    base = {
        "_env_file": None,
        "enable_web_search": "on",
        "enable_places": "on",
        "tavily_api_key": SecretStr("tvly-test-key"),
    }
    return Settings(**{**base, **kw})


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latest_info, "_cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(latest_info, "_cache_put", lambda *_a, **_k: None)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _hit(n: int = 1, score: float = 0.8, **kw: Any) -> dict[str, Any]:
    base = {"title": f"잠실 맛집 베스트{n}", "url": f"https://blog.example/{n}",
            "content": "가", "score": score}
    return {**base, **kw}


def _run(results: list[dict[str, Any]], seen: list[httpx.Request] | None = None, **skw: Any):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json={"results": results})

    return places.search(stadium="서울종합운동장 야구장", stadium_short="잠실",
                         settings=_settings(**skw), client=_client(handler))


# --------------------------------------------------------------------------- 요청 계약

def test_topic_is_general_never_news() -> None:
    """LIVE_WEB_RE 에 연고지·홈구장이 있어 자동 판정은 news 로 샌다. 맛집은 뉴스가 아니다."""
    seen: list[httpx.Request] = []
    _run([_hit()], seen)
    body = json.loads(seen[0].content)
    assert body["topic"] == "general"


def test_include_domains_is_never_sent() -> None:
    """허용 도메인은 KBO·언론사다. 맛집 블로그가 없으므로 제한을 걸면 0건이 확정이다."""
    seen: list[httpx.Request] = []
    _run([_hit()], seen)
    assert len(seen) == 1                                  # 도메인 제한 시도를 건너뛴다
    assert "include_domains" not in json.loads(seen[0].content)


def test_exclude_domains_still_applies() -> None:
    """robots 로 직접 수집을 금지한 곳은 검색 결과에서도 받지 않는다."""
    seen: list[httpx.Request] = []
    _run([_hit()], seen)
    assert "namu.wiki" in json.loads(seen[0].content)["exclude_domains"]


def test_key_travels_in_the_header_not_the_body() -> None:
    seen: list[httpx.Request] = []
    _run([_hit()], seen)
    assert seen[0].headers["Authorization"] == "Bearer tvly-test-key"
    assert "api_key" not in json.loads(seen[0].content)


def test_query_is_deterministic_not_the_user_sentence() -> None:
    seen: list[httpx.Request] = []
    _run([_hit()], seen)
    assert json.loads(seen[0].content)["query"] == "잠실야구장 근처 맛집 추천"


# --------------------------------------------------------------------------- 방어 다섯 겹

def test_low_score_results_are_dropped() -> None:
    """Tavily 는 맞는 게 없어도 채움용 결과를 준다. 맛집에서는 그게 가짜 가게다."""
    entries, cards, _ = _run([_hit(1, score=0.55)])         # 기본 임계값 0.6 미만
    assert entries == [] and cards == []


def test_results_without_a_url_are_dropped() -> None:
    entries, cards, _ = _run([_hit(1, url=""), _hit(2)])
    assert [c["url"] for c in cards] == ["https://blog.example/2"]


def test_name_is_the_provider_title_verbatim() -> None:
    """상호를 본문에서 추출하지 않는다. 추출 단계가 곧 할루시네이션이다."""
    _, cards, _ = _run([_hit(1)])
    assert cards[0]["name"] == "잠실 맛집 베스트1"


def test_cards_carry_no_address_or_phone_fields() -> None:
    """없는 필드는 틀릴 수 없다."""
    _, cards, _ = _run([_hit(1)])
    assert set(cards[0]) == {"id", "name", "url", "snippet", "confidence"}


def test_notice_entry_comes_first() -> None:
    """프롬프트를 건드리지 않고 컨텍스트 블록으로 지시를 얹는다."""
    entries, _, _ = _run([_hit(1)])
    assert entries[0].label == places.NOTICE_LABEL
    assert "확인되지 않았습니다" in entries[0].text


def test_snippet_is_capped_for_the_context_budget() -> None:
    """600자를 그대로 쓰면 4건이 latest_context_max_tokens(800) 을 넘어 조용히 잘린다."""
    entries, _, _ = _run([_hit(1, content="가" * 5000)])
    body = [e for e in entries if e.label != places.NOTICE_LABEL][0]
    assert len(body.text) == places.PLACES_SNIPPET_MAX_CHARS


def test_results_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    _, cards, _ = _run([_hit(i) for i in range(1, 9)])
    assert len(cards) == 4                                   # places_max_results 기본값


def test_duplicate_urls_are_shown_once() -> None:
    _, cards, _ = _run([_hit(1), _hit(1)])
    assert len(cards) == 1


# --------------------------------------------------------------------------- 게이트·실패

def test_disabled_returns_nothing() -> None:
    entries, cards, media = _run([_hit(1)], enable_places="off")
    assert (entries, cards, media) == ([], [], [])


def test_web_search_off_disables_places() -> None:
    entries, cards, media = _run([_hit(1)], enable_web_search="off")
    assert (entries, cards, media) == ([], [], [])


@pytest.mark.parametrize("status", [403, 429, 500, 503])
def test_upstream_errors_degrade_quietly(status: int) -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={})

    out = places.search(stadium="사직야구장", stadium_short="사직",
                        settings=_settings(), client=_client(handler))
    assert out == ([], [], [])


def test_connection_failure_degrades_quietly() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    out = places.search(stadium="사직야구장", stadium_short="사직",
                        settings=_settings(), client=_client(handler))
    assert out == ([], [], [])


# --------------------------------------------------------------------------- 지도

def test_map_is_omitted_without_a_key() -> None:
    """키가 없으면 지도를 생략한다. 기능 저하는 허용하되 오류는 내지 않는다."""
    _, cards, media = _run([_hit(1)], enable_places_map="on")
    assert cards and media == []


def test_map_carries_only_a_query_never_a_url() -> None:
    """Embed API 키는 클라이언트에 노출된다. 응답 본문에 키도 URL 도 넣지 않는다."""
    _, _, media = _run([_hit(1)], enable_places_map="on",
                       google_maps_embed_api_key=SecretStr("AIza-test"))
    assert len(media) == 1
    assert media[0]["kind"] == "map"
    assert media[0]["query"] == "서울종합운동장 야구장 근처 맛집"
    assert "url" not in media[0]


def test_map_is_omitted_when_there_are_no_places() -> None:
    _, _, media = _run([], enable_places_map="on",
                       google_maps_embed_api_key=SecretStr("AIza-test"))
    assert media == []
