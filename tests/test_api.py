from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from baseball.prompts import NOT_IN_CONTEXT_REFUSAL, OFF_TOPIC_REFUSAL, PROMPT_SHA


@pytest.fixture
def app(service):
    """ASGITransport 는 lifespan 을 실행하지 않으므로 state 를 직접 주입한다."""
    from baseball.api import create_app

    application = create_app(service.settings)
    application.state.settings = service.settings
    application.state.service = service
    application.state.ready_error = None
    return application


async def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_healthz_and_readyz(app) -> None:
    async with await _client(app) as c:
        health = (await c.get("/healthz")).json()
        assert health["prompt_sha"] == PROMPT_SHA and health["status"] == "ok"
        ready = (await c.get("/readyz")).json()
        assert ready["status"] == "ready" and ready["bm25_ready"] is True


async def test_chat_schema(app) -> None:
    async with await _client(app) as c:
        r = await c.post("/chat", json={"question": "보크가 뭐야?", "session_id": "t1"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "answered"
        assert body["llm_called"] is True
        assert body["partial_refusal"] is False
        assert body["prompt_sha"] == PROMPT_SHA
        assert body["route"] == {"kind": "rule", "by": "keyword"}
        assert len(body["sources"]) == 1
        assert [c_["rule_id"] for c_ in body["citations"]] == ["5.09"]
        # 답변 스키마가 내보내는 세 필드. 기본값이 있어 구버전 클라이언트도 깨지지 않는다.
        assert body["format_ok"] is True
        assert body["format_issues"] == []
        assert body["answer_kind"] in {"term_rule", "situation", "entity", "latest", None}


async def test_empty_question_is_422(app) -> None:
    async with await _client(app) as c:
        assert (await c.post("/chat", json={"question": ""})).status_code == 422


async def test_off_topic_never_calls_answer_llm(app, service) -> None:
    async with await _client(app) as c:
        body = (await c.post("/chat", json={"question": "오늘 서울 날씨 어때?"})).json()
    assert body["answer"] == OFF_TOPIC_REFUSAL
    assert body["status"] == "out_of_scope"
    assert body["llm_called"] is False


async def test_abstain_returns_fixed_refusal(app, service) -> None:
    service.retriever.abstain = True
    async with await _client(app) as c:
        body = (await c.post("/chat", json={"question": "인필드 플라이가 뭐야?"})).json()
    assert body["answer"] == NOT_IN_CONTEXT_REFUSAL
    assert body["status"] == "not_in_rulebook"
    assert body["llm_called"] is False


async def test_stream_event_order(app) -> None:
    async with await _client(app) as c:
        r = await c.post("/chat/stream", json={"question": "보크가 뭐야?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [l.split(": ", 1)[1] for l in r.text.splitlines() if l.startswith("event: ")]
    assert events[0] == "route"
    assert events[1] == "status"
    assert events[2] == "sources"
    assert "token" in events
    assert events[-2:] == ["final", "done"]


async def test_stream_final_payload(app) -> None:
    async with await _client(app) as c:
        r = await c.post("/chat/stream", json={"question": "보크가 뭐야?"})
    lines = r.text.splitlines()
    idx = lines.index("event: final")
    payload = json.loads(lines[idx + 1].split("data: ", 1)[1])
    assert payload["status"] == "answered"
    assert payload["partial_refusal"] is False
    assert payload["llm_called"] is True
    assert set(payload["usage"]) >= {"input_tokens", "output_tokens", "cache_read", "cost_usd"}
    assert payload["format_ok"] is True
    assert payload["format_issues"] == []
    assert "answer_kind" in payload


async def test_session_reset(app, service) -> None:
    async with await _client(app) as c:
        await c.post("/chat", json={"question": "보크가 뭐야?", "session_id": "s"})
        assert service.history("s")
        await c.post("/sessions/s/reset")
    assert service.history("s") == []


# --------------------------------------------------------------------------- 미디어 채널

def test_media_survives_the_chat_response_boundary() -> None:
    """POST /chat 만 pydantic 을 거친다. 여기서 잘리면 SSE 와 동작이 갈린다.

    Source 와 ChatResponse 는 model_config 가 없어 extra="ignore" 다. 그래서
    선언하지 않은 키는 예외 없이 사라진다 — tools.py 가 source dict 에 넣는
    page·rule_id 가 지금도 이렇게 버려지고 있다. 미디어를 sources 안에 동봉하지
    않고 최상위 필드로 올린 이유가 이것이다.
    """
    from baseball.chain import TurnResult, _final_payload
    from baseball.schemas import ChatResponse

    result = TurnResult(
        answer="9월 13일 경기 하이라이트입니다.", status="answered",
        route={"kind": "latest", "by": "keyword"}, model="gpt-4o-mini", session_id="s1",
        media=[{"kind": "video", "id": "abc", "title": "하이라이트",
                "url": "https://youtu.be/abc", "embeddable": True, "duration": "12:41"},
               {"kind": "map", "id": "m1", "query": "잠실야구장 근처 맛집"}],
        places=[{"id": "p1", "name": "잠실 맛집 베스트10",
                 "url": "https://blog.example/1", "confidence": "likely"}],
    )
    out = ChatResponse(**_final_payload(result)).model_dump()

    assert [m["kind"] for m in out["media"]] == ["video", "map"]
    assert out["places"][0]["name"] == "잠실 맛집 베스트10"
    # 지도는 query 만 싣는다. Embed API 키가 클라이언트에 노출되므로 URL 은 UI 가 조립한다.
    assert out["media"][1]["url"] is None
    assert out["media"][1]["query"] == "잠실야구장 근처 맛집"


def test_new_source_kinds_do_not_raise_on_chat() -> None:
    """api.py 의 ChatResponse(...) 는 try 블록 밖이라 ValidationError 가 500 이 된다.

    Source.kind 가 닫힌 Literal 이므로 새 출처 종류는 반드시 여기에 등록해야 한다.
    """
    from baseball.chain import TurnResult, _final_payload
    from baseball.schemas import ChatResponse

    for kind in ("static", "snapshot", "web", "kbo", "model", "place", "video"):
        result = TurnResult(
            answer="x", status="answered", route={"kind": "latest", "by": "keyword"},
            model="gpt-4o-mini", sources=[{"kind": kind, "label": "L", "url": None}],
        )
        assert ChatResponse(**_final_payload(result)).sources[0].kind == kind


def test_media_defaults_to_empty_and_keeps_the_old_contract() -> None:
    """기존 응답에는 media 키가 없었다. 기본값이 빈 배열이라 계약이 깨지지 않는다."""
    from baseball.chain import TurnResult, _final_payload
    from baseball.schemas import ChatResponse

    result = TurnResult(answer="x", status="answered",
                        route={"kind": "rule", "by": "keyword"}, model="gpt-4o-mini")
    out = ChatResponse(**_final_payload(result)).model_dump()
    assert out["media"] == [] and out["places"] == []
