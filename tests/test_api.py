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
