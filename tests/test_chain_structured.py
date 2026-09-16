"""구조화 출력 경로: 렌더링·폴백·강등·usage 보존·스트림 재조립."""
from __future__ import annotations

import json

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage

from baseball.chain import SCHEMA_FALLBACK, RagService, _token_pieces
from baseball.config import Settings
from baseball.prompts import NOT_IN_CONTEXT_REFUSAL
from conftest import CANNED_ANSWER, fake_structured_model
from helpers import make_answer_payload

QUESTION = "보크가 뭐야?"


def _service(settings, fake_retriever, responses: list[str]) -> RagService:
    return RagService(
        settings, fake_retriever,
        llm=fake_structured_model(responses),
        router_llm=FakeListChatModel(responses=['{"kind":"off_topic"}']),
    )


# --- 정상 경로 ----------------------------------------------------------------

def test_structured_answer_is_rendered_markdown(service) -> None:
    r = service.answer(QUESTION)
    assert r.status == "answered"
    assert r.answer.startswith("**보크는 투수의 반칙 투구입니다.**")
    assert "\n\n**어떤 상황에서 적용되나요**\n\n- **주자가 있을 때**" in r.answer
    assert r.answer_kind == "term_rule"
    assert r.format_issues == []
    assert r.format_ok is True


def test_citations_still_come_from_the_rendered_text(service) -> None:
    """근거 섹션이 있으므로 citations.extract 가 그대로 동작한다."""
    r = service.answer(QUESTION)
    assert [c["rule_id"] for c in r.citations] == ["5.09"]
    assert r.dropped_citations == []


def test_usage_survives_structured_output(service) -> None:
    """include_raw=True 가 아니면 AIMessage 가 사라져 토큰·비용이 조용히 0 이 된다."""
    prepared = service.prepare(QUESTION, model="gpt-4o-mini")
    raw = AIMessage(content="", usage_metadata={
        "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
        "input_token_details": {"cache_read": 80},
    })
    gen = service._structured_result(
        {"raw": raw, "parsed": make_answer_payload(), "parsing_error": None}, prepared
    )
    assert gen is not None
    assert gen.usage == {"input_tokens": 100, "output_tokens": 50, "cache_read": 80}


# --- 폴백 ---------------------------------------------------------------------

def test_broken_json_falls_back_to_a_plain_call(settings, fake_retriever) -> None:
    svc = _service(settings, fake_retriever, ["{ 반쪽짜리 JSON"])
    r = svc.answer(QUESTION)
    assert r.format_issues == [SCHEMA_FALLBACK]
    assert r.format_ok is False
    assert r.answer == "{ 반쪽짜리 JSON"      # 평문 재호출 결과를 그대로 쓴다
    assert r.answer_kind is None


def test_schema_violating_json_falls_back(settings, fake_retriever) -> None:
    """파싱은 되지만 필드가 빠진 경우도 폴백이다."""
    payload = make_answer_payload()
    del payload["why"]
    svc = _service(settings, fake_retriever, [json.dumps(payload, ensure_ascii=False)])
    assert svc.answer(QUESTION).format_issues == [SCHEMA_FALLBACK]


# --- blocking 린트 강등 ---------------------------------------------------------

def test_refusal_inside_answer_is_demoted_to_a_pure_refusal(settings, fake_retriever) -> None:
    payload = make_answer_payload(why=NOT_IN_CONTEXT_REFUSAL)
    svc = _service(settings, fake_retriever, [json.dumps(payload, ensure_ascii=False)])
    r = svc.answer(QUESTION)
    assert r.answer == NOT_IN_CONTEXT_REFUSAL     # 반쪽짜리를 보여 주지 않는다
    assert r.status == "not_in_rulebook"
    assert r.partial_refusal is False             # ⚠ 배너가 뜨지 않는다
    assert "REFUSAL_INSIDE_ANSWER" in r.format_issues
    assert r.format_ok is False


def test_non_blocking_lint_keeps_the_answer(settings, fake_retriever) -> None:
    payload = make_answer_payload(why="일반적으로 그렇게 봅니다.")
    svc = _service(settings, fake_retriever, [json.dumps(payload, ensure_ascii=False)])
    r = svc.answer(QUESTION)
    assert r.status == "answered"
    assert "HEDGE" in r.format_issues
    assert r.format_ok is True                    # 경고는 답변을 버리지 않는다


# --- 레거시 플래그 --------------------------------------------------------------

def test_flag_off_uses_the_plain_path(fake_retriever) -> None:
    settings = Settings(enable_answer_schema="off")
    svc = _service(settings, fake_retriever, ["보크는 투수의 반칙 투구입니다. 규칙 5.09 참조."])
    r = svc.answer(QUESTION)
    assert r.answer == "보크는 투수의 반칙 투구입니다. 규칙 5.09 참조."
    assert r.answer_kind is None
    assert r.format_issues == []
    assert [c["rule_id"] for c in r.citations] == ["5.09"]


# --- 스트리밍 계약 --------------------------------------------------------------

def test_token_pieces_reassemble_into_the_answer(service) -> None:
    prepared = service.prepare(QUESTION, model="gpt-4o-mini")
    gen = service._generate(prepared, question=QUESTION, session_id=None, model="gpt-4o-mini")
    assert "".join(_token_pieces(gen)) == gen.text
    assert len(_token_pieces(gen)) > 1            # 섹션 단위로 흘려 보낸다


def test_stream_matches_sync_answer(service) -> None:
    events = list(service.stream(QUESTION))
    streamed = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    final = [e for e in events if e["event"] == "final"][0]["data"]
    assert streamed == final["answer"]
    assert final["answer_kind"] == "term_rule"
    assert final["format_ok"] is True


def test_sse_event_order_is_unchanged(service) -> None:
    order = [e["event"] for e in service.stream(QUESTION)]
    assert order[0] == "route"
    assert order[1] == "status"
    assert order[2] == "sources"
    assert "token" in order
    assert order[-2:] == ["final", "done"]


@pytest.mark.asyncio
async def test_astream_matches_stream(service) -> None:
    events = [e async for e in service.astream(QUESTION)]
    streamed = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    assert streamed == [e for e in events if e["event"] == "final"][0]["data"]["answer"]


# --- 게이트 경로는 그대로 --------------------------------------------------------

def test_gate_path_is_untouched_by_the_schema(service) -> None:
    r = service.answer("파이썬 리스트 정렬하는 법 알려줘")
    assert r.llm_called is False
    assert r.answer_kind is None
    assert r.format_ok is True
    assert r.format_issues == []


def test_flag_off_restores_token_streaming(fake_retriever) -> None:
    """롤백 레버는 형식뿐 아니라 스트리밍까지 되돌려야 반쪽이 아니다."""
    settings = Settings(enable_answer_schema="off")
    svc = _service(settings, fake_retriever, ["보크는 투수의 반칙 투구입니다. 규칙 5.09 참조."])
    tokens = [e["data"]["text"] for e in svc.stream(QUESTION) if e["event"] == "token"]
    assert len(tokens) > 1, "플래그를 꺼도 한 덩어리로만 온다 — 스트리밍이 죽어 있다"
    assert "".join(tokens) == "보크는 투수의 반칙 투구입니다. 규칙 5.09 참조."


@pytest.mark.asyncio
async def test_flag_off_restores_token_streaming_async(fake_retriever) -> None:
    settings = Settings(enable_answer_schema="off")
    svc = _service(settings, fake_retriever, ["보크는 투수의 반칙 투구입니다. 규칙 5.09 참조."])
    tokens = [e["data"]["text"] async for e in svc.astream(QUESTION) if e["event"] == "token"]
    assert len(tokens) > 1
    assert "".join(tokens) == "보크는 투수의 반칙 투구입니다. 규칙 5.09 참조."
