"""턴 흐름: route → (최신정보) → 검색 → {context} 조립 → verbatim 프롬프트 1회 호출."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from baseball import citations as cite
from baseball import latest_info
from baseball.config import Settings, get_settings
from baseball.context import KboEntry, LatestEntry, format_context
from baseball.prompts import (
    NOT_IN_CONTEXT_REFUSAL,
    OFF_TOPIC_REFUSAL,
    PROMPT_SHA,
    answer_prompt,
    detect_status,
    has_partial_refusal,
)
from baseball.router import Route, route as route_question

# (input, output, cached_input) USD per 1M tokens
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "gpt-5.6-luna": (0.20, 1.20, 0.02),
}
RULEBOOK_AS_OF = "2026-01-01"

QUOTA_MESSAGE = (
    "OpenAI 사용 한도를 초과해 답변을 만들 수 없습니다. "
    "platform.openai.com 의 프로젝트 한도(Limits)를 올리거나 결제 수단을 확인해 주세요."
)


class QuotaExceeded(RuntimeError):
    """OpenAI 지출 한도·쿼터 초과 — 재시도가 아니라 사용자 조치가 필요하다."""


def is_quota_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(k in text for k in ("insufficient_quota", "spend limit", "project_spend_limit_exceeded",
                                   "exceeded your current quota", "billing_hard_limit_reached"))


def build_llm(model: str, settings: Settings) -> Any:
    from langchain_openai import ChatOpenAI

    kw: dict[str, Any] = {
        "model": model, "timeout": 30, "max_retries": 3,
        "streaming": True, "stream_usage": True, "api_key": settings.openai_api_key,
    }
    if model.startswith("gpt-5."):
        kw["reasoning_effort"] = "none"     # gpt-5.x 는 temperature 미지원
    else:
        kw["temperature"] = 0
    return ChatOpenAI(**kw).with_config(tags=["final_answer"], run_name="final_answer")


def _cost(model: str, usage: dict[str, Any]) -> float:
    price_in, price_out, price_cached = PRICES.get(model, PRICES["gpt-4o-mini"])
    cached = int(usage.get("cache_read", 0))
    fresh_in = max(int(usage.get("input_tokens", 0)) - cached, 0)
    return (
        fresh_in / 1e6 * price_in
        + cached / 1e6 * price_cached
        + int(usage.get("output_tokens", 0)) / 1e6 * price_out
    )


@dataclass
class TurnResult:
    answer: str
    status: str
    partial_refusal: bool = False
    freshness: str = "static"
    needs_web: bool = False
    route: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    llm_called: bool = True
    context: str = ""
    model: str = ""
    session_id: str | None = None
    prompt_sha: str = PROMPT_SHA
    retrieval: Any | None = None


@dataclass
class _Prepared:
    route: Route
    latest: list[LatestEntry]
    freshness: str
    docs: list[dict[str, Any]]
    context: str
    gate: TurnResult | None = None


def _rule_sources(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "kind": "static",
        "label": f"2026 공식야구규칙 {d['breadcrumb']} p.{d['page_start']}",
        "url": None, "as_of": RULEBOOK_AS_OF, "confidence": "confirmed",
    } for d in docs]


def _latest_sources(entries: Iterable[LatestEntry]) -> list[dict[str, Any]]:
    return [{
        "kind": e.kind, "label": e.label, "url": e.source_url,
        "as_of": e.as_of, "confidence": e.confidence,
    } for e in entries]


class RagService:
    def __init__(
        self, settings: Settings, retriever: Any, *,
        llm: Any | None = None, router_llm: Any | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self._llm = llm
        self._router_llm = router_llm
        self.sessions: dict[str, deque[BaseMessage]] = {}

    @classmethod
    def create(cls, settings: Settings | None = None) -> "RagService":
        from baseball.retriever import build_retriever

        settings = settings or get_settings()
        return cls(settings, build_retriever(settings))

    # --- 세션 -------------------------------------------------------------
    def history(self, session_id: str | None) -> list[BaseMessage]:
        if not session_id:
            return []
        return list(self.sessions.get(session_id, ()))

    def remember(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        buf = self.sessions.setdefault(
            session_id, deque(maxlen=self.settings.chat_history_max_messages)
        )
        buf.append(HumanMessage(content=question))
        buf.append(AIMessage(content=answer))

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def llm(self, model: str) -> Any:
        if self._llm is not None:
            return self._llm
        return build_llm(model, self.settings)

    # --- 턴 준비(동기/스트림 공용) -----------------------------------------
    def prepare(self, question: str, *, model: str) -> _Prepared:
        r = route_question(question, llm=self._router_llm, settings=self.settings)

        if r.kind == "off_topic":
            gate = TurnResult(
                answer=OFF_TOPIC_REFUSAL, status="out_of_scope", freshness="static",
                route=r.to_dict(), llm_called=False, model=model,
            )
            return _Prepared(r, [], "static", [], "", gate)

        latest: list[LatestEntry] = []
        freshness = "static"
        if r.kind in {"latest", "mixed"}:
            freshness, latest = latest_info.route(question, self.settings)
            if freshness == "phase2_pending":
                gate = TurnResult(
                    answer=NOT_IN_CONTEXT_REFUSAL, status="phase2_pending", freshness="static",
                    needs_web=True, route=r.to_dict(), llm_called=False, model=model,
                )
                return _Prepared(r, [], "static", [], "", gate)

        retrieval = self.retriever.retrieve(question, k=6)
        docs = list(retrieval.docs)
        if latest and r.kind == "latest" and not r.rule_hit:
            docs = []          # 순위·일정 같은 질문에 규칙집 조항이 섞이면 모델이 답을 거부한다
        if retrieval.abstain:
            if not latest:
                gate = TurnResult(
                    answer=NOT_IN_CONTEXT_REFUSAL, status="not_in_rulebook", freshness=freshness,
                    route=r.to_dict(), llm_called=False, model=model,
                    sources=_latest_sources(latest), retrieval=retrieval,
                )
                return _Prepared(r, latest, freshness, [], "", gate)
            docs = []                      # 규칙집 근거 없음 → 최신정보 블록만 사용

        context = format_context(
            docs, latest,
            max_tokens=self.settings.context_max_tokens,
            latest_max_tokens=self.settings.latest_context_max_tokens,
        )
        if not context:
            gate = TurnResult(
                answer=NOT_IN_CONTEXT_REFUSAL, status="not_in_rulebook", freshness=freshness,
                route=r.to_dict(), llm_called=False, model=model, retrieval=retrieval,
            )
            return _Prepared(r, latest, freshness, [], "", gate)

        prepared = _Prepared(r, latest, freshness, docs, context)
        prepared.retrieval = retrieval                      # type: ignore[attr-defined]
        return prepared

    def _finish(
        self, prepared: _Prepared, *, question: str, answer_text: str, usage: dict[str, Any],
        model: str, started: float, session_id: str | None,
    ) -> TurnResult:
        status = detect_status(answer_text, tail_max_chars=self.settings.refusal_tail_max_chars)
        found, dropped = cite.extract(answer_text, prepared.docs)
        usage = dict(usage)
        usage["cost_usd"] = round(_cost(model, usage), 6)
        result = TurnResult(
            answer=answer_text,
            status=status,
            partial_refusal=has_partial_refusal(answer_text),
            freshness=prepared.freshness,
            needs_web=False,
            route=prepared.route.to_dict(),
            citations=[c.to_dict() for c in found],
            dropped_citations=dropped,
            sources=_rule_sources(prepared.docs) + _latest_sources(prepared.latest),
            usage=usage,
            latency_ms=int((time.time() - started) * 1000),
            llm_called=True,
            context=prepared.context,
            model=model,
            session_id=session_id,
            retrieval=getattr(prepared, "retrieval", None),
        )
        self.remember(session_id, question, answer_text)
        return result

    def _gate_result(
        self, gate: TurnResult, *, question: str, session_id: str | None, started: float
    ) -> TurnResult:
        gate.latency_ms = int((time.time() - started) * 1000)
        gate.session_id = session_id
        gate.context = ""
        self.remember(session_id, question, gate.answer)
        return gate

    # --- 동기 ---------------------------------------------------------------
    def answer(
        self, question: str, session_id: str | None = None, model: str | None = None
    ) -> TurnResult:
        started = time.time()
        model = model or self.settings.openai_chat_model
        prepared = self.prepare(question, model=model)
        if prepared.gate is not None:
            return self._gate_result(prepared.gate, question=question, session_id=session_id, started=started)

        messages = answer_prompt(self.settings.chat_history_max_messages).format_messages(
            question=question, context=prepared.context, chat_history=self.history(session_id),
        )
        ai = self.llm(model).invoke(messages)
        text = ai.content if isinstance(ai.content, str) else str(ai.content)
        usage = _usage_of(ai)
        return self._finish(
            prepared, question=question, answer_text=text, usage=usage,
            model=model, started=started, session_id=session_id,
        )

    # --- 스트리밍 -----------------------------------------------------------
    async def astream(
        self, question: str, session_id: str | None = None, model: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        started = time.time()
        model = model or self.settings.openai_chat_model
        prepared = self.prepare(question, model=model)
        yield {"event": "route", "data": prepared.route.to_dict()}

        if prepared.gate is not None:
            result = self._gate_result(prepared.gate, question=question, session_id=session_id, started=started)
            yield {"event": "status", "data": {"status": result.status, "llm_called": False}}
            yield {"event": "sources", "data": {"sources": result.sources}}
            yield {"event": "token", "data": {"text": result.answer}}
            yield {"event": "final", "data": _final_payload(result)}
            yield {"event": "done", "data": {}}
            return

        yield {"event": "status", "data": {"status": "retrieving", "llm_called": True}}
        sources = _rule_sources(prepared.docs) + _latest_sources(prepared.latest)
        yield {"event": "sources", "data": {"sources": sources}}

        messages = answer_prompt(self.settings.chat_history_max_messages).format_messages(
            question=question, context=prepared.context, chat_history=self.history(session_id),
        )
        parts: list[str] = []
        usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0}
        async for chunk in self.llm(model).astream(messages):
            piece = chunk.content if isinstance(chunk.content, str) else ""
            if piece:
                parts.append(piece)
                yield {"event": "token", "data": {"text": piece}}
            chunk_usage = _usage_of(chunk)
            if chunk_usage.get("output_tokens") or chunk_usage.get("input_tokens"):
                usage = chunk_usage
        text = "".join(parts)
        result = self._finish(
            prepared, question=question, answer_text=text, usage=usage,
            model=model, started=started, session_id=session_id,
        )
        yield {"event": "final", "data": _final_payload(result)}
        yield {"event": "done", "data": {}}


    def stream(
        self, question: str, session_id: str | None = None, model: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """astream() 의 동기 쌍둥이.

        Streamlit 스크립트 러너에는 실행 중인 이벤트 루프가 없어 async 제너레이터를
        그대로 쓸 수 없다. 이벤트 순서와 페이로드는 astream() 과 동일하다.
        """
        started = time.time()
        model = model or self.settings.openai_chat_model
        prepared = self.prepare(question, model=model)
        yield {"event": "route", "data": prepared.route.to_dict()}

        if prepared.gate is not None:
            result = self._gate_result(prepared.gate, question=question, session_id=session_id, started=started)
            yield {"event": "status", "data": {"status": result.status, "llm_called": False}}
            yield {"event": "sources", "data": {"sources": result.sources}}
            yield {"event": "token", "data": {"text": result.answer}}
            yield {"event": "final", "data": _final_payload(result)}
            yield {"event": "done", "data": {}}
            return

        yield {"event": "status", "data": {"status": "retrieving", "llm_called": True}}
        sources = _rule_sources(prepared.docs) + _latest_sources(prepared.latest)
        yield {"event": "sources", "data": {"sources": sources}}

        messages = answer_prompt(self.settings.chat_history_max_messages).format_messages(
            question=question, context=prepared.context, chat_history=self.history(session_id),
        )
        parts: list[str] = []
        usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0}
        for chunk in self.llm(model).stream(messages):
            piece = chunk.content if isinstance(chunk.content, str) else ""
            if piece:
                parts.append(piece)
                yield {"event": "token", "data": {"text": piece}}
            chunk_usage = _usage_of(chunk)
            if chunk_usage.get("output_tokens") or chunk_usage.get("input_tokens"):
                usage = chunk_usage
        text = "".join(parts)
        result = self._finish(
            prepared, question=question, answer_text=text, usage=usage,
            model=model, started=started, session_id=session_id,
        )
        yield {"event": "final", "data": _final_payload(result)}
        yield {"event": "done", "data": {}}


def _usage_of(message: Any) -> dict[str, Any]:
    meta = getattr(message, "usage_metadata", None) or {}
    details = meta.get("input_token_details") or {}
    return {
        "input_tokens": int(meta.get("input_tokens", 0) or 0),
        "output_tokens": int(meta.get("output_tokens", 0) or 0),
        "cache_read": int(details.get("cache_read", 0) or 0),
    }


def _final_payload(result: TurnResult) -> dict[str, Any]:
    return {
        "answer": result.answer, "status": result.status,
        "partial_refusal": result.partial_refusal, "freshness": result.freshness,
        "needs_web": result.needs_web, "route": result.route,
        "citations": result.citations, "dropped_citations": result.dropped_citations,
        "sources": result.sources, "usage": result.usage,
        "latency_ms": result.latency_ms, "prompt_sha": result.prompt_sha,
        "model": result.model, "session_id": result.session_id,
        "llm_called": result.llm_called,
    }
