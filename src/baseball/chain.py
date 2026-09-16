"""턴 흐름: route → (최신정보) → 검색 → {context} 조립 → verbatim 프롬프트 1회 호출."""
from __future__ import annotations

import logging
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Iterator, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError

from baseball import answer_lint, answer_render
from baseball import citations as cite
from baseball import latest_info, scope
from baseball.answer_schema import RESPONSE_FORMAT, AnswerDoc
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

log = logging.getLogger(__name__)

# (input, output, cached_input) USD per 1M tokens
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "gpt-5.6-luna": (0.20, 1.20, 0.02),
}
RULEBOOK_AS_OF = "2026-01-01"
SCHEMA_FALLBACK = "SCHEMA_FALLBACK"

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


def build_llm(
    model: str, settings: Settings, *, streaming: bool = True, max_tokens: int | None = None
) -> Any:
    from langchain_openai import ChatOpenAI

    kw: dict[str, Any] = {
        "model": model, "timeout": 30, "max_retries": 3,
        "streaming": streaming, "stream_usage": True, "api_key": settings.openai_api_key,
    }
    if max_tokens:
        kw["max_tokens"] = max_tokens
    if model.startswith("gpt-5."):
        kw["reasoning_effort"] = "none"     # gpt-5.x 는 temperature 미지원
    else:
        kw["temperature"] = 0
    return ChatOpenAI(**kw).with_config(tags=["final_answer"], run_name="final_answer")


def build_structured_llm(model: str, settings: Settings) -> Any:
    """답변 구조를 강제하는 LLM. 프롬프트가 아니라 response_format 으로 건다.

    streaming=False 가 중요하다. 생성자에 streaming=True 를 주면 model_fields_set 에
    남아 _should_stream 이 참이 되고 .invoke() 조차 스트림 경로로 간다.
    include_raw=True 도 중요하다. False 면 llm | parser 가 되어 AIMessage 가 사라지고
    _usage_of 가 붙을 곳이 없어 토큰·비용이 조용히 0 이 된다.
    """
    llm = build_llm(
        model, settings, streaming=False, max_tokens=settings.answer_max_output_tokens
    )
    return llm.with_structured_output(
        RESPONSE_FORMAT, method="json_schema", strict=True, include_raw=True
    )


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
    answer_kind: str | None = None
    format_issues: list[str] = field(default_factory=list)

    @property
    def format_ok(self) -> bool:
        """스키마 경로가 온전히 돌았는가. 폴백과 blocking 린트는 실패로 본다."""
        return not any(
            code in answer_lint.BLOCKING_CODES or code == SCHEMA_FALLBACK
            for code in self.format_issues
        )


@dataclass
class _Generated:
    """한 번의 생성 결과. blocks 는 SSE token 이벤트 단위다."""

    text: str
    usage: dict[str, Any]
    blocks: list[str] = field(default_factory=list)
    kind: str | None = None
    issues: list[str] = field(default_factory=list)


@dataclass
class _Prepared:
    route: Route
    latest: list[LatestEntry]
    freshness: str
    docs: list[dict[str, Any]]
    context: str
    gate: TurnResult | None = None
    kbo: list[Any] = field(default_factory=list)
    knowledge_only: bool = False


def _rule_sources(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "kind": "static",
        "label": f"2026 공식야구규칙 {d['breadcrumb']} p.{d['page_start']}",
        "url": None, "as_of": RULEBOOK_AS_OF, "confidence": "confirmed",
    } for d in docs]


def _model_source() -> list[dict[str, Any]]:
    """근거 문서가 없다는 사실을 출처로 남긴다.

    화면에서 구분해 보여 주기 위해서이고, sources 가 비지 않아야 평가 게이트
    (sources_nonempty_ratio == 1.0)도 그대로 유지된다.
    """
    return [{
        "kind": "model", "label": "모델 일반 지식 (규칙집·웹에서 확인되지 않음)",
        "url": None, "as_of": None, "confidence": "uncertain",
    }]


def _latest_sources(entries: Iterable[LatestEntry]) -> list[dict[str, Any]]:
    return [{
        "kind": e.kind, "label": e.label, "url": e.source_url,
        "as_of": e.as_of, "confidence": e.confidence,
    } for e in entries]


class RagService:
    def __init__(
        self, settings: Settings, retriever: Any, *,
        llm: Any | None = None, router_llm: Any | None = None, scope_llm: Any | None = None,
        web_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self._llm = llm
        self._router_llm = router_llm
        self._scope_llm = scope_llm
        self._web_client = web_client
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

    def structured_llm(self, model: str) -> Any:
        if self._llm is not None:                       # 테스트 주입 seam
            return self._llm.with_structured_output(
                RESPONSE_FORMAT, method="json_schema", strict=True, include_raw=True
            )
        return build_structured_llm(model, self.settings)

    # --- 생성(동기/비동기 공용 조각) ----------------------------------------
    def _render_messages(
        self, prepared: _Prepared, *, question: str, session_id: str | None
    ) -> list[BaseMessage]:
        return answer_prompt(self.settings.chat_history_max_messages).format_messages(
            question=question, context=prepared.context, chat_history=self.history(session_id),
        )

    def _plain_result(self, ai: Any, *, issues: list[str]) -> _Generated:
        text = ai.content if isinstance(ai.content, str) else str(ai.content)
        return _Generated(
            text=text, usage=_usage_of(ai), blocks=[text] if text else [], issues=issues
        )

    def _structured_result(self, out: Any, prepared: _Prepared) -> _Generated | None:
        """구조화 응답을 린트·렌더까지 마친다. 파싱·검증에 실패하면 None."""
        if not isinstance(out, dict):
            return None
        usage = _usage_of(out.get("raw"))
        parsed = out.get("parsed")
        if parsed is None:
            return None
        try:
            doc = AnswerDoc.model_validate(parsed)
        except ValidationError:
            return None

        issues = answer_lint.check(doc, prepared.docs, knowledge_only=prepared.knowledge_only)
        if answer_lint.blocking(issues):
            # 본문에 거부 문장이 섞이거나 headline 이 비면 화면이 망가진다.
            # 반쪽짜리를 보여 주느니 순수 거부 문장으로 떨어뜨린다.
            doc = doc.model_copy(update={"answerable": False, "refusal": "not_in_context"})
        return _Generated(
            text=answer_render.render(doc), usage=usage,
            blocks=answer_render.render_blocks(doc), kind=doc.kind,
            issues=answer_lint.codes(issues),
        )

    def _stream_plain(
        self, messages: list[BaseMessage], model: str, parts: list[str], usage: dict[str, Any]
    ) -> Iterator[str]:
        """레거시 토큰 단위 스트리밍. parts·usage 를 채우면서 조각을 흘린다."""
        for chunk in self.llm(model).stream(messages):
            piece = chunk.content if isinstance(chunk.content, str) else ""
            if piece:
                parts.append(piece)
                yield piece
            chunk_usage = _usage_of(chunk)
            if chunk_usage.get("output_tokens") or chunk_usage.get("input_tokens"):
                usage.update(chunk_usage)

    async def _astream_plain(
        self, messages: list[BaseMessage], model: str, parts: list[str], usage: dict[str, Any]
    ) -> AsyncIterator[str]:
        async for chunk in self.llm(model).astream(messages):
            piece = chunk.content if isinstance(chunk.content, str) else ""
            if piece:
                parts.append(piece)
                yield piece
            chunk_usage = _usage_of(chunk)
            if chunk_usage.get("output_tokens") or chunk_usage.get("input_tokens"):
                usage.update(chunk_usage)

    def _generate(
        self, prepared: _Prepared, *, question: str, session_id: str | None, model: str
    ) -> _Generated:
        messages = self._render_messages(prepared, question=question, session_id=session_id)
        if not self.settings.answer_schema_enabled:
            return self._plain_result(self.llm(model).invoke(messages), issues=[])
        gen = self._structured_result(self.structured_llm(model).invoke(messages), prepared)
        if gen is not None:
            return gen
        # 파싱 실패의 대표 원인은 출력 절단이고, 그때 raw.content 는 반쪽 JSON 이라 쓸 수 없다.
        # response_format 없이 같은 메시지로 딱 한 번 다시 부른다(턴당 최대 1회).
        return self._plain_result(self.llm(model).invoke(messages), issues=[SCHEMA_FALLBACK])

    async def _agenerate(
        self, prepared: _Prepared, *, question: str, session_id: str | None, model: str
    ) -> _Generated:
        """_generate 의 비동기 쌍둥이. SSE 경로가 이벤트 루프를 막지 않게 한다."""
        messages = self._render_messages(prepared, question=question, session_id=session_id)
        if not self.settings.answer_schema_enabled:
            return self._plain_result(await self.llm(model).ainvoke(messages), issues=[])
        out = await self.structured_llm(model).ainvoke(messages)
        gen = self._structured_result(out, prepared)
        if gen is not None:
            return gen
        return self._plain_result(
            await self.llm(model).ainvoke(messages), issues=[SCHEMA_FALLBACK]
        )

    # --- 턴 준비(동기/스트림 공용) -----------------------------------------
    def prepare(self, question: str, *, model: str) -> _Prepared:
        # 야구 외의 질문은 검색도 답변도 하지 않는다. 대부분 사전에서 끝나 비용이 들지 않는다.
        verdict = scope.check(question, llm=self._scope_llm, settings=self.settings)
        if verdict.blocked:
            log.info("scope 차단: by=%s hits=%s", verdict.by, verdict.hits)
            blocked = Route("off_topic", "scope", False, False)
            gate = TurnResult(
                answer=OFF_TOPIC_REFUSAL, status="out_of_scope", freshness="static",
                route=blocked.to_dict(), llm_called=False, model=model,
            )
            return _Prepared(blocked, [], "static", [], "", gate)

        # 사전이 못 거른 오프토픽은 라우터가 한 번 더 본다.
        r = route_question(question, llm=self._router_llm, settings=self.settings)

        if r.kind == "off_topic":
            gate = TurnResult(
                answer=OFF_TOPIC_REFUSAL, status="out_of_scope", freshness="static",
                route=r.to_dict(), llm_called=False, model=model,
            )
            return _Prepared(r, [], "static", [], "", gate)

        latest: list[LatestEntry] = []
        kbo_entries: list[Any] = []
        freshness = "static"

        # 0단계 — 가진 데이터. KBO 실데이터와 리그 규정 스냅샷은 공짜이고 확실하다.
        if r.kind in {"latest", "mixed"}:
            if r.topics and self.settings.kbo_data_enabled:
                from baseball import kbo as kbo_data

                kbo_entries = kbo_data.entries_for(question, r, settings=self.settings)
            if kbo_entries:
                freshness = ("live" if any(e.freshness in {"live", "cached"} for e in kbo_entries)
                             else "snapshot")
            else:
                snapshot = latest_info.match_snapshot(question)
                if snapshot:
                    latest, freshness = list(snapshot), "snapshot"

        # 1단계 — 규칙집. 이게 1차 근거다.
        retrieval = self.retriever.retrieve(question, k=6)
        docs = list(retrieval.docs)
        if retrieval.abstain:
            # 근거가 약하다는 뜻이지 거부하라는 뜻이 아니다. 아래 단계가 받는다.
            docs = []

        # 2단계 — 웹 검색. 규칙집이 빈손일 때만 나간다.
        if not docs and not kbo_entries and not latest and self.settings.web_search_enabled:
            web = latest_info.web_search(question, self.settings, client=self._web_client)
            if web:
                latest, freshness = web, "web"

        # 순위·일정 질문에 규칙집 조항이 섞이면 모델이 답을 거부한다. 웹 단계 뒤에 판단해야
        # 2단계가 채운 자료까지 포함해 같은 규칙이 걸린다.
        if (latest or kbo_entries) and r.kind == "latest" and not r.rule_hit:
            docs = []

        # 3단계 — 어디에서도 근거를 못 찾았다.
        knowledge_only = False
        if not docs and not latest and not kbo_entries:
            needs_live = bool(
                latest_info.LIVE_WEB_RE.search(unicodedata.normalize("NFKC", question))
            )
            # 실시간 값(순위·기록·일정)은 모델 지식으로 답하면 안 된다. 틀린 숫자가 나온다.
            if self.settings.model_knowledge_enabled and not needs_live:
                knowledge_only, freshness = True, "model"
            elif r.kind in {"latest", "mixed"} and needs_live:
                # 실시간 정보가 필요한데 웹 경로가 닫혀 있거나 빈손이다.
                gate = TurnResult(
                    answer=NOT_IN_CONTEXT_REFUSAL, status="phase2_pending", freshness="static",
                    needs_web=True, route=r.to_dict(), llm_called=False, model=model,
                    retrieval=retrieval,
                )
            else:
                gate = TurnResult(
                    answer=NOT_IN_CONTEXT_REFUSAL, status="not_in_rulebook", freshness=freshness,
                    route=r.to_dict(), llm_called=False, model=model, retrieval=retrieval,
                )
            if not knowledge_only:
                return _Prepared(r, [], freshness, [], "", gate, kbo_entries)

        context = format_context(
            docs, latest, kbo_entries,
            max_tokens=self.settings.context_max_tokens,
            latest_max_tokens=self.settings.latest_context_max_tokens,
            kbo_max_tokens=self.settings.kbo_context_max_tokens,
            knowledge_only=knowledge_only,
        )
        if not context:
            gate = TurnResult(
                answer=NOT_IN_CONTEXT_REFUSAL, status="not_in_rulebook", freshness=freshness,
                route=r.to_dict(), llm_called=False, model=model, retrieval=retrieval,
            )
            return _Prepared(r, latest, freshness, [], "", gate, kbo_entries)

        prepared = _Prepared(r, latest, freshness, docs, context, None, kbo_entries,
                             knowledge_only=knowledge_only)
        prepared.retrieval = retrieval                      # type: ignore[attr-defined]
        return prepared

    def _finish(
        self, prepared: _Prepared, *, question: str, answer_text: str, usage: dict[str, Any],
        model: str, started: float, session_id: str | None,
        answer_kind: str | None = None, issues: list[str] | None = None,
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
            sources=_rule_sources(prepared.docs) + _latest_sources(prepared.latest)
                    + _kbo_sources(prepared.kbo) + (_model_source() if prepared.knowledge_only else []),
            usage=usage,
            latency_ms=int((time.time() - started) * 1000),
            llm_called=True,
            context=prepared.context,
            model=model,
            session_id=session_id,
            retrieval=getattr(prepared, "retrieval", None),
            answer_kind=answer_kind,
            format_issues=list(issues or []),
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

        gen = self._generate(prepared, question=question, session_id=session_id, model=model)
        return self._finish(
            prepared, question=question, answer_text=gen.text, usage=gen.usage,
            model=model, started=started, session_id=session_id,
            answer_kind=gen.kind, issues=gen.issues,
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
        sources = (_rule_sources(prepared.docs) + _latest_sources(prepared.latest)
                   + _kbo_sources(prepared.kbo)
                   + (_model_source() if prepared.knowledge_only else []))
        yield {"event": "sources", "data": {"sources": sources}}

        yield {"event": "status", "data": {"status": "generating", "llm_called": True}}
        if self.settings.answer_schema_enabled:
            gen = await self._agenerate(
                prepared, question=question, session_id=session_id, model=model
            )
            for piece in _token_pieces(gen):
                yield {"event": "token", "data": {"text": piece}}
        else:
            # 롤백 레버는 형식뿐 아니라 토큰 단위 스트리밍까지 되돌려야 의미가 있다.
            messages = self._render_messages(prepared, question=question, session_id=session_id)
            parts, usage = [], {"input_tokens": 0, "output_tokens": 0, "cache_read": 0}
            async for piece in self._astream_plain(messages, model, parts, usage):
                yield {"event": "token", "data": {"text": piece}}
            gen = _Generated(text="".join(parts), usage=usage)
        result = self._finish(
            prepared, question=question, answer_text=gen.text, usage=gen.usage,
            model=model, started=started, session_id=session_id,
            answer_kind=gen.kind, issues=gen.issues,
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
        sources = (_rule_sources(prepared.docs) + _latest_sources(prepared.latest)
                   + _kbo_sources(prepared.kbo)
                   + (_model_source() if prepared.knowledge_only else []))
        yield {"event": "sources", "data": {"sources": sources}}

        yield {"event": "status", "data": {"status": "generating", "llm_called": True}}
        if self.settings.answer_schema_enabled:
            gen = self._generate(prepared, question=question, session_id=session_id, model=model)
            for piece in _token_pieces(gen):
                yield {"event": "token", "data": {"text": piece}}
        else:
            messages = self._render_messages(prepared, question=question, session_id=session_id)
            parts, usage = [], {"input_tokens": 0, "output_tokens": 0, "cache_read": 0}
            for piece in self._stream_plain(messages, model, parts, usage):
                yield {"event": "token", "data": {"text": piece}}
            gen = _Generated(text="".join(parts), usage=usage)
        result = self._finish(
            prepared, question=question, answer_text=gen.text, usage=gen.usage,
            model=model, started=started, session_id=session_id,
            answer_kind=gen.kind, issues=gen.issues,
        )
        yield {"event": "final", "data": _final_payload(result)}
        yield {"event": "done", "data": {}}


def _kbo_sources(entries: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        if e.freshness == "unavailable":
            continue
        if e.freshness == "stale":
            status = f"오래된 저장본({e.as_of} 기준)"
        else:
            status = {"live": "실시간", "cached": "저장본(최신)"}.get(e.freshness, "기본정보")
        out.append({"kind": "kbo", "label": e.label, "url": e.source_url,
                    "as_of": e.as_of, "confidence": status})
    return out


def _token_pieces(gen: _Generated) -> list[str]:
    """블록을 이어 붙이면 정확히 gen.text 가 되도록 구분자를 붙여 쪼갠다.

    UI 는 token 조각을 그대로 이어 붙이므로(st.write_stream) 블록 사이 빈 줄을
    여기서 넣지 않으면 마크다운 리스트가 앞 문단에 붙어 버린다.
    """
    if not gen.blocks:
        return [gen.text] if gen.text else []
    last = len(gen.blocks) - 1
    return [b if i == last else b + "\n\n" for i, b in enumerate(gen.blocks)]


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
        "answer_kind": result.answer_kind, "format_ok": result.format_ok,
        "format_issues": result.format_issues,
    }
