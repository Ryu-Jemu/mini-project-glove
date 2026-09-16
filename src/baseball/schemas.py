"""API 요청/응답 스키마."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["answered", "not_in_rulebook", "out_of_scope", "phase2_pending"]
AnswerKind = Literal["term_rule", "situation", "entity", "latest"]
Freshness = Literal["static", "snapshot", "web", "live"]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    model: str | None = None


class Citation(BaseModel):
    label: str
    rule_id: str
    doc_type: str
    page: int
    snippet: str


class Source(BaseModel):
    kind: Literal["static", "snapshot", "web", "kbo"]
    label: str
    url: str | None = None
    as_of: str | None = None
    confidence: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_read: int = 0


class RouteInfo(BaseModel):
    kind: str
    by: str


class ChatResponse(BaseModel):
    answer: str
    status: Status
    partial_refusal: bool = False
    freshness: Freshness = "static"
    needs_web: bool = False
    route: RouteInfo
    citations: list[Citation] = []
    dropped_citations: list[str] = []
    sources: list[Source] = []
    usage: Usage = Usage()
    latency_ms: int = 0
    prompt_sha: str
    model: str
    session_id: str | None = None
    llm_called: bool = True
    # 답변 스키마. 전부 기본값이라 기존 응답 계약을 깨지 않는다.
    answer_kind: AnswerKind | None = None
    format_ok: bool = True
    format_issues: list[str] = []
