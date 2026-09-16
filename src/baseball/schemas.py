"""API 요청/응답 스키마."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["answered", "not_in_rulebook", "out_of_scope", "phase2_pending"]
AnswerKind = Literal["term_rule", "situation", "entity", "latest"]
Freshness = Literal["static", "snapshot", "web", "live", "model"]


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
    kind: Literal["static", "snapshot", "web", "kbo", "model", "place", "video"]
    label: str
    url: str | None = None
    as_of: str | None = None
    confidence: str | None = None


class Media(BaseModel):
    """채팅 안에서 재생·표시할 것.

    본문(answer)과 달리 이 값들은 LLM 을 통과하지 않는다. 도구가 만든 것이 그대로
    온다. 영상 주소나 지도 질의를 모델이 지어낼 경로 자체를 두지 않기 위해서다.

    지도는 url 이 아니라 query 만 싣는다. Embed API 키가 클라이언트에 노출되므로
    URL 조립은 UI 에서 하고, 응답 본문에는 키를 넣지 않는다.
    """

    kind: Literal["video", "map"]
    id: str
    title: str | None = None
    url: str | None = None
    query: str | None = None
    published_at: str | None = None
    duration: str | None = None
    embeddable: bool = True
    thumbnail_url: str | None = None


class Place(BaseModel):
    """맛집 카드.

    name 은 검색 제공자가 준 페이지 제목 그대로다. 주소·전화·영업시간 필드를 두지
    않는 것이 의도다 — 블로그 본문에서 뽑으면 틀릴 수밖에 없고, 없는 필드는 틀릴 수 없다.
    url 은 필수다. 출처로 확인할 수 없는 항목은 애초에 만들지 않는다.
    """

    id: str
    name: str
    url: str
    snippet: str | None = None
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
    # 미디어 채널. 기본값이 빈 배열이라 기존 응답 계약을 깨지 않는다.
    # ChatResponse 는 extra="ignore" 라, 여기 선언하지 않으면 POST /chat 에서 조용히 잘린다.
    media: list[Media] = []
    places: list[Place] = []
