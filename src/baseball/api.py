"""FastAPI 백엔드. 모듈 import 시 Settings 를 평가하지 않는다(키 없이 import 가능)."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent

from baseball.config import Settings, get_settings
from baseball.prompts import PROMPT_SHA
from baseball.schemas import ChatRequest, ChatResponse

log = logging.getLogger(__name__)


def get_service(request: Request) -> Any:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="서비스가 준비되지 않았습니다")
    return service


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or get_settings()
        app.state.settings = cfg
        app.state.service = None
        app.state.ready_error = None
        try:
            from baseball.chain import RagService

            app.state.service = RagService.create(cfg)          # Kiwi/BM25 인덱스 1회 빌드
            log.info("retriever ready: %s", app.state.service.retriever.document_id)
        except Exception as exc:                                 # 인덱스 없음 등 → /readyz 503
            app.state.ready_error = str(exc)
            log.warning("서비스 초기화 실패: %s", exc)
        yield

    app = FastAPI(title="KBO 야구 규칙 도우미 API", version="0.1.0", lifespan=lifespan)
    # 미들웨어는 시작 전에만 추가할 수 있다(Starlette 제약).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=(settings or get_settings()).cors_origin_list,
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        cfg: Settings = request.app.state.settings
        return {"status": "ok", "chat_model": cfg.openai_chat_model, "prompt_sha": PROMPT_SHA}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, Any]:
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(status_code=503, detail=request.app.state.ready_error or "not ready")
        cfg: Settings = request.app.state.settings
        retriever = service.retriever
        return {
            "status": "ready",
            "document": {
                "id": retriever.document_id,
                "chunks": len(retriever.lexical.chunks),
            },
            "bm25_ready": retriever.lexical.bm25 is not None,
            "web_search": "on" if cfg.web_search_enabled else "off",
            "prompt_sha": PROMPT_SHA,
            "chat_model": cfg.openai_chat_model,
        }

    @app.get("/documents/active/url")
    async def document_url(request: Request) -> dict[str, str]:
        from baseball import db
        from baseball.storage import MinioStorage

        cfg: Settings = request.app.state.settings
        with db.connect(cfg) as conn:
            doc = db.active_document(conn)
        if not doc:
            raise HTTPException(status_code=404, detail="active 문서가 없습니다")
        return {"url": MinioStorage(cfg).presign(doc["object_key"], 15)}

    @app.get("/search")
    async def search(q: str = Query(min_length=1), k: int = 6, service: Any = Depends(get_service)) -> dict[str, Any]:
        res = service.retriever.retrieve(q, k=k)
        return {
            "abstain": res.abstain, "dense_top": res.dense_top, "bm25_ratio": res.bm25_ratio,
            "exact_hits": res.exact_hits, "channels": res.channels,
            "docs": [
                {"rule_id": d["rule_id"], "breadcrumb": d["breadcrumb"], "page": d["page_start"],
                 "tokens": d["tokens"]}
                for d in res.docs
            ],
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, service: Any = Depends(get_service)) -> ChatResponse:
        from baseball.chain import QUOTA_MESSAGE, is_quota_error

        try:
            result = service.answer(req.question, session_id=req.session_id, model=req.model)
        except Exception as exc:
            log.exception("chat 실패")
            if is_quota_error(exc):
                raise HTTPException(status_code=402, detail=QUOTA_MESSAGE) from exc
            raise HTTPException(status_code=503, detail="모델 호출 실패 — 잠시 후 다시 시도해 주세요") from exc
        from baseball.chain import _final_payload

        return ChatResponse(**_final_payload(result))

    @app.post("/chat/stream", response_class=EventSourceResponse)
    async def chat_stream(req: ChatRequest, service: Any = Depends(get_service)):
        try:
            async for ev in service.astream(req.question, session_id=req.session_id, model=req.model):
                yield ServerSentEvent(event=ev["event"], data=ev["data"])
        except Exception as exc:
            log.exception("chat/stream 실패")
            from baseball.chain import QUOTA_MESSAGE, is_quota_error

            detail = QUOTA_MESSAGE if is_quota_error(exc) else "모델 호출 실패 — 잠시 후 다시 시도해 주세요"
            yield ServerSentEvent(event="error", data={"detail": detail, "kind": "quota" if is_quota_error(exc) else "error"})

    @app.post("/sessions/{session_id}/reset")
    async def reset_session(session_id: str, service: Any = Depends(get_service)) -> dict[str, str]:
        service.reset(session_id)
        return {"status": "reset", "session_id": session_id}

    return app


app = create_app()
