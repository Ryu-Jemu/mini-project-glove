"""단일 프로세스 어댑터.

Streamlit Community Cloud 처럼 포트를 하나만 열 수 있는 곳에서는 FastAPI 백엔드를
따로 띄울 수 없다. 이 모듈은 api_client 와 같은 (event, data) 튜플 계약을 유지한 채
RagService 를 같은 프로세스에서 직접 호출한다. ui/app.py 는 바뀌지 않는다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:                       # 패키지 설치 없이 소스에서 바로 쓴다.
    sys.path.insert(0, _SRC)

# Settings 가 읽는 이름들. secrets 를 환경변수로 먼저 옮겨야 get_settings 의
# lru_cache 가 키 없는 설정을 캐싱하는 사고를 막을 수 있다.
_SECRET_KEYS = (
    "OPENAI_API_KEY", "PG_DSN", "ENABLE_WEB_SEARCH", "TAVILY_API_KEY",
    "OPENAI_CHAT_MODEL", "OPENAI_CHAT_MODEL_FALLBACK", "OPENAI_EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS", "CONTEXT_MAX_TOKENS", "LATEST_CONTEXT_MAX_TOKENS",
    "CHAT_HISTORY_MAX_MESSAGES", "CONTEXT_MIN_SCORE_RATIO",
    "ABSTAIN_DENSE_THRESHOLD", "ABSTAIN_BM25_RATIO",
    "ENABLE_KBO_DATA", "KBO_SEASON_YEAR", "KBO_HTTP_TIMEOUT_SECONDS",
    "KBO_STANDINGS_TTL_SECONDS", "KBO_SCHEDULE_TTL_SECONDS",
    "KBO_CONTEXT_MAX_TOKENS", "KBO_SCHEDULE_MAX_GAMES",
    "APP_ACCESS_PIN",
)

RULEBOOK_URL = (
    "https://github.com/Ryu-Jemu/mini-project-glove/blob/main/"
    "data/2026_%EC%95%BC%EA%B5%AC%EA%B7%9C%EC%B9%99.pdf"
)


def _export_secrets() -> None:
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in _SECRET_KEYS:
        if os.environ.get(key):
            continue
        try:
            value = secrets.get(key)
        except Exception:
            value = None
        if value not in (None, ""):
            os.environ[key] = str(value)


@st.cache_resource(show_spinner="규칙집 색인을 준비하는 중…")
def _build_service() -> Any:
    _export_secrets()
    from baseball.chain import RagService      # 지연 import: API 모드는 비용을 치르지 않는다.

    return RagService.create()


def get_service() -> Any | None:
    """실패해도 예외를 올리지 않는다. 사이드바가 '준비 안 됨'을 보여 주면 된다."""
    try:
        return _build_service()
    except Exception as exc:                   # noqa: BLE001
        st.session_state["service_error"] = f"{type(exc).__name__}: {exc}"
        return None


def readyz() -> dict[str, Any] | None:
    service = get_service()
    if service is None:
        return None
    try:
        from baseball.prompts import PROMPT_SHA

        retriever = service.retriever
        return {
            "status": "ready",
            "document": {"id": retriever.document_id, "chunks": len(retriever.by_id)},
            "bm25_ready": retriever.lexical.bm25 is not None,
            "web_search": "on" if service.settings.web_search_enabled else "off",
            "prompt_sha": PROMPT_SHA,
            "chat_model": service.settings.openai_chat_model,
        }
    except Exception:
        return None


def document_url() -> str | None:
    """MinIO 없이 규칙집 원문을 연다. PDF 는 저장소에 함께 들어 있다."""
    return os.getenv("RULEBOOK_PDF_URL") or RULEBOOK_URL


def reset_session(session_id: str) -> None:
    service = get_service()
    if service is not None:
        try:
            service.reset(session_id)
        except Exception:
            pass


def stream_answer(
    question: str, session_id: str | None = None
) -> Iterator[tuple[str, dict[str, Any]]]:
    """api_client.stream_answer 와 같은 튜플을 낸다."""
    service = get_service()
    if service is None:
        detail = st.session_state.get("service_error", "서비스를 준비하지 못했습니다.")
        yield "error", {"detail": detail, "kind": "error"}
        return
    try:
        from baseball.chain import QUOTA_MESSAGE, is_quota_error
    except Exception:                          # noqa: BLE001
        QUOTA_MESSAGE, is_quota_error = "사용 한도를 초과했습니다.", lambda _e: False
    try:
        for item in service.stream(question, session_id):
            yield item["event"], item.get("data", {})
    except Exception as exc:                   # noqa: BLE001
        if is_quota_error(exc):
            yield "error", {"detail": QUOTA_MESSAGE, "kind": "quota"}
        else:
            yield "error", {"detail": "모델 호출 실패 — 잠시 후 다시 시도해 주세요", "kind": "error"}
