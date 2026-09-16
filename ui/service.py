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


_HINTS = (
    ("could not translate host name", "PG_DSN 의 호스트 이름이 잘못되었습니다."),
    ("connection refused", "데이터베이스에 연결하지 못했습니다. PG_DSN 을 확인하세요."),
    ("password authentication", "데이터베이스 비밀번호가 틀렸습니다. PG_DSN 을 확인하세요."),
    ("does not exist", "데이터베이스 또는 테이블이 없습니다. 색인을 먼저 넣어야 합니다."),
    ("timeout", "데이터베이스 응답이 없습니다. 주소와 방화벽을 확인하세요."),
    ("no such file", "필요한 데이터 파일이 저장소에 없습니다."),
    ("sslmode", "PG_DSN 의 SSL 설정을 확인하세요."),
)


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def _dsn_host() -> str:
    """설정된 데이터베이스 호스트만 꺼낸다. 사용자·비밀번호는 건드리지 않는다."""
    dsn = os.environ.get("PG_DSN", "")
    if not dsn:
        try:
            from baseball.config import get_settings

            dsn = get_settings().pg_dsn
        except Exception:                      # noqa: BLE001
            return ""
    try:
        from urllib.parse import urlsplit

        return urlsplit(dsn).hostname or ""
    except Exception:                          # noqa: BLE001
        return ""


def dsn_is_loopback() -> bool:
    return _dsn_host() in LOOPBACK_HOSTS


def diagnose(message: str) -> str:
    """원인을 한 줄로 바꾼다. 비밀값은 절대 담지 않는다."""
    # 호스트가 루프백이면 오류 문구를 따질 것 없이 원인이 확정된다.
    # .env.example 의 PG_DSN 을 그대로 배포 설정에 붙여 넣으면 여기에 걸린다.
    if dsn_is_loopback():
        return ("PG_DSN 이 로컬 주소를 가리킵니다. .env.example 의 기본값은 내 컴퓨터의 도커용입니다. "
                "배포에서는 외부 데이터베이스 주소를 넣어야 합니다.")
    low = message.lower()
    for needle, hint in _HINTS:
        if needle in low:
            return hint
    return "설정을 확인하세요."


def get_service() -> Any | None:
    """실패해도 예외를 올리지 않는다. 대신 원인을 남겨 사이드바가 보여 준다."""
    try:
        return _build_service()
    except Exception as exc:                   # noqa: BLE001
        raw = f"{type(exc).__name__}: {exc}"
        st.session_state["service_error"] = raw
        st.session_state["service_hint"] = diagnose(raw)
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
