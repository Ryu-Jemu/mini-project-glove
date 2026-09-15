"""백엔드 접근 모드 전환기.

API_BASE_URL 이 설정되어 있으면 FastAPI 백엔드를 SSE 로 호출하고(로컬 개발),
없으면 같은 프로세스에서 RagService 를 직접 호출한다(무료 배포).
APP_MODE 로 강제할 수 있다: "api" 또는 "inprocess".
"""
from __future__ import annotations

import os

import api_client

_forced = (os.getenv("APP_MODE") or "").strip().lower()
if _forced in ("api", "inprocess"):
    MODE = _forced
else:
    MODE = "api" if api_client.configured_base() else "inprocess"

if MODE == "api":
    from api_client import document_url, readyz, reset_session, stream_answer
else:
    from service import document_url, readyz, reset_session, stream_answer

__all__ = ["MODE", "document_url", "readyz", "reset_session", "stream_answer"]
