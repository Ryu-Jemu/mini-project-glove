"""FastAPI 백엔드 호출(요청당 짧은 SSE 스트림 1개). 장기 구독은 만들지 않는다."""
from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

def _default_base() -> str:
    """우선순위: 환경변수 API_BASE_URL → Streamlit secrets → 로컬 백엔드."""
    import os

    env = os.getenv("API_BASE_URL")
    if env:
        return env.rstrip("/")
    try:
        import streamlit as st

        value = st.secrets.get("API_BASE_URL")          # Streamlit Cloud 배포용
        if value:
            return str(value).rstrip("/")
    except Exception:
        pass
    return "http://127.0.0.1:8000"


DEFAULT_BASE = _default_base()
TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)


def readyz(base_url: str = DEFAULT_BASE) -> dict[str, Any] | None:
    try:
        r = httpx.get(f"{base_url}/readyz", timeout=5.0)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def document_url(base_url: str = DEFAULT_BASE) -> str | None:
    try:
        r = httpx.get(f"{base_url}/documents/active/url", timeout=10.0)
        return r.json().get("url") if r.status_code == 200 else None
    except Exception:
        return None


def reset_session(session_id: str, base_url: str = DEFAULT_BASE) -> None:
    try:
        httpx.post(f"{base_url}/sessions/{session_id}/reset", timeout=5.0)
    except Exception:
        pass


def stream_answer(
    question: str, session_id: str | None = None, base_url: str = DEFAULT_BASE
) -> Iterator[tuple[str, dict[str, Any]]]:
    """(event, data) 튜플을 순서대로 낸다: route → status → sources → token* → final → done."""
    payload = {"question": question, "session_id": session_id}
    with httpx.Client(timeout=TIMEOUT) as client:
        with client.stream("POST", f"{base_url}/chat/stream", json=payload) as response:
            if response.status_code != 200:
                response.read()
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    detail = ""
                kind = "quota" if response.status_code == 402 else "error"
                yield "error", {
                    "detail": detail or f"HTTP {response.status_code}", "kind": kind,
                }
                return
            event = "message"
            for line in response.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event: "):
                    event = line[7:].strip()
                elif line.startswith("data: "):
                    raw = line[6:]
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = {"text": raw}
                    yield event, data if isinstance(data, dict) else {"value": data}
