"""실행 전 사전 점검. 비밀 값은 절대 출력하지 않는다(존재·형식·길이만)."""
from __future__ import annotations

import re
import sys
from typing import Callable

from baseball.config import Settings, get_settings

YOUTUBE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{35}")


def _openai(settings: Settings) -> str:
    if settings.openai_api_key is None:
        return "OPENAI missing (.env 확인) — FAIL"
    return f"OPENAI ok (model={settings.openai_chat_model}, dims={settings.embedding_dimensions})"


def _postgres(settings: Settings) -> str:
    from baseball import db

    with db.connect(settings, register=False) as conn:
        row = conn.execute("select extversion as v from pg_extension where extname='vector'").fetchone()
        meta = conn.execute("select version, embedding_dim from schema_meta").fetchone()
    if not row:
        return "PG vector 확장 없음 — python -m baseball.db ensure 필요"
    if not meta:
        return f"PG ok vector {row['v']} (schema 미생성)"
    if meta["embedding_dim"] != settings.embedding_dimensions:
        return (f"PG schema mismatch: {meta['embedding_dim']} != {settings.embedding_dimensions} "
                f"— run: python -m baseball.db reset --yes")
    return f"PG ok vector {row['v']} (schema {meta['version']}, dim {meta['embedding_dim']})"


def _minio(settings: Settings) -> str:
    from baseball.storage import MinioStorage

    store = MinioStorage(settings)
    created = store.ensure_bucket()
    keys = store.list_keys()
    return f"MinIO ok ({'bucket created' if created else 'bucket exists'}, objects={len(keys)})"


def _kiwi(_: Settings) -> str:
    from baseball.lexical import tokenize

    return f"Kiwi ok (예: {tokenize('인필드 플라이가 뭐야')[:3]})"


def _prompts(_: Settings) -> str:
    from baseball.prompts import HUMAN_SHA256, SYSTEM_SHA256

    return f"prompts ok ({SYSTEM_SHA256[:8]} / {HUMAN_SHA256[:8]})"


def _index(settings: Settings) -> str:
    from baseball import db

    with db.connect(settings) as conn:
        doc = db.active_document(conn)
        if not doc:
            return "index 없음 — python -m baseball.ingest 필요"
        n = db.count_chunks(conn, doc["id"])
    return f"index ok (chunks={n}, {doc['index_version']})"


def _tavily(settings: Settings) -> str:
    if settings.tavily_api_key is None:
        return "Tavily absent (웹 최신정보 비활성)"
    return f"Tavily ok (web_search={'on' if settings.web_search_enabled else 'off'})"


def _langsmith(settings: Settings) -> str:
    if settings.langchain_api_key is None:
        return "LangSmith absent (트레이싱 없음)"
    return f"LangSmith ok (project={settings.langchain_project}, tracing={settings.langchain_tracing_v2})"


def _youtube(settings: Settings) -> str:
    if settings.youtube_kbo_api_key is None:
        return "YouTube key absent (Phase 2 optional)"
    raw = settings.youtube_kbo_api_key.get_secret_value()
    if YOUTUBE_KEY_RE.fullmatch(raw):
        return f"YouTube key present (format ok, {len(raw)} chars)"
    return f"YouTube key present (format unexpected, {len(raw)} chars) — soft warning"



def _kbo(_: Settings) -> str:
    """네트워크를 타지 않는다. make setup 이 상류 장애로 실패하면 안 된다."""
    try:
        from baseball.kbo import load_teams, teams_by_code

        meta, codes = load_teams(), teams_by_code()
        if len(codes) != 10:
            return f"KBO teams 구단 수 이상 ({len(codes)}팀)"
        return f"KBO teams ok (10팀, as_of={meta.get('as_of', '?')})"
    except Exception as exc:                       # noqa: BLE001
        return f"KBO teams 파일 없음 — 구단 정보 비활성 ({type(exc).__name__})"



def _access(settings: Settings) -> str:
    """값은 출력하지 않는다. 설정 여부와 형식만 본다."""
    pin = settings.app_access_pin
    if pin is None:
        return "접근 코드 없음 — 누구나 사용 가능"
    raw = pin.get_secret_value()
    if len(raw) == 4 and raw.isdigit():
        return "접근 코드 설정됨 (숫자 4자리)"
    return f"접근 코드 형식 이상 ({len(raw)}자) — 숫자 4자리여야 한다"


CHECKS: list[tuple[str, Callable[[Settings], str], bool]] = [
    ("openai", _openai, True),
    ("postgres", _postgres, True),
    ("minio", _minio, True),
    ("kiwi", _kiwi, True),
    ("index", _index, False),
    ("tavily", _tavily, False),
    ("langsmith", _langsmith, False),
    ("youtube", _youtube, False),
    ("kbo", _kbo, False),
    ("access", _access, False),
    ("prompts", _prompts, True),
]


def main() -> int:
    settings = get_settings()
    failed = False
    for name, fn, required in CHECKS:
        try:
            line = fn(settings)
        except Exception as exc:
            line = f"{name} ERROR: {type(exc).__name__}: {exc}"
            if required:
                failed = True
        else:
            if required and ("FAIL" in line or "missing" in line or "mismatch" in line):
                failed = True
        print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
