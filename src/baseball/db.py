"""Postgres(pgvector) 접근 계층. 스키마 생성은 비파괴, DROP은 reset --yes 경로만."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from baseball.config import Settings, get_settings

SCHEMA_VERSION = "phase1-v1"

APP_TABLES = (
    "rule_chunks",
    "rule_documents",
    "source_snapshots",
    "assistant_cache",
    "chat_sessions",
    "schema_meta",
)


class SchemaMismatch(RuntimeError):
    """기존 스키마가 현재 설정과 호환되지 않음. 자동 DROP 금지."""


@contextmanager
def connect(settings: Settings | None = None, *, register: bool = True) -> Iterator[psycopg.Connection]:
    settings = settings or get_settings()
    with psycopg.connect(settings.pg_dsn, row_factory=dict_row) as conn:
        if register:
            try:
                register_vector(conn)
            except Exception:  # vector 확장 미설치 상태(ensure_schema 이전)
                pass
        yield conn


def _ddl(dims: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_meta (
    id              boolean PRIMARY KEY DEFAULT true CHECK (id),
    version         text    NOT NULL,
    embedding_dim   integer NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rule_documents (
    id              text PRIMARY KEY,
    title           text NOT NULL,
    version         text NOT NULL,
    sha256          text NOT NULL,
    object_key      text NOT NULL,
    embedding_model text NOT NULL,
    embedding_dim   integer NOT NULL,
    chunker_version text NOT NULL,
    index_version   text NOT NULL,
    page_count      integer NOT NULL CHECK (page_count > 0),
    chunk_count     integer NOT NULL CHECK (chunk_count > 0),
    active          boolean NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS rule_documents_one_active
    ON rule_documents (active) WHERE active;
CREATE UNIQUE INDEX IF NOT EXISTS rule_documents_index_version
    ON rule_documents (index_version);

CREATE TABLE IF NOT EXISTS rule_chunks (
    id            text PRIMARY KEY,
    document_id   text NOT NULL REFERENCES rule_documents(id) ON DELETE CASCADE,
    chunk_index   integer NOT NULL,
    doc_type      text NOT NULL,
    rule_chapter  text,
    rule_no       text,
    rule_title    text,
    sub_item      text,
    rule_id       text NOT NULL,
    parent_id     text,
    annotation    text,
    term_no       integer,
    term_en       text,
    term_ko       text,
    page_start    integer NOT NULL CHECK (page_start > 0),
    page_end      integer NOT NULL CHECK (page_end > 0),
    breadcrumb    text NOT NULL,
    content       text NOT NULL,
    tokens        integer NOT NULL,
    embedding     vector({dims}) NOT NULL
);
CREATE INDEX IF NOT EXISTS rule_chunks_hnsw
    ON rule_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS rule_chunks_rule_no_idx  ON rule_chunks (document_id, rule_no);
CREATE INDEX IF NOT EXISTS rule_chunks_parent_idx   ON rule_chunks (document_id, parent_id);
CREATE INDEX IF NOT EXISTS rule_chunks_doc_idx      ON rule_chunks (document_id, chunk_index);

CREATE TABLE IF NOT EXISTS source_snapshots (
    id         bigserial PRIMARY KEY,
    kind       text NOT NULL,
    payload    jsonb NOT NULL,
    as_of      date,
    source_url text,
    fetched_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS source_snapshots_kind_idx ON source_snapshots (kind, fetched_at DESC);

CREATE TABLE IF NOT EXISTS assistant_cache (
    key          text PRIMARY KEY,
    payload      jsonb NOT NULL,
    fetched_at   timestamptz NOT NULL DEFAULT now(),
    ttl_seconds  integer NOT NULL DEFAULT 21600
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id         text PRIMARY KEY,
    turns      jsonb NOT NULL DEFAULT '[]'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);
"""


def _embedding_typmod(conn: psycopg.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS t
        FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'rule_chunks' AND a.attname = 'embedding' AND a.attnum > 0
        """
    ).fetchone()
    return row["t"] if row else None


def ensure_schema(settings: Settings | None = None) -> str:
    """CREATE ... IF NOT EXISTS 만 수행. 불일치 시 SchemaMismatch(자동 DROP 없음)."""
    settings = settings or get_settings()
    dims = settings.embedding_dimensions
    with connect(settings, register=False) as conn:
        conn.execute(_ddl(dims))
        conn.commit()
        typmod = _embedding_typmod(conn)
        expected = f"vector({dims})"
        if typmod and typmod != expected:
            raise SchemaMismatch(
                f"rule_chunks.embedding={typmod} != {expected} — run: python -m baseball.db reset --yes"
            )
        conn.execute(
            """
            INSERT INTO schema_meta (id, version, embedding_dim) VALUES (true, %s, %s)
            ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version,
                embedding_dim = EXCLUDED.embedding_dim, updated_at = now()
            """,
            (SCHEMA_VERSION, dims),
        )
        conn.commit()
    return SCHEMA_VERSION


SNAPSHOT_KEEP = 3


def latest_snapshot(conn: psycopg.Connection, kind: str) -> dict[str, Any] | None:
    """kind 의 가장 최근 저장본 한 건. 없으면 None."""
    return conn.execute(
        "SELECT payload, as_of, source_url, fetched_at FROM source_snapshots "
        "WHERE kind = %s ORDER BY fetched_at DESC LIMIT 1",
        (kind,),
    ).fetchone()


def put_snapshot(conn: psycopg.Connection, kind: str, payload: Any, *,
                 as_of: Any = None, source_url: str | None = None,
                 keep: int = SNAPSHOT_KEEP) -> None:
    """저장본을 넣고 같은 kind 의 오래된 행을 keep 개만 남기고 지운다."""
    conn.execute(
        "INSERT INTO source_snapshots (kind, payload, as_of, source_url) VALUES (%s, %s, %s, %s)",
        (kind, json.dumps(payload, ensure_ascii=False, default=str), as_of, source_url),
    )
    conn.execute(
        "DELETE FROM source_snapshots WHERE kind = %s AND id NOT IN "
        "(SELECT id FROM source_snapshots WHERE kind = %s ORDER BY fetched_at DESC LIMIT %s)",
        (kind, kind, keep),
    )


def reset(settings: Settings | None = None) -> dict[str, int]:
    """모든 앱 테이블 DROP 후 재생성. 호출자가 --yes 를 확인해야 한다."""
    settings = settings or get_settings()
    counts: dict[str, int] = {}
    with connect(settings, register=False) as conn:
        for t in APP_TABLES:
            try:
                row = conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()
                counts[t] = int(row["n"])
            except Exception:
                conn.rollback()
                counts[t] = 0
        conn.rollback()
        conn.execute("DROP TABLE IF EXISTS " + ", ".join(APP_TABLES) + " CASCADE")
        conn.commit()
    ensure_schema(settings)
    return counts


# --------------------------------------------------------------------------- 문서/청크

def active_document(conn: psycopg.Connection) -> dict[str, Any] | None:
    return conn.execute("SELECT * FROM rule_documents WHERE active").fetchone()


def document_by_index_version(conn: psycopg.Connection, index_version: str) -> dict[str, Any] | None:
    return conn.execute(
        "SELECT * FROM rule_documents WHERE index_version = %s", (index_version,)
    ).fetchone()


def insert_document(conn: psycopg.Connection, doc: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO rule_documents
            (id, title, version, sha256, object_key, embedding_model, embedding_dim,
             chunker_version, index_version, page_count, chunk_count, active)
        VALUES (%(id)s, %(title)s, %(version)s, %(sha256)s, %(object_key)s, %(embedding_model)s,
                %(embedding_dim)s, %(chunker_version)s, %(index_version)s, %(page_count)s,
                %(chunk_count)s, false)
        ON CONFLICT (id) DO UPDATE SET
            chunk_count = EXCLUDED.chunk_count, page_count = EXCLUDED.page_count
        """,
        doc,
    )


CHUNK_COLUMNS = (
    "id", "document_id", "chunk_index", "doc_type", "rule_chapter", "rule_no", "rule_title",
    "sub_item", "rule_id", "parent_id", "annotation", "term_no", "term_en", "term_ko",
    "page_start", "page_end", "breadcrumb", "content", "tokens", "embedding",
)


def upsert_chunks(conn: psycopg.Connection, rows: Sequence[dict[str, Any]]) -> int:
    cols = ", ".join(CHUNK_COLUMNS)
    placeholders = ", ".join(f"%({c})s" for c in CHUNK_COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in CHUNK_COLUMNS if c != "id")
    sql = f"INSERT INTO rule_chunks ({cols}) VALUES ({placeholders}) ON CONFLICT (id) DO UPDATE SET {updates}"
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def delete_stale_chunks(conn: psycopg.Connection, document_id: str, keep_ids: Sequence[str]) -> int:
    """같은 문서를 재인덱싱할 때 이번에 생성되지 않은 옛 청크를 제거한다."""
    row = conn.execute(
        """
        WITH d AS (
            DELETE FROM rule_chunks
            WHERE document_id = %s AND NOT (id = ANY(%s)) RETURNING id
        ) SELECT count(*) AS n FROM d
        """,
        (document_id, list(keep_ids)),
    ).fetchone()
    return int(row["n"]) if row else 0


def activate_document(conn: psycopg.Connection, document_id: str) -> None:
    """부분 UNIQUE 인덱스를 위반하지 않도록 단일 문으로 전환."""
    conn.execute(
        "UPDATE rule_documents SET active = (id = %s) WHERE active OR id = %s",
        (document_id, document_id),
    )


def prune_inactive(conn: psycopg.Connection) -> int:
    row = conn.execute(
        """
        WITH d AS (DELETE FROM rule_documents WHERE NOT active RETURNING id)
        SELECT count(*) AS n FROM d
        """
    ).fetchone()
    return int(row["n"]) if row else 0


def count_chunks(conn: psycopg.Connection, document_id: str) -> int:
    row = conn.execute(
        "SELECT count(*) AS n FROM rule_chunks WHERE document_id = %s", (document_id,)
    ).fetchone()
    return int(row["n"]) if row else 0


# --------------------------------------------------------------------------- 검색

SELECT_FIELDS = """
    id, document_id, chunk_index, doc_type, rule_chapter, rule_no, rule_title, sub_item,
    rule_id, parent_id, annotation, term_no, term_en, term_ko, page_start, page_end,
    breadcrumb, content, tokens
"""


def dense_search(
    conn: psycopg.Connection, document_id: str, embedding: Sequence[float], limit: int = 20
) -> list[dict[str, Any]]:
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.ef_search = 80")
        rows = conn.execute(
            f"""
            SELECT {SELECT_FIELDS}, 1 - (embedding <=> %s::vector) AS score
            FROM rule_chunks WHERE document_id = %s
            ORDER BY embedding <=> %s::vector LIMIT %s
            """,
            (list(embedding), document_id, list(embedding), limit),
        ).fetchall()
    return rows


def by_rule_no(conn: psycopg.Connection, document_id: str, rule_nos: Sequence[str]) -> list[dict[str, Any]]:
    if not rule_nos:
        return []
    return conn.execute(
        f"""
        SELECT {SELECT_FIELDS} FROM rule_chunks
        WHERE document_id = %s AND rule_no = ANY(%s)
        ORDER BY chunk_index
        """,
        (document_id, list(rule_nos)),
    ).fetchall()


def siblings(conn: psycopg.Connection, document_id: str, parent_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not parent_ids:
        return []
    return conn.execute(
        f"""
        SELECT {SELECT_FIELDS} FROM rule_chunks
        WHERE document_id = %s AND parent_id = ANY(%s)
        ORDER BY chunk_index
        """,
        (document_id, list(parent_ids)),
    ).fetchall()


def all_chunks(conn: psycopg.Connection, document_id: str) -> list[dict[str, Any]]:
    return conn.execute(
        f"SELECT {SELECT_FIELDS} FROM rule_chunks WHERE document_id = %s ORDER BY chunk_index",
        (document_id,),
    ).fetchall()


# --------------------------------------------------------------------------- CLI

def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m baseball.db")
    ap.add_argument("command", choices=["ensure", "reset", "stats"])
    ap.add_argument("--yes", action="store_true", help="reset 확인 플래그")
    args = ap.parse_args(argv)
    settings = get_settings()

    if args.command == "ensure":
        print(f"schema ok ({ensure_schema(settings)})")
        return 0
    if args.command == "reset":
        if not args.yes:
            with connect(settings, register=False) as conn:
                for t in APP_TABLES:
                    try:
                        n = conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"]
                    except Exception:
                        conn.rollback()
                        n = "-"
                    print(f"  DROP 예정: {t} ({n} rows)")
            print("reset 하려면 --yes 플래그가 필요합니다.")
            return 1
        counts = reset(settings)
        print("dropped:", ", ".join(f"{k}={v}" for k, v in counts.items()))
        print(f"schema ok ({SCHEMA_VERSION})")
        return 0
    with connect(settings) as conn:
        doc = active_document(conn)
        if not doc:
            print("active document 없음")
            return 0
        print(f"active={doc['id']} index_version={doc['index_version']} chunks={count_chunks(conn, doc['id'])}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
