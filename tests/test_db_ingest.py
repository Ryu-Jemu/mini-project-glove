"""실제 Postgres 가 필요한 멱등성 테스트. 운영 인덱스를 오염시키지 않도록 별도 DB(baseball_test)를 쓴다."""
from __future__ import annotations

import psycopg
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from baseball import db
from baseball.config import get_settings

pytestmark = pytest.mark.db

TEST_DB = "baseball_test"


def _ensure_test_database(admin_dsn: str) -> str:
    """운영 DB(baseball)와 분리된 테스트 DB를 만든다."""
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        exists = conn.execute("select 1 from pg_database where datname = %s", (TEST_DB,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    return admin_dsn.rsplit("/", 1)[0] + f"/{TEST_DB}"


@pytest.fixture
def pg_settings(monkeypatch: pytest.MonkeyPatch):
    get_settings.cache_clear()
    base = get_settings()
    try:
        test_dsn = _ensure_test_database(base.pg_dsn)
    except Exception:
        pytest.skip("Postgres 컨테이너 없음")
    monkeypatch.setenv("PG_DSN", test_dsn)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.pg_dsn.endswith(TEST_DB)          # 운영 DB 로 절대 쓰지 않는다
    db.ensure_schema(settings)
    yield settings
    get_settings.cache_clear()


def test_ingest_is_idempotent_and_activates_single_document(pg_settings) -> None:
    from baseball.ingest import run

    fake = DeterministicFakeEmbedding(size=pg_settings.embedding_dimensions)

    assert run(pg_settings, force=True, embedder=fake) == 0
    with db.connect(pg_settings) as conn:
        first = db.active_document(conn)
        assert first is not None
        assert db.count_chunks(conn, first["id"]) > 0
        assert conn.execute("select count(*) as n from rule_documents where active").fetchone()["n"] == 1

    assert run(pg_settings, embedder=fake) == 0        # 두 번째 실행은 skipped
    with db.connect(pg_settings) as conn:
        again = db.active_document(conn)
        assert again["id"] == first["id"]
        orphan = conn.execute(
            "select count(*) as n from rule_chunks c left join rule_documents d"
            " on c.document_id = d.id where d.id is null or not d.active"
        ).fetchone()
        assert orphan["n"] == 0                        # prune 후 inactive 청크 없음


def test_schema_mismatch_is_detected(pg_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    get_settings.cache_clear()
    mismatched = get_settings()
    assert mismatched.pg_dsn.endswith(TEST_DB)
    with pytest.raises(db.SchemaMismatch):
        db.ensure_schema(mismatched)
    get_settings.cache_clear()
