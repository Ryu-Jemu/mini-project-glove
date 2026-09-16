"""PDF → MinIO → 구조 청킹 → 임베딩 → pgvector 적재 (멱등)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from langchain_openai import OpenAIEmbeddings

from baseball import db
from baseball.chunker import CHUNKER_VERSION, build, glossary
from baseball.config import Settings, get_settings
from baseball.pdf_parser import parse
from baseball.storage import MinioStorage, RULEBOOK_FILENAME, allowlisted_paths, sha256_file

DOC_TITLE = "2026 공식야구규칙"
DOC_VERSION = "2026"
EMBED_BATCH = 256
PRICE_PER_1M = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}


def make_index_version(sha256: str, settings: Settings) -> str:
    return f"{sha256[:12]}:{CHUNKER_VERSION}:{settings.openai_embedding_model}:{settings.embedding_dimensions}"


def document_id_for(index_version: str) -> str:
    return "rulebook-" + hashlib.sha1(index_version.encode("utf-8")).hexdigest()[:16]


def run(settings: Settings | None = None, *, force: bool = False, dry_run: bool = False,
        embedder: Any | None = None) -> int:
    settings = settings or get_settings()
    path = allowlisted_paths(settings)[0]
    if not path.exists():
        print(f"missing: {path}")
        return 1

    store = MinioStorage(settings)
    key, status = store.upload_if_changed(path)
    digest = sha256_file(path)
    index_version = make_index_version(digest, settings)
    doc_id = document_id_for(index_version)
    print(f"source: {status} key={key}")

    with db.connect(settings) as conn:
        existing = db.document_by_index_version(conn, index_version)
        if existing and existing["active"] and not force:
            print(f"skipped (0 embeddings) index_version={index_version}")
            return 0

    local = store.fetch_to_tmp(key)
    sections, pstats = parse(local)
    chunks = build(sections, index_version=index_version, document_id=doc_id)
    rules = sum(1 for s in sections if s.doc_type == "rule")
    terms = sum(1 for s in sections if s.doc_type == "term")
    total_tokens = sum(c.tokens for c in chunks)
    price = PRICE_PER_1M.get(settings.openai_embedding_model, 0.13)
    cost = total_tokens / 1_000_000 * price
    print(f"parsed: rules={rules} terms={terms} chunks={len(chunks)} tokens={total_tokens}")

    if dry_run:
        print(f"dry-run: embedded=0 cost≈${cost:.4f}")
        return 0

    if embedder is None:                     # 테스트는 가짜 임베더를 주입한다
        settings.require_openai_key()
        embedder = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            dimensions=settings.embedding_dimensions,
            chunk_size=EMBED_BATCH,
            max_retries=5,
            api_key=settings.openai_api_key,
        )
    started = time.time()
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i:i + EMBED_BATCH]
        vectors.extend(embedder.embed_documents([c.content for c in batch]))
        print(f"  embedded {min(i + EMBED_BATCH, len(chunks))}/{len(chunks)}", flush=True)
    elapsed = time.time() - started
    if len(vectors) != len(chunks):
        print(f"embedding count mismatch: {len(vectors)} != {len(chunks)}")
        return 1
    dims = len(vectors[0])
    if dims != settings.embedding_dimensions:
        print(f"dimension mismatch: {dims} != {settings.embedding_dimensions}")
        return 1

    with db.connect(settings) as conn:
        db.insert_document(conn, {
            "id": doc_id, "title": DOC_TITLE, "version": DOC_VERSION, "sha256": digest,
            "object_key": key, "embedding_model": settings.openai_embedding_model,
            "embedding_dim": settings.embedding_dimensions, "chunker_version": CHUNKER_VERSION,
            "index_version": index_version, "page_count": pstats.pages, "chunk_count": len(chunks),
        })
        rows = [c.to_row(v) for c, v in zip(chunks, vectors)]
        db.upsert_chunks(conn, rows)
        stale = db.delete_stale_chunks(conn, doc_id, [r["id"] for r in rows])
        stored = db.count_chunks(conn, doc_id)
        if stored == 0 or any(not r["rule_id"] or not r["page_start"] for r in rows):
            conn.rollback()
            print("구조 검증 실패: chunks=0 또는 rule_id/page 누락")
            return 1
        db.activate_document(conn, doc_id)
        conn.commit()
        removed = db.prune_inactive(conn)
        conn.commit()

    gl = glossary(chunks)
    gl_path = settings.base_dir / "data" / "glossary.generated.json"
    gl_path.write_text(json.dumps(gl, ensure_ascii=False, indent=2), encoding="utf-8")
    with db.connect(settings) as conn:
        conn.execute(
            "INSERT INTO source_snapshots (kind, payload, source_url) VALUES (%s, %s, %s)",
            ("glossary", json.dumps({"items": gl, "index_version": index_version}, ensure_ascii=False), key),
        )
        conn.commit()

    print(
        f"rules={rules} terms={terms} chunks={len(chunks)} embedded={len(vectors)} "
        f"cost≈${cost:.4f} elapsed={elapsed:.1f}s stale_chunks_removed={stale} "
        f"pruned_docs={removed} glossary={len(gl)}"
    )
    return 0


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m baseball.ingest")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return run(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(_main())
