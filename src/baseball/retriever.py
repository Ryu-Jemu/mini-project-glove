"""하이브리드 검색: exact(규칙번호) + dense(pgvector) + BM25(Kiwi) → RRF → parent expansion."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from langchain_openai import OpenAIEmbeddings

from baseball import db
from baseball.config import Settings, get_settings
from baseball.lexical import LexicalIndex

RULE_NO_RE = re.compile(r"(?<![\d.])([1-9]\.\d{2})(?![\d.])")
RRF_K = 60
DENSE_LIMIT = 20
BM25_LIMIT = 20
EXACT_LIMIT = 8
PARENT_EXPAND_TOP = 3
PARENT_BUDGET_TOKENS = 900
DEFAULT_TOP_K = 6


@dataclass
class QueryPlan:
    exact_rule_nos: list[str] = field(default_factory=list)
    bm25_extra: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    docs: list[dict[str, Any]]
    abstain: bool
    dense_top: float
    bm25_ratio: float
    exact_hits: list[str]
    channels: dict[str, list[str]]
    context_tokens: int


def load_glossary(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    path = settings.base_dir / "data" / "glossary.generated.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def plan_query(query: str, glossary: Sequence[dict[str, Any]]) -> QueryPlan:
    norm = unicodedata.normalize("NFKC", query or "")
    plan = QueryPlan(exact_rule_nos=sorted(set(RULE_NO_RE.findall(norm))))
    for item in glossary:
        for alias in item.get("aliases", []):
            if alias and len(alias) >= 2 and alias in norm:
                plan.bm25_extra.extend([item["term_en"], item["term_ko"]])
                break
    plan.bm25_extra = sorted(set(a for a in plan.bm25_extra if a))
    return plan


def rrf_fuse(channels: dict[str, list[str]], weights: dict[str, float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, ids in channels.items():
        w = weights.get(name, 1.0)
        for rank, cid in enumerate(ids):
            scores[cid] = scores.get(cid, 0.0) + w / (RRF_K + rank + 1)
    return scores


class HybridRetriever:
    def __init__(
        self, settings: Settings, *, document_id: str, lexical: LexicalIndex,
        glossary: Sequence[dict[str, Any]] | None = None, embedder: Any | None = None,
    ) -> None:
        self.settings = settings
        self.document_id = document_id
        self.lexical = lexical
        self.glossary = list(glossary or [])
        self._embedder = embedder
        self.by_id = {c["id"]: c for c in lexical.chunks}

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                dimensions=self.settings.embedding_dimensions,
                max_retries=5,
                api_key=self.settings.openai_api_key,
            )
        return self._embedder

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> RetrievalResult:
        plan = plan_query(query, self.glossary)
        channels: dict[str, list[str]] = {}

        exact_rows: list[dict[str, Any]] = []
        if plan.exact_rule_nos:
            with db.connect(self.settings) as conn:
                exact_rows = db.by_rule_no(conn, self.document_id, plan.exact_rule_nos)[:EXACT_LIMIT]
        channels["exact"] = [r["id"] for r in exact_rows]

        dense_rows: list[dict[str, Any]] = []
        dense_failed = False
        try:
            vector = self.embedder.embed_query(query)
            with db.connect(self.settings) as conn:
                dense_rows = db.dense_search(conn, self.document_id, vector, DENSE_LIMIT)
        except Exception as exc:          # 임베딩 불가(쿼터·네트워크) → BM25·exact 로 계속
            dense_failed = True
            logging.getLogger(__name__).warning("dense 검색 생략: %s", type(exc).__name__)
        channels["dense"] = [r["id"] for r in dense_rows]
        dense_top = float(dense_rows[0]["score"]) if dense_rows else 0.0

        hits = self.lexical.search(query, BM25_LIMIT, extra_terms=plan.bm25_extra)
        channels["bm25"] = [h.chunk_id for h in hits]
        bm25_max = hits[0].score if hits else 0.0
        bm25_ratio = bm25_max / self.lexical.ref_score if self.lexical.ref_score else 0.0

        abstain = (
            not exact_rows
            and (dense_failed or dense_top < self.settings.abstain_dense_threshold)
            and bm25_ratio < self.settings.abstain_bm25_ratio
        )

        pool: dict[str, dict[str, Any]] = {r["id"]: dict(r) for r in dense_rows}
        for r in exact_rows:
            pool.setdefault(r["id"], dict(r))
        for h in hits:
            if h.chunk_id in self.by_id:
                pool.setdefault(h.chunk_id, dict(self.by_id[h.chunk_id]))

        weights = {
            "exact": 1.0,
            "dense": self.settings.rrf_weight_dense,
            "bm25": self.settings.rrf_weight_bm25,
        }
        fused = rrf_fuse(channels, weights)
        ranked_ids = sorted(fused, key=lambda i: fused[i], reverse=True)
        ranked = []
        for cid in ranked_ids:
            if cid not in pool:
                continue
            doc = pool[cid]
            doc["score_fused"] = fused[cid]
            ranked.append(doc)
        ranked = self._expand_parents(ranked)

        top_score = max((d.get("score_fused", 0.0) for d in ranked), default=0.0)
        floor = top_score * self.settings.context_min_score_ratio
        docs: list[dict[str, Any]] = []
        used = 0
        for doc in ranked:
            if len(docs) >= k or used + doc["tokens"] > self.settings.context_max_tokens:
                break
            # 관련성이 확연히 낮은 문서는 넣지 않는다(최소 1건은 유지).
            if docs and doc.get("score_fused", 0.0) < floor:
                continue
            docs.append(doc)
            used += doc["tokens"]
        return RetrievalResult(
            docs=docs, abstain=abstain, dense_top=dense_top, bm25_ratio=bm25_ratio,
            exact_hits=[r["rule_id"] for r in exact_rows], channels=channels, context_tokens=used,
        )

    def _expand_parents(self, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """상위 결과가 하위 항목이면 같은 조항 형제를 묶어 조항 단위로 대체한다."""
        parents = [d["parent_id"] for d in ranked[:PARENT_EXPAND_TOP] if d.get("sub_item") and d.get("parent_id")]
        parents = sorted(set(parents))
        if not parents:
            return ranked
        with db.connect(self.settings) as conn:
            sibling_rows = db.siblings(conn, self.document_id, parents)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in sibling_rows:
            grouped.setdefault(row["parent_id"], []).append(dict(row))

        merged_by_parent: dict[str, dict[str, Any]] = {}
        for parent, rows in grouped.items():
            rows.sort(key=lambda r: r["chunk_index"])
            picked: list[dict[str, Any]] = []
            total = 0
            for r in rows:
                if total + r["tokens"] > PARENT_BUDGET_TOKENS:
                    break
                picked.append(r)
                total += r["tokens"]
            if not picked:
                continue
            head = picked[0]
            body_parts: list[str] = []
            for r in picked:
                body = r["content"].split("\n", 1)[-1].strip()
                marker = r.get("sub_item") or ""
                body_parts.append(f"{marker} {body}".strip() if marker else body)
            breadcrumb = f"[{head['rule_chapter']} ] {head['rule_no']}".replace(" ]", "]")
            if head.get("rule_title"):
                breadcrumb = f"{breadcrumb} {head['rule_title']}"
            content = head["content"].split("\n", 1)[0] + "\n" + "\n".join(body_parts)
            merged = dict(head)
            merged["score_fused"] = max(
                (r.get("score_fused", 0.0) for r in picked), default=head.get("score_fused", 0.0)
            )
            merged.update({
                "id": f"parent:{parent}", "rule_id": parent, "sub_item": None,
                "content": content, "tokens": total,
                "page_start": min(r["page_start"] for r in picked),
                "page_end": max(r["page_end"] for r in picked),
                "breadcrumb": head["breadcrumb"].rsplit(" ", 1)[0] if head.get("sub_item") else head["breadcrumb"],
            })
            merged_by_parent[parent] = merged

        out: list[dict[str, Any]] = []
        seen_parent: set[str] = set()
        for doc in ranked:
            parent = doc.get("parent_id")
            if parent and parent in merged_by_parent:
                if parent in seen_parent:
                    continue
                seen_parent.add(parent)
                merged = merged_by_parent[parent]
                merged["score_fused"] = max(
                    merged.get("score_fused", 0.0), doc.get("score_fused", 0.0)
                )
                out.append(merged)
                continue
            out.append(doc)
        return out


def build_retriever(settings: Settings | None = None) -> HybridRetriever:
    settings = settings or get_settings()
    with db.connect(settings) as conn:
        doc = db.active_document(conn)
        if not doc:
            raise RuntimeError("active 문서가 없습니다 — python -m baseball.ingest 를 먼저 실행하세요")
        chunks = db.all_chunks(conn, doc["id"])
    return HybridRetriever(
        settings, document_id=doc["id"], lexical=LexicalIndex(chunks), glossary=load_glossary(settings),
    )


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m baseball.retriever")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args(argv)
    r = build_retriever()
    res = r.retrieve(args.query, args.k)
    print(f"abstain={res.abstain} dense_top={res.dense_top:.3f} bm25_ratio={res.bm25_ratio:.3f} "
          f"exact={res.exact_hits} context_tokens={res.context_tokens}")
    for i, d in enumerate(res.docs, 1):
        print(f"  [{i}] {d['rule_id']:<16} p.{d['page_start']:<4} tokens={d['tokens']:<4} {d['breadcrumb'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
