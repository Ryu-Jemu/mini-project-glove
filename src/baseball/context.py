"""{context} 문자열의 단일 조립점. 규칙집·최신정보·(Phase 2)KBO 블록을 같은 번호 체계로 렌더."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import tiktoken

_ENC = tiktoken.get_encoding("o200k_base")

BLOCK_SEPARATOR = "\n\n"


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass(frozen=True)
class LatestEntry:
    kind: Literal["snapshot", "web"]
    label: str
    text: str
    as_of: str                                   # "YYYY-MM-DD"
    source_url: str | None = None
    confidence: Literal["confirmed", "likely", "uncertain"] | None = None


@dataclass(frozen=True)
class KboEntry:
    kind: Literal["standings", "schedule", "team_info", "roster", "highlight", "season"]
    label: str
    text: str
    as_of: str
    source_label: str
    source_url: str | None = None
    freshness: Literal["live", "cached", "stale", "static", "unavailable"] = "static"


_STATUS_TEXT = {
    "live": "실시간",
    "cached": "저장본(최신)",
    "static": "기본정보",
}


def _rule_block(index: int, doc: dict[str, Any]) -> str:
    body = doc["content"].split("\n", 1)[-1].strip()
    return (
        f"[자료 {index} | 2026 공식야구규칙 | {doc['breadcrumb']} | p.{doc['page_start']}]\n{body}"
    )


def _latest_block(index: int, entry: LatestEntry) -> str:
    source = entry.source_url or "내부 스냅샷"
    confidence = entry.confidence or "-"
    return (
        f"[자료 {index} | 최신정보 | {entry.label} | 기준일 {entry.as_of} | "
        f"출처 {source} | 신뢰도 {confidence}]\n{entry.text.strip()}"
    )


def _kbo_block(index: int, entry: KboEntry) -> str:
    if entry.freshness == "stale":
        status = f"오래된 저장본(최신 조회 실패, {entry.as_of} 기준)"
    else:
        status = _STATUS_TEXT.get(entry.freshness, "기본정보")
    return (
        f"[자료 {index} | KBO 데이터 | {entry.label} | 기준 {entry.as_of} | "
        f"출처 {entry.source_label} | 상태 {status}]\n{entry.text.strip()}"
    )


def format_context(
    rule_docs: Sequence[dict[str, Any]],
    latest: Sequence[LatestEntry] = (),
    kbo: Sequence[KboEntry] = (),
    *,
    max_tokens: int = 3000,
    latest_max_tokens: int = 800,
    kbo_max_tokens: int = 2200,
) -> str:
    """번호는 규칙집 → 최신정보 → KBO 순서로 연속. 예산 초과 시 해당 블록에서 중단."""
    blocks: list[str] = []
    index = 1

    used = 0
    for doc in rule_docs:
        block = _rule_block(index, doc)
        cost = count_tokens(block)
        if used + cost > max_tokens:
            break
        blocks.append(block)
        used += cost
        index += 1

    used = 0
    for entry in latest:
        block = _latest_block(index, entry)
        cost = count_tokens(block)
        if used + cost > latest_max_tokens:
            break
        blocks.append(block)
        used += cost
        index += 1

    used = 0
    for entry in kbo:
        if entry.freshness == "unavailable":
            continue
        block = _kbo_block(index, entry)
        cost = count_tokens(block)
        if used + cost > kbo_max_tokens:
            break
        blocks.append(block)
        used += cost
        index += 1

    return BLOCK_SEPARATOR.join(blocks)
