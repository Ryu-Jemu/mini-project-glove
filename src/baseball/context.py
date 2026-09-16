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
    kind: Literal["snapshot", "web", "model", "place", "video"]
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


# 이 문자열은 system_prompt.txt 2절이 찾는 표지다. 바꾸면 그 지시가 켜지지 않는다.
MODEL_BLOCK_LABEL = "일반 지식"
MODEL_BLOCK_TEXT = (
    "이 질문의 답은 규칙집·리그 규정·KBO 데이터에서 확인되지 않았습니다. "
    "web_search 도구를 쓸 수 있으면 먼저 확인하고, 그래도 없으면 널리 합의된 야구 상식으로 답하세요. "
    "순위·기록·일정 같은 값은 확인된 자료 없이 말하지 말고, 규칙 번호는 인용하지 마세요."
)


def _model_block(index: int) -> str:
    """규칙집에도 웹에도 근거가 없을 때만 들어가는 블록.

    _latest_block 을 재사용하지 않는 이유는 그쪽이 출처 없음을 '내부 스냅샷' 으로 적기 때문이다.
    모델 지식에는 출처가 없다는 사실 자체를 그대로 적어야 한다.
    """
    return (
        f"[자료 {index} | {MODEL_BLOCK_LABEL} | 규칙집·웹에서 확인되지 않은 질문 | "
        f"출처 없음 | 신뢰도 uncertain]\n{MODEL_BLOCK_TEXT}"
    )


def format_context(
    rule_docs: Sequence[dict[str, Any]],
    latest: Sequence[LatestEntry] = (),
    kbo: Sequence[KboEntry] = (),
    *,
    max_tokens: int = 3000,
    latest_max_tokens: int = 800,
    kbo_max_tokens: int = 2200,
    knowledge_only: bool = False,
) -> str:
    """번호는 규칙집 → 최신정보 → KBO 순서로 연속. 예산 초과 시 해당 블록에서 중단."""
    blocks: list[str] = []
    index = 1

    if knowledge_only:
        return _model_block(index)

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
