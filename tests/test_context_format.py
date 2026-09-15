from __future__ import annotations

from baseball.context import LatestEntry, format_context
from baseball.prompts import answer_prompt

from helpers import make_doc


def _latest(kind: str = "snapshot") -> LatestEntry:
    return LatestEntry(
        kind=kind, label="KBO 리그 규정 — 피치클락", text="주자 없을 때 18초, 있을 때 23초.",
        as_of="2026-09-15", source_url="https://example.invalid/kbo", confidence="confirmed",
    )


def test_block_shapes_and_numbering() -> None:
    docs = [make_doc("5.09(a)"), make_doc("5.09(b)", id="id-b")]
    out = format_context(docs, [_latest("snapshot"), _latest("web")])
    blocks = out.split("\n\n")
    assert len(blocks) == 4
    assert blocks[0].startswith("[자료 1 | 2026 공식야구규칙 | [5.00 경기의 진행] 5.09 아웃 ⒜ | p.87]")
    assert blocks[2].startswith("[자료 3 | 최신정보 | KBO 리그 규정 — 피치클락 | 기준일 2026-09-15 | 출처 https://example.invalid/kbo | 신뢰도 confirmed]")
    assert blocks[3].startswith("[자료 4 |")


def test_budgets_cut_blocks() -> None:
    docs = [make_doc(f"5.0{i}", id=f"id{i}") for i in range(4)]
    assert len(format_context(docs, [], max_tokens=50).split("\n\n")) == 1
    assert format_context(docs, [_latest()], latest_max_tokens=1).count("최신정보") == 0


def test_empty_returns_empty_string() -> None:
    assert format_context([], []) == ""


def test_braces_survive_and_render() -> None:
    doc = make_doc(content="[5.00 경기의 진행] 5.09 아웃 ⒜\n중괄호 {x} 와 }y{ 포함")
    ctx = format_context([doc], [])
    assert "{x}" in ctx
    msgs = answer_prompt(8).format_messages(question="Q", context=ctx)
    assert "{x}" in msgs[1].content
