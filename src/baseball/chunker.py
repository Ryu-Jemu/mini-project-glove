"""섹션 → 검색용 청크. 구조(규칙번호·하위항목)를 메타데이터로 보존한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from baseball.pdf_parser import Section, default_pdf_path, parse

CHUNKER_VERSION = "struct-v2"
ENCODING_NAME = "o200k_base"
MERGE_BUDGET = 500       # 같은 chapter 내 연속 소섹션 병합 상한
SPLIT_THRESHOLD = 900    # 이 이상이면 sub_item 경계로 분할
FALLBACK_CHUNK = 450
FALLBACK_OVERLAP = 60
FALLBACK_SEPARATORS = ["\n\n", "\n", "다. ", ". ", " ", ""]

# 하위 항목 마커
SUB_A_RE = re.compile(r"^\s*([⒜-⒵])")      # ⒜-⒵
SUB_N_RE = re.compile(r"^\s*([⑴-⒇])")      # ⑴-⒇
SUB_U_RE = re.compile(r"^\s*\(([A-Z])\)")
ANNOT_RE = re.compile(r"^\s*\[(주|원주|부기|문|답)\]")

_ENC = tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _letter(ch: str) -> str:
    return chr(ord(ch) - 0x249C + ord("a"))


def _number(ch: str) -> int:
    return ord(ch) - 0x2474 + 1


@dataclass
class Chunk:
    id: str
    document_id: str
    chunk_index: int
    doc_type: str
    rule_chapter: str | None
    rule_no: str | None
    rule_title: str | None
    sub_item: str | None
    rule_id: str
    parent_id: str | None
    annotation: str | None
    term_no: int | None
    term_en: str | None
    term_ko: str | None
    page_start: int
    page_end: int
    breadcrumb: str
    content: str
    tokens: int

    def to_row(self, embedding: list[float]) -> dict:
        row = asdict(self)
        row["embedding"] = embedding
        return row


@dataclass
class _Unit:
    """규칙 섹션 내부의 하위 항목 단위."""
    lines: list[tuple[int, str]] = field(default_factory=list)
    sub_item: str | None = None
    suffix: str = ""
    annotation: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(l for _, l in self.lines).strip()


def iter_units(section: Section) -> Iterator[_Unit]:
    """⒜ / ⑴ / (A) 경계로 분해. [주]·[원주]·[부기]는 직전 항목에 부착."""
    letter: str | None = None
    number: int | None = None
    upper: str | None = None
    raw_letter: str = ""
    raw_number: str = ""
    current = _Unit()

    def marker() -> tuple[str | None, str]:
        raw = f"{raw_letter}{raw_number}" + (f"({upper})" if upper else "")
        suffix = ""
        if letter:
            suffix += f"({letter})"
        if number:
            suffix += f"({number})"
        if upper:
            suffix += f"({upper})"
        return (raw or None), suffix

    for page, line in section.lines:
        m_a, m_n, m_u = SUB_A_RE.match(line), SUB_N_RE.match(line), SUB_U_RE.match(line)
        m_an = ANNOT_RE.match(line)
        if m_a or m_n or m_u:
            if current.lines:
                current.sub_item, current.suffix = marker()
                yield current
            if m_a:
                letter, number, upper = _letter(m_a.group(1)), None, None
                raw_letter, raw_number = m_a.group(1), ""
            elif m_n:
                number, upper = _number(m_n.group(1)), None
                raw_number = m_n.group(1)
            else:
                upper = m_u.group(1)
            current = _Unit()
        elif m_an and current.annotation is None:
            current.annotation = m_an.group(1)
        current.lines.append((page, line))

    if current.lines:
        current.sub_item, current.suffix = marker()
        yield current


def _breadcrumb(section: Section, sub_item: str | None) -> str:
    if section.doc_type == "term":
        return f"[용어의 정의] {section.term_no}. {section.term_en} ({section.term_ko})"
    if section.doc_type == "front":
        return "[2026년 공식야구규칙 변경 요약]"
    chapter = section.rule_chapter or ""
    chapter_title = section.chapter_title or ""
    head = f"[{chapter} {chapter_title}]".replace("  ", " ").strip()
    parts = [head, section.rule_no or ""]
    if section.rule_title:
        parts.append(section.rule_title)
    if sub_item:
        parts.append(sub_item)
    return " ".join(p for p in parts if p)


def _chunk_id(index_version: str, rule_id: str, ordinal: int) -> str:
    raw = f"{index_version}|{rule_id}|{ordinal}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _make(
    section: Section, *, index_version: str, document_id: str, chunk_index: int,
    rule_id: str, sub_item: str | None, annotation: str | None,
    lines: list[tuple[int, str]], ordinal: int,
) -> Chunk:
    body = "\n".join(l for _, l in lines).strip()
    breadcrumb = _breadcrumb(section, sub_item)
    content = f"{breadcrumb}\n{body}" if body else breadcrumb
    pages = [p for p, _ in lines] or [section.page_start]
    parent = section.rule_no if (section.doc_type == "rule" and sub_item) else None
    return Chunk(
        id=_chunk_id(index_version, rule_id, ordinal),
        document_id=document_id,
        chunk_index=chunk_index,
        doc_type=section.doc_type,
        rule_chapter=section.rule_chapter,
        rule_no=section.rule_no,
        rule_title=section.rule_title,
        sub_item=sub_item,
        rule_id=rule_id,
        parent_id=parent,
        annotation=annotation,
        term_no=section.term_no,
        term_en=section.term_en,
        term_ko=section.term_ko,
        page_start=min(pages),
        page_end=max(pages),
        breadcrumb=breadcrumb,
        content=content,
        tokens=count_tokens(content),
    )


def _fallback_split(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME,
        chunk_size=FALLBACK_CHUNK,
        chunk_overlap=FALLBACK_OVERLAP,
        separators=FALLBACK_SEPARATORS,
    )
    return splitter.split_text(text)


def build(sections: Iterable[Section], *, index_version: str, document_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = list(sections)
    idx = 0

    # --- front / term: 항목당 1청크 ---------------------------------------
    simple = [s for s in sections if s.doc_type in {"front", "term"}]
    rules = [s for s in sections if s.doc_type == "rule"]

    for s in simple:
        rule_id = "CHG-2026" if s.doc_type == "front" else f"DEF-{s.term_no}"
        chunks.append(_make(
            s, index_version=index_version, document_id=document_id, chunk_index=idx,
            rule_id=rule_id, sub_item=None, annotation=None, lines=s.lines, ordinal=0,
        ))
        idx += 1

    # --- rule ---------------------------------------------------------------
    pending: list[Section] = []          # 병합 대기(같은 chapter의 작은 섹션)
    pending_tokens = 0

    def flush_pending() -> None:
        nonlocal pending, pending_tokens, idx
        if not pending:
            return
        head = pending[0]
        lines: list[tuple[int, str]] = []
        for sec in pending:
            lines.extend(sec.lines)
        rule_id = head.rule_no or "?"
        if len(pending) > 1:
            rule_id = f"{head.rule_no}~{pending[-1].rule_no}"
        chunks.append(_make(
            head, index_version=index_version, document_id=document_id, chunk_index=idx,
            rule_id=rule_id, sub_item=None, annotation=None, lines=lines, ordinal=0,
        ))
        idx += 1
        pending, pending_tokens = [], 0

    for s in rules:
        tokens = count_tokens(s.text)
        is_chapter = bool(s.rule_no and s.rule_no.endswith(".00"))

        if tokens >= SPLIT_THRESHOLD:
            flush_pending()
            ordinal = 0
            for unit in iter_units(s):
                utext = unit.text
                if not utext:
                    continue
                rule_id = f"{s.rule_no}{unit.suffix}"
                if count_tokens(utext) <= SPLIT_THRESHOLD:
                    chunks.append(_make(
                        s, index_version=index_version, document_id=document_id, chunk_index=idx,
                        rule_id=rule_id, sub_item=unit.sub_item, annotation=unit.annotation,
                        lines=unit.lines, ordinal=ordinal,
                    ))
                    idx += 1
                    ordinal += 1
                    continue
                page = unit.lines[0][0]
                for piece in _fallback_split(utext):
                    chunks.append(_make(
                        s, index_version=index_version, document_id=document_id, chunk_index=idx,
                        rule_id=rule_id, sub_item=unit.sub_item, annotation=unit.annotation,
                        lines=[(page, piece)], ordinal=ordinal,
                    ))
                    idx += 1
                    ordinal += 1
            continue

        # 병합 대상: chapter 경계(x.00)에서 끊고, 예산 초과 시 끊는다
        if is_chapter or (pending and pending[0].rule_chapter != s.rule_chapter):
            flush_pending()
        if pending_tokens + tokens > MERGE_BUDGET:
            flush_pending()
        pending.append(s)
        pending_tokens += tokens

    flush_pending()

    # --- 후처리: 최종 content 기준으로 상한 초과 청크를 재분할 ----------------
    final: list[Chunk] = []
    for c in chunks:
        if c.tokens <= SPLIT_THRESHOLD:
            final.append(c)
            continue
        body = c.content[len(c.breadcrumb):].lstrip("\n")
        for ordinal, piece in enumerate(_fallback_split(body)):
            content = f"{c.breadcrumb}\n{piece}"
            final.append(Chunk(
                **{**asdict(c),
                   "id": _chunk_id(index_version, c.rule_id, 1000 + ordinal),
                   "content": content,
                   "tokens": count_tokens(content)}
            ))
    for i, c in enumerate(final):
        c.chunk_index = i
    return final


def glossary(chunks: Iterable[Chunk]) -> list[dict]:
    out: list[dict] = []
    seen: set[int] = set()
    for c in chunks:
        if c.doc_type != "term" or not c.term_en or c.term_no in seen:
            continue
        seen.add(c.term_no)
        aliases = sorted({c.term_en, c.term_en.title(), c.term_ko, *(c.term_ko or "").split("·")} - {None, ""})
        out.append({
            "term_no": c.term_no, "term_en": c.term_en, "term_ko": c.term_ko,
            "rule_id": c.rule_id, "page": c.page_start, "aliases": aliases,
        })
    return out


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m baseball.chunker")
    ap.add_argument("command", choices=["stats", "dump"])
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--rule", default=None)
    args = ap.parse_args(argv)

    sections, pstats = parse(Path(args.pdf) if args.pdf else default_pdf_path())
    chunks = build(sections, index_version="dev", document_id="dev")
    if args.command == "dump":
        for c in chunks:
            if args.rule and (c.rule_no != args.rule and c.rule_id != args.rule):
                continue
            print(f"--- [{c.chunk_index}] {c.rule_id} tokens={c.tokens} p.{c.page_start}-{c.page_end}")
            print(c.content[:400])
        return 0

    tokens = sorted(c.tokens for c in chunks)
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.doc_type] = by_type.get(c.doc_type, 0) + 1
    crossing = [c for c in chunks if c.doc_type == "rule" and c.rule_id.count("~") and
                c.rule_id.split("~")[0][0] != c.rule_id.split("~")[1][0]]
    print(
        f"chunks={len(chunks)} rule={by_type.get('rule',0)} term={by_type.get('term',0)} "
        f"front={by_type.get('front',0)} max_tokens={tokens[-1]} "
        f"p50={tokens[len(tokens)//2]} p90={tokens[int(len(tokens)*0.9)]} "
        f"chapter_crossing={len(crossing)} glossary={len(glossary(chunks))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
