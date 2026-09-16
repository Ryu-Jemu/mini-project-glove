"""답변 본문에서 규칙 번호·용어를 추출해 검색 결과와 대조(best-effort)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# 한글은 \w 이므로 \b 를 쓸 수 없다 → 전후 숫자/마침표만 배제
RULE_NO_RE = re.compile(r"(?<![\d.])([1-9]\.\d{2})(?![\d.])")
TERM_EN_RE = re.compile(r"\b([A-Z][A-Z'’`\-]{2,}(?:\s+(?:or\s+)?[A-Z][A-Z'’`\-]{2,})*)\b")
SUB_MARKER_RE = re.compile(r"[⒜-⒵⑴-⒇]+")


@dataclass(frozen=True)
class Citation:
    label: str
    rule_id: str
    doc_type: str
    page: int
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "rule_id": self.rule_id, "doc_type": self.doc_type,
            "page": self.page, "snippet": self.snippet,
        }


def normalize_rule_id(text: str) -> str:
    """'규칙 5.09⒜⑴' / '5.09 (a)(1)' → '5.09(a)(1)'."""
    t = text.strip()
    m = RULE_NO_RE.search(t)
    if not m:
        return t
    rule_no = m.group(1)
    tail = t[m.end():]
    parts: list[str] = []
    for ch in tail:
        if "⒜" <= ch <= "⒵":
            parts.append(f"({chr(ord(ch) - 0x249C + ord('a'))})")
        elif "⑴" <= ch <= "⒇":
            parts.append(f"({ord(ch) - 0x2474 + 1})")
    for m2 in re.finditer(r"\(\s*([a-zA-Z0-9]{1,2})\s*\)", tail):
        parts.append(f"({m2.group(1)})")
    seen: set[str] = set()
    uniq = [p for p in parts if not (p in seen or seen.add(p))]
    return rule_no + "".join(uniq)


def _snippet(content: str, limit: int = 160) -> str:
    body = content.split("\n", 1)[-1].strip().replace("\n", " ")
    return body[:limit]


def extract(answer: str, retrieved: Sequence[dict[str, Any]]) -> tuple[list[Citation], list[str]]:
    """반환 (검색 결과로 뒷받침되는 인용, 뒷받침되지 않아 버린 라벨)."""
    by_rule_no: dict[str, dict[str, Any]] = {}
    by_term: dict[str, dict[str, Any]] = {}
    for c in retrieved:
        if c.get("rule_no") and c["rule_no"] not in by_rule_no:
            by_rule_no[c["rule_no"]] = c
        if c.get("term_en"):
            by_term.setdefault(c["term_en"].upper(), c)

    citations: list[Citation] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for m in RULE_NO_RE.finditer(answer or ""):
        rule_no = m.group(1)
        tail = (answer[m.end():m.end() + 12] or "")
        sub = SUB_MARKER_RE.match(tail.lstrip())
        label = f"규칙 {rule_no}{sub.group(0) if sub else ''}"
        if label in seen:
            continue
        seen.add(label)
        hit = by_rule_no.get(rule_no)
        if hit is None:
            dropped.append(label)
            continue
        citations.append(Citation(
            label=label, rule_id=normalize_rule_id(label), doc_type=hit.get("doc_type", "rule"),
            page=int(hit.get("page_start") or 0), snippet=_snippet(hit.get("content", "")),
        ))

    for m in TERM_EN_RE.finditer(answer or ""):
        name = m.group(1).strip().upper()
        hit = by_term.get(name)
        if hit is None or name in seen:
            continue
        seen.add(name)
        citations.append(Citation(
            label=f"용어의 정의 {hit['term_no']}. {hit['term_en']}",
            rule_id=f"DEF-{hit['term_no']}", doc_type="term",
            page=int(hit.get("page_start") or 0), snippet=_snippet(hit.get("content", "")),
        ))
    return citations, dropped
