"""AnswerDoc 내용 검사. 렌더러가 보장하는 건 구조뿐이고 내용은 모델이 쓴다.

blocking 은 사용자 화면을 실제로 망가뜨리는 두 가지뿐이다. 나머지는 경고로 남겨
응답(format_issues)과 평가 하네스로 흘린다. 린트 트집 때문에 답변을 버리지 않는다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from baseball.answer_render import render
from baseball.answer_schema import AnswerDoc
# 거부 문장 탐지는 prompts 의 공백 관용 정규식을 그대로 쓴다. 복제하면 드리프트한다.
from baseball.prompts import _NOT_IN_CONTEXT_RE, _OFF_TOPIC_RE

BLOCKING_CODES = frozenset({"REFUSAL_INSIDE_ANSWER", "EMPTY_HEADLINE"})

MIN_HEADLINE_CHARS = 10
MIN_DETAIL_CHARS = 8
TERM_RULE_BULLETS = (2, 5)
MAX_RENDERED_CHARS = 2500

_MD_INJECT_RE = re.compile(r"^\s*[#>*+]|^\s*-\s|\||<\w+|\$", re.MULTILINE)
_RULE_NO_RE = re.compile(r"^[1-9]\.\d{2}")
_DEF_RE = re.compile(r"^DEF-(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Issue:
    code: str
    message: str

    @property
    def blocking(self) -> bool:
        return self.code in BLOCKING_CODES


def _norm(text: str | None) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())


def _text_fields(doc: AnswerDoc) -> Iterator[tuple[str, str]]:
    """모델이 자유롭게 쓴 모든 문자열을 (경로, 값)으로 훑는다."""
    for name in ("headline", "definition", "ruling", "outcome", "example", "why"):
        yield name, getattr(doc, name) or ""
    for i, note in enumerate(doc.extra_notes):
        yield f"extra_notes[{i}]", note
    for i, caveat in enumerate(doc.caveats):
        yield f"caveats[{i}]", caveat
    for group in ("points", "variations"):
        for i, p in enumerate(getattr(doc, group)):
            yield f"{group}[{i}].label", p.label
            yield f"{group}[{i}].detail", p.detail
    for i, f in enumerate(doc.facts):
        yield f"facts[{i}].label", f.label
        yield f"facts[{i}].value", f.value
    for i, s in enumerate(doc.sub_answers):
        yield f"sub_answers[{i}].question", s.question
        yield f"sub_answers[{i}].headline", s.headline
        for j, p in enumerate(s.points):
            yield f"sub_answers[{i}].points[{j}].label", p.label
            yield f"sub_answers[{i}].points[{j}].detail", p.detail


def _backed_refs(retrieved: Sequence[dict[str, Any]]) -> set[str]:
    backing: set[str] = set()
    for doc in retrieved:
        for key in ("rule_no", "rule_id"):
            if doc.get(key):
                backing.add(str(doc[key]))
        if doc.get("term_no"):
            backing.add(f"DEF-{doc['term_no']}")
    return backing


def _claimed_refs(doc: AnswerDoc) -> list[str]:
    refs = list(doc.evidence)
    for group in ("points", "variations"):
        refs += [p.rule_ref for p in getattr(doc, group) if p.rule_ref]
    for sub in doc.sub_answers:
        refs += list(sub.evidence)
        refs += [p.rule_ref for p in sub.points if p.rule_ref]
    return refs


def _is_ref_backed(ref: str, backing: set[str]) -> bool:
    token = _norm(ref)
    if token in backing:
        return True
    term = _DEF_RE.match(token)
    if term:
        return f"DEF-{term.group(1)}" in backing
    if _RULE_NO_RE.match(token):
        # '5.09(a)' 은 '5.09' 로 뒷받침된다. 반대 방향도 허용한다.
        return any(b.startswith(token) or token.startswith(b) for b in backing)
    return False


def check(
    doc: AnswerDoc, retrieved: Sequence[dict[str, Any]] = (), *, knowledge_only: bool = False
) -> list[Issue]:
    if doc.is_refusal:
        return []                       # 거부는 렌더러가 상수를 내므로 검사할 내용이 없다

    issues: list[Issue] = []

    for path, text in _text_fields(doc):
        norm = _norm(text)
        if _OFF_TOPIC_RE.search(norm) or _NOT_IN_CONTEXT_RE.search(norm):
            issues.append(Issue("REFUSAL_INSIDE_ANSWER",
                                f"{path} 에 거부 문장이 섞여 있다"))
            break
    # 비어 있는 것과 짧은 것은 다르다. "9명입니다" 는 다섯 자지만 완전한 답이다.
    # 둘을 한 코드로 묶어 두면 정답이 거부 문장으로 강등된다(chain 이 블로킹 코드를 그렇게 쓴다).
    head = _norm(doc.headline)
    if not head:
        issues.append(Issue("EMPTY_HEADLINE", "headline 이 비었다"))
    elif len(head) < MIN_HEADLINE_CHARS:
        issues.append(Issue("SHORT_HEADLINE",
                            f"headline 이 {len(head)}자다({MIN_HEADLINE_CHARS}자 이상 기대)"))

    injected = [p for p, t in _text_fields(doc) if t and _MD_INJECT_RE.search(t)]
    if injected:
        issues.append(Issue("MARKDOWN_INJECTION",
                            f"마크다운 기호가 섞였다(렌더러가 정화): {', '.join(injected[:3])}"))

    stubs = [f"{g}[{i}]" for g in ("points", "variations")
             for i, p in enumerate(getattr(doc, g))
             if len(_norm(p.detail)) < MIN_DETAIL_CHARS or _norm(p.detail) == _norm(p.label)]
    if stubs:
        issues.append(Issue("STUB_SECTION", f"내용 없는 불릿: {', '.join(stubs)}"))

    low, high = TERM_RULE_BULLETS
    if doc.kind == "term_rule" and not low <= len(doc.points) <= high:
        issues.append(Issue("BULLET_COUNT",
                            f"term_rule 의 points 가 {len(doc.points)}개다({low}~{high} 기대)"))

    if knowledge_only:
        # 근거 문서가 아예 없다. 그러면 backing 이 비어 아래 검사가 통째로 건너뛰어지고,
        # 규칙 번호를 지어내도 아무도 잡지 못한다. 이 경로의 주된 실패 방식이므로 뒤집는다.
        claimed = sorted(set(_claimed_refs(doc)))
        if claimed:
            issues.append(Issue("UNBACKED_RULE_REF",
                                f"근거 없이 인용한 규칙 번호: {', '.join(claimed)}"))
    else:
        backing = _backed_refs(retrieved)
        if backing:
            unbacked = sorted({r for r in _claimed_refs(doc) if not _is_ref_backed(r, backing)})
            if unbacked:
                issues.append(Issue("UNBACKED_RULE_REF",
                                    f"검색 결과에 없는 근거: {', '.join(unbacked)}"))

    situation_only = bool(doc.ruling or doc.outcome or doc.variations)
    if doc.kind == "situation" and not _norm(doc.ruling):
        issues.append(Issue("KIND_FIELD_MISMATCH", "situation 인데 ruling 이 비었다"))
    elif doc.kind != "situation" and situation_only:
        issues.append(Issue("KIND_FIELD_MISMATCH",
                            f"{doc.kind} 인데 situation 전용 필드가 채워졌다"))
    if doc.kind not in {"entity", "latest"} and doc.facts:
        issues.append(Issue("KIND_FIELD_MISMATCH", f"{doc.kind} 인데 facts 가 채워졌다"))

    if doc.kind == "latest" and not _norm(doc.as_of):
        issues.append(Issue("LATEST_NO_AS_OF", "latest 인데 기준일이 없다"))

    if _norm(doc.definition) and _norm(doc.definition) == _norm(doc.headline):
        issues.append(Issue("DUP_TEXT", "definition 이 headline 과 같다"))

    rendered = len(render(doc))
    if rendered > MAX_RENDERED_CHARS:
        issues.append(Issue("TOO_LONG", f"렌더 결과가 {rendered}자다({MAX_RENDERED_CHARS} 초과)"))

    return issues


def blocking(issues: Sequence[Issue]) -> bool:
    return any(i.blocking for i in issues)


def codes(issues: Sequence[Issue]) -> list[str]:
    return [i.code for i in issues]
