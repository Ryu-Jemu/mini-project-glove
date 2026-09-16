"""AnswerDoc → 마크다운. 형식은 여기서 결정되며 모델은 내용만 채운다.

불변식(tests/test_answer_render.py 가 고정한다):
1. 거부면 출력은 prompts 의 거부 상수 **바이트 그대로**, 부가 문구 0자.
   덕분에 detect_status 의 꼬리 예산(refusal_tail_max_chars) 휴리스틱이 무력해진다.
2. 블록을 "\n\n" 로 잇는다 → 모든 리스트 앞에 빈 줄이 보장된다(Streamlit 요구사항).
3. `#` 헤딩을 쓰지 않는다. 채팅 버블에서는 굵은 라벨이면 충분하다.
4. `$` 는 이스케이프한다. Streamlit 이 `$...$` 를 LaTeX 로 먹는다.
5. 원시 HTML 을 만들지 않는다.
6. 모델이 쓴 텍스트는 전부 _clean() 을 지난다. 결정적인 건 구조뿐이고 내용은 모델이 쓴다.
7. 규칙번호 모양 문자열을 **고정 문구**에 넣지 않는다. citations.extract 가 근거 없는
   인용으로 오인해 dropped_citations 를 오염시킨다.
8. evidence 가 있으면 항상 근거 섹션을 낸다 → citations.extract 가 계속 작동한다.
"""
from __future__ import annotations

import re

from baseball.answer_schema import AnswerDoc, Fact, Point, SubAnswer
from baseball.prompts import NOT_IN_CONTEXT_REFUSAL, OFF_TOPIC_REFUSAL

_LINE_LEAD_RE = re.compile(r"^[\s>#*\-+|]+", re.MULTILINE)
_WS_RE = re.compile(r"\s+")
_TERM_RE = re.compile(r"^DEF-(\d+)$")
_RULE_RE = re.compile(r"^\d+\.\d+")

MAX_SUB_ANSWERS = 3


def _clean(text: str | None) -> str:
    """모델 텍스트에서 마크다운 구조를 빼앗는다. 내용은 그대로 둔다."""
    if not text:
        return ""
    out = _LINE_LEAD_RE.sub(" ", text.replace("\r\n", "\n"))
    out = _WS_RE.sub(" ", out).strip()
    return out.replace("<", "&lt;").replace("$", r"\$")


def _bullet(label: str, detail: str) -> str:
    label, detail = _clean(label), _clean(detail)
    if label and detail:
        return f"- **{label}** — {detail}"
    return f"- {detail or label}"


def _points_block(points: list[Point]) -> str:
    return "\n".join(_bullet(p.label, p.detail) for p in points if _clean(p.label) or _clean(p.detail))


def _facts_block(facts: list[Fact]) -> str:
    lines = []
    for f in facts:
        value = _clean(f.value)
        if f.as_of:
            value = f"{value} ({_clean(f.as_of)} 기준)"
        lines.append(_bullet(f.label, value))
    return "\n".join(lines)


def _notes_block(notes: list[str]) -> str:
    return "\n".join(f"- {_clean(n)}" for n in notes if _clean(n))


def _evidence_label(raw: str) -> str:
    token = _clean(raw)
    term = _TERM_RE.match(token)
    if term:
        return f"용어의 정의 {term.group(1)}"
    if _RULE_RE.match(token):
        return f"규칙 {token}"
    return token


def _evidence_block(evidence: list[str]) -> str:
    seen: list[str] = []
    for raw in evidence:
        label = _evidence_label(raw)
        if label and label not in seen:
            seen.append(label)
    return "\n".join(f"- {label}" for label in seen)


def _sub_answers_block(subs: list[SubAnswer]) -> str:
    lines: list[str] = []
    for sub in subs[:MAX_SUB_ANSWERS]:
        lines.append(_bullet(sub.question, sub.headline))
        lines += [f"  {_bullet(p.label, p.detail)}" for p in sub.points]
    return "\n".join(lines)


def _section(blocks: list[str], label: str | None, body: str) -> None:
    """라벨과 본문을 별개 블록으로 넣는다. 본문이 비면 라벨도 넣지 않는다."""
    if not body:
        return
    if label:
        blocks.append(f"**{label}**")
    blocks.append(body)


def render_blocks(doc: AnswerDoc) -> list[str]:
    """SSE token 이벤트 단위이자 render() 의 재료."""
    if doc.is_refusal:
        return [OFF_TOPIC_REFUSAL if doc.refusal == "off_topic" else NOT_IN_CONTEXT_REFUSAL]

    blocks: list[str] = []
    headline = _clean(doc.headline)
    if headline:
        blocks.append(f"**{headline}**")

    if doc.kind == "situation":
        _section(blocks, "판정", _clean(doc.ruling))
        _section(blocks, "왜 이렇게 판정하나요", _points_block(doc.points))
        _section(blocks, "경기는 어떻게 되나요", _clean(doc.outcome))
        _section(blocks, "상황이 달라지면", _points_block(doc.variations))
        _section(blocks, "실제 경기에서는", _clean(doc.example))
        _section(blocks, "왜 중요한가요", _clean(doc.why))
    elif doc.kind in {"entity", "latest"}:
        if doc.kind == "latest" and _clean(doc.as_of):
            blocks.append(f"기준일 {_clean(doc.as_of)}")
        _section(blocks, None, _facts_block(doc.facts))
        _section(blocks, "조금 더 설명하면", _points_block(doc.points))
        _section(blocks, "왜 중요한가요", _clean(doc.why))
    else:                                    # term_rule
        _section(blocks, None, _clean(doc.definition))
        _section(blocks, "어떤 상황에서 적용되나요", _points_block(doc.points))
        _section(blocks, "실제 경기에서는", _clean(doc.example))
        _section(blocks, "왜 중요한가요", _clean(doc.why))

    _section(blocks, "함께 알아두면 좋아요", _notes_block(doc.extra_notes))
    _section(blocks, "다른 질문에 대해서는", _sub_answers_block(doc.sub_answers))
    blocks += [f"※ {_clean(c)}" for c in doc.caveats if _clean(c)]
    _section(blocks, "근거", _evidence_block(doc.evidence))
    return blocks


def render(doc: AnswerDoc) -> str:
    return "\n\n".join(render_blocks(doc))
