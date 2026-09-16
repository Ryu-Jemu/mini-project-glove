"""테스트 공용 헬퍼(픽스처가 아닌 순수 함수)."""
from __future__ import annotations

from typing import Any


def make_doc(rule_id: str = "5.09(a)", **kw: Any) -> dict[str, Any]:
    base = {
        "id": f"id-{rule_id}", "document_id": "doc", "chunk_index": 0, "doc_type": "rule",
        "rule_chapter": "5.00", "rule_no": rule_id.split("(")[0], "rule_title": "아웃",
        "sub_item": "⒜", "rule_id": rule_id, "parent_id": "5.09", "annotation": None,
        "term_no": None, "term_en": None, "term_ko": None, "page_start": 87, "page_end": 87,
        "breadcrumb": "[5.00 경기의 진행] 5.09 아웃 ⒜",
        "content": "[5.00 경기의 진행] 5.09 아웃 ⒜\n타자 아웃인 경우는 다음과 같다.",
        "tokens": 30,
    }
    base.update(kw)
    return base


def make_answer_payload(**kw: Any) -> dict[str, Any]:
    """AnswerDoc 모양의 완전한 dict. 모든 필드가 required 라 빠짐없이 채운다."""
    base: dict[str, Any] = {
        "kind": "term_rule",
        "answerable": True,
        "refusal": "none",
        "headline": "보크는 투수의 반칙 투구입니다.",
        "definition": "투수가 정해진 투구 동작을 어기거나 주자를 속이는 동작을 하면 보크가 선언됩니다.",
        "ruling": None,
        "outcome": None,
        "points": [
            {"label": "주자가 있을 때", "detail": "누상에 주자가 있을 때만 선언됩니다.",
             "rule_ref": "5.09"},
            {"label": "동작을 멈추면", "detail": "투구 동작을 시작한 뒤 중간에 멈추면 보크입니다.",
             "rule_ref": "5.09"},
        ],
        "variations": [],
        "facts": [],
        "example": "1루에 주자가 있을 때 투수가 세트포지션에서 어깨를 움직이다 멈추면 보크가 선언됩니다.",
        "why": "주자를 속이는 동작을 막아 주자 쪽이 일방적으로 손해 보지 않게 하려는 규칙입니다.",
        "extra_notes": [],
        "caveats": [],
        "evidence": ["5.09"],
        "as_of": None,
        "sub_answers": [],
    }
    base.update(kw)
    return base


def make_refusal_payload(refusal: str = "not_in_context") -> dict[str, Any]:
    return make_answer_payload(
        answerable=False, refusal=refusal, headline="", definition=None,
        points=[], example=None, why=None, evidence=[],
    )
