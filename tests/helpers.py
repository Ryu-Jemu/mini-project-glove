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
