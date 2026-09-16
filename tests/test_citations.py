from __future__ import annotations

from baseball.citations import extract, normalize_rule_id

from helpers import make_doc


def test_normalize_rule_id() -> None:
    assert normalize_rule_id("규칙 5.09⒜⑴") == "5.09(a)(1)"
    assert normalize_rule_id("5.09 (a)(1)") == "5.09(a)(1)"
    assert normalize_rule_id("규칙 6.02") == "6.02"
    assert normalize_rule_id("규칙집") == "규칙집"


def test_extract_keeps_only_grounded_citations() -> None:
    docs = [make_doc("5.09(a)")]
    found, dropped = extract("규칙 5.09⒜ 에 따라 아웃입니다. 또한 7.77 을 보세요.", docs)
    assert [c.rule_id for c in found] == ["5.09(a)"]
    assert found[0].page == 87
    assert dropped == ["규칙 7.77"]


def test_extract_matches_term_names() -> None:
    docs = [make_doc(
        "DEF-40", doc_type="term", rule_no=None, term_no=40, term_en="INFIELD FLY",
        term_ko="인필드 플라이", breadcrumb="[용어의 정의] 40. INFIELD FLY (인필드 플라이)",
        content="[용어의 정의] 40. INFIELD FLY (인필드 플라이)\n내야수가 평범한 수비로 포구할 수 있는 뜬공.",
    )]
    found, dropped = extract("이것을 INFIELD FLY 라고 합니다.", docs)
    assert [c.rule_id for c in found] == ["DEF-40"]
    assert dropped == []


def test_no_citation_is_fine() -> None:
    found, dropped = extract("보크는 반칙 투구입니다.", [make_doc()])
    assert found == [] and dropped == []
