"""린터. blocking 은 화면을 망가뜨리는 두 가지뿐이어야 한다."""
from __future__ import annotations

from baseball.answer_lint import BLOCKING_CODES, Issue, blocking, check, codes
from baseball.answer_schema import AnswerDoc
from baseball.prompts import NOT_IN_CONTEXT_REFUSAL
from helpers import make_answer_payload, make_doc, make_refusal_payload


def _doc(**kw: object) -> AnswerDoc:
    return AnswerDoc.model_validate(make_answer_payload(**kw))


def _point(label: str, detail: str, rule_ref: str | None = None) -> dict[str, object]:
    return {"label": label, "detail": detail, "rule_ref": rule_ref}


def test_clean_answer_has_no_issues() -> None:
    assert check(_doc(), [make_doc()]) == []


def test_refusal_is_not_linted() -> None:
    doc = AnswerDoc.model_validate(make_refusal_payload())
    assert check(doc, [make_doc()]) == []


# --- blocking ----------------------------------------------------------------

def test_refusal_sentence_inside_answer_blocks() -> None:
    issues = check(_doc(why=NOT_IN_CONTEXT_REFUSAL), [make_doc()])
    assert "REFUSAL_INSIDE_ANSWER" in codes(issues)
    assert blocking(issues) is True


def test_refusal_sentence_is_found_in_nested_points() -> None:
    doc = _doc(points=[_point("라벨", NOT_IN_CONTEXT_REFUSAL, "5.09"),
                       _point("다른 라벨", "정상 설명입니다.", "5.09")])
    assert "REFUSAL_INSIDE_ANSWER" in codes(check(doc, [make_doc()]))


def test_empty_headline_blocks() -> None:
    issues = check(_doc(headline="짧다"), [make_doc()])
    assert "EMPTY_HEADLINE" in codes(issues)
    assert blocking(issues) is True


def test_only_two_codes_are_blocking() -> None:
    assert BLOCKING_CODES == {"REFUSAL_INSIDE_ANSWER", "EMPTY_HEADLINE"}
    assert Issue("HEDGE", "x").blocking is False


# --- 경고 --------------------------------------------------------------------

def test_markdown_injection_is_warned_not_blocked() -> None:
    issues = check(_doc(definition="## 제목"), [make_doc()])
    assert "MARKDOWN_INJECTION" in codes(issues)
    assert blocking(issues) is False


def test_stub_bullet() -> None:
    assert "STUB_SECTION" in codes(check(_doc(points=[
        _point("라벨", "짧음", "5.09"), _point("또", "또", "5.09")]), [make_doc()]))


def test_bullet_count_out_of_range() -> None:
    assert "BULLET_COUNT" in codes(check(_doc(points=[]), [make_doc()]))
    many = [_point(f"라벨{i}", f"충분히 긴 설명입니다 {i}", "5.09") for i in range(6)]
    assert "BULLET_COUNT" in codes(check(_doc(points=many), [make_doc()]))


def test_hedge_phrases_forbidden_by_the_system_prompt() -> None:
    assert "HEDGE" in codes(check(_doc(why="일반적으로 그렇게 봅니다."), [make_doc()]))


def test_unbacked_reference() -> None:
    issues = check(_doc(evidence=["9.99"]), [make_doc()])
    assert "UNBACKED_RULE_REF" in codes(issues)
    assert blocking(issues) is False


def test_sub_item_reference_counts_as_backed() -> None:
    assert check(_doc(evidence=["5.09(a)"]), [make_doc()]) == []


def test_no_retrieved_docs_means_no_reference_check() -> None:
    """게이트가 아닌데 검색 결과가 비는 경로(최신정보 전용)에서 오탐하지 않는다."""
    assert "UNBACKED_RULE_REF" not in codes(check(_doc(evidence=["9.99"]), []))


def test_kind_field_mismatch_both_directions() -> None:
    assert "KIND_FIELD_MISMATCH" in codes(check(_doc(ruling="포스 아웃"), [make_doc()]))
    situation = _doc(kind="situation", ruling=None, outcome="주자가 아웃됩니다.")
    assert "KIND_FIELD_MISMATCH" in codes(check(situation, [make_doc()]))


def test_facts_outside_entity_or_latest() -> None:
    doc = _doc(facts=[{"label": "소속", "value": "LG", "as_of": None}])
    assert "KIND_FIELD_MISMATCH" in codes(check(doc, [make_doc()]))


def test_latest_without_as_of() -> None:
    doc = _doc(kind="latest", points=[], as_of=None,
               facts=[{"label": "피치클락", "value": "18초", "as_of": None}])
    assert "LATEST_NO_AS_OF" in codes(check(doc, [make_doc()]))


def test_duplicate_headline_and_definition() -> None:
    doc = _doc(definition="보크는 투수의 반칙 투구입니다.")
    assert "DUP_TEXT" in codes(check(doc, [make_doc()]))


def test_too_long() -> None:
    doc = _doc(why="길다. " * 700)
    assert "TOO_LONG" in codes(check(doc, [make_doc()]))
