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
    issues = check(_doc(headline="   "), [make_doc()])
    assert "EMPTY_HEADLINE" in codes(issues)
    assert blocking(issues) is True


def test_short_headline_warns_but_does_not_block() -> None:
    """"9명입니다" 는 다섯 자지만 완전한 답이다. 이걸 막으면 정답이 거부로 바뀐다."""
    issues = check(_doc(headline="9명입니다"), [make_doc()])
    assert "SHORT_HEADLINE" in codes(issues)
    assert "EMPTY_HEADLINE" not in codes(issues)
    assert blocking(issues) is False


def test_only_two_codes_are_blocking() -> None:
    assert BLOCKING_CODES == {"REFUSAL_INSIDE_ANSWER", "EMPTY_HEADLINE"}
    assert Issue("SHORT_HEADLINE", "x").blocking is False


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


def test_knowledge_only_rejects_every_rule_reference() -> None:
    """근거 문서가 없으면 backing 이 비어 UNBACKED_RULE_REF 가 통째로 꺼진다.

    그 상태에서 규칙 번호를 지어내도 아무도 못 잡는다. 모델 지식 경로의 주된 실패 방식이라
    검사를 뒤집어 인용 자체를 잡는다.
    """
    doc = _doc(points=[_point("라벨", "정상 설명입니다.", "5.09")])
    assert "UNBACKED_RULE_REF" not in codes(check(doc, []))
    assert "UNBACKED_RULE_REF" in codes(check(doc, [], knowledge_only=True))


def test_hedging_is_no_longer_flagged() -> None:
    """프롬프트가 더는 "일반적으로" 를 금지하지 않는다.

    모델 지식으로 답하는 경로가 정식 경로가 된 이상 그 어투는 오히려 정확한 표시다.
    금지하지 않는 것을 린트만 계속 잡으면 경고가 의미를 잃는다.
    """
    doc = _doc(why="일반적으로 그렇게 봅니다.", headline="일반적으로 아홉 명이 뜁니다.")
    assert "HEDGE" not in codes(check(doc, [make_doc()]))
    assert "HEDGE" not in codes(check(doc, [make_doc()], knowledge_only=True))


def test_web_citation_in_evidence_is_not_a_rule_claim() -> None:
    """evidence 는 이제 웹 출처도 담는다. 그걸 규칙 인용으로 세면 오탐이 된다.

    실제로 "박해민 어디 팀 소속이야?" 가 연합뉴스를 근거로 정확히 답했는데도
    UNBACKED_RULE_REF 가 붙었다.
    """
    web_only = {
        "evidence": ["2026-09-07 기준, 박해민은 LG 소속이다. 출처: 연합뉴스"],
        "points": [_point("소속 팀", "박해민은 LG 트윈스에서 뛰고 있습니다."),
                   _point("출처", "연합뉴스 기사로 확인했습니다.")],
    }
    assert "UNBACKED_RULE_REF" not in codes(check(_doc(**web_only), [make_doc()]))
    assert "UNBACKED_RULE_REF" not in codes(check(_doc(**web_only), [], knowledge_only=True))


def test_real_rule_numbers_are_still_checked() -> None:
    assert "UNBACKED_RULE_REF" in codes(check(_doc(evidence=["9.99"]), [make_doc()]))
    assert "UNBACKED_RULE_REF" in codes(check(_doc(evidence=["5.09"]), [], knowledge_only=True))
