"""렌더러 불변식. 형식은 여기서 결정되므로 골든 문자열로 고정한다."""
from __future__ import annotations

import pytest

from baseball import citations as cite
from baseball.answer_render import render, render_blocks
from baseball.answer_schema import AnswerDoc
from baseball.prompts import (
    NOT_IN_CONTEXT_REFUSAL,
    OFF_TOPIC_REFUSAL,
    detect_status,
    has_partial_refusal,
)
from helpers import make_answer_payload, make_doc, make_refusal_payload


def _doc(**kw: object) -> AnswerDoc:
    return AnswerDoc.model_validate(make_answer_payload(**kw))


def _point(label: str, detail: str, rule_ref: str | None = None) -> dict[str, object]:
    return {"label": label, "detail": detail, "rule_ref": rule_ref}


# --- 골든 출력 ---------------------------------------------------------------

def test_term_rule_golden() -> None:
    assert render(_doc()) == (
        "**보크는 투수의 반칙 투구입니다.**\n"
        "\n"
        "투수가 정해진 투구 동작을 어기거나 주자를 속이는 동작을 하면 보크가 선언됩니다.\n"
        "\n"
        "**어떤 상황에서 적용되나요**\n"
        "\n"
        "- **주자가 있을 때** — 누상에 주자가 있을 때만 선언됩니다.\n"
        "- **동작을 멈추면** — 투구 동작을 시작한 뒤 중간에 멈추면 보크입니다.\n"
        "\n"
        "**실제 경기에서는**\n"
        "\n"
        "1루에 주자가 있을 때 투수가 세트포지션에서 어깨를 움직이다 멈추면 보크가 선언됩니다.\n"
        "\n"
        "**왜 중요한가요**\n"
        "\n"
        "주자를 속이는 동작을 막아 주자 쪽이 일방적으로 손해 보지 않게 하려는 규칙입니다.\n"
        "\n"
        "**근거**\n"
        "\n"
        "- 규칙 5.09"
    )


def test_situation_section_order() -> None:
    doc = _doc(
        kind="situation", headline="인필드 플라이입니다.", definition=None,
        ruling="인필드 플라이", outcome="타자는 아웃이고 주자는 그대로 남습니다.",
        points=[_point("무사 또는 1사", "아웃카운트가 2개면 선언하지 않습니다.")],
        variations=[_point("파울이 되면", "선언이 취소됩니다.")],
        example="2루수 앞 뜬공에 심판이 손을 듭니다.", why="병살을 막기 위한 규칙입니다.",
        evidence=["DEF-40"],
    )
    assert [b for b in render_blocks(doc) if b.startswith("**")] == [
        "**인필드 플라이입니다.**", "**판정**", "**왜 이렇게 판정하나요**",
        "**경기는 어떻게 되나요**", "**상황이 달라지면**", "**실제 경기에서는**",
        "**왜 중요한가요**", "**근거**",
    ]
    assert "- 용어의 정의 40" in render(doc)


def test_latest_shows_as_of_and_facts() -> None:
    doc = _doc(
        kind="latest", headline="피치클락은 18초와 23초입니다.", definition=None,
        points=[], example=None, why=None, as_of="2026-09-15", evidence=[],
        facts=[{"label": "주자 없을 때", "value": "18초", "as_of": "2026-09-15"},
               {"label": "주자 있을 때", "value": "23초", "as_of": None}],
    )
    assert render(doc) == (
        "**피치클락은 18초와 23초입니다.**\n"
        "\n"
        "기준일 2026-09-15\n"
        "\n"
        "- **주자 없을 때** — 18초 (2026-09-15 기준)\n"
        "- **주자 있을 때** — 23초"
    )


def test_entity_has_no_as_of_line() -> None:
    doc = _doc(kind="entity", definition=None, points=[], example=None, why=None,
               as_of="2026-09-15", evidence=[],
               facts=[{"label": "소속 팀", "value": "LG 트윈스", "as_of": None}])
    assert "기준일" not in render(doc)


def test_empty_sections_are_omitted() -> None:
    doc = _doc(definition=None, example=None, why=None, points=[], evidence=[])
    assert render(doc) == "**보크는 투수의 반칙 투구입니다.**"


# --- 거부(불변식 1) -----------------------------------------------------------

@pytest.mark.parametrize(
    ("refusal", "expected", "status"),
    [("not_in_context", NOT_IN_CONTEXT_REFUSAL, "not_in_rulebook"),
     ("off_topic", OFF_TOPIC_REFUSAL, "out_of_scope")],
)
def test_refusal_is_byte_exact(refusal: str, expected: str, status: str) -> None:
    doc = AnswerDoc.model_validate(make_refusal_payload(refusal))
    rendered = render(doc)
    assert rendered == expected                      # 부가 문구 0자
    assert detect_status(rendered) == status         # 꼬리 예산 휴리스틱이 무력해진다
    assert has_partial_refusal(rendered) is False


def test_refusal_ignores_filled_fields() -> None:
    """모델이 거부를 고르고도 본문을 채우면 본문은 버린다."""
    doc = _doc(answerable=False, refusal="not_in_context")
    assert render(doc) == NOT_IN_CONTEXT_REFUSAL


# --- 마크다운 안전성 -----------------------------------------------------------

def _rendered_variants() -> list[str]:
    return [render(_doc()), render(_doc(kind="situation", ruling="포스 아웃",
                                        outcome="주자가 아웃됩니다.")),
            render(_doc(extra_notes=["메모"], caveats=["기준일이 지났을 수 있습니다."],
                        sub_answers=[{"question": "그럼 주자는?", "kind": "term_rule",
                                      "headline": "그대로 남습니다.",
                                      "points": [_point("이유", "타자가 이미 아웃입니다.")],
                                      "evidence": ["5.09"]}]))]


@pytest.mark.parametrize("text", _rendered_variants())
def test_every_list_has_a_blank_line_before_it(text: str) -> None:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("- "):
            prev = lines[i - 1]
            assert prev == "" or prev.startswith("- ") or prev.startswith("  - "), (
                f"{i}행 리스트 앞에 빈 줄이 없다: {prev!r}"
            )


@pytest.mark.parametrize("text", _rendered_variants())
def test_no_headings(text: str) -> None:
    assert not any(line.startswith("#") for line in text.split("\n"))


def test_model_markdown_is_neutralized() -> None:
    doc = _doc(definition="## 가짜 제목\n- 가짜 불릿\n| 표 |",
               example="a < b 이고 $OPS$ 가 높다",
               points=[_point("라벨", "> 인용\n\n* 별표")])
    text = render(doc)
    assert "## 가짜 제목" not in text
    assert "| 표 |" not in text
    assert "&lt;" in text and "\\$OPS\\$" in text
    assert "가짜 제목 가짜 불릿" in text          # 한 줄로 접혀 불릿이 살아나지 않았다
    assert text.count("\n- ") == 2              # 렌더러가 만든 불릿만 남는다
    for line in text.split("\n"):
        assert not line.startswith(">")
        assert not line.lstrip().startswith("|")
    # 개행이 전부 공백으로 접히므로 표 구분행(|---|)이 만들어질 수 없다
    assert "---" not in text


# --- citations 연동(불변식 7·8) -------------------------------------------------

def test_evidence_section_feeds_citations() -> None:
    found, dropped = cite.extract(render(_doc()), [make_doc()])
    assert [c.rule_id for c in found] == ["5.09"]
    assert dropped == []


def test_fixed_text_adds_no_phantom_citations() -> None:
    """렌더러 고정 문구에 규칙번호 모양이 없어야 dropped_citations 가 깨끗하다."""
    doc = _doc(evidence=[], points=[_point("라벨", "규칙 번호를 말하지 않는 설명")])
    _, dropped = cite.extract(render(doc), [make_doc()])
    assert dropped == []


def test_evidence_is_deduped_and_labeled() -> None:
    doc = _doc(evidence=["5.09", "5.09", "DEF-40", "기타근거"])
    tail = render(doc).split("**근거**\n\n")[1]
    assert tail == "- 규칙 5.09\n- 용어의 정의 40\n- 기타근거"


# --- 복합 질문 ----------------------------------------------------------------

def test_sub_answers_nest_and_cap_at_three() -> None:
    sub = {"question": "질문", "kind": "term_rule", "headline": "답",
           "points": [_point("라벨", "설명")], "evidence": []}
    text = render(_doc(sub_answers=[dict(sub, question=f"질문{i}") for i in range(5)]))
    assert text.count("  - **라벨**") == 3        # 4번째부터는 버린다
    assert "질문4" not in text


def test_caveats_render_as_separate_blocks() -> None:
    doc = _doc(caveats=["기준일이 지났을 수 있습니다.", "일부만 확인됩니다."])
    assert "\n\n※ 기준일이 지났을 수 있습니다.\n\n※ 일부만 확인됩니다.\n\n" in render(doc)
