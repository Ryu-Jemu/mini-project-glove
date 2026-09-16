"""답변 스키마가 OpenAI strict json_schema 요구사항을 만족하는지 고정한다."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from baseball.answer_schema import (
    ANSWER_KINDS,
    JSON_SCHEMA,
    REFUSAL_KINDS,
    RESPONSE_FORMAT,
    SCHEMA_NAME,
    AnswerDoc,
    strictify,
)
from helpers import make_answer_payload, make_refusal_payload


def _walk(node: object, path: str = "$") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            props, req = set(node.get("properties", {})), set(node.get("required", []))
            if props != req:
                bad.append(f"{path}: required != properties ({props ^ req})")
            if node.get("additionalProperties") is not False:
                bad.append(f"{path}: additionalProperties is not False")
        if "title" in node:
            bad.append(f"{path}: title 이 남아 있다(토큰 낭비)")
        for k, v in node.items():
            bad += _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad += _walk(v, f"{path}[{i}]")
    return bad


def test_schema_is_strict_everywhere() -> None:
    assert _walk(JSON_SCHEMA) == []


def test_root_is_object() -> None:
    # OpenAI Structured Outputs 는 루트가 object 여야 한다(루트 anyOf 는 400).
    assert JSON_SCHEMA["type"] == "object"


def test_response_format_shape() -> None:
    assert RESPONSE_FORMAT["name"] == SCHEMA_NAME
    assert RESPONSE_FORMAT["strict"] is True
    assert RESPONSE_FORMAT["schema"] is JSON_SCHEMA


def test_every_field_has_a_description() -> None:
    """description 이 곧 형식 지시문이다. 빈 필드가 있으면 그 필드는 통제되지 않는다."""
    missing = [
        f"{owner}.{name}"
        for owner, node in [("AnswerDoc", JSON_SCHEMA), *JSON_SCHEMA["$defs"].items()]
        for name, prop in node["properties"].items()
        if not prop.get("description")
    ]
    assert missing == []


def test_no_defaults_leak_into_schema() -> None:
    """default 가 있으면 '모델이 필드를 빠뜨렸다'를 검증으로 못 잡는다."""
    assert "default" not in json.dumps(JSON_SCHEMA)


def test_kind_enums_match_constants() -> None:
    assert JSON_SCHEMA["properties"]["kind"]["enum"] == list(ANSWER_KINDS)
    assert JSON_SCHEMA["properties"]["refusal"]["enum"] == list(REFUSAL_KINDS)


def test_schema_stays_within_openai_limits() -> None:
    blob = json.dumps(RESPONSE_FORMAT, ensure_ascii=False)
    assert len(blob) < 15_000                       # 문자열 총량 상한
    assert len(JSON_SCHEMA["properties"]) <= 100    # 속성 수 상한


def test_valid_payload_round_trips() -> None:
    doc = AnswerDoc.model_validate(make_answer_payload())
    assert doc.kind == "term_rule"
    assert doc.is_refusal is False
    assert [p.label for p in doc.points] == ["주자가 있을 때", "동작을 멈추면"]


@pytest.mark.parametrize("refusal", ["not_in_context", "off_topic"])
def test_refusal_payload_is_flagged(refusal: str) -> None:
    assert AnswerDoc.model_validate(make_refusal_payload(refusal)).is_refusal is True


def test_answerable_false_alone_is_a_refusal() -> None:
    """refusal 을 none 으로 둔 채 answerable 만 내리는 모델도 거부로 취급한다."""
    payload = make_answer_payload(answerable=False, headline="")
    assert AnswerDoc.model_validate(payload).is_refusal is True


def test_missing_field_is_rejected() -> None:
    payload = make_answer_payload()
    del payload["why"]
    with pytest.raises(ValidationError):
        AnswerDoc.model_validate(payload)


def test_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AnswerDoc.model_validate(make_answer_payload(surprise="x"))


def test_strictify_is_idempotent() -> None:
    once = strictify(json.loads(json.dumps(JSON_SCHEMA)))
    assert strictify(json.loads(json.dumps(once))) == once
