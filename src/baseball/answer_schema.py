"""답변 구조 스키마. 형식 지시문은 프롬프트가 아니라 여기 description 에 있다.

`prompts/` 두 파일은 sha 로 동결되어 있어 형식 지시를 넣을 수 없다. 대신 OpenAI
`response_format`(JSON Schema)으로 구조를 강제한다. `with_structured_output` 은
메시지를 건드리지 않고 `bind(response_format=...)` 만 하므로 프롬프트 바이트·sha·
골든 픽스처가 그대로 유지된다(tests/test_prompt_untouched_by_schema.py 가 단언).

각 필드의 description 이 곧 모델에게 주는 형식 지시다. 문장을 고치면 답변 형태가
바뀌므로 tests/test_answer_schema.py 의 스냅샷도 함께 갱신해야 한다.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import GenerateJsonSchema

SCHEMA_NAME = "baseball_answer_v1"
ANSWER_KINDS: tuple[str, ...] = ("term_rule", "situation", "entity", "latest")
REFUSAL_KINDS: tuple[str, ...] = ("none", "not_in_context", "off_topic")

AnswerKind = Literal["term_rule", "situation", "entity", "latest"]
RefusalKind = Literal["none", "not_in_context", "off_topic"]

_KIND_DESC = (
    "질문의 지배적인 유형 하나를 고른다. "
    "term_rule = 야구 용어나 규칙의 뜻을 묻는 질문. "
    "situation = 구체적인 경기 장면을 주고 판정이나 결과를 묻는 질문. "
    "entity = 선수·팀·리그의 정보처럼 시간이 지나도 잘 변하지 않는 사실을 묻는 질문. "
    "latest = 규정 변경, 순위, 일정처럼 기준일에 따라 값이 달라지는 정보를 묻는 질문. "
    "시간이 지나면 값이 달라지는 질문은 entity 가 아니라 latest 다. "
    "질문이 여러 개 섞여 있으면 가장 비중이 큰 하나만 여기에 넣고 나머지는 sub_answers 로 보낸다."
)


class _NoTitles(GenerateJsonSchema):
    """필드 title 억제. description 만 남겨 스키마 토큰을 줄인다."""

    def field_title_should_be_set(self, schema: Any) -> bool:  # noqa: ARG002
        return False


class Point(BaseModel):
    """굵은 라벨이 붙는 불릿 한 줄."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description=(
        "불릿 맨 앞에 굵게 표시할 짧은 제목. 2~12자 명사구로 쓰고 문장으로 쓰지 않는다."
    ))
    detail: str = Field(description=(
        "label 을 풀어 쓴 한두 문장. 야구를 처음 보는 사람이 이해할 수 있는 쉬운 말로 쓴다. "
        "줄바꿈, 머리글 기호(#, -, *), 표 기호(|)를 쓰지 않는다."
    ))
    rule_ref: str | None = Field(description=(
        "이 항목의 근거가 되는 규칙 번호나 용어 번호. context 블록에 실제로 있는 값만 쓴다"
        '(예: "5.09", "DEF-40"). 특정할 수 없으면 null.'
    ))


class Fact(BaseModel):
    """이름-값으로 나열하는 사실 한 줄."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description='항목 이름(예: "소속 팀", "순위", "피치클락").')
    value: str = Field(description=(
        "context 에서 그대로 확인되는 값. 숫자를 바꾸거나 반올림하지 않는다."
    ))
    as_of: str | None = Field(description=(
        "이 값의 기준일(YYYY-MM-DD). context 블록의 기준일을 그대로 쓴다. 없으면 null."
    ))


class SubAnswer(BaseModel):
    """복합 질문에서 갈라져 나온 하위 답변."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description=(
        "사용자 질문에서 갈라져 나온 하위 질문을 한 문장으로 다시 쓴 것."
    ))
    kind: AnswerKind = Field(description="이 하위 질문의 유형. 상위 kind 와 같아도 된다.")
    headline: str = Field(description=(
        "이 하위 질문에 대한 핵심 답 한 문장. 40자 이내."
    ))
    points: list[Point] = Field(description=(
        "이 하위 질문을 설명하는 불릿 1~3개. 상위 답변의 points 와 내용이 겹치지 않게 쓴다."
    ))
    evidence: list[str] = Field(description=(
        "이 하위 답변의 근거 라벨. context 에 있는 규칙 번호·용어 번호만 넣는다."
    ))


class AnswerDoc(BaseModel):
    """답변 한 건의 전체 구조. 렌더러가 이걸 마크다운으로 조립한다."""

    model_config = ConfigDict(extra="forbid")

    kind: AnswerKind = Field(description=_KIND_DESC)
    answerable: bool = Field(description=(
        "제공된 context 만으로 질문에 답할 수 있으면 true. context 에 근거가 없거나 "
        "질문이 야구와 무관하면 false. false 이면 headline 은 빈 문자열로 두고 "
        "나머지 서술 필드는 전부 null 또는 빈 배열로 둔다."
    ))
    refusal: RefusalKind = Field(description=(
        'answerable 이 true 면 "none". context 에 근거가 없어 답할 수 없으면 "not_in_context". '
        '질문이 야구와 무관하면 "off_topic". '
        "거부 문장 자체는 어느 필드에도 쓰지 않는다. 이 값만 고르면 프로그램이 정해진 문장을 대신 넣는다."
    ))
    headline: str = Field(description=(
        "질문에 대한 핵심 답을 한 문장으로. 40자 이내. "
        '용어 질문이면 "○○는 …이다" 형태로, 상황 질문이면 판정 이름이 드러나게 쓴다. '
        "이 문장은 화면 맨 위에 굵게 표시되므로 인사말이나 서론을 넣지 않는다."
    ))
    definition: str | None = Field(description=(
        "용어나 규칙의 뜻을 한두 문장으로 쉽게 풀어 쓴다. context 에 적힌 정의만 쓰고, "
        "headline 을 그대로 반복하지 않는다. 해당 내용이 없으면 null."
    ))
    ruling: str | None = Field(description=(
        "kind 가 situation 일 때만 채운다. 그 장면에 적용되는 판정이나 용어 이름만 짧게"
        '(예: "포스 아웃", "인필드 플라이"). 설명은 여기 쓰지 않는다. 그 외 kind 에서는 null.'
    ))
    outcome: str | None = Field(description=(
        "kind 가 situation 일 때만 채운다. 그 판정에 따라 주자·아웃카운트·점수가 어떻게 되는지 "
        "한두 문장. 그 외 kind 에서는 null."
    ))
    points: list[Point] = Field(description=(
        '"어떤 상황에서 적용되는가"를 설명하는 불릿 2~5개. 정의를 되풀이하지 말고 '
        "적용 조건, 예외, 구분 기준을 서로 겹치지 않게 나눠 담는다. "
        "context 에서 확인되는 내용만 쓴다. 담을 내용이 없으면 빈 배열."
    ))
    variations: list[Point] = Field(description=(
        "kind 가 situation 일 때만 채운다. 주자 위치·아웃카운트·점수에 따라 결과가 달라지는 "
        "경우를 0~3개. context 에 그런 설명이 없으면 빈 배열."
    ))
    facts: list[Fact] = Field(description=(
        "kind 가 entity 나 latest 일 때만 채운다. 이름과 값으로 나열할 수 있는 사실 0~6개. "
        "규칙 설명에는 쓰지 않는다."
    ))
    example: str | None = Field(description=(
        "이 규칙이나 판정이 실제 경기에서 어떻게 적용되는지 두세 문장으로 장면을 그려 보여 준다. "
        "context 에 있는 예시만 쓴다. context 에 예시가 없으면 지어내지 말고 null."
    ))
    why: str | None = Field(description=(
        "이 규칙이 왜 있는지, 또는 이 개념을 알면 경기를 볼 때 무엇이 보이는지 한두 문장. "
        "context 에 이유가 적혀 있지 않으면 null."
    ))
    extra_notes: list[str] = Field(description=(
        "알아 두면 좋은 짧은 문장 0~3개. 앞의 내용과 겹치는 문장은 넣지 않는다."
    ))
    caveats: list[str] = Field(description=(
        "자료의 한계를 알려야 할 때만 쓰는 짧은 문장 0~2개"
        "(예: 기준일이 지났을 수 있다, 일부만 확인된다). 해당 없으면 빈 배열."
    ))
    evidence: list[str] = Field(description=(
        "답변 전체의 근거 라벨. context 블록에 실제로 있는 규칙 번호나 용어 번호만 넣는다"
        '(예: "5.09", "DEF-40"). context 에 없는 번호는 절대 넣지 않는다. 없으면 빈 배열.'
    ))
    as_of: str | None = Field(description=(
        "답변 내용의 기준일을 YYYY-MM-DD 로. context 블록의 '기준일' 또는 '기준' 값을 그대로 옮긴다. "
        "규칙집만 근거라면 null."
    ))
    sub_answers: list[SubAnswer] = Field(description=(
        "질문이 여러 개를 한꺼번에 묻고 있을 때만 0~3개. 위의 kind 로 담기지 않은 나머지 질문을 "
        "하나씩 담는다. 질문이 하나면 빈 배열."
    ))

    @property
    def is_refusal(self) -> bool:
        return (not self.answerable) or self.refusal != "none"


def strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI strict json_schema 요구사항을 재귀로 강제한다.

    모든 object 는 additionalProperties=false 이고 required 가 properties 전체여야 한다.
    title 은 정보가 없어 토큰만 먹으므로 제거한다.
    """
    schema.pop("title", None)
    for key in ("$defs", "properties"):
        for sub in schema.get(key, {}).values():
            strictify(sub)
    for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
        for sub in schema.get(key, []):
            strictify(sub)
    if isinstance(schema.get("items"), dict):
        strictify(schema["items"])
    if schema.get("type") == "object" or "properties" in schema:
        props = schema.setdefault("properties", {})
        schema["additionalProperties"] = False
        schema["required"] = list(props)
    return schema


JSON_SCHEMA: dict[str, Any] = strictify(
    AnswerDoc.model_json_schema(schema_generator=_NoTitles)
)
RESPONSE_FORMAT: dict[str, Any] = {
    "name": SCHEMA_NAME,
    "strict": True,
    "schema": JSON_SCHEMA,
}
