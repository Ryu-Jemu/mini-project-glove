#!/usr/bin/env python
"""답변 품질 LLM 심판.

심판 프롬프트는 이 파일 안에만 있다. prompts/ 두 파일은 sha 로 동결되어 있고
답변 생성 전용이므로 평가 지시가 섞이면 안 된다.

router.py 의 실패 관용 스타일을 따른다 — 심판이 죽어도 평가 전체를 멈추지 않고
그 케이스만 verdict=None 으로 남긴다.
"""
from __future__ import annotations

from typing import Any, TypedDict

AXES = ("grounded", "schema_format", "readability", "conciseness")

SYSTEM = """당신은 야구 규칙 안내 답변을 채점하는 평가자다. 답변을 새로 쓰지 말고 채점만 한다.

주어진 [자료]는 답변이 근거로 쓸 수 있었던 전부다. [자료]에 없는 내용은 모두 근거 없는 서술이다.

네 항목을 각각 1~5점으로 매긴다.

grounded (근거성)
5 = 모든 주장이 [자료]로 확인된다
3 = 대체로 확인되나 한두 문장이 [자료]를 넘어선다
1 = [자료]에 없는 내용이 답변의 중심이다

schema_format (형식 준수)
5 = 핵심 답이 맨 앞에 굵게 오고, 적용 상황이 불릿으로 나뉘며, 예시와 이유가 각각 제자리에 있고, 빈 섹션이나 중복이 없다
3 = 구성은 있으나 섹션 하나가 비었거나 내용이 서로 겹친다
1 = 한 문단으로 뭉쳐 있거나 섹션 순서가 뒤엉켰다

readability (초보자 가독성)
5 = 야구를 처음 보는 사람이 한 번에 이해한다. 전문 용어가 나오면 그 자리에서 풀어 준다
3 = 대체로 읽히나 풀이 없는 용어가 남아 있다
1 = 규칙집 문장을 그대로 옮겨 놓아 이해하기 어렵다

conciseness (간결성)
5 = 군더더기가 없다
3 = 같은 말이 두 번 나오거나 불필요하게 길다
1 = 장황해서 핵심이 묻힌다

reason 에는 가장 크게 감점한 이유를 한 문장으로 쓴다. 감점이 없으면 무엇이 좋았는지 한 문장으로 쓴다."""

HUMAN = """[질문]
{question}

[자료]
{context}

[채점할 답변]
{answer}"""


class JudgeVerdict(TypedDict):
    grounded: int
    schema_format: int
    readability: int
    conciseness: int
    reason: str


def build_judge(settings: Any, model: str | None = None) -> Any:
    """구조화 출력을 쓰는 심판 러너블. 키가 없으면 None."""
    if settings.openai_api_key is None:
        return None
    from langchain_openai import ChatOpenAI

    kw: dict[str, Any] = {
        "model": model or settings.openai_chat_model,
        "timeout": 30, "max_retries": 2, "api_key": settings.openai_api_key,
    }
    if str(kw["model"]).startswith("gpt-5."):
        kw["reasoning_effort"] = "none"
    else:
        kw["temperature"] = 0
    llm = ChatOpenAI(**kw).with_config(tags=["judge"], run_name="answer_judge")
    return llm.with_structured_output(JudgeVerdict, method="json_schema", strict=True)


def judge(runner: Any, *, question: str, context: str, answer: str) -> dict[str, Any] | None:
    """채점 결과 dict 또는 None(심판 실패). 실패해도 평가를 멈추지 않는다."""
    if runner is None:
        return None
    messages = [
        ("system", SYSTEM),
        ("human", HUMAN.format(question=question, context=context, answer=answer)),
    ]
    try:
        verdict = runner.invoke(messages)
    except Exception as exc:                      # noqa: BLE001 — 심판 실패는 치명적이지 않다
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    if not isinstance(verdict, dict):
        return {"error": "심판이 dict 를 내지 않았다"}
    out: dict[str, Any] = {"reason": str(verdict.get("reason", ""))[:300]}
    for axis in AXES:
        try:
            out[axis] = max(1, min(5, int(verdict[axis])))
        except (KeyError, TypeError, ValueError):
            return {"error": f"{axis} 점수가 없다"}
    return out


def averages(verdicts: list[dict[str, Any] | None]) -> dict[str, Any]:
    """축별 평균·최저. 실패한 심판은 분모에서 뺀다."""
    scored = [v for v in verdicts if v and "error" not in v]
    out: dict[str, Any] = {"judge_scored": len(scored), "judge_failed": len(verdicts) - len(scored)}
    for axis in AXES:
        values = [v[axis] for v in scored]
        out[f"judge_{axis}_avg"] = round(sum(values) / len(values), 2) if values else None
        out[f"judge_{axis}_min"] = min(values) if values else None
    return out
