"""구조화 출력이 프롬프트를 건드리지 않음을 기계적으로 증명한다.

설계서 금지 목록은 '시스템 메시지 앞 가변 텍스트 삽입'과 '두 번째 시스템 프롬프트로
답변 생성'을 막는다. response_format 은 메시지가 아니라 요청 파라미터이므로 여기에
걸리지 않는다 — 다만 그 주장을 말로 하지 않고 이 파일로 단언한다.
"""
from __future__ import annotations

import json
from typing import Any

from baseball.answer_schema import RESPONSE_FORMAT, SCHEMA_NAME
from baseball.chain import RagService, build_structured_llm
from baseball.config import Settings
from baseball.prompts import HUMAN_TEXT, SYSTEM_TEXT
from helpers import make_doc

# 도구는 response_format 과 같은 요청 파라미터다. 메시지를 만들거나 바꾸지 않으므로
# 프롬프트 조작이 아니다. 메시지 관련 키가 들어오면 여전히 실패해야 한다.
ALLOWED_BIND_KEYS = {
    "response_format", "ls_structured_output_format", "max_tokens",
    "tools", "parallel_tool_calls", "tool_choice",
}


def _messages(schema_on: bool, fake_retriever: Any) -> list[Any]:
    settings = Settings(enable_answer_schema="on" if schema_on else "off")
    service = RagService(settings, fake_retriever)
    prepared = service.prepare("보크가 뭐야?", model=settings.openai_chat_model)
    return service._render_messages(prepared, question="보크가 뭐야?", session_id=None)


def test_rendered_prompt_is_identical_with_schema_on_and_off(fake_retriever) -> None:
    on = [(m.type, m.content) for m in _messages(True, fake_retriever)]
    off = [(m.type, m.content) for m in _messages(False, fake_retriever)]
    assert on == off
    assert on[0][1] == SYSTEM_TEXT          # 시스템 메시지는 바이트 그대로다


def test_schema_text_never_reaches_the_messages(fake_retriever) -> None:
    blob = "\n".join(str(m.content) for m in _messages(True, fake_retriever))
    assert SCHEMA_NAME not in blob
    for field in RESPONSE_FORMAT["schema"]["properties"]:
        assert f'"{field}"' not in blob
    # description 문장이 프롬프트로 새지 않았는지도 확인한다
    sample = RESPONSE_FORMAT["schema"]["properties"]["headline"]["description"][:20]
    assert sample not in blob


def test_human_prompt_still_has_exactly_two_placeholders() -> None:
    assert HUMAN_TEXT.count("{") == 2 and HUMAN_TEXT.count("}") == 2


def _bound_kwargs(runnable: Any, seen: set[int] | None = None) -> list[set[str]]:
    seen = seen if seen is not None else set()
    if id(runnable) in seen:
        return []
    seen.add(id(runnable))
    found: list[set[str]] = []
    kwargs = getattr(runnable, "kwargs", None)
    if isinstance(kwargs, dict) and kwargs:
        found.append(set(kwargs))
    for attr in ("bound", "first", "last", "default"):
        child = getattr(runnable, attr, None)
        if child is not None:
            found += _bound_kwargs(child, seen)
    for attr in ("steps", "steps__"):
        steps = getattr(runnable, attr, None)
        if isinstance(steps, dict):
            for child in steps.values():
                found += _bound_kwargs(child, seen)
        elif isinstance(steps, list):
            for child in steps:
                found += _bound_kwargs(child, seen)
    return found


def test_structured_llm_binds_only_request_parameters() -> None:
    """메시지 관련 키를 bind 하면 그건 프롬프트 조작이다. 없어야 한다."""
    settings = Settings(enable_answer_schema="on")
    bound = _bound_kwargs(build_structured_llm(settings.openai_chat_model, settings))
    assert bound, "bind 된 kwargs 를 찾지 못했다 — 테스트가 구조를 놓치고 있다"
    for keys in bound:
        assert keys <= ALLOWED_BIND_KEYS, f"허용되지 않은 bind 키: {keys - ALLOWED_BIND_KEYS}"


def test_response_format_is_a_request_parameter_not_a_message() -> None:
    blob = json.dumps(RESPONSE_FORMAT, ensure_ascii=False)
    assert SYSTEM_TEXT not in blob and HUMAN_TEXT not in blob
