"""사용자 프롬프트 파일이 바이트 그대로 쓰이는지 고정한다."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from baseball.prompts import (
    HUMAN_SHA256,
    HUMAN_TEXT,
    NOT_IN_CONTEXT_REFUSAL,
    OFF_TOPIC_REFUSAL,
    PROMPT_DIR,
    SYSTEM_SHA256,
    SYSTEM_TEXT,
    answer_prompt,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_sha256_pins() -> None:
    for name, pin, size in [
        ("system_prompt.txt", SYSTEM_SHA256, 9895),
        ("human_prompt.txt", HUMAN_SHA256, 2483),
    ]:
        raw = (PROMPT_DIR / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin
        assert len(raw) == size


def test_system_has_no_braces() -> None:
    assert SYSTEM_TEXT.count("{") == 0 and SYSTEM_TEXT.count("}") == 0


def test_human_placeholders() -> None:
    assert sorted(set(re.findall(r"{([^}]*)}", HUMAN_TEXT))) == ["context", "question"]


def test_header_lines_kept() -> None:
    assert SYSTEM_TEXT.startswith("#system prompt\n")
    assert HUMAN_TEXT.startswith("#human prompt\n")


def test_system_is_literal_message() -> None:
    p = answer_prompt(8)
    assert isinstance(p.messages[0], SystemMessage)
    assert p.messages[0].content == SYSTEM_TEXT


def test_input_variables() -> None:
    p = answer_prompt(8)
    assert sorted(p.input_variables) == ["context", "question"]
    ph = p.messages[1]
    assert ph.variable_name == "chat_history"
    assert ph.optional is True
    assert ph.n_messages == 8


def test_render_is_verbatim() -> None:
    p = answer_prompt(8)
    msgs = p.format_messages(question="Q", context="C {not_a_var}")
    assert len(msgs) == 2
    assert msgs[0].content == SYSTEM_TEXT
    assert msgs[1].content == HUMAN_TEXT.replace("{question}", "Q").replace("{context}", "C {not_a_var}")

    with_history = p.format_messages(
        question="Q", context="C", chat_history=[HumanMessage(content="a"), AIMessage(content="b")]
    )
    assert len(with_history) == 4
    assert with_history[0].content == SYSTEM_TEXT
    assert [m.content for m in with_history[1:3]] == ["a", "b"]
    assert with_history[3].content == HUMAN_TEXT.replace("{question}", "Q").replace("{context}", "C")


def test_refusal_sentences_in_files() -> None:
    assert OFF_TOPIC_REFUSAL in SYSTEM_TEXT
    assert NOT_IN_CONTEXT_REFUSAL in SYSTEM_TEXT
    assert NOT_IN_CONTEXT_REFUSAL in HUMAN_TEXT
    assert OFF_TOPIC_REFUSAL not in HUMAN_TEXT      # 오프토픽 문장은 system 에만 있다


def test_golden_render() -> None:
    msgs = answer_prompt(8).format_messages(
        question="인필드 플라이가 뭐야?",
        context="[자료 1 | 2026 공식야구규칙 | [용어의 정의] 40. INFIELD FLY (인필드 플라이) | p.201]\n본문 {중괄호} 포함",
        chat_history=[HumanMessage(content="이전 질문"), AIMessage(content="이전 답변")],
    )
    rendered = "\n<<<MSG>>>\n".join(f"{m.type}\n{m.content}" for m in msgs)
    assert rendered == (FIXTURES / "rendered_prompt.golden.txt").read_text(encoding="utf-8")
