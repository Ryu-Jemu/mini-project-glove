"""사용자 프롬프트 파일을 바이트 그대로 사용한다. 수정·재구성 금지."""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
SYSTEM_SHA256 = "13a28529f2539dd3577aa71290084108e2ec392cd4f374d933c65ec5b0c18980"
HUMAN_SHA256 = "718e8ea13c330d9247affd12c14eaa140049f80f16d79bc915d6b35e3a2ac705"

OFF_TOPIC_REFUSAL = "죄송하지만 저는 야구와 관련된 질문에 대해서만 답변할 수 있습니다."
NOT_IN_CONTEXT_REFUSAL = "죄송하지만 제공된 야구 자료에서는 해당 내용을 확인할 수 없습니다."

_OFF_TOPIC_RE = re.compile(r"죄송하지만\s*저는\s*야구와\s*관련된\s*질문에\s*대해서만\s*답변할\s*수\s*있습니다\s*\.?")
_NOT_IN_CONTEXT_RE = re.compile(r"죄송하지만\s*제공된\s*야구\s*자료에서는\s*해당\s*내용을\s*확인할\s*수\s*없습니다\s*\.?")
_NOISE_RE = re.compile(r"[\s\"'“”‘’.。!?,\-–—*_`>#:()\[\]]+")

Status = Literal["answered", "not_in_rulebook", "out_of_scope"]


def read_verbatim(name: str, expected_sha256: str) -> str:
    text = (PROMPT_DIR / name).read_text(encoding="utf-8")   # strip/dedent/replace 금지
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if got != expected_sha256:
        raise RuntimeError(f"{name} modified: sha256={got[:8]} expected={expected_sha256[:8]}")
    return text


SYSTEM_TEXT = read_verbatim("system_prompt.txt", SYSTEM_SHA256)
HUMAN_TEXT = read_verbatim("human_prompt.txt", HUMAN_SHA256)
PROMPT_SHA = f"{SYSTEM_SHA256[:8]}-{HUMAN_SHA256[:8]}"       # "13a28529-718e8ea1"


@lru_cache(maxsize=4)
def answer_prompt(history_max_messages: int = 8) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_TEXT),                                   # 리터럴 → 템플릿 파싱 없음
        MessagesPlaceholder("chat_history", optional=True, n_messages=history_max_messages),
        ("human", HUMAN_TEXT),                                                # {question}, {context}
    ])


def detect_status(answer: str, *, tail_max_chars: int = 40) -> Status:
    text = unicodedata.normalize("NFKC", answer or "")
    if not text.strip():
        return "not_in_rulebook"
    if _OFF_TOPIC_RE.search(text):
        rest = _NOISE_RE.sub("", _OFF_TOPIC_RE.sub("", text))
        return "out_of_scope" if len(rest) <= tail_max_chars else "answered"
    if _NOT_IN_CONTEXT_RE.search(text):
        rest = _NOISE_RE.sub("", _NOT_IN_CONTEXT_RE.sub("", text))
        return "not_in_rulebook" if len(rest) <= tail_max_chars else "answered"
    return "answered"


def has_partial_refusal(answer: str) -> bool:
    text = unicodedata.normalize("NFKC", answer or "")
    return detect_status(text) == "answered" and bool(
        _NOT_IN_CONTEXT_RE.search(text) or _OFF_TOPIC_RE.search(text)
    )


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m baseball.prompts")
    ap.add_argument("command", choices=["verify", "tokens"])
    args = ap.parse_args(argv)

    if args.command == "verify":
        p = answer_prompt(8)
        print(f"system sha256 OK ({SYSTEM_SHA256[:8]})")
        print(f"human sha256 OK ({HUMAN_SHA256[:8]})")
        print(f"braces_in_system={SYSTEM_TEXT.count('{') + SYSTEM_TEXT.count('}')}")
        print(f"system_is_literal={isinstance(p.messages[0], SystemMessage)}")
        print(f"input_variables={sorted(p.input_variables)}")
        ph = p.messages[1]
        print(f"placeholder={ph.variable_name} optional={ph.optional} n_messages={ph.n_messages}")
        return 0

    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    s, h = len(enc.encode(SYSTEM_TEXT)), len(enc.encode(HUMAN_TEXT))
    print(f"system={s} human={h} fixed_overhead={s + h} (o200k_base)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
