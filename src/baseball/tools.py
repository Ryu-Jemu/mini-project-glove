"""LangChain @tool 정의 — 체인·API·(Phase 2) 에이전트가 공유하는 단일 진입점."""
from __future__ import annotations

import argparse
import sys
from typing import Any

from langchain_core.tools import tool

from baseball.config import get_settings
from baseball.context import format_context
from baseball.latest_info import match_snapshot

_retriever: Any | None = None


def get_retriever() -> Any:
    global _retriever
    if _retriever is None:
        from baseball.retriever import build_retriever

        _retriever = build_retriever(get_settings())
    return _retriever


@tool(response_format="content_and_artifact")
def search_baseball_rules(query: str, k: int = 6) -> tuple[str, dict[str, Any]]:
    """2026 공식야구규칙(PDF)에서 질문과 관련된 조항·용어를 검색한다.

    사용 시점: 야구 규칙·용어·판정·경기 상황에 대한 질문일 때.
    사용하지 말 것: 순위·일정·선수 명단·하이라이트처럼 시간에 따라 변하는 정보.

    Args:
        query: 사용자의 질문 원문.
        k: 반환할 조항 수(기본 6).
    """
    res = get_retriever().retrieve(query, k=k)
    text = format_context(res.docs, [])
    artifact = {
        "sources": [
            {"kind": "static", "label": d["breadcrumb"], "page": d["page_start"], "rule_id": d["rule_id"]}
            for d in res.docs
        ],
        "retrieved_ids": [d["id"] for d in res.docs],
        "abstain": res.abstain,
    }
    return (text or "규칙집에서 관련 조항을 찾지 못했습니다.", artifact)


@tool(response_format="content_and_artifact")
def league_regulation_lookup(query: str) -> tuple[str, dict[str, Any]]:
    """KBO 리그 운영 규정 스냅샷(ABS·피치클락·엔트리 등)을 조회한다.

    사용 시점: 공식야구규칙에는 없는 KBO 리그 운영 규정(ABS, 피치클락, 체크스윙 판독,
        엔트리, 아시아쿼터, 포스트시즌 방식)을 물을 때.
    사용하지 말 것: 규칙집 본문 조항(5.09 등)이나 실시간 순위·일정.

    Args:
        query: 사용자의 질문 원문.
    """
    entries = match_snapshot(query)
    if not entries:
        return ("해당 리그 규정 스냅샷을 찾지 못했습니다.", {"sources": [], "retrieved_ids": []})
    text = format_context([], entries)
    artifact = {
        "sources": [
            {"kind": "snapshot", "label": e.label, "as_of": e.as_of, "url": e.source_url}
            for e in entries
        ],
        "retrieved_ids": [e.label for e in entries],
    }
    return (text, artifact)



@tool(response_format="content_and_artifact")
def kbo_data_lookup(query: str) -> tuple[str, dict[str, Any]]:
    """KBO 리그의 순위·승률·남은 경기 일정·구단 연고지와 홈구장을 조회한다.

    사용 시점: "LG 순위", "두산 남은 경기", "기아 연고지" 처럼 현재 시즌의
        구단 성적·일정·기본 정보를 묻는 질문.
    사용하지 말 것: 규칙집 조항(5.09 등)이나 리그 운영 규정(ABS·피치클락)은
        search_baseball_rules 를 쓴다.
    Args:
        query: 사용자 질문 원문.
    """
    from baseball import kbo
    from baseball.context import format_context
    from baseball.router import kbo_topics

    topics, teams = kbo_topics(query)
    if not topics:
        return "KBO 데이터로 답할 수 있는 질문이 아닙니다.", {"sources": [], "retrieved_ids": []}

    class _R:
        pass

    r = _R()
    r.topics, r.teams = topics, teams
    entries = kbo.entries_for(query, r)
    if not entries:
        return "KBO 데이터를 가져오지 못했습니다.", {"sources": [], "retrieved_ids": []}
    text = format_context([], [], entries)
    sources = [{"kind": "kbo", "label": e.label, "url": e.source_url, "as_of": e.as_of}
               for e in entries]
    return text, {"sources": sources, "retrieved_ids": [e.label for e in entries]}


TOOLS = [search_baseball_rules, league_regulation_lookup, kbo_data_lookup]
BY_NAME = {t.name: t for t in TOOLS}


def demo(question: str = "인필드 플라이가 뭐야?") -> int:
    """강의 4단계 루프: bind_tools → tool_calls 분기 → ToolMessage → 재호출."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0, api_key=settings.openai_api_key)
    bound = llm.bind_tools(TOOLS)
    print(f"bind_tools {len(TOOLS)}: {[t.name for t in TOOLS]}")

    messages: list[Any] = [HumanMessage(content=question)]
    ai = bound.invoke(messages)
    if ai.tool_calls:                                   # 흔한 실수: tool_calls 가 빈 리스트일 수 있다
        messages.append(ai)
        for call in ai.tool_calls:
            print(f"  tool_call {call['name']} args={call['args']}")
            messages.append(BY_NAME[call["name"]].invoke(call))
        final = bound.invoke(messages)
        print("--- 최종 답변(도구 경유) ---")
        print((final.content or "")[:400])
    else:
        print("--- 도구 없이 직접 답변 ---")
        print((ai.content or "")[:400])
    return 0


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m baseball.tools")
    ap.add_argument("command", choices=["demo", "list"])
    ap.add_argument("--question", default="인필드 플라이가 뭐야?")
    args = ap.parse_args(argv)
    if args.command == "list":
        for t in TOOLS:
            print(f"{t.name}: {(t.description or '').splitlines()[0]}")
        return 0
    return demo(args.question)


if __name__ == "__main__":
    sys.exit(_main())
