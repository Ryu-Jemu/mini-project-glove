"""LangChain @tool 정의 — 체인·API·(Phase 2) 에이전트가 공유하는 단일 진입점."""
from __future__ import annotations

import argparse
import sys
from typing import Any

from langchain_core.tools import tool

from baseball import latest_info
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


@tool(response_format="content_and_artifact")
def web_search(query: str) -> tuple[str, dict[str, Any]]:
    """웹에서 야구 관련 최신 사실을 검색한다.

    사용 시점: 선수·감독·구단·인물의 소속·생년월일·경력, 최근 경기·이적·부상·수상,
        올해의 규정 변경처럼 규칙집에 없거나 시간이 지나면 바뀌는 사실을 물을 때.
        제공된 자료에 규칙 조항만 있고 질문과 맞지 않을 때도 여기로 확인한다.
    사용하지 말 것: 규칙 조항이나 용어의 뜻 자체(보크, 인필드 플라이, 5.09 등).
        그런 질문은 이미 제공된 자료가 답이다.
    한 번에 한 가지 사실만 짧은 한국어 검색어로 찾는다(예: "박해민 소속 구단").

    Args:
        query: 검색어. 질문을 그대로 넣지 말고 찾을 사실을 명사구로 쓴다.
    """
    settings = get_settings()
    entries = latest_info.web_search(query, settings)
    if not entries:
        return ("검색 결과 없음: 관련된 웹 문서를 찾지 못했습니다.", {"sources": [], "entries": []})
    text = format_context([], entries, latest_max_tokens=settings.latest_context_max_tokens)
    # _latest_sources 와 같은 모양이어야 화면·API 가 그대로 받는다. chain 을 import 하면 순환이다.
    sources = [
        {"kind": e.kind, "label": e.label, "url": e.source_url,
         "as_of": e.as_of, "confidence": e.confidence}
        for e in entries
    ]
    header = f"[웹 검색 결과 {len(entries)}건 | 검색어: {query}]"
    return (f"{header}\n\n{text}", {"sources": sources, "entries": list(entries)})


def _when(d: Any) -> str:
    """날짜를 "9월 13일" 로 적는다.

    "9.13" 으로 적으면 citations.RULE_NO_RE 가 규칙 9.13 인용으로 오인한다(실측).
    모델이 본문에 그대로 옮겨 적으면 dropped_citations 가 오염된다.
    """
    return f"{d.month}월 {d.day}일"


def _game_line(game: Any, relation: str) -> str:
    when = _when(game.game_date)
    clock_ = game.game_date_time.strftime("%H:%M")
    place = game.stadium or "구장 미정"
    if relation == "upcoming":
        return f"{when} {clock_} {game.away_name} 대 {game.home_name} ({place})"
    score = f" {game.away_score}-{game.home_score}" if game.status_code == "RESULT" else ""
    return f"{when} {game.away_name} 대 {game.home_name}{score} ({place})"


def _resolve(query: str, settings: Any, *, prefer: str) -> tuple[Any, Any, str, Any]:
    """질의 -> (경기, 구단, 관계, 구장이름). 도구 내부 연쇄의 공통 앞단이다."""
    from baseball import clock, kbo

    team = kbo.find_team(query)
    found = kbo.find_venue(query)
    venue_name = found[0] if found else None
    codes: tuple[str, ...] = ()
    if team is not None:
        codes = (team["code"],)
    elif found is not None:
        codes = found[1]

    snap, _fresh, _as_of = kbo.schedule_snapshot(settings)
    game, relation = kbo.pick_game(
        snap, now=clock.now_kst_naive(), codes=codes, venue=venue_name, prefer=prefer)
    return game, team, relation, venue_name


@tool(response_format="content_and_artifact")
def kbo_schedule_lookup(query: str) -> tuple[str, dict[str, Any]]:
    """KBO 경기 일정을 질문한 날짜(한국 시간) 기준으로 찾는다.

    사용 시점: 다음 경기·가까운 경기·오늘 경기·어제 경기가 언제 어디서 열리는지 물을 때.
        구단명이나 구장 이름이 있으면 그 팀·그 구장 경기를 찾는다.
    사용하지 말 것: 규칙 조항이나 용어의 뜻. 순위·승률(그건 이미 자료에 있다).

    Args:
        query: 구단명이나 구장 이름을 포함한 짧은 한국어 문구(예: "LG 다음 경기").
    """
    from baseball import kbo

    settings = get_settings()
    if not settings.schedule_tool_enabled:
        return ("일정 조회를 쓸 수 없습니다.", {"sources": []})

    game, team, relation, _venue = _resolve(query, settings, prefer="upcoming")
    if game is None:
        return ("다가오는 경기를 찾지 못했습니다. 비시즌이거나 아직 일정이 나오지 않았습니다. "
                "다음 시즌 일정이 공개되면 확인할 수 있다고 안내하세요.",
                {"sources": []})

    from baseball import kbo_naver

    label = "다가오는 경기" if relation == "upcoming" else "가장 최근 끝난 경기"
    text = f"[KBO {label}]\n{_game_line(game, relation)}"
    if team is not None:
        text += f"\n기준 구단: {team['full']} (연고지 {team['hometown']}, 홈구장 {team['stadium']})"
    sources = [{"kind": "kbo", "label": f"KBO 일정 — {label}", "url": kbo_naver.SOURCE_URL,
                "as_of": game.game_date.isoformat(), "confidence": "실시간"}]
    return (text, {"sources": sources, "game_id": game.game_id})


@tool(response_format="content_and_artifact")
def find_restaurants(query: str) -> tuple[str, dict[str, Any]]:
    """경기가 열리는 구장 주변(연고지)의 맛집을 웹에서 찾는다.

    사용 시점: 경기장 근처에서 먹을 곳·맛집·식당을 물을 때. 구장 이름이 없으면
        가까운 경기를 먼저 찾아 그 홈구장 주변으로 찾는다.
    사용하지 말 것: 음식의 조리법이나 야구와 무관한 지역의 맛집.

    Args:
        query: 구장·구단·지역 이름이 있으면 함께 넣는다(예: "잠실 근처 맛집").
    """
    from baseball import kbo, places

    settings = get_settings()
    if not settings.places_enabled:
        return ("맛집 검색을 쓸 수 없습니다.", {"sources": []})

    team = kbo.find_team(query)
    found = kbo.find_venue(query)
    stadium = stadium_short = None
    game = None
    if found is not None:
        stadium_short = found[0]
        code = (team or {}).get("code") or found[1][0]
        info = kbo.teams_by_code().get(code) or {}
        stadium = info.get("stadium") or stadium_short
    elif team is not None:
        stadium, stadium_short = team.get("stadium"), team.get("stadium_short")
    else:
        # 도구 내부 연쇄 — 구장이 안 적혔으면 가까운 경기의 홈구장을 쓴다.
        game, _team, _rel, _v = _resolve(query, settings, prefer="upcoming")
        if game is not None:
            info = kbo.teams_by_code().get(game.home_code) or {}
            stadium, stadium_short = info.get("stadium"), info.get("stadium_short")

    if not stadium or not stadium_short:
        return ("어느 구장 근처인지 알 수 없어 맛집을 찾지 못했습니다. "
                "구단이나 구장 이름을 함께 물어봐 달라고 안내하세요.", {"sources": []})

    entries, cards, media = places.search(
        stadium=stadium, stadium_short=stadium_short, settings=settings)
    if not entries:
        return (f"{stadium_short} 주변 맛집 정보를 찾지 못했습니다.", {"sources": []})

    head = f"[{stadium_short} 주변 맛집 {len(cards)}건]"
    if game is not None:
        head = f"[{_when(game.game_date)} {game.away_name} 대 {game.home_name} 경기 · {head[1:]}"
    text = head + "\n\n" + format_context(
        [], entries, latest_max_tokens=get_settings().latest_context_max_tokens)
    sources = [{"kind": "place", "label": e.label, "url": e.source_url,
                "as_of": e.as_of, "confidence": e.confidence}
               for e in entries if e.source_url]
    return (text, {"sources": sources, "places": cards, "media": media})


@tool(response_format="content_and_artifact")
def find_game_highlight(query: str) -> tuple[str, dict[str, Any]]:
    """특정 경기의 하이라이트 영상을 찾아 채팅에서 재생할 수 있게 한다.

    사용 시점: 경기 하이라이트·다시 보기·명장면 영상을 물을 때.
    사용하지 말 것: 경기 결과나 기록 자체(그건 일정·순위 자료가 답이다).
    찾은 영상의 제목과 주소는 화면에 그대로 표시되므로 본문에 옮겨 적지 않는다.

    Args:
        query: 구단명과 시점을 담은 짧은 한국어 문구(예: "어제 LG 경기 하이라이트").
    """
    from baseball import highlights

    settings = get_settings()
    if not settings.highlights_enabled:
        return ("하이라이트 검색을 쓸 수 없습니다.", {"sources": []})

    game, _team, relation, _venue = _resolve(query, settings, prefer="last_finished")
    if game is None:
        return ("어느 경기의 하이라이트인지 찾지 못했습니다.", {"sources": []})
    if relation != "last_finished":
        return (f"{_when(game.game_date)} 경기는 아직 열리지 않아 하이라이트가 없습니다.",
                {"sources": []})

    # 구단 코드를 함께 넘긴다 — 발견이 그 구단의 유튜브 채널을 고르는 데 쓴다.
    entries, media = highlights.find(
        game_date=game.game_date, home_name=game.home_name, away_name=game.away_name,
        home_code=game.home_code, away_code=game.away_code, settings=settings)
    if not entries:
        return (f"{_game_line(game, relation)} 경기의 YouTube 하이라이트를 찾지 못했습니다. "
                f"찾지 못했다는 사실을 그대로 알리고, 없는 영상을 지어내지 마세요.",
                {"sources": []})

    entry = entries[0]
    sources = [{"kind": "video", "label": entry.label, "url": entry.source_url,
                "as_of": entry.as_of, "confidence": "실시간"}]
    return (f"[{entry.label}]\n{entry.text}", {"sources": sources, "media": media})


# 답변 모델에 붙이는 도구. 규칙집 조회는 이미 턴 앞단에서 끝났다.
# 게이트는 도구마다 따로 본다 — 예전에는 전부 web_search_enabled 하나에 묶여 있어
# Tavily 를 끄면 일정 조회까지 함께 죽었다.
ANSWER_TOOLS = [web_search]
OPTIONAL_TOOLS: tuple[tuple[str, Any], ...] = (
    ("schedule_tool_enabled", kbo_schedule_lookup),
    ("places_enabled", find_restaurants),
    ("highlights_enabled", find_game_highlight),
)
TOOLS = [search_baseball_rules, league_regulation_lookup, kbo_data_lookup, web_search,
         kbo_schedule_lookup, find_restaurants, find_game_highlight]
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
