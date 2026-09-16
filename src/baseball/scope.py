"""입력 즉시 범위 검증. 야구 외의 질문을 검색·답변 경로에 들이기 전에 막는다.

router.route() 와 나누어 둔 이유는 계약이 다르기 때문이다. route() 는 네 갈래 의도
분류기이고 그 반환 모양은 여러 테스트와 골든셋이 고정하고 있다. 여기 필요한 것은
'야구인가 아닌가' 라는 이진 판정과 확신도뿐이다.

판정은 싼 것부터 본다. 사전에서 결론이 나면 LLM 을 부르지 않는다(대부분이 여기서 끝난다).
애매할 때만 한 번 물어보고, 그 호출이 어떤 이유로든 실패하면 통과시킨다(fail-open).
막는 쪽으로 실패하면 멀쩡한 야구 질문이 조용히 사라지는데, 그게 애초에 고치려던 문제다.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from baseball.config import Settings, get_settings

SCOPE_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "scope_v1.md"

ScopeVerdict = Literal["in_scope", "out_of_scope", "unclear"]
ScopeBy = Literal["rule_no", "lexicon", "negative", "llm", "default", "disabled"]

# 경기의 기본 구조를 묻는 말. router.RULE_RE 와 겹쳐도 무방하다(둘 다 통과 신호다).
BASEBALL_CONCEPT_RE = re.compile(
    r"몇\s*명|인원|정원|선수\s*수|포지션|수비\s*위치|공격\s*(?:과|와)\s*수비"
    r"|경기\s*방법|진행\s*방식|경기의?\s*목적|룰북|규칙집|몇\s*이닝|연장전|경기\s*시간"
)

# 다른 도메인임이 분명한 신호. 긍정 신호가 하나도 없을 때만 본다.
# ("야구 영화 추천" 처럼 야구 단어가 함께 있으면 여기까지 오지 않는다.)
OFF_DOMAIN_RE = re.compile(
    r"날씨|기온|미세먼지|장마|태풍"
    r"|파이썬|자바스크립트|자바|타입스크립트|리액트|알고리즘|프로그래밍|소스\s*코드|깃허브|SQL"
    r"|요리|레시피|맛집|김치찌개|파스타|볶음밥"
    r"|주식|코인|비트코인|환율|부동산|금리|적금"
    r"|번역해|영어로\s*해|일본어로"
    r"|축구|농구|배구|골프|테니스|배드민턴|리그오브레전드|롤드컵|이스포츠"
    r"|다이어트|병원|증상|처방"
    r"|웹툰|아이돌",
    re.I,
)

_MIN_NAME_LEN = 2


@dataclass(frozen=True)
class ScopeResult:
    verdict: ScopeVerdict
    by: ScopeBy
    hits: tuple[str, ...] = ()          # 로그·평가용. 사용자에게 보여 주지 않는다.

    @property
    def blocked(self) -> bool:
        return self.verdict == "out_of_scope"


@lru_cache(maxsize=1)
def _names() -> frozenset[str]:
    """용어집 별칭·구단 별칭·선수 이름. 레코드 전체가 아니라 이름만 올린다(메모리)."""
    from baseball.retriever import load_glossary

    out: set[str] = set()
    try:
        for item in load_glossary():
            out.update(a for a in item.get("aliases", []) if len(a) >= _MIN_NAME_LEN)
            for key in ("term_ko", "term_en"):
                if len(item.get(key, "")) >= _MIN_NAME_LEN:
                    out.add(item[key])
    except Exception:                                        # noqa: BLE001
        pass

    try:
        from baseball.kbo import teams_by_code

        for team in teams_by_code().values():
            out.update(a for a in team.get("aliases", []) if len(a) >= _MIN_NAME_LEN)
            for key in ("full", "short", "stadium", "stadium_short"):
                if len(team.get(key, "")) >= _MIN_NAME_LEN:
                    out.add(team[key])
    except Exception:                                        # noqa: BLE001
        pass

    try:
        path = get_settings().base_dir / "data" / "kbo_players.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("_meta"):
                continue
            name = row.get("name", "")
            if len(name) >= _MIN_NAME_LEN:
                out.add(name)
    except Exception:                                        # noqa: BLE001
        pass

    return frozenset(out)


def _positive_hits(text: str) -> tuple[str, ...]:
    """싼 순서대로 본다. 정규식 먼저, 이름 사전은 마지막."""
    from baseball.latest_info import LEAGUE_ONLY_RE, LIVE_WEB_RE
    from baseball.router import BASEBALL_DOMAIN_RE, KBO_TOPIC_RE, RULE_RE

    hits: list[str] = []
    for label, pattern in (
        ("domain", BASEBALL_DOMAIN_RE), ("rule", RULE_RE),
        ("concept", BASEBALL_CONCEPT_RE), ("league", LEAGUE_ONLY_RE), ("live", LIVE_WEB_RE),
    ):
        m = pattern.search(text)
        if m:
            hits.append(f"{label}:{m.group(0).strip()}")
    for topic, pattern in KBO_TOPIC_RE:
        if pattern.search(text):
            hits.append(f"topic:{topic}")
    if not hits:
        lowered = text.lower()
        for name in _names():
            if name.lower() in lowered:
                hits.append(f"name:{name}")
                break
    return tuple(hits)


def _is_strong(hit: str) -> bool:
    """야구를 특정하는 신호인지, 어느 종목에나 있는 말인지 가른다.

    "몇 명"·"인원"·"경기 방법" 같은 말은 축구에도 농구에도 있다. 이런 신호만으로
    타 도메인 신호를 이길 수는 없다. 반대로 "인필드"·"보크"·"트윈스" 는 야구를 특정한다.
    """
    _, _, value = hit.partition(":")
    return not BASEBALL_CONCEPT_RE.fullmatch(value.strip())


def scope_system_prompt() -> str:
    return SCOPE_PROMPT_PATH.read_text(encoding="utf-8")


def render_scope_messages(question: str) -> list[tuple[str, str]]:
    return [("system", scope_system_prompt()), ("human", f"질문: {question}")]


def build_scope_llm(settings: Settings | None = None) -> Any:
    from langchain_openai import ChatOpenAI

    settings = settings or get_settings()
    llm = ChatOpenAI(
        model=settings.openai_chat_model, temperature=0, max_tokens=16,
        timeout=8, max_retries=1, api_key=settings.openai_api_key,
    )
    return llm.with_structured_output(
        {"type": "object", "properties": {"in_scope": {"type": "boolean"}},
         "required": ["in_scope"], "additionalProperties": False},
        method="json_schema", strict=True,
    )


def _parse_in_scope(out: Any) -> bool | None:
    if isinstance(out, dict) and isinstance(out.get("in_scope"), bool):
        return out["in_scope"]
    content = getattr(out, "content", out)
    try:
        value = json.loads(content)["in_scope"]
    except Exception:                                        # noqa: BLE001
        return None
    return value if isinstance(value, bool) else None


def check(
    question: str, *, llm: Any | None = None, settings: Settings | None = None
) -> ScopeResult:
    """야구 질문인지 판정한다. 확신이 없으면 통과시킨다."""
    settings = settings or get_settings()
    if settings.enable_scope_gate == "off":
        return ScopeResult("in_scope", "disabled")

    text = unicodedata.normalize("NFKC", question or "")

    from baseball.retriever import RULE_NO_RE

    m = RULE_NO_RE.search(text)
    if m:
        return ScopeResult("in_scope", "rule_no", (f"rule_no:{m.group(0)}",))

    # 타 도메인 표현을 먼저 가린 뒤에 긍정 신호를 찾는다. 가리지 않으면 "리그오브레전드" 안의
    # '리그' 같은 부분 문자열이 야구 신호로 잡힌다. 가려도 "야구랑 축구 중 뭐가 재밌어?" 는
    # '야구' 가 남으므로 통과한다.
    off = OFF_DOMAIN_RE.search(text)
    masked = OFF_DOMAIN_RE.sub(" ", text) if off else text

    hits = _positive_hits(masked)
    strong = tuple(h for h in hits if _is_strong(h))
    if strong:
        return ScopeResult("in_scope", "lexicon", strong)

    if off:
        # 남은 긍정 신호가 종목을 가리지 않는 말뿐이다("축구 경기 몇 명이서 해?").
        return ScopeResult("out_of_scope", "negative", (f"off:{off.group(0)}",))

    if hits:
        return ScopeResult("in_scope", "lexicon", hits)

    # 여기까지 왔으면 사전으로는 모른다. 물어볼 수 있을 때만 물어본다.
    if llm is None:
        if settings.scope_gate_llm == "off" or settings.openai_api_key is None:
            return ScopeResult("in_scope", "default")
        try:
            llm = build_scope_llm(settings)
        except Exception:                                    # noqa: BLE001
            return ScopeResult("in_scope", "default")
    try:
        decided = _parse_in_scope(llm.invoke(render_scope_messages(text)))
    except Exception:                                        # noqa: BLE001
        decided = None
    if decided is None:
        return ScopeResult("in_scope", "default")
    return ScopeResult("in_scope" if decided else "out_of_scope", "llm")
