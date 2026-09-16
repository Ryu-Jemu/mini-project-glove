"""의도 라우터: 키워드 표(정본) → 미매칭 시에만 LLM(router_v1.md). off_topic 은 LLM 만 산출."""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from baseball.config import Settings, get_settings
from baseball.latest_info import LEAGUE_ONLY_RE, LIVE_WEB_RE

RouteKind = Literal["rule", "latest", "mixed", "off_topic"]
VALID_KINDS: tuple[str, ...] = ("rule", "latest", "mixed", "off_topic")

ROUTER_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "router_v1.md"


class RouteTD(TypedDict):
    kind: RouteKind


RULE_RE = re.compile(
    # 규칙 번호(5.09 등)가 있으면 항상 규칙집 질문이다
    r"(?<![\d.])[1-9]\.\d{2}(?![\d.])|"
    r"규칙|룰|판정|아웃|세이프|스트라이크|볼넷|인필드|보크|용어|뜻|무엇|뭐야|어떻게\s*되"
    r"|타점|도루|번트|플라이|홈런|주루|파울|병살|낫아웃|담장|스트라이크\s*존|포스\s*아웃"
    # 규칙집 어휘(야구 단어가 없어 LLM 라우터가 오분류하는 것을 막는다)
    r"|투수|타자|주자|포수|내야|외야|심판|구장|베이스|배트|글러브|유니폼|마운드|덕아웃"
    r"|이닝|타석|투구|송구|포구|득점|안타|만루|번트|누상|타순|대타|대주자|몰수|라인"
    # 경기의 기본 구조를 묻는 말. 이게 없어서 "야구 몇 명이서 플레이해?" 처럼 규칙집 1.00 에
    # 답이 있는 질문이 '야구'라는 도메인 단어 하나만 걸려 최신정보 경로로 새 나갔다.
    r"|몇\s*명|인원|정원|선수\s*수|포지션|수비\s*위치|경기\s*방법|진행\s*방식"
    r"|경기의?\s*목적|룰북|규칙집|몇\s*이닝|연장전|플레이"
)

# 야구 도메인 고유명사(구단·리그). 이 단어가 있으면 오프토픽일 수 없다.
BASEBALL_DOMAIN_RE = re.compile(
    r"야구|KBO|MLB|프로야구|메이저리그|리그|구단|연고지|홈구장|선수|감독|코치|시즌|우승|한국시리즈"
    r"|트윈스|베어스|히어로즈|랜더스|위즈|이글스|라이온즈|자이언츠|타이거즈|다이노스"
    r"|두산|키움|SSG|한화|삼성|롯데|기아|KIA|엘지|엔씨|케이티|쓱",
    re.I,
)
LATEST_RE = re.compile(f"(?:{LEAGUE_ONLY_RE.pattern})|(?:{LIVE_WEB_RE.pattern})", re.I)
KEYWORDS_P1: dict[str, re.Pattern[str]] = {"rule": RULE_RE, "latest": LATEST_RE}


KboTopic = Literal["standings", "schedule", "team_info", "roster"]

KBO_TOPIC_RE: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("standings", re.compile(r"순위|승률|몇\s*위|선두|1위|꼴찌|게임\s*차|순위표|가을야구")),
    ("schedule", re.compile(
        r"일정|남은\s*경기|잔여\s*경기|다음\s*경기|경기\s*일정|몇\s*경기\s*남"
        r"|오늘.{0,8}경기|내일.{0,8}경기|언제\s*경기|경기\s*언제")),
    ("team_info", re.compile(r"연고지|본거지|홈\s*구장|구장|창단|구단\s*정보|어느\s*도시|어디\s*연고")),
    ("roster", re.compile(
        r"명단|로스터|선수단|엔트리|주요\s*선수|투수진|타선|불펜|선발진"
        r"|타율\s*1위|홈런\s*1위|누가\s*뛰")),
)
MAX_TOPICS = 2


@lru_cache(maxsize=1)
def _team_aliases() -> tuple[tuple[str, str], ...]:
    """(별칭, 구단코드) 를 긴 별칭부터. 파일이 없으면 빈 튜플."""
    try:
        from baseball.kbo import teams_by_code

        pairs = [(a, code) for code, t in teams_by_code().items() for a in t.get("aliases", [])]
    except Exception:                              # noqa: BLE001
        return ()
    return tuple(sorted(pairs, key=lambda p: -len(p[0])))


def kbo_topics(question: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """질문에서 KBO 주제와 구단 코드를 뽑는다. kind 결정에는 관여하지 않는다."""
    t = unicodedata.normalize("NFKC", question or "")
    topics = tuple(name for name, rx in KBO_TOPIC_RE if rx.search(t))[:MAX_TOPICS]
    teams: list[str] = []
    upper = t.upper()
    for alias, code in _team_aliases():
        if code in teams:
            continue
        if alias.upper() in upper:
            teams.append(code)
    return topics, tuple(teams)


@dataclass(frozen=True)
class Route:
    kind: RouteKind
    by: Literal["keyword", "llm"]
    rule_hit: bool
    latest_hit: bool
    domain_hit: bool = False
    # KBO 데이터 태그. kind 와 직교한다 — kind 는 네 값 그대로이고
    # to_dict() 도 두 키 그대로라 기존 계약(API 응답·골든셋)이 바뀌지 않는다.
    topics: tuple[KboTopic, ...] = ()
    teams: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "by": self.by}


def router_system_prompt() -> str:
    return ROUTER_PROMPT_PATH.read_text(encoding="utf-8")


def render_router_messages(question: str) -> list[tuple[str, str]]:
    return [("system", router_system_prompt()), ("human", f"질문: {question}")]


def build_router_llm(settings: Settings | None = None) -> Any:
    from langchain_openai import ChatOpenAI

    settings = settings or get_settings()
    llm = ChatOpenAI(
        model=settings.openai_chat_model, temperature=0, max_tokens=80,
        timeout=20, max_retries=2, api_key=settings.openai_api_key,
    )
    return llm.with_structured_output(RouteTD, method="json_schema", strict=True)


def _parse_kind(out: Any) -> str:
    if isinstance(out, dict):
        return str(out.get("kind", ""))
    content = getattr(out, "content", out)
    try:
        return str(json.loads(content)["kind"])
    except Exception:
        return ""



def _tags(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        return kbo_topics(text)
    except Exception:                              # noqa: BLE001
        return (), ()


def route(question: str, *, llm: Any | None = None, settings: Settings | None = None) -> Route:
    """rule∧latest → mixed / latest / rule / 무매칭 → LLM. llm 은 테스트 주입용."""
    t = unicodedata.normalize("NFKC", question or "")
    r, l = bool(RULE_RE.search(t)), bool(LATEST_RE.search(t))
    d = bool(BASEBALL_DOMAIN_RE.search(t))
    topics, teams = _tags(t)
    if topics and teams:
        # 구단명과 KBO 주제가 함께 있으면 구단 데이터 질문이다.
        # '투수'·'타자' 같은 단어가 규칙 정규식에도 있어 규칙 질문으로 빠지는 것을 막는다.
        return Route("latest", "keyword", r, True, d, topics, teams)
    if r and l:
        return Route("mixed", "keyword", r, l, d, *_tags(t))
    if l:
        return Route("latest", "keyword", r, l, d, *_tags(t))
    if r:
        return Route("rule", "keyword", r, l, d, *_tags(t))
    if d:
        # 야구 도메인 단어만 있는 질문은 규칙집을 먼저 본다.
        # 예전에는 최신정보로 보냈는데, 그건 웹 검색이 규칙집 검색보다 먼저 돌던 시절의 선택이다.
        # 지금은 rule 이 "규칙집 → 없으면 웹 → 그래도 없으면 모델 지식" 을 뜻하므로 이쪽이 낫다.
        # 구단 데이터 질문은 위의 topics∧teams 분기가 이미 걷어 간다.
        return Route("rule", "keyword", r, l, d, *_tags(t))

    model = llm if llm is not None else build_router_llm(settings)
    try:
        out = model.invoke(render_router_messages(t))
        kind = _parse_kind(out)
    except Exception:
        kind = ""
    if kind not in VALID_KINDS:
        kind = "rule"                      # 안전 폴백: 규칙집 검색 → 근거 없으면 abstain
    # off_topic 에는 KBO 태그를 붙이지 않는다. 거부 경로가 데이터 조회로 새지 않도록.
    tags = ((), ()) if kind == "off_topic" else _tags(t)
    return Route(kind, "llm", r, l, d, *tags)     # type: ignore[arg-type]
