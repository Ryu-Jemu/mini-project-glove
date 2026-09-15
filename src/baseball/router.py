"""의도 라우터: 키워드 표(정본) → 미매칭 시에만 LLM(router_v1.md). off_topic 은 LLM 만 산출."""
from __future__ import annotations

import json
import re
import unicodedata
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


@dataclass(frozen=True)
class Route:
    kind: RouteKind
    by: Literal["keyword", "llm"]
    rule_hit: bool
    latest_hit: bool
    domain_hit: bool = False

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


def route(question: str, *, llm: Any | None = None, settings: Settings | None = None) -> Route:
    """rule∧latest → mixed / latest / rule / 무매칭 → LLM. llm 은 테스트 주입용."""
    t = unicodedata.normalize("NFKC", question or "")
    r, l = bool(RULE_RE.search(t)), bool(LATEST_RE.search(t))
    d = bool(BASEBALL_DOMAIN_RE.search(t))
    if r and l:
        return Route("mixed", "keyword", r, l, d)
    if l:
        return Route("latest", "keyword", r, l, d)
    if r:
        return Route("rule", "keyword", r, l, d)
    if d:
        # 구단·리그 고유명사만 있는 질문(예: "LG 트윈스는 어떤 팀이야?")은 규칙집에 답이 없다.
        # 웹 최신정보 경로로 보내고, 웹이 꺼져 있으면 거부 문장으로 떨어진다.
        return Route("latest", "keyword", r, l, d)

    model = llm if llm is not None else build_router_llm(settings)
    try:
        out = model.invoke(render_router_messages(t))
        kind = _parse_kind(out)
    except Exception:
        kind = ""
    if kind not in VALID_KINDS:
        kind = "rule"                      # 안전 폴백: 규칙집 검색 → 근거 없으면 abstain
    return Route(kind, "llm", r, l, d)     # type: ignore[arg-type]
