"""답변 모델이 직접 web_search 도구를 부르는 경로.

왜 필요한가: BM25 는 "선수"·"야구" 같은 단어로 항상 무언가를 찾아낸다. 그래서 규칙집 검색이
무관한 조항을 들고 와도 docs 가 비지 않고, 시스템은 그게 무관한지 알 방법이 없다.
"박해민 어느 팀 소속이야?" 가 규칙 5.11·8.04 를 근거로 거부되던 이유다.
docs 가 질문과 맞는지는 모델이 판단할 수 있으므로, 검색 여부를 모델에게 맡긴다.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from baseball import latest_info
from baseball.chain import RagService
from baseball.config import Settings
from baseball.context import LatestEntry

from conftest import CANNED_ANSWER, FakeRetriever


def fake_tool_model(replies: list[Any], seen: list[dict[str, Any]]):
    """tool_calls 를 낼 수 있는 가짜 모델.

    FakeListChatModel 은 문자열만 낼 수 있어 tool_calls 를 못 만든다.
    GenericFakeChatModel 은 AIMessage 를 그대로 돌려주므로 여기에
    conftest 와 같은 with_structured_output 껍데기를 얹는다.
    """
    from operator import itemgetter

    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.runnables import RunnableMap, RunnablePassthrough

    class _Fake(GenericFakeChatModel):
        def with_structured_output(self, schema=None, *, include_raw=False, **kw):  # noqa: ANN001
            seen.append(kw)
            parser = JsonOutputParser()
            if not include_raw:
                return self | parser
            assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser, parsing_error=lambda _: None
            )
            none = RunnablePassthrough.assign(parsed=lambda _: None)
            return RunnableMap(raw=self) | assign.with_fallbacks(
                [none], exception_key="parsing_error"
            )

    return _Fake(messages=iter(replies))


def _tool_call(query: str = "박해민 소속 구단", call_id: str = "call_1", name: str = "web_search"):
    return AIMessage(
        content="",                                  # JSON 파싱 실패 → parsed=None (프로덕션과 같은 모양)
        tool_calls=[{"name": name, "args": {"query": query}, "id": call_id, "type": "tool_call"}],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )


def _final():
    return AIMessage(
        content=CANNED_ANSWER,
        usage_metadata={"input_tokens": 300, "output_tokens": 50, "total_tokens": 350},
    )


def _settings(**kw: Any) -> Settings:
    base = {"_env_file": None, "openai_api_key": SecretStr("test-key"),
            "enable_kbo_data": "off", "enable_web_search": "off",
            "enable_model_knowledge": "on"}
    return Settings(**{**base, **kw})


def _web_on(**kw: Any) -> Settings:
    return _settings(enable_web_search="on", tavily_api_key=SecretStr("tvly-test"), **kw)


def _service(settings: Settings, replies: list[Any], seen: list[dict[str, Any]]) -> RagService:
    return RagService(settings, FakeRetriever(abstain=True),
                      llm=fake_tool_model(replies, seen), router_llm=None)


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch):
    """latest_info.web_search 만 갈아끼운다. @tool 래퍼는 진짜로 돈다."""

    class Recorder:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.results: list[LatestEntry] = []
            self.per_query: dict[str, list[LatestEntry]] = {}

        def __call__(self, query: str, settings=None, *, client=None) -> list[LatestEntry]:
            self.queries.append(query)
            if self.per_query:
                return list(self.per_query.get(query, []))
            return list(self.results)

    rec = Recorder()
    monkeypatch.setattr(latest_info, "web_search", rec)
    return rec


def _entry(n: int = 1) -> LatestEntry:
    return LatestEntry(kind="web", label=f"웹 검색: 연합뉴스 {n}", text="박해민은 LG 트윈스 소속이다.",
                       as_of="2026-09-07", source_url=f"https://example.com/{n}", confidence="likely")


# --------------------------------------------------------------------------- 정상 경로

def test_tool_round_then_answer(web) -> None:
    web.results = [_entry()]
    seen: list[dict[str, Any]] = []
    out = _service(_web_on(), [_tool_call(), _final()], seen).answer("박해민 어디 팀 소속이야?")

    assert web.queries == ["박해민 소속 구단"]        # 모델이 고른 검색어가 그대로 갔다
    assert out.status == "answered"
    assert out.llm_called is True
    assert [s["kind"] for s in out.sources] == ["web"]
    assert out.freshness == "web"
    assert out.format_ok is True
    assert "SCHEMA_FALLBACK" not in out.format_issues


def test_usage_is_summed_across_rounds(web) -> None:
    """루프는 LLM 을 두 번 부른다. 덮어쓰면 마지막 호출분만 청구된다."""
    web.results = [_entry()]
    out = _service(_web_on(), [_tool_call(), _final()], []).answer("박해민 소속")

    assert out.usage["input_tokens"] == 400          # 100 + 300
    assert out.usage["output_tokens"] == 60          # 10 + 50
    assert out.usage["cost_usd"] > 0


def test_tools_are_bound_as_request_parameters(web) -> None:
    web.results = [_entry()]
    seen: list[dict[str, Any]] = []
    _service(_web_on(), [_tool_call(), _final()], seen).answer("박해민 소속")

    assert [t.name for t in seen[0]["tools"]] == ["web_search"]
    assert seen[0]["parallel_tool_calls"] is False   # 병렬 호출이면 구조화 출력이 보장되지 않는다
    assert "tool_choice" not in seen[0]              # 첫 라운드는 모델이 자유롭게 고른다


def test_no_tool_call_keeps_the_old_path(web) -> None:
    out = _service(_web_on(), [_final()], []).answer("인필드 플라이가 뭐야?")

    assert web.queries == []
    assert out.status == "answered"
    assert [s["kind"] for s in out.sources] == ["model"]
    assert out.freshness == "model"


# --------------------------------------------------------------------------- 상한과 실패

def test_last_round_forbids_further_tool_calls(web) -> None:
    """미해결 도구 호출로 턴이 끝나면 스키마 파싱이 깨진다. 마지막 라운드는 호출을 막는다."""
    web.results = [_entry()]
    seen: list[dict[str, Any]] = []
    _service(_web_on(max_tool_rounds=1), [_tool_call(), _final()], seen).answer("박해민 소속")

    assert len(web.queries) == 1
    assert seen[-1]["tool_choice"] == "none"


def test_two_rounds_are_allowed(web) -> None:
    web.per_query = {"박해민 소속": [_entry(1)], "박해민 생년월일": [_entry(2)]}
    seen: list[dict[str, Any]] = []
    out = _service(
        _web_on(max_tool_rounds=2),
        [_tool_call("박해민 소속"), _tool_call("박해민 생년월일", "call_2"), _final()],
        seen,
    ).answer("박해민 소속과 생년월일")

    assert web.queries == ["박해민 소속", "박해민 생년월일"]
    assert seen[-1]["tool_choice"] == "none"
    assert out.status == "answered"
    assert len(out.sources) == 2                     # 라운드마다 출처가 쌓인다


def test_empty_search_still_answers(web) -> None:
    web.results = []
    out = _service(_web_on(), [_tool_call(), _final()], []).answer("박해민 소속")

    assert web.queries == ["박해민 소속 구단"]
    assert out.status == "answered"
    assert [s["kind"] for s in out.sources] == ["model"]   # 찾은 게 없으니 웹 출처도 없다
    assert out.freshness == "model"


def test_unknown_tool_name_does_not_break_the_turn(web) -> None:
    out = _service(_web_on(), [_tool_call(name="없는도구"), _final()], []).answer("박해민 소속")

    assert web.queries == []
    assert out.status == "answered"


def test_tool_failure_does_not_break_the_turn(web, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any):
        raise RuntimeError("tavily down")

    monkeypatch.setattr(latest_info, "web_search", boom)
    out = _service(_web_on(), [_tool_call(), _final()], []).answer("박해민 소속")
    assert out.status == "answered"


# --------------------------------------------------------------------------- 도구 비활성

@pytest.mark.parametrize(
    "settings",
    [pytest.param(_settings(), id="web-off"),
     pytest.param(_web_on(max_tool_rounds=0), id="rounds-0")],
)
def test_no_tools_bound_when_disabled(settings: Settings, web) -> None:
    seen: list[dict[str, Any]] = []
    _service(settings, [_final()], seen).answer("인필드 플라이가 뭐야?")

    assert "tools" not in seen[0]
    assert "tool_choice" not in seen[0]
    assert web.queries == []


# --------------------------------------------------------------------------- SSE

def test_stream_emits_a_tool_event_before_tokens(web) -> None:
    web.results = [_entry()]
    service = _service(_web_on(), [_tool_call(), _final()], [])
    events = list(service.stream("박해민 소속"))
    order = [e["event"] for e in events]

    assert order[:3] == ["route", "status", "sources"]       # 기존 계약
    assert order[-2:] == ["final", "done"]
    assert order.index("sources") < order.index("tool") < order.index("token")

    tool_event = next(e["data"] for e in events if e["event"] == "tool")
    assert tool_event == {"name": "web_search", "query": "박해민 소속 구단", "results": 1}

    final = next(e["data"] for e in events if e["event"] == "final")
    assert [s["kind"] for s in final["sources"]] == ["web"]
    assert final["freshness"] == "web"


async def test_astream_matches_stream(web) -> None:
    web.results = [_entry()]
    service = _service(_web_on(), [_tool_call(), _final()], [])
    events = [e async for e in service.astream("박해민 소속")]
    order = [e["event"] for e in events]

    assert order[:3] == ["route", "status", "sources"]
    assert order[-2:] == ["final", "done"]
    assert "tool" in order
    tokens = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    final = next(e["data"] for e in events if e["event"] == "final")
    assert tokens == final["answer"]


def test_tool_messages_never_enter_session_history(web) -> None:
    """도구 왕복은 한 턴의 내부 사정이다. 다음 턴 프롬프트에 새면 안 된다."""
    web.results = [_entry()]
    service = _service(_web_on(), [_tool_call(), _final()], [])
    service.answer("박해민 소속", session_id="s1")

    history = service.history("s1")
    assert [m.type for m in history] == ["human", "ai"]


def test_repeated_query_does_not_search_twice(web) -> None:
    """모델이 원하는 값을 못 찾으면 같은 검색어를 되풀이한다. 실측으로 확인한 행동이다.

    결과가 같으므로 왕복만 낭비되고 출처가 중복된다. 두 번째는 나가지 않는다.
    """
    web.results = [_entry()]
    out = _service(
        _web_on(max_tool_rounds=2),
        [_tool_call("박해민 생년월일"), _tool_call("박해민 생년월일", "call_2"), _final()],
        [],
    ).answer("박해민 생년월일")

    assert web.queries == ["박해민 생년월일"]          # 한 번만 실제로 나갔다
    assert [s["kind"] for s in out.sources] == ["web"]  # 출처도 하나뿐이다
    assert out.status == "answered"


def test_same_document_from_two_queries_is_shown_once(web) -> None:
    web.per_query = {"질의 A": [_entry(1)], "질의 B": [_entry(1)]}
    out = _service(
        _web_on(max_tool_rounds=2),
        [_tool_call("질의 A"), _tool_call("질의 B", "call_2"), _final()],
        [],
    ).answer("박해민")

    assert web.queries == ["질의 A", "질의 B"]         # 서로 다른 질의라 둘 다 나갔다
    assert len(out.sources) == 1                        # 같은 URL 이라 한 번만 보여 준다


# --------------------------------------------------------------------------- 도구 사용 사후 검증

def test_live_question_without_any_tool_call_is_downgraded(web) -> None:
    """실시간 값을 물었는데 도구를 한 번도 안 부르면 기억으로 답한 것이다.

    prepare 3단계의 can_tool 은 "도구를 붙일 수 있다"는 사실만 보고 거부 게이트를 연다.
    tool_choice 는 마지막 라운드를 빼면 None 이라 호출은 모델의 선택이다. 부르지 않고
    학습 시점 기억으로 답해도 예전에는 status=answered 로 나갔다.
    """
    from baseball.prompts import NOT_IN_CONTEXT_REFUSAL

    out = _service(_web_on(), [_final()], []).answer("박해민 소속")

    assert web.queries == []                       # 도구를 부르지 않았다
    assert out.status == "phase2_pending"
    assert out.answer == NOT_IN_CONTEXT_REFUSAL
    assert out.needs_web is True
    assert out.sources == []                       # 배지 밑에 출처가 붙는 모순을 만들지 않는다
    assert out.freshness == "static"
    assert out.citations == []


def test_calling_the_tool_and_getting_nothing_still_answers(web) -> None:
    """부른 뒤 빈손인 것은 강등하지 않는다. 이미 설계된 별도 경로다.

    freshness="model" 과 "모델 일반 지식" 출처 칩으로 화면에 드러난다.
    """
    web.results = []
    out = _service(_web_on(), [_tool_call(), _final()], []).answer("박해민 소속")

    assert web.queries == ["박해민 소속 구단"]      # 실제로 불렀다
    assert out.status == "answered"
    assert out.freshness == "model"
    assert [s["kind"] for s in out.sources] == ["model"]


def test_non_live_question_without_tool_call_is_not_downgraded(web) -> None:
    """실시간 값이 아니면 도구를 안 불러도 강등하지 않는다. 과발동 방지."""
    out = _service(_web_on(), [_final()], []).answer("인필드 플라이가 뭐야?")

    assert web.queries == []
    assert out.status == "answered"
    assert out.freshness == "model"


def test_downgrade_applies_to_the_stream_path_too(web) -> None:
    """강등은 _finish 한 곳에 있으므로 answer·stream·astream 세 경로가 함께 따른다."""
    from baseball.prompts import NOT_IN_CONTEXT_REFUSAL

    events = list(_service(_web_on(), [_final()], []).stream("박해민 소속"))
    final = next(e["data"] for e in events if e["event"] == "final")

    assert final["status"] == "phase2_pending"
    assert final["answer"] == NOT_IN_CONTEXT_REFUSAL
    assert final["sources"] == []


# --------------------------------------------------------------------------- 무관한 규칙집 출처

def _final_without_rule_refs():
    """맛집 답변의 모양. 규칙 번호를 인용하지 않는다."""
    import json

    from helpers import make_answer_payload

    payload = make_answer_payload(
        kind="latest",
        headline="잠실 야구장 근처 맛집입니다.",
        definition=None,
        points=[{"label": "가게 A", "detail": "블로그에서 소개된 곳입니다.", "rule_ref": None}],
        example=None, why=None,
        evidence=["웹 검색 | 가게 A | 2026-09-17"],      # 규칙 번호를 인용하지 않는다
    )
    return AIMessage(
        content=json.dumps(payload, ensure_ascii=False),
        usage_metadata={"input_tokens": 300, "output_tokens": 50, "total_tokens": 350},
    )


def _place_tool(query: str = "잠실 맛집", call_id: str = "call_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "find_restaurants", "args": {"query": query},
                     "id": call_id, "type": "tool_call"}],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )


@pytest.fixture
def places_tool(monkeypatch: pytest.MonkeyPatch):
    """find_restaurants 가 카드 한 건을 돌려주도록 고정한다.

    도구 본문은 서비스에 주입한 settings 가 아니라 전역 get_settings() 를 읽는다
    (web_search 도 같다). 그래서 tools.get_settings 까지 함께 갈아끼워야 한다.
    """
    from baseball import places, tools

    def fake(**_kw: Any):
        from baseball.context import LatestEntry

        entries = [LatestEntry(kind="place", label="맛집 정보: 가게 A", text="설명",
                               as_of="2026-09-17", source_url="https://blog.example/a",
                               confidence="likely")]
        cards = [{"id": "a", "name": "가게 A", "url": "https://blog.example/a",
                  "snippet": None, "confidence": "likely"}]
        return entries, cards, []

    monkeypatch.setattr(places, "search", fake)
    settings = _web_on(enable_places="on", enable_kbo_data="off")
    monkeypatch.setattr(tools, "get_settings", lambda: settings)
    return settings


def test_restaurant_answer_drops_unrelated_rulebook_sources(places_tool) -> None:
    """실측 회귀: "잠실 맛집" 답변에 규칙 5.10·4.03 이 근거로 붙었다.

    BM25 는 "야구"·"구장" 같은 단어로 늘 무언가를 찾아낸다. prepare 의 docs 비우기는
    KBO 엔트리가 이미 있을 때만 돌아서, 도구 결과가 뒤에 오는 맛집에는 걸리지 않았다.
    """
    service = RagService(places_tool, FakeRetriever(abstain=False),
                         llm=fake_tool_model([_place_tool(), _final_without_rule_refs()], []), router_llm=None)
    out = service.answer("잠실 야구장 근처 맛집")

    kinds = [s["kind"] for s in out.sources]
    assert "static" not in kinds, f"무관한 규칙집 출처가 남았다: {kinds}"
    assert "place" in kinds
    assert out.sources, "출처가 통째로 비면 평가 게이트(sources_nonempty_ratio)가 깨진다"


def test_plain_rule_answer_keeps_its_rulebook_sources(web) -> None:
    """도구 출처가 없으면 규칙집 출처는 그대로 둔다. 과잉 제거 방지."""
    settings = _web_on(enable_kbo_data="off")
    service = RagService(settings, FakeRetriever(abstain=False),
                         llm=fake_tool_model([_final()], []), router_llm=None)
    out = service.answer("인필드 플라이가 뭐야?")
    assert "static" in [s["kind"] for s in out.sources]


def test_web_search_answer_keeps_rulebook_sources(web) -> None:
    """web_search 는 규칙 답변과 함께 쓰일 수 있다. place·video 만 무관하다고 본다."""
    web.results = [_entry()]
    settings = _web_on(enable_kbo_data="off")
    service = RagService(settings, FakeRetriever(abstain=False),
                         llm=fake_tool_model([_tool_call(), _final()], []), router_llm=None)
    out = service.answer("박해민 소속")
    kinds = [s["kind"] for s in out.sources]
    assert "web" in kinds and "static" in kinds


# --------------------------------------------------------------------------- 확인되지 않은 답변 표시

def test_unused_rulebook_chunks_mark_the_answer_as_unverified(web) -> None:
    """실측 회귀: "박해민 생년월일" 이 무관한 조항 6건 + 무관한 기사 3건을 근거로 달고
    기억으로 지어낸 날짜를 "웹 검색 · 근거 9건" 으로 내보냈다. 네 번 물으면 네 번 다른
    날짜가 나왔고 전부 틀렸다(실제 1990-02-24).

    knowledge_only 는 docs 가 비었는지만 보는데 BM25 는 늘 무언가를 찾아내므로,
    잡음이 들어온 순간 False 가 되어 모델 지식 경고가 사라졌다.
    """
    web.results = [_entry()]
    service = RagService(_web_on(enable_kbo_data="off"), FakeRetriever(abstain=False),
                         llm=fake_tool_model([_tool_call(), _final_without_rule_refs()], []),
                         router_llm=None)
    out = service.answer("박해민 생년월일")

    kinds = [s["kind"] for s in out.sources]
    assert "static" not in kinds, f"인용하지 않은 규칙집이 근거로 남았다: {kinds}"
    assert "model" in kinds, "확인되지 않은 답변에 모델 지식 마커가 없다"
    assert "web" in kinds


def test_cited_rulebook_answer_is_not_marked(web) -> None:
    """규칙 조항을 실제로 인용했으면 확인된 답변이다. 과발동 방지."""
    web.results = [_entry()]
    service = RagService(_web_on(enable_kbo_data="off"), FakeRetriever(abstain=False),
                         llm=fake_tool_model([_tool_call(), _final()], []), router_llm=None)
    out = service.answer("박해민 소속")

    kinds = [s["kind"] for s in out.sources]
    assert "model" not in kinds
    assert "static" in kinds          # 인용했으므로 규칙집 출처도 남는다


def test_web_answer_without_rulebook_noise_is_not_marked(web) -> None:
    """규칙집이 애초에 아무것도 못 찾았으면(abstain) 착각할 것도 없다."""
    web.results = [_entry()]
    service = RagService(_web_on(enable_kbo_data="off"), FakeRetriever(abstain=True),
                         llm=fake_tool_model([_tool_call(), _final()], []), router_llm=None)
    out = service.answer("박해민 소속")
    assert [s["kind"] for s in out.sources] == ["web"]
