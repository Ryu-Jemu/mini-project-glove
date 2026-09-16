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

        def __call__(self, query: str, settings=None, *, client=None) -> list[LatestEntry]:
            self.queries.append(query)
            return list(self.results)

    rec = Recorder()
    monkeypatch.setattr(latest_info, "web_search", rec)
    return rec


def _entry() -> LatestEntry:
    return LatestEntry(kind="web", label="웹 검색: 연합뉴스", text="박해민은 LG 트윈스 소속이다.",
                       as_of="2026-09-07", source_url="https://example.com/1", confidence="likely")


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
    web.results = [_entry()]
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
