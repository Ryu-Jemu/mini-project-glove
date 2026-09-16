from __future__ import annotations

import json
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import Any

import pytest

from helpers import make_answer_payload, make_doc

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """키 없이도 전부 통과해야 한다. 실제 키가 있어도 테스트 값으로 덮어쓴다.
    단 -m live 는 실제 키·인덱스를 써야 하므로 그대로 둔다."""
    if "live" in request.keywords:
        from baseball.config import get_settings

        get_settings.cache_clear()
        yield
        get_settings.cache_clear()
        return
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "off")
    monkeypatch.setenv("ENABLE_KBO_DATA", "off")
    from baseball.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings():
    from baseball.config import get_settings

    return get_settings()


@pytest.fixture
def pages_sample() -> dict[str, str]:
    return json.loads((FIXTURES / "pages_sample.json").read_text(encoding="utf-8"))["pages"]


@dataclass
class FakeRetrieval:
    docs: list[dict[str, Any]]
    abstain: bool = False
    dense_top: float = 0.7
    bm25_ratio: float = 0.5
    exact_hits: list[str] = field(default_factory=list)
    channels: dict[str, list[str]] = field(default_factory=dict)
    context_tokens: int = 100
    dense_failed: bool = False


class FakeRetriever:
    document_id = "doc"

    def __init__(self, docs: list[dict[str, Any]] | None = None, abstain: bool = False) -> None:
        self.docs = docs if docs is not None else [make_doc()]
        self.abstain = abstain
        self.lexical = type("L", (), {"chunks": self.docs, "bm25": object()})()

    def retrieve(self, query: str, k: int = 6) -> FakeRetrieval:
        return FakeRetrieval(docs=[] if self.abstain else self.docs[:k], abstain=self.abstain)


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()


def fake_structured_model(responses: list[str]):
    """with_structured_output 을 지원하는 가짜 채팅 모델.

    FakeListChatModel 은 이걸 구현하지 않는다(BaseChatModel 기본이 NotImplementedError).
    langchain-openai 가 include_raw=True 에서 만드는 모양
    (RunnableMap(raw=llm) | assign(parsed, parsing_error) + 폴백)을 그대로 재현한다.
    덕분에 깨진 JSON 을 주면 폴백 분기까지 가짜로 검사할 수 있다.
    """
    from langchain_core.language_models import FakeListChatModel
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.runnables import RunnableMap, RunnablePassthrough

    class _Fake(FakeListChatModel):
        def with_structured_output(self, schema=None, *, include_raw=False, **kw):  # noqa: ANN001
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

    return _Fake(responses=responses)


CANNED_ANSWER = json.dumps(make_answer_payload(), ensure_ascii=False)


@pytest.fixture
def service(settings, fake_retriever):
    """답변 LLM·라우터 LLM 모두 가짜로 주입한 RagService."""
    from langchain_core.language_models import FakeListChatModel

    from baseball.chain import RagService

    return RagService(
        settings, fake_retriever,
        llm=fake_structured_model([CANNED_ANSWER]),
        router_llm=FakeListChatModel(responses=['{"kind":"off_topic"}']),
    )
