from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from helpers import make_doc

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


@pytest.fixture
def service(settings, fake_retriever):
    """답변 LLM·라우터 LLM 모두 가짜로 주입한 RagService."""
    from langchain_core.language_models import FakeListChatModel

    from baseball.chain import RagService

    return RagService(
        settings, fake_retriever,
        llm=FakeListChatModel(responses=["보크는 투수의 반칙 투구입니다. 규칙 5.09 참조."]),
        router_llm=FakeListChatModel(responses=['{"kind":"off_topic"}']),
    )
