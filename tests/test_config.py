from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr


def test_env_var_names_match_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """tests/fixtures/test.env 의 변수명이 Settings 필드명과 1:1로 대응한다."""
    from baseball.config import Settings

    env_path = Path(__file__).resolve().parent / "fixtures" / "test.env"
    settings = Settings(_env_file=str(env_path))
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.chat_history_max_messages == 6
    assert len(settings.tavily_include_domain_list) == 5
    # 새 노브도 같은 약속을 지켜야 한다. 이름이 어긋나면 배포 시크릿이 조용히 무시된다.
    assert settings.answer_schema_enabled is False
    assert settings.answer_max_output_tokens == 900
    assert settings.enable_scope_gate == "off"
    assert settings.scope_gate_llm == "off"
    assert settings.model_knowledge_enabled is False
    assert settings.abstain_on_dense_failure is True
    assert settings.abstain_bm25_reference == "sentence"
    assert settings.context_min_docs == 5
    assert settings.max_tool_rounds == 1


def test_secrets_are_masked(settings) -> None:
    assert isinstance(settings.openai_api_key, SecretStr)
    assert "test-key" not in repr(settings)
    assert "test-key" not in str(settings)


def test_import_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """키가 전혀 없어도 모듈 import 와 Settings 생성이 가능해야 한다."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from baseball.config import Settings

    s = Settings(_env_file=None)
    assert s.openai_api_key is None
    import baseball.api  # noqa: F401  (import 만으로 예외가 없어야 한다)


def test_web_search_auto_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from baseball.config import Settings

    off = Settings(_env_file=None, enable_web_search="auto")
    assert off.web_search_enabled is False
    on = Settings(_env_file=None, enable_web_search="auto", tavily_api_key=SecretStr("x"))
    assert on.web_search_enabled is True
