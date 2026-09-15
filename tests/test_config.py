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
