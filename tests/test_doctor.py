"""doctor 의 OPENAI 점검이 무효 키를 실제로 잡아내는지 고정한다.

존재만 검사하던 시절에는 401 키에도 'OPENAI ok' 가 나왔다. retriever 는 임베딩 실패를
dense_failed 로 삼키고 abstain 이 그것을 '자료에 없음'으로 해석하므로, 죽은 키가 조용히
내용 거부로 둔갑했다. 그 회귀를 막는다.
"""
from __future__ import annotations

import pytest
from pydantic import SecretStr


def _settings(**kw):
    from baseball.config import Settings

    base = {"_env_file": None, "openai_api_key": SecretStr("sk-test"), "embedding_dimensions": 1024}
    return Settings(**{**base, **kw})


def _patch_embeddings(monkeypatch: pytest.MonkeyPatch, *, result=None, error=None) -> dict:
    """doctor._openai 이 함수 안에서 import 하므로 모듈 속성을 갈아끼운다."""
    import langchain_openai

    seen: dict = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

        def embed_query(self, text: str) -> list[float]:
            seen["query"] = text
            if error is not None:
                raise error
            return result

    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", FakeEmbeddings)
    return seen


def test_missing_key_fails() -> None:
    from baseball.doctor import _openai

    line = _openai(_settings(openai_api_key=None))
    assert "FAIL" in line and "missing" in line


def test_live_call_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from baseball.doctor import _openai

    seen = _patch_embeddings(monkeypatch, result=[0.0] * 1024)
    line = _openai(_settings())

    assert line.startswith("OPENAI ok")
    assert "FAIL" not in line
    assert seen["dimensions"] == 1024
    assert seen["max_retries"] == 0          # 401 을 재시도하지 않는다
    assert seen["query"]                     # 실제로 호출했다


def _auth_error() -> Exception:
    """openai.AuthenticationError 처럼 클래스 이름으로만 식별되는 경우."""
    return type("AuthenticationError", (Exception,), {})("401")


def _status_error() -> Exception:
    """openai.APIStatusError 처럼 status_code 를 들고 오는 경우."""
    exc = Exception("nope")
    exc.status_code = 403
    return exc


@pytest.mark.parametrize("make_error", [_auth_error, _status_error],
                         ids=["by-class-name", "by-status-code"])
def test_rejected_key_fails(monkeypatch: pytest.MonkeyPatch, make_error) -> None:
    from baseball.doctor import _openai

    _patch_embeddings(monkeypatch, error=make_error())

    line = _openai(_settings())
    assert "FAIL" in line
    assert "거부" in line


def test_network_error_is_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    """네트워크·한도 장애는 키의 문제가 아니다. make setup 을 여기서 멈추면 안 된다."""
    from baseball.doctor import _openai

    _patch_embeddings(monkeypatch, error=ConnectionError("dns"))
    line = _openai(_settings())

    # main() 이 required 체크를 판정하는 세 단어가 모두 없어야 soft 다.
    assert "FAIL" not in line and "missing" not in line and "mismatch" not in line
    assert "soft warning" in line


def test_dimension_mismatch_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    from baseball.doctor import _openai

    _patch_embeddings(monkeypatch, result=[0.0] * 1536)
    line = _openai(_settings())
    assert "mismatch" in line                # main() 이 이 단어로 실패 처리한다


def test_secret_never_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    from baseball.doctor import _openai

    _patch_embeddings(monkeypatch, error=Exception("boom"))
    line = _openai(_settings(openai_api_key=SecretStr("sk-supersecret-value")))
    assert "supersecret" not in line
