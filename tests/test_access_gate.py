"""접근 코드 게이트. 값은 어떤 단언에서도 출력하지 않는다."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
if str(UI) not in sys.path:
    sys.path.insert(0, str(UI))

APP = str(UI / "app.py")
PIN = "1234"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_PIN", raising=False)
    import gate as module

    monkeypatch.setattr(module, "_from_env_file", lambda: None)
    return module


def test_pin_format_rules(gate) -> None:
    assert gate.pin_is_valid("0000")
    assert gate.pin_is_valid("9999")
    assert not gate.pin_is_valid("123")
    assert not gate.pin_is_valid("12345")
    assert not gate.pin_is_valid("12a4")
    assert not gate.pin_is_valid("")
    assert not gate.pin_is_valid(None)


def test_env_wins_over_env_file(gate, monkeypatch) -> None:
    monkeypatch.setattr(gate, "_from_env_file", lambda: "9999")
    monkeypatch.setenv("APP_ACCESS_PIN", PIN)
    assert gate.configured_pin() == PIN


def test_env_file_is_used_when_env_absent(gate, monkeypatch) -> None:
    monkeypatch.setattr(gate, "_from_env_file", lambda: PIN)
    assert gate.configured_pin() == PIN


def test_no_pin_means_no_gate(gate) -> None:
    assert gate.configured_pin() is None
    assert gate.gate_enabled() is False


def test_app_is_open_without_pin(gate) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.chat_input) == 1                 # 바로 질문할 수 있다


def test_app_is_locked_with_pin(monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("APP_ACCESS_PIN", PIN)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.chat_input) == 0                 # 질문 입력창이 없다
    assert len(at.text_input) == 1                 # 코드 입력창만 있다
    assert not at.sidebar.button                   # 예시 질문 버튼도 노출되지 않는다


def test_correct_pin_unlocks(monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("APP_ACCESS_PIN", PIN)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.text_input[0].set_value(PIN)
    at.button[0].click().run()
    assert not at.exception
    assert len(at.chat_input) == 1                 # 통과 후 질문 가능


def test_wrong_pin_keeps_lock(monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("APP_ACCESS_PIN", PIN)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.text_input[0].set_value("0000")
    at.button[0].click().run()
    assert not at.exception
    assert len(at.chat_input) == 0
    assert at.warning                              # 남은 횟수 안내


def test_lockout_after_repeated_failures(monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    import gate as module

    monkeypatch.setenv("APP_ACCESS_PIN", PIN)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    for _ in range(module.MAX_ATTEMPTS):
        at.text_input[0].set_value("0000")
        at.button[0].click().run()
    assert at.error                                # 마지막 실패에서 대기 안내가 나온다
    at.run()                                       # 다음 실행부터 입력창 자체가 사라진다
    assert len(at.chat_input) == 0
    assert not at.text_input                       # 대기 중에는 올바른 코드도 받지 않는다
    assert at.error


def test_malformed_pin_locks_everyone_out(monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("APP_ACCESS_PIN", "abc")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.chat_input) == 0
    assert at.error                                # 운영자에게 형식 오류를 알린다


def test_doctor_never_prints_the_pin() -> None:
    from baseball.config import Settings
    from baseball.doctor import _access

    line = _access(Settings(app_access_pin="1234"))
    assert "1234" not in line
    assert "설정됨" in line
    assert "없음" in _access(Settings(app_access_pin=None))

def test_reads_pin_from_a_real_env_file(monkeypatch, tmp_path) -> None:
    """사용자는 .env 에 적는다. UI 프로세스 환경에는 자동으로 실리지 않으므로 파일을 직접 읽는다."""
    import gate as module

    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-dummy\nAPP_ACCESS_PIN=4321\n", encoding="utf-8")
    monkeypatch.delenv("APP_ACCESS_PIN", raising=False)
    monkeypatch.setattr(module, "ENV_PATH", env)
    assert module._from_env_file() == "4321"
    assert module.configured_pin() == "4321"
    assert module.gate_enabled() is True


def test_blank_pin_in_env_file_means_no_gate(monkeypatch, tmp_path) -> None:
    import gate as module

    env = tmp_path / ".env"
    env.write_text("APP_ACCESS_PIN=\n", encoding="utf-8")
    monkeypatch.delenv("APP_ACCESS_PIN", raising=False)
    monkeypatch.setattr(module, "ENV_PATH", env)
    assert module.configured_pin() is None


def test_missing_env_file_is_not_an_error(monkeypatch, tmp_path) -> None:
    import gate as module

    monkeypatch.delenv("APP_ACCESS_PIN", raising=False)
    monkeypatch.setattr(module, "ENV_PATH", tmp_path / "없는파일")
    assert module.configured_pin() is None
