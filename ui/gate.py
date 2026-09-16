"""접근 제한 게이트.

4자리 숫자를 맞춰야 질문할 수 있다. 값은 APP_ACCESS_PIN 으로 준다.
읽는 순서: 환경변수 → Streamlit secrets → 저장소 루트의 .env 파일.

한계를 분명히 해 둔다. 4자리는 경우의 수가 1만 가지라 강한 인증이 아니다.
아는 사람만 들어오게 하는 가벼운 문턱이며, 무차별 대입을 늦추기 위해
시도 횟수 제한과 대기 시간을 함께 둔다. 더 강한 보호가 필요하면 자릿수를 늘린다.
"""
from __future__ import annotations

import hmac
import os
import time
from pathlib import Path

import streamlit as st

PIN_KEY = "APP_ACCESS_PIN"
PIN_LENGTH = 4
MAX_ATTEMPTS = 5              # 세션당 연속 실패 허용 횟수
LOCKOUT_SECONDS = 300         # 초과 시 대기 시간
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

_UNLOCKED = "access_unlocked"
_FAILS = "access_fails"
_LOCKED_UNTIL = "access_locked_until"


def _from_env_file() -> str | None:
    """로컬 개발용. .env 는 UI 프로세스 환경에 자동으로 실리지 않는다."""
    try:
        from dotenv import dotenv_values

        value = dotenv_values(ENV_PATH).get(PIN_KEY)
    except Exception:                                  # noqa: BLE001
        return None
    return str(value).strip() if value else None


def configured_pin() -> str | None:
    value = os.getenv(PIN_KEY)
    if value and value.strip():
        return value.strip()
    try:
        secret = st.secrets.get(PIN_KEY)
        if secret is not None and str(secret).strip():
            return str(secret).strip()
    except Exception:                                  # noqa: BLE001
        pass
    return _from_env_file()


def pin_is_valid(pin: str | None) -> bool:
    return bool(pin) and len(pin) == PIN_LENGTH and pin.isdigit()


def gate_enabled() -> bool:
    return configured_pin() is not None


def is_unlocked() -> bool:
    return bool(st.session_state.get(_UNLOCKED))


def _remaining_lockout() -> int:
    until = st.session_state.get(_LOCKED_UNTIL, 0.0)
    return max(0, int(until - time.time()))


def _accept(entered: str, expected: str) -> bool:
    # 자릿수별 비교 시간 차이를 없앤다.
    return hmac.compare_digest(entered, expected)


def ensure_access() -> bool:
    """통과하면 True. 아니면 입력 화면을 그리고 False 를 돌려준다."""
    pin = configured_pin()
    if pin is None:                                    # 미설정이면 제한하지 않는다
        return True
    if is_unlocked():
        return True

    if not pin_is_valid(pin):
        st.error(
            f"{PIN_KEY} 설정이 잘못되었습니다. 숫자 {PIN_LENGTH}자리여야 합니다. "
            "운영자가 설정을 고칠 때까지 사용할 수 없습니다.",
            icon=":material/error:",
        )
        return False

    st.markdown("### 접근 코드를 입력하세요")
    st.caption(f"허가된 인원에게 공유된 숫자 {PIN_LENGTH}자리를 입력합니다.")

    wait = _remaining_lockout()
    if wait:
        st.error(
            f"시도 횟수를 초과했습니다. {wait // 60}분 {wait % 60}초 뒤에 다시 시도하세요.",
            icon=":material/lock_clock:",
        )
        return False

    with st.form("access-gate", clear_on_submit=True):
        entered = st.text_input(
            "접근 코드", max_chars=PIN_LENGTH, type="password",
            placeholder="0000", label_visibility="collapsed",
        )
        submitted = st.form_submit_button("입장", width="stretch")

    if not submitted:
        return False

    entered = (entered or "").strip()
    if _accept(entered, pin):
        st.session_state[_UNLOCKED] = True
        st.session_state[_FAILS] = 0
        st.rerun()
        return True

    fails = int(st.session_state.get(_FAILS, 0)) + 1
    st.session_state[_FAILS] = fails
    if fails >= MAX_ATTEMPTS:
        st.session_state[_LOCKED_UNTIL] = time.time() + LOCKOUT_SECONDS
        st.session_state[_FAILS] = 0
        st.error(
            f"{MAX_ATTEMPTS}회 틀렸습니다. {LOCKOUT_SECONDS // 60}분 뒤에 다시 시도하세요.",
            icon=":material/lock:",
        )
    else:
        st.warning(
            f"접근 코드가 맞지 않습니다. {MAX_ATTEMPTS - fails}회 남았습니다.",
            icon=":material/key_off:",
        )
    return False


def lock() -> None:
    """세션을 다시 잠근다."""
    for key in (_UNLOCKED, _FAILS):
        st.session_state.pop(key, None)
