"""상류가 실제로 바뀌었는지 확인하는 유일한 네트워크 테스트. make test 에서 제외된다."""
from __future__ import annotations

from datetime import date

import pytest

pytestmark = pytest.mark.net


def test_standings_shape_is_still_valid() -> None:
    from baseball.kbo_naver import fetch_standings

    today = date.today()
    snap = fetch_standings(today.year, today=today, timeout=10.0)
    assert len(snap.teams) == 10
    assert {t.ranking for t in snap.teams} == set(range(1, 11))


def test_schedule_shape_is_still_valid() -> None:
    from baseball.kbo_naver import fetch_schedule

    today = date.today()
    snap = fetch_schedule(start=today, end=date(today.year, 12, 31), today=today, timeout=10.0)
    assert snap.games
    assert all(g.is_regular or g.round_code for g in snap.games)
