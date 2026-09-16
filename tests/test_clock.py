"""KST 시계. 서버 시간대와 무관하게 한국 날짜를 정하는지 고정한다.

이 테스트가 없으면 개발 머신(KST)에서는 아무 문제가 없어 보인다. 배포처인
Streamlit Community Cloud 는 UTC 라 매일 00:00~09:00 KST 구간에만 틀린다.
그래서 단언은 전부 로컬 시간대와 무관한 입력으로 만든다.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from baseball import clock


def test_kst_is_fixed_plus_nine() -> None:
    """ZoneInfo 가 아니라 고정 오프셋이다. tzdata 없는 이미지에서도 동작해야 한다."""
    assert clock.KST.utcoffset(None) == timedelta(hours=9)


def test_utc_evening_is_already_tomorrow_in_kst() -> None:
    """9시간 차이가 '가까운 일정' 을 하루 틀리게 만든다. 그 경계를 고정한다."""
    utc_evening = datetime(2026, 9, 16, 15, 30, tzinfo=timezone.utc)
    assert clock.to_kst(utc_evening).date() == date(2026, 9, 17)


def test_utc_midday_is_still_today_in_kst() -> None:
    utc_midday = datetime(2026, 9, 16, 2, 0, tzinfo=timezone.utc)
    assert clock.to_kst(utc_midday).date() == date(2026, 9, 16)


def test_boundary_is_exactly_fifteen_utc() -> None:
    """15:00 UTC 정각부터 KST 는 다음 날이다."""
    assert clock.to_kst(datetime(2026, 9, 16, 14, 59, 59, tzinfo=timezone.utc)).date() == date(2026, 9, 16)
    assert clock.to_kst(datetime(2026, 9, 16, 15, 0, 0, tzinfo=timezone.utc)).date() == date(2026, 9, 17)


def test_naive_input_is_treated_as_kst_wall_clock() -> None:
    """네이버가 주는 game_date_time 은 KST 벽시계의 naive datetime 이다."""
    naive = datetime(2026, 9, 16, 18, 30)
    assert clock.to_kst(naive) == naive


def test_now_kst_naive_is_comparable_to_game_date_time() -> None:
    """aware 와 naive 를 비교하면 TypeError 가 난다. 경기 시각 비교용 진입점을 고정한다."""
    assert clock.now_kst_naive().tzinfo is None
    game_time = datetime(2026, 9, 16, 18, 30)           # Game.game_date_time 과 같은 모양
    assert isinstance(clock.now_kst_naive() >= game_time, bool)   # TypeError 가 나지 않는다


def test_now_kst_is_aware() -> None:
    assert clock.now_kst().tzinfo is not None
    assert clock.now_kst().utcoffset() == timedelta(hours=9)


def test_today_kst_matches_now_kst_date() -> None:
    assert clock.today_kst() == clock.now_kst().date()
