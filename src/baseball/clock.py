"""KBO 시각의 단일 출처. 서버 시간대와 무관하게 한국 표준시로 오늘을 정한다.

왜 필요한가
-----------
답변 경로 곳곳이 `date.today()` 로 "오늘" 을 정했다. 개발 머신이 KST 라 로컬에서는
드러나지 않지만, 배포처인 Streamlit Community Cloud 는 UTC 로 돈다. 그래서 매일
00:00~09:00 KST 구간에 `date.today()` 가 전날을 돌려주고, 그 시간대에 "오늘 경기" 를
물으면 이미 끝난 어제 경기가 '남은 경기' 로 잡힌다.

왜 ZoneInfo 가 아니라 고정 오프셋인가
--------------------------------------
`zoneinfo` 는 표준 라이브러리지만 IANA tzdata 파일이 없는 최소 이미지에서
`ZoneInfoNotFoundError` 를 던진다. 막으려면 `tzdata` 를 의존성에 넣어야 하는데,
`ui/requirements.txt` 는 배포 메모리를 줄이려고 의존성을 깎아 둔 파일이다.
한국은 1988년 이후 서머타임이 없고 계획도 없으므로 고정 +09:00 이 KBO 전 경기에
대해 정확하다. 이 단순함이 의존성 추가보다 낫다.

naive 와 aware
--------------
네이버 스포츠가 주는 `Game.game_date_time` 은 KST 벽시계의 naive datetime 이다.
aware 와 비교하면 `TypeError: can't compare offset-naive and offset-aware datetimes`
가 즉시 난다. 경기 시각과 비교할 때는 반드시 `now_kst_naive()` 를 쓴다.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

KST = timezone(timedelta(hours=9), "KST")


def now_kst() -> datetime:
    """현재 시각(KST, aware)."""
    return datetime.now(KST)


def today_kst() -> date:
    """오늘 날짜(KST). 서버가 UTC 여도 한국 날짜를 준다."""
    return now_kst().date()


def now_kst_naive() -> datetime:
    """현재 시각(KST 벽시계, naive). Game.game_date_time 과 비교할 때 쓴다."""
    return now_kst().replace(tzinfo=None)


def to_kst(moment: datetime) -> datetime:
    """aware datetime 을 KST 로 옮긴다. naive 는 이미 KST 벽시계로 본다."""
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(KST)
