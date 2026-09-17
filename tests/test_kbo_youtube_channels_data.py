"""유튜브 채널 파일. 하이라이트 발견이 통째로 이 파일에 기댄다.

코드가 하나라도 빠지면 그 구단 경기는 KBO 공식 채널에만 의존하게 되고,
채널 id 표기가 틀리면 조용히 0건이 된다. 둘 다 네트워크 없이 잡을 수 있다.
"""
from __future__ import annotations

import re

import pytest

from baseball import highlights, kbo, youtube

CHANNELS = highlights.load_channels()
ITEMS = CHANNELS.get("channels", [])
UC_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")


def test_file_loads_and_is_not_empty() -> None:
    assert ITEMS, "data/kbo_youtube_channels.json 을 읽지 못했다"
    assert CHANNELS.get("as_of")


def test_ten_clubs_plus_the_league_channel() -> None:
    codes = {c["code"] for c in ITEMS}
    assert codes == set(kbo.teams_by_code()) | {highlights.LEAGUE_CODE}


def test_codes_are_unique() -> None:
    codes = [c["code"] for c in ITEMS]
    assert len(codes) == len(set(codes))


@pytest.mark.parametrize("chan", ITEMS, ids=[c["code"] for c in ITEMS])
def test_each_entry_can_be_looked_up(chan: dict) -> None:
    """핸들이든 채널 id 든, 해석할 실마리가 하나는 있어야 한다."""
    assert chan.get("handle") or chan.get("username") or chan.get("channel_id"), chan["code"]


@pytest.mark.parametrize("chan", ITEMS, ids=[c["code"] for c in ITEMS])
def test_handles_start_with_an_at_sign(chan: dict) -> None:
    handle = chan.get("handle")
    assert handle is None or handle.startswith("@"), chan["code"]


@pytest.mark.parametrize("chan", ITEMS, ids=[c["code"] for c in ITEMS])
def test_channel_id_shape_when_present(chan: dict) -> None:
    """적혀 있으면 UC + 22자여야 한다. 틀리면 UU 변환이 조용히 None 이 된다."""
    cid = chan.get("channel_id")
    if cid:
        assert UC_RE.match(cid), f"{chan['code']}: {cid}"
        assert youtube.uploads_playlist_id(cid)


def test_the_league_channel_is_present() -> None:
    """구단 핸들이 전부 어긋나도 이 채널 하나면 회수가 선다."""
    assert highlights.LEAGUE_CODE in highlights.channels_by_code()


def test_channel_order_always_ends_with_the_league_channel() -> None:
    for home, away in (("LG", "SS"), ("", ""), ("없는코드", "OB")):
        assert highlights.channel_order(home, away)[-1]["code"] == highlights.LEAGUE_CODE
