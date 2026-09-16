"""경기 하이라이트. 없을 때가 정상 경로이고, 제목이 규칙 번호로 오인되면 안 된다."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from baseball import citations as cite
from baseball import highlights, latest_info, youtube
from baseball.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"
GAME_DATE = date(2026, 9, 13)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(latest_info, "_cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(latest_info, "_cache_put", lambda *_a, **_k: None)
    youtube._UNITS.clear()
    yield
    youtube._UNITS.clear()


def _settings(**kw: Any) -> Settings:
    base = {"_env_file": None, "enable_web_search": "on", "enable_highlights": "on",
            "tavily_api_key": SecretStr("tvly-test"),
            "youtube_kbo_api_key": SecretStr("AIza-test")}
    return Settings(**{**base, **kw})


def _yt_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "youtube_videos_list.json").read_text(encoding="utf-8"))


def _tavily(urls: list[str]) -> httpx.Client:
    body = {"results": [{"title": f"t{i}", "url": u, "content": "c", "score": 0.9}
                        for i, u in enumerate(urls)]}

    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _yt(body: dict[str, Any] | None = None, status: int = 200) -> httpx.Client:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body if body is not None else _yt_payload())

    return httpx.Client(transport=httpx.MockTransport(handler))


def _find(urls: list[str], *, yt: httpx.Client | None = None, **skw: Any):
    return highlights.find(
        game_date=GAME_DATE, home_name="LG 트윈스", away_name="삼성 라이온즈",
        settings=_settings(**skw), client=yt or _yt(), tavily_client=_tavily(urls),
    )


# --------------------------------------------------------------------------- 매칭

def test_exact_game_scores_high() -> None:
    v = {"title": "[9.13 vs 삼성] LG 트윈스 하이라이트", "published_at": "2026-09-13"}
    assert highlights.score_match(v, game_date=GAME_DATE,
                                  opponents=("LG 트윈스", "삼성 라이온즈")) >= 2


def test_non_highlight_video_scores_zero() -> None:
    """하이라이트 키워드가 없으면 볼 것도 없다."""
    v = {"title": "프로야구 뉴스 9월 13일", "published_at": "2026-09-13"}
    assert highlights.score_match(v, game_date=GAME_DATE, opponents=("LG 트윈스",)) == 0


def test_other_game_falls_below_the_threshold() -> None:
    v = {"title": "[8.02 vs 두산] 하이라이트", "published_at": "2026-08-02"}
    score = highlights.score_match(v, game_date=GAME_DATE,
                                   opponents=("LG 트윈스", "삼성 라이온즈"))
    assert score < highlights.MIN_MATCH_SCORE


# --------------------------------------------------------------------------- 없을 때가 정상

def test_no_youtube_urls_returns_nothing() -> None:
    """TVING 독점이라 흔한 경우다. 예외가 아니라 빈 결과다."""
    assert _find(["https://blog.example/1", "https://news.example/2"]) == ([], [])


def test_no_candidate_scores_high_enough_returns_nothing() -> None:
    body = _yt_payload()
    body["items"][0]["snippet"]["title"] = "야구 규칙 설명 영상"
    body["items"][1]["snippet"]["title"] = "훈련 브이로그"
    out = _find(["https://youtu.be/aAAAAAAAAA1"], yt=_yt(body))
    assert out == ([], [])


def test_youtube_failure_degrades_quietly() -> None:
    out = _find(["https://youtu.be/aAAAAAAAAA1"], yt=_yt({"error": {}}, status=500))
    assert out == ([], [])


def test_disabled_returns_nothing() -> None:
    assert _find(["https://youtu.be/aAAAAAAAAA1"], enable_highlights="off") == ([], [])


def test_missing_youtube_key_disables_the_feature() -> None:
    assert _find(["https://youtu.be/aAAAAAAAAA1"], youtube_kbo_api_key=None) == ([], [])


# --------------------------------------------------------------------------- 찾았을 때

def test_found_videos_become_media_not_prose() -> None:
    entries, media = _find(["https://youtu.be/aAAAAAAAAA1", "https://youtu.be/bBBBBBBBBB2"])
    assert len(entries) == 1 and entries[0].kind == "highlight"
    assert [m["id"] for m in media] == ["aAAAAAAAAA1", "bBBBBBBBBB2"]
    assert all(m["url"] for m in media)          # dedupe 가 삼키지 않도록 url 이 있어야 한다


def test_non_embeddable_is_flagged_for_the_ui() -> None:
    _, media = _find(["https://youtu.be/aAAAAAAAAA1", "https://youtu.be/bBBBBBBBBB2"])
    flags = {m["id"]: m["embeddable"] for m in media}
    assert flags == {"aAAAAAAAAA1": True, "bBBBBBBBBB2": False}


def test_body_never_produces_a_fake_rule_citation() -> None:
    """citations.RULE_NO_RE 가 '9.13' 을 규칙 번호로 잡는다(실측).

    그래서 모델이 읽는 본문에는 '9월 13일' 로 적고 원래 제목은 media 로만 보낸다.
    """
    entries, media = _find(["https://youtu.be/aAAAAAAAAA1"])
    text = entries[0].text + entries[0].label
    assert "9.13" not in text
    assert "9월 13일" in text
    assert cite.extract(text, [])[1] == []       # dropped_citations 가 비어야 한다
    # 원래 제목은 화면용으로 살아 있다
    assert media[0]["title"] == "[9.13 vs 삼성] LG 트윈스 하이라이트"


def test_result_count_is_capped() -> None:
    _, media = _find(["https://youtu.be/aAAAAAAAAA1", "https://youtu.be/bBBBBBBBBB2"],
                     highlight_max_videos=1)
    assert len(media) == 1


def test_query_is_deterministic() -> None:
    assert highlights.build_query(GAME_DATE, "LG 트윈스", "삼성 라이온즈") == \
        "삼성 라이온즈 LG 트윈스 9월 13일 야구 하이라이트"


# --------------------------------------------------------------------------- 오탐 (실측 회귀)

def test_same_day_other_game_is_rejected() -> None:
    """실측 회귀: "LG 경기 하이라이트" 에 "[KT위즈 vs 한화이글스] 9.16(수)" 가 걸렸다.

    날짜 토큰만으로 임계값을 넘었기 때문이다. 구단 일치를 필수 조건으로 바꿨다.
    """
    opp = highlights._name_variants("LG 트윈스", "NC 다이노스")
    v = {"title": "[KT위즈 vs 한화이글스] 9.16(수) 야구 하이라이트｜2026 KBO 리그",
         "published_at": "2026-09-16"}
    assert highlights.score_match(v, game_date=date(2026, 9, 16), opponents=opp) == 0


@pytest.mark.parametrize(
    "title",
    ["[9.16 vs NC] LG 트윈스 하이라이트", "[LG vs NC] 9.16 하이라이트",
     "엘지 트윈스 9월 16일 하이라이트", "LG 하이라이트 9.16"],
)
def test_short_forms_still_match(title: str) -> None:
    """네이버는 "LG 트윈스" 로 주는데 영상 제목은 "LG"·"엘지"·"트윈스" 로 쓴다."""
    opp = highlights._name_variants("LG 트윈스", "NC 다이노스")
    v = {"title": title, "published_at": "2026-09-16"}
    assert highlights.score_match(v, game_date=date(2026, 9, 16),
                                  opponents=opp) >= highlights.MIN_MATCH_SCORE


def test_name_variants_include_aliases() -> None:
    got = highlights._name_variants("LG 트윈스", "NC 다이노스")
    assert {"LG", "엘지", "트윈스", "NC", "다이노스"} <= set(got)
