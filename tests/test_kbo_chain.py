"""KBO 데이터가 붙었을 때의 체인 동작과 Phase 1 경로 보존."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from baseball.context import KboEntry, format_context
from baseball.kbo_models import StandingsSnapshot, TeamStanding

FIXTURES = Path(__file__).parent / "fixtures"


def _standings() -> StandingsSnapshot:
    r = json.loads((FIXTURES / "kbo_standings.json").read_text(encoding="utf-8"))["result"]
    return StandingsSnapshot(
        teams=[TeamStanding.model_validate(t) for t in r["seasonTeamStats"]],
        season=2026, as_of=date(2026, 9, 15), game_type=r["gameType"],
    )


def _entry(freshness: str = "live") -> KboEntry:
    return KboEntry(kind="standings", label="KBO 2026 정규시즌 순위",
                    text="1위 KT 76승 46패", as_of="2026-09-15",
                    source_label="네이버 스포츠", source_url="https://example.invalid",
                    freshness=freshness)


def test_kbo_block_header_is_exact() -> None:
    out = format_context([], [], [_entry()])
    assert out.startswith(
        "[자료 1 | KBO 데이터 | KBO 2026 정규시즌 순위 | 기준 2026-09-15 | "
        "출처 네이버 스포츠 | 상태 실시간]"
    )


def test_stale_status_is_visible_in_context() -> None:
    """오래된 저장본이라는 사실이 컨텍스트 안에 들어가야 모델이 말할 수 있다."""
    out = format_context([], [], [_entry("stale")])
    assert "상태 오래된 저장본(최신 조회 실패, 2026-09-15 기준)" in out


def test_unavailable_entry_is_dropped() -> None:
    assert format_context([], [], [_entry("unavailable")]) == ""


def test_kbo_budget_drops_blocks() -> None:
    assert format_context([], [], [_entry()], kbo_max_tokens=1) == ""


def test_numbering_runs_across_block_types() -> None:
    doc = {"content": "머리\n본문", "breadcrumb": "[5.00] 5.09", "page_start": 74, "tokens": 10}
    out = format_context([doc], [], [_entry()])
    assert "[자료 1 | 2026 공식야구규칙" in out
    assert "[자료 2 | KBO 데이터" in out


def test_braces_in_kbo_text_survive_prompt_render() -> None:
    from baseball.prompts import answer_prompt

    e = KboEntry(kind="standings", label="x", text="{중괄호} 포함", as_of="2026-09-15",
                 source_label="네이버 스포츠")
    ctx = format_context([], [], [e])
    msgs = answer_prompt(8).format_messages(question="Q", context=ctx)
    assert "{중괄호}" in msgs[1].content


def test_kbo_sources_shape() -> None:
    from baseball.chain import _kbo_sources

    src = _kbo_sources([_entry("live"), _entry("stale"), _entry("unavailable")])
    assert len(src) == 2                       # unavailable 은 빠진다
    assert src[0]["kind"] == "kbo"
    assert src[0]["confidence"] == "실시간"
    assert "오래된 저장본" in src[1]["confidence"]


def test_standings_text_lists_ten_teams() -> None:
    from baseball.kbo import _standings_text

    text = _standings_text(_standings())
    assert text.count("\n") == 9
    assert text.startswith("1위 KT")
    assert "10위 키움" in text


def test_phase1_latest_path_untouched_when_kbo_disabled(monkeypatch) -> None:
    """ENABLE_KBO_DATA=off 면 Phase 1 스냅샷 경로가 그대로여야 한다."""
    from baseball import latest_info
    from baseball.config import Settings

    freshness, entries = latest_info.route("피치클락 몇 초야?", Settings(enable_web_search="off"))
    assert freshness == "snapshot"
    assert entries
