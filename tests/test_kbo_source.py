"""HTTP 어댑터: 요청 형태와 실패 처리. 네트워크를 타지 않는다."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from baseball import kbo_naver
from baseball.kbo_models import KboUpstreamChanged

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 9, 15)
BLOCKED = ("koreabaseball.com", "statiz", "espn.com", "namu.wiki",
           "m.sports.naver.com/kbaseball/schedule", "feeds/videos.xml")


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _client(body: dict, captured: list[httpx.Request], status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_standings_request_shape() -> None:
    seen: list[httpx.Request] = []
    with _client(_fixture("kbo_standings"), seen) as c:
        snap = kbo_naver.fetch_standings(2026, today=TODAY, client=c)
    assert len(snap.teams) == 10
    url = str(seen[0].url)
    assert url.endswith("/statistics/categories/kbo/seasons/2026/teams")
    assert "api-gw.sports.naver.com" in url
    # 브라우저 위장 헤더를 보내지 않는다.
    assert "Referer" not in seen[0].headers


def test_schedule_request_params() -> None:
    seen: list[httpx.Request] = []
    with _client(_fixture("kbo_schedule_remaining"), seen) as c:
        kbo_naver.fetch_schedule(start=date(2026, 9, 16), end=date(2026, 10, 7),
                                 today=TODAY, client=c)
    q = dict(seen[0].url.params)
    assert q["categoryId"] == "kbo"
    assert q["upperCategoryId"] == "kbaseball"
    assert q["size"] == "1000"
    assert "roundCode" in q["fields"]          # 정규시즌 필터에 필요하다
    assert q["fromDate"] == "2026-09-16"


@pytest.mark.parametrize("status", [403, 500, 503])
def test_http_error_becomes_source_error(status: int) -> None:
    seen: list[httpx.Request] = []
    with _client({}, seen, status=status) as c:
        with pytest.raises(kbo_naver.KboSourceError):
            kbo_naver.fetch_standings(2026, today=TODAY, client=c)


def test_unsuccessful_body_becomes_source_error() -> None:
    seen: list[httpx.Request] = []
    with _client({"code": 500, "success": False}, seen) as c:
        with pytest.raises(kbo_naver.KboSourceError):
            kbo_naver.fetch_standings(2026, today=TODAY, client=c)


def test_connection_failure_becomes_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(kbo_naver.KboSourceError):
            kbo_naver.fetch_standings(2026, today=TODAY, client=c)


def test_shape_change_is_distinct_from_transport_failure() -> None:
    seen: list[httpx.Request] = []
    with _client({"code": 200, "success": True, "result": {"seasonTeamStats": "nope"}}, seen) as c:
        with pytest.raises(KboUpstreamChanged):
            kbo_naver.fetch_standings(2026, today=TODAY, client=c)


def test_module_never_targets_blocked_hosts() -> None:
    """금지 소스가 '요청 코드'에 없어야 한다.

    모듈 독스트링과 주석은 그 금지 사항을 적어 두는 자리이므로 검사에서 뺀다.
    실제로 요청이 나가는 곳은 문자열 상수뿐이므로 AST 로 상수만 훑는다.
    """
    import ast

    tree = ast.parse(Path("src/baseball/kbo_naver.py").read_text(encoding="utf-8"))
    # clean=False 로 읽어야 원본 상수와 같은 문자열이 나온다.
    docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value not in docstrings]
    joined = "\n".join(literals)
    for host in BLOCKED:
        assert host not in joined, f"금지 소스가 요청 코드에 있다: {host}"
