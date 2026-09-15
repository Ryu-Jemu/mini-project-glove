"""구단별 선수 명단 데이터.

상류에는 구단별 등록선수 엔드포인트가 없다. 대신 시즌 기록 목록을 전부 받아
구단으로 나눈다. 받아 둔 결과는 data/kbo_players.jsonl 에 한 줄 한 명으로 저장하고
앱은 그 파일만 읽는다. 매 요청마다 상류를 호출하지 않는다.

상류 함정:
  - pageSize 가 500 이하일 때만 응답한다. 1000 이상은 조용히 0건이 된다.
  - teamId·page 인자는 무시된다. 구단 분리는 받은 뒤에 한다.
  - 이닝은 '138 2/3' 같은 문자열이다. 정렬하려면 숫자로 바꿔야 한다.
  - 포지션은 profile 이라는 JSON '문자열' 안에 들어 있다.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "kbo_players.jsonl"
PLAYERS_PATH = "/statistics/categories/kbo/seasons/{year}/players"
MAX_PAGE_SIZE = 500                      # 이 값을 넘기면 0건이 된다
PLAYER_TYPES = ("HITTER", "PITCHER")

_INNING_RE = re.compile(r"^\s*(?:(\d+)\s*)?(?:(\d)\s*/\s*(\d))?\s*$")

# 입문자에게 의미가 통하는 기록만 남긴다. WAR·wOBA·wRC+ 같은 파생 지표는 뺀다.
HITTER_STATS = {
    "games": "hitterGameCount", "avg": "hitterHra", "hits": "hitterHit",
    "hr": "hitterHr", "rbi": "hitterRbi", "runs": "hitterRun",
    "sb": "hitterSb", "bb": "hitterBb", "so": "hitterKk",
    "obp": "hitterObp", "slg": "hitterSlg", "ops": "hitterOps",
}
PITCHER_STATS = {
    "games": "pitcherGameCount", "era": "pitcherEra", "win": "pitcherWin",
    "lose": "pitcherLose", "save": "pitcherSave", "hold": "pitcherHold",
    "so": "pitcherKk", "bb": "pitcherBb", "whip": "pitcherWhip",
    "quality_starts": "pitcherQs",
}


def parse_innings(raw: str | None) -> float | None:
    """'138 2/3' → 138.667. 정렬과 비교에 쓴다. 표시는 원문을 그대로 둔다."""
    if not raw:
        return None
    m = _INNING_RE.match(str(raw))
    if not m:
        return None
    whole, num, den = m.groups()
    total = float(whole) if whole else 0.0
    if num and den and float(den):
        total += float(num) / float(den)
    return round(total, 3)


def _position(row: dict[str, Any], player_type: str) -> str:
    raw = row.get("profile")
    if raw:
        try:
            value = (json.loads(raw).get("position") or "").strip()
            if value:
                return value
        except Exception:                                  # noqa: BLE001
            pass
    return "투수" if player_type == "PITCHER" else "야수"


def _round(value: Any, digits: int) -> Any:
    return round(float(value), digits) if isinstance(value, (int, float)) else value


def normalize(row: dict[str, Any], player_type: str) -> dict[str, Any]:
    """상류 한 행 → 저장할 한 줄. 필드 이름은 이 프로젝트의 표기를 따른다."""
    out: dict[str, Any] = {
        "player_id": str(row.get("playerId") or ""),
        "name": row.get("playerName") or "",
        "team_code": row.get("teamId") or "",
        "player_type": player_type,
        "position": _position(row, player_type),
        "back_number": row.get("backNumber"),
        "height_cm": row.get("height"),
        "weight_kg": row.get("weight"),
        "is_qualified": bool(row.get("isQualified")),
        "is_retired": row.get("isRetire") == "Y",
        "league_rank": row.get("ranking"),
    }
    table = HITTER_STATS if player_type == "HITTER" else PITCHER_STATS
    stats: dict[str, Any] = {}
    for name, source in table.items():
        value = row.get(source)
        if name in {"avg", "obp", "slg", "ops"}:
            value = _round(value, 3)
        elif name in {"era", "whip"}:
            value = _round(value, 2)
        stats[name] = value
    if player_type == "PITCHER":
        raw_innings = row.get("pitcherInning")
        stats["innings"] = raw_innings
        stats["innings_value"] = parse_innings(raw_innings)
    out["stats"] = stats
    return out


def fetch_all(year: int, *, timeout: float = 8.0, client: Any = None) -> list[dict[str, Any]]:
    """두 번 호출해 전 구단 선수를 모두 받는다."""
    import httpx

    from baseball.kbo_naver import BASE, KboSourceError

    own = client is None
    c = client or httpx.Client(timeout=timeout, follow_redirects=False)
    rows: list[dict[str, Any]] = []
    try:
        for player_type in PLAYER_TYPES:
            r = c.get(BASE + PLAYERS_PATH.format(year=year),
                      params={"playerType": player_type, "pageSize": MAX_PAGE_SIZE})
            if r.status_code != 200:
                raise KboSourceError(f"HTTP {r.status_code} ({player_type})")
            body = r.json()
            if body.get("success") is not True:
                raise KboSourceError(f"응답이 성공이 아님 ({player_type})")
            found = body.get("result", {}).get("seasonPlayerStats") or []
            if not found:
                raise KboSourceError(f"{player_type} 0건 — pageSize 상한을 확인하라")
            rows += [normalize(row, player_type) for row in found]
    finally:
        if own:
            c.close()
    return rows



def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """한 선수가 두 목록에 실리는 경우를 정리한다.

    상류는 타석에 한 번이라도 선 투수를 타자 목록에도 넣는다(실측 62명).
    소속은 목록이 아니라 포지션으로 정한다. 포지션이 투수면 투수 기록만 남긴다.
    """
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = row["player_id"]
        keep = best.get(pid)
        if keep is None:
            best[pid] = row
            continue
        wanted = "PITCHER" if row.get("position") == "투수" else "HITTER"
        row_fits = row["player_type"] == wanted
        keep_fits = keep["player_type"] == wanted
        if row_fits and not keep_fits:
            best[pid] = row
        elif row_fits == keep_fits:
            # 둘 다 맞거나 둘 다 아니면 출전 경기가 많은 쪽을 남긴다.
            if (row["stats"].get("games") or 0) > (keep["stats"].get("games") or 0):
                best[pid] = row
    return list(best.values())


def write_jsonl(rows: list[dict[str, Any]], *, year: int, path: Path = DATA_PATH,
                captured: date | None = None) -> dict[str, Any]:
    """첫 줄은 메타(언제·어디서 받았는지), 그다음부터 선수 한 명씩."""
    rows = dedupe(rows)
    captured = captured or date.today()
    meta = {
        "_meta": True,
        "season": year,
        "captured_at": captured.isoformat(),
        "source": "네이버 스포츠",
        "source_url": "https://m.sports.naver.com/kbaseball/record/index",
        "player_count": len(rows),
        "note": "구단별 등록선수 명단이 아니라 해당 시즌 출전 기록이 있는 선수 목록이다.",
    }
    order = {"HITTER": 0, "PITCHER": 1}
    rows = sorted(rows, key=lambda r: (r["team_code"], order.get(r["player_type"], 9),
                                       -(r["stats"].get("games") or 0), r["name"]))
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return meta


def iter_lines(path: Path = DATA_PATH) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


@lru_cache(maxsize=1)
def load() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """(메타, 구단코드 → 선수 목록). 파일이 없으면 빈 결과."""
    meta: dict[str, Any] = {}
    by_team: dict[str, list[dict[str, Any]]] = {}
    for row in iter_lines():
        if row.get("_meta"):
            meta = row
            continue
        by_team.setdefault(row.get("team_code", ""), []).append(row)
    return meta, by_team


def roster(team_code: str, player_type: str | None = None,
           *, include_retired: bool = False) -> list[dict[str, Any]]:
    _, by_team = load()
    rows = by_team.get(team_code, [])
    if not include_retired:
        rows = [r for r in rows if not r.get("is_retired")]
    if player_type:
        rows = [r for r in rows if r.get("player_type") == player_type]
    return rows


def captured_at() -> str:
    return str(load()[0].get("captured_at", ""))


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m baseball.kbo_players")
    ap.add_argument("command", choices=["build", "stats"],
                    help="build 는 상류에서 받아 파일을 새로 쓴다. stats 는 저장된 파일을 요약한다")
    ap.add_argument("--year", type=int, default=date.today().year)
    args = ap.parse_args(argv)

    if args.command == "build":
        fetched = fetch_all(args.year)
        rows = dedupe(fetched)
        meta = write_jsonl(rows, year=args.year)
        dropped = len(fetched) - len(rows)
        if dropped:
            print(f"  중복 {dropped}명 제거 (타석에 선 투수가 타자 목록에도 실린 경우)")
        by_team: dict[str, int] = {}
        for r in rows:
            by_team[r["team_code"]] = by_team.get(r["team_code"], 0) + 1
        print(f"wrote {DATA_PATH.relative_to(DATA_PATH.parents[1])} — "
              f"{meta['player_count']}명, {len(by_team)}구단, 기준 {meta['captured_at']}")
        print("  " + " ".join(f"{k}:{v}" for k, v in sorted(by_team.items())))
        return 0

    meta, by_team = load()
    if not meta:
        print("파일이 없다. python -m baseball.kbo_players build 를 먼저 실행하라.")
        return 1
    print(f"season={meta.get('season')} captured_at={meta.get('captured_at')} "
          f"players={meta.get('player_count')}")
    for code in sorted(by_team):
        rows = by_team[code]
        h = sum(1 for r in rows if r["player_type"] == "HITTER")
        p = len(rows) - h
        print(f"  {code}: 타자 {h:2d}  투수 {p:2d}  합 {len(rows):2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
