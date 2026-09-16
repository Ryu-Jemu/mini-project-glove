"""네이버 스포츠 KBO 응답의 Pydantic 모델과 불변식.

상류는 비공개 API 라 예고 없이 형태가 바뀔 수 있다. 여기의 불변식은 그 변화를
조용히 넘기지 않고 즉시 드러내기 위한 것이다. 사용자 경로로는 절대 전파되지 않으며
캐시 계층이 붙잡아 저장본으로 물러난다.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

KBO_TEAM_CODES: frozenset[str] = frozenset({"KT", "SS", "LG", "HT", "OB", "NC", "SK", "HH", "LT", "WO"})
REGULAR_ROUND_CODE = "kbo_r"
GAME_ID_RE = re.compile(r"^\d{8}([A-Z]{2})([A-Z]{2})\d{5}$")
KNOWN_STATUS = {"BEFORE", "STARTED", "RESULT", "CANCEL", "POSTPONE"}
WRA_TOLERANCE = 0.0015


class KboUpstreamChanged(ValueError):
    """상류 스키마가 '조용히' 바뀐 것으로 판단될 때만 올린다."""


class TeamStanding(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    team_id: str = Field(alias="teamId")
    team_name: str = Field(alias="teamName")
    team_short: str = Field(alias="teamShortName")
    keyword: str = Field(alias="keyword")
    ranking: int = Field(alias="ranking")
    wra: float = Field(alias="wra")
    games: int = Field(alias="gameCount")
    wins: int = Field(alias="winGameCount")
    draws: int = Field(alias="drawnGameCount")
    losses: int = Field(alias="loseGameCount")
    game_behind: float = Field(alias="gameBehind")
    streak: str | None = Field(default=None, alias="continuousGameResult")
    last_five: str | None = Field(default=None, alias="lastFiveGames")

    @field_validator("team_id")
    @classmethod
    def _known_team(cls, v: str) -> str:
        if v not in KBO_TEAM_CODES:                       # 구단 코드 변경·구단 수 변동
            raise KboUpstreamChanged(f"알 수 없는 구단 코드: {v}")
        return v

    @field_validator("ranking")
    @classmethod
    def _rank_range(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError(f"순위 범위 벗어남: {v}")
        return v

    @field_validator("wra")
    @classmethod
    def _wra_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:                           # 비율 → 백분율 변경 감지
            raise KboUpstreamChanged(f"승률이 0~1 범위를 벗어남: {v}")
        return v

    @field_validator("last_five")
    @classmethod
    def _last_five(cls, v: str | None) -> str | None:
        if v and (len(v) > 5 or set(v) - set("WLD-")):
            raise ValueError(f"최근 5경기 표기 이상: {v!r}")
        return v

    @field_validator("keyword")
    @classmethod
    def _keyword_present(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("구단 표기(keyword)가 비어 있다")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "TeamStanding":
        if self.games != self.wins + self.draws + self.losses:
            raise KboUpstreamChanged(
                f"{self.team_id}: 경기수 {self.games} != 승{self.wins}+무{self.draws}+패{self.losses}"
            )
        decided = self.wins + self.losses
        if decided and abs(self.wra - self.wins / decided) > WRA_TOLERANCE:
            # 승률 정의가 바뀌면(무승부 포함 등) 조용히 다른 수를 말하게 된다.
            raise KboUpstreamChanged(f"{self.team_id}: 승률 정의가 승/(승+패)와 다르다")
        return self


class StandingsSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    teams: list[TeamStanding]
    season: int
    as_of: date
    game_type: str = "REGULAR_SEASON"

    @model_validator(mode="after")
    def _league_shape(self) -> "StandingsSnapshot":
        if len(self.teams) != 10:
            raise KboUpstreamChanged(f"구단 수가 10이 아님: {len(self.teams)}")
        if {t.ranking for t in self.teams} != set(range(1, 11)):
            raise KboUpstreamChanged("순위가 1~10의 순열이 아님")
        by_wra = [t.team_id for t in sorted(self.teams, key=lambda t: -t.wra)]
        by_rank = [t.team_id for t in sorted(self.teams, key=lambda t: t.ranking)]
        if by_wra != by_rank:
            raise KboUpstreamChanged("승률 정렬과 순위가 어긋남")
        if sum(t.wins for t in self.teams) != sum(t.losses for t in self.teams):
            raise KboUpstreamChanged("리그 전체 승수와 패수가 맞지 않음")
        if self.game_type != "REGULAR_SEASON":
            raise KboUpstreamChanged(f"정규시즌이 아님: {self.game_type}")
        return self

    def by_code(self, code: str) -> TeamStanding | None:
        return next((t for t in self.teams if t.team_id == code), None)


class Game(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    game_id: str = Field(alias="gameId")
    game_date: date = Field(alias="gameDate")
    game_date_time: datetime = Field(alias="gameDateTime")
    round_code: str = Field(default="", alias="roundCode")
    stadium: str | None = Field(default=None, alias="stadium")
    home_code: str = Field(alias="homeTeamCode")
    away_code: str = Field(alias="awayTeamCode")
    home_name: str = Field(default="", alias="homeTeamName")
    away_name: str = Field(default="", alias="awayTeamName")
    home_score: int = Field(default=0, alias="homeTeamScore")
    away_score: int = Field(default=0, alias="awayTeamScore")
    status_code: str = Field(alias="statusCode")
    cancel: bool = Field(default=False, alias="cancel")
    suspended: bool = Field(default=False, alias="suspended")

    @field_validator("game_date", mode="before")
    @classmethod
    def _date_str(cls, v: Any) -> Any:
        if isinstance(v, str) and len(v) == 8 and v.isdigit():
            return f"{v[:4]}-{v[4:6]}-{v[6:]}"
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "Game":
        m = GAME_ID_RE.match(self.game_id)
        if not m:
            raise KboUpstreamChanged(f"경기 ID 형식 변경: {self.game_id}")
        if (m.group(1), m.group(2)) != (self.away_code, self.home_code):
            # 경기 ID 는 원정+홈 순서다. 어긋나면 홈/원정이 뒤바뀐 것이다.
            raise KboUpstreamChanged(f"{self.game_id}: 홈/원정 코드가 경기 ID와 어긋남")
        if self.home_code == self.away_code:
            raise ValueError(f"{self.game_id}: 홈과 원정이 같다")
        if self.game_date != self.game_date_time.date():
            raise KboUpstreamChanged(f"{self.game_id}: 날짜와 일시가 어긋남")
        if self.status_code == "BEFORE" and (self.home_score or self.away_score):
            raise KboUpstreamChanged(f"{self.game_id}: 시작 전인데 점수가 있다")
        if self.round_code == REGULAR_ROUND_CODE:
            # 올스타전(kbo_as)은 EA/WE 코드를 쓰므로 정규시즌에만 코드 검사를 건다.
            if self.home_code not in KBO_TEAM_CODES or self.away_code not in KBO_TEAM_CODES:
                raise KboUpstreamChanged(f"{self.game_id}: 정규시즌에 알 수 없는 구단 코드")
        return self

    @property
    def is_regular(self) -> bool:
        return self.round_code == REGULAR_ROUND_CODE


class ScheduleSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    games: list[Game]
    as_of: date
    total: int | None = None

    @model_validator(mode="after")
    def _shape(self) -> "ScheduleSnapshot":
        if self.total is not None and len(self.games) != self.total:
            raise KboUpstreamChanged(f"경기 수 불일치: {len(self.games)} != {self.total}")
        if self.games:
            known = sum(1 for g in self.games if g.status_code in KNOWN_STATUS)
            if known / len(self.games) < 0.9:
                raise KboUpstreamChanged("경기 상태 코드 체계가 바뀐 것으로 보임")
        return self

    def remaining(self, today: date) -> list[Game]:
        """정규시즌, 시작 전, 취소 아님, 오늘 이후. 넷 다 필요하다.

        취소된 우천 경기도 statusCode 는 BEFORE 로 남으므로 cancel 을 함께 봐야 한다.
        """
        return [
            g for g in self.games
            if g.is_regular and g.status_code == "BEFORE" and not g.cancel and g.game_date >= today
        ]

    def upcoming(self, now: datetime) -> list["Game"]:
        """아직 시작하지 않은 경기를 이른 순으로. remaining 과 달리 시각 단위다.

        remaining 은 날짜 단위(game_date >= today)라 세 시간 전에 시작한 오늘 경기도
        '남은 경기' 로 친다. "가장 가까운 경기" 에는 그 구분이 필요하다.

        now 는 반드시 naive KST 여야 한다(clock.now_kst_naive()). game_date_time 이
        네이버가 준 naive KST 벽시계라, aware 를 넣으면 TypeError 가 난다.
        """
        return sorted(
            (g for g in self.games
             if g.is_regular and not g.cancel and g.status_code == "BEFORE"
             and g.game_date_time >= now),
            key=lambda g: g.game_date_time,
        )

    def last_finished(self, now: datetime) -> "Game | None":
        """이미 끝난 경기 중 가장 최근. 하이라이트가 가리킬 경기다."""
        done = [
            g for g in self.games
            if g.is_regular and not g.cancel and g.game_date_time <= now
            and g.status_code in {"RESULT", "STARTED"}
        ]
        return max(done, key=lambda g: g.game_date_time) if done else None

    def for_team(self, code: str) -> list["Game"]:
        return [g for g in self.games if code in (g.home_code, g.away_code)]
