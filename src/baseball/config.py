"""애플리케이션 설정. env 변수명 == Settings 필드명(대문자)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- 비밀 키(전부 optional: 키 없이 import·테스트 가능) ---
    openai_api_key: SecretStr | None = None
    youtube_kbo_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    langchain_api_key: SecretStr | None = None
    app_access_pin: SecretStr | None = None      # UI 접근 코드(숫자 4자리). 없으면 제한 없음
    langchain_tracing_v2: bool = False
    langchain_project: str = "baseball-rules-phase1"

    @field_validator(
        "openai_api_key", "youtube_kbo_api_key", "tavily_api_key", "langchain_api_key",
        mode="before",
    )
    @classmethod
    def _blank_key_is_absent(cls, value: object) -> object:
        """빈 문자열은 '키 없음'으로 취급한다.

        `.env` 에 `TAVILY_API_KEY=` 처럼 줄만 남기면 값이 ""(빈 문자열)로 들어와
        `is None` 가드가 전부 통과해 버린다. 그 결과 doctor 가 없는 키를 ok 로
        표시하고 web_search_enabled 가 참이 되어 매번 실패하는 호출이 나간다.
        """
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, SecretStr) and not value.get_secret_value().strip():
            return None
        return value

    # --- 모델 ---
    openai_chat_model: str = "gpt-4o-mini"
    openai_chat_model_fallback: str = "gpt-5.6-luna"
    openai_embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1024

    # --- 인프라 ---
    pg_dsn: str = "postgresql://baseball:baseball-local@127.0.0.1:55432/baseball"
    minio_endpoint: str = "127.0.0.1:59000"
    minio_access_key: str = "baseball-local"
    minio_secret_key: SecretStr = SecretStr("baseball-local-secret")
    minio_bucket: str = "baseball-rules"

    # --- 검색 ---
    abstain_dense_threshold: float = 0.30
    abstain_bm25_ratio: float = 0.15
    rrf_weight_dense: float = 0.5
    rrf_weight_bm25: float = 0.5
    # 융합 점수가 1위의 이 비율 미만인 문서는 컨텍스트에서 제외(무관 자료가 답변 거부를 유발) tunable
    context_min_score_ratio: float = 0.55

    # --- 최신정보 ---
    enable_web_search: Literal["auto", "on", "off"] = "auto"
    tavily_include_domains: str = (
        "koreabaseball.com,sports.news.naver.com,sports.naver.com,yna.co.kr,lgtwins.com"
    )

    # --- 컨텍스트/세션 (tunable) ---
    context_max_tokens: int = 3000
    latest_context_max_tokens: int = 800
    chat_history_max_messages: int = 8
    refusal_tail_max_chars: int = 40

    # --- 답변 스키마 (tunable) ---
    # off 면 구조화 출력 없이 평문으로 답한다. 롤백 레버이자 품질 A/B 레버다.
    enable_answer_schema: Literal["on", "off"] = "on"
    # JSON 은 평문보다 길다. 너무 낮으면 잘려서 폴백 재호출이 돌아 지연이 두 배가 된다.
    answer_max_output_tokens: int = 1600

    # --- API/UI ---
    api_base_url: str = "http://127.0.0.1:8000"
    cors_origins: str = "http://127.0.0.1:8501,http://localhost:8501"

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def tavily_include_domain_list(self) -> list[str]:
        return [d.strip() for d in self.tavily_include_domains.split(",") if d.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # --- KBO 데이터 (tunable) ---
    enable_kbo_data: Literal["on", "off"] = "on"
    kbo_season_year: int = 0                       # 0 이면 오늘 날짜의 연도를 쓴다
    kbo_http_timeout_seconds: float = 4.0
    kbo_standings_ttl_seconds: int = 600
    kbo_schedule_ttl_seconds: int = 1800
    kbo_context_max_tokens: int = 1000
    kbo_schedule_max_games: int = 10

    @property
    def kbo_data_enabled(self) -> bool:
        return self.enable_kbo_data == "on"

    @property
    def answer_schema_enabled(self) -> bool:
        return self.enable_answer_schema == "on"

    @property
    def web_search_enabled(self) -> bool:
        if self.enable_web_search == "off":
            return False
        if self.enable_web_search == "on":
            return True
        return self.tavily_api_key is not None

    def require_openai_key(self) -> str:
        if self.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다 (.env 확인)")
        return self.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
