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
        "google_maps_embed_api_key",
        mode="before",
    )
    @classmethod
    def _blank_key_is_absent(cls, value: object) -> object:
        """빈 문자열은 '키 없음'으로 취급한다.

        `.env` 에 `TAVILY_API_KEY=` 처럼 줄만 남기면 값이 ""(빈 문자열)로 들어와
        `is None` 가드가 전부 통과해 버린다. 그 결과 doctor 가 없는 키를 ok 로
        표시하고 web_search_enabled 가 참이 되어 매번 실패하는 호출이 나간다.

        `#` 로 시작하는 값도 '없음' 으로 본다. python-dotenv 는 값이 비어 있을 때만
        인라인 주석을 떼지 못해서, `KEY=            # 설명` 이 통째로 값이 된다.
        실측으로 확인했다 — 그러면 키가 있는 것처럼 보여 호출이 나가고 401 이 난다.
        """
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(raw, str) and (not raw.strip() or raw.lstrip().startswith("#")):
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
    # 임베딩 호출 실패(키·쿼터·네트워크)는 '자료에 없음'의 증거가 아니다. 기본은 abstain 하지 않는다.
    abstain_on_dense_failure: bool = False
    # bm25_ratio 의 분모. "short" 는 질문 길이 기준선(권장), "sentence" 는 예전 문장 기준선. tunable
    abstain_bm25_reference: Literal["sentence", "short"] = "short"
    rrf_weight_dense: float = 0.5
    rrf_weight_bm25: float = 0.5
    # 융합 점수가 1위의 이 비율 미만인 문서는 컨텍스트에서 제외(무관 자료가 답변 거부를 유발) tunable
    context_min_score_ratio: float = 0.55
    # 하한을 적용하기 전에 남겨 둘 상위 문서 수. 단채널만 찾은 정답이 잘려 나가는 것을 막는다. tunable
    context_min_docs: int = 3

    # --- 범위 게이트 ---
    # 야구 외 질문을 검색·답변 이전에 막는다. off 면 라우터의 off_topic 분기만 남는다.
    enable_scope_gate: Literal["on", "off"] = "on"
    # 사전으로 판정이 서지 않을 때만 저가 LLM 을 한 번 부른다. off 면 그런 질문은 통과시킨다.
    scope_gate_llm: Literal["auto", "off"] = "auto"

    # 규칙집과 웹 어디에도 근거가 없는 야구 질문에 모델의 일반 지식으로 답한다.
    # 출처가 없다는 사실을 배지로 구분해 보여 준다. off 면 그런 질문은 거부한다.
    enable_model_knowledge: Literal["on", "off"] = "on"

    # --- 최신정보 ---
    enable_web_search: Literal["auto", "on", "off"] = "auto"
    # 답변 모델이 web_search 도구를 부를 수 있는 최대 횟수. 0 이면 도구를 붙이지 않는다.
    # 라운드마다 LLM 호출 1회와 Tavily 최대 3시도가 붙는다. tunable
    max_tool_rounds: int = 2
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
    # 하이라이트는 지난 경기다. start=today 면 과거 경기가 스냅샷에 아예 없다.
    # 창은 전역에 하나여야 한다 — 캐시 키가 kind 문자열뿐이라 창이 둘이면 서로 덮어쓴다.
    kbo_schedule_lookback_days: int = 14

    # --- 신규 도구 (기본 off. tests/conftest.py 의 "테스트는 외부 API 를 부르지 않는다" 계약) ---
    enable_schedule_tool: Literal["on", "off"] = "off"
    enable_places: Literal["on", "off"] = "off"
    enable_places_map: Literal["on", "off"] = "off"
    enable_highlights: Literal["on", "off"] = "off"
    google_maps_embed_api_key: SecretStr | None = None
    places_max_results: int = 4                    # tunable
    # 기본 0.5 보다 높다. Tavily 는 맞는 게 없어도 채움용 결과를 주는데,
    # 맛집에서는 그게 곧 그럴듯한 가짜 가게다. 무응답이 오답보다 낫다.
    places_min_score: float = 0.6                  # tunable
    places_cache_ttl_seconds: int = 604800         # 7일. 가게는 천천히 바뀐다
    youtube_http_timeout_seconds: float = 4.0
    # YouTube ToS: 저장한 API 데이터는 30일 내 갱신하거나 지워야 한다. 7일이면 넉넉히 만족한다.
    youtube_cache_ttl_seconds: int = 604800
    highlight_max_videos: int = 4

    @property
    def kbo_data_enabled(self) -> bool:
        return self.enable_kbo_data == "on"

    @property
    def model_knowledge_enabled(self) -> bool:
        return self.enable_model_knowledge == "on"

    @property
    def answer_schema_enabled(self) -> bool:
        return self.enable_answer_schema == "on"

    @property
    def schedule_tool_enabled(self) -> bool:
        """일정 도구는 KBO 데이터에만 기댄다. Tavily 와 무관하다."""
        return self.enable_schedule_tool == "on" and self.kbo_data_enabled

    @property
    def places_enabled(self) -> bool:
        """맛집은 Tavily 를 쓰므로 웹 검색이 살아 있어야 한다."""
        return self.enable_places == "on" and self.web_search_enabled

    @property
    def places_map_enabled(self) -> bool:
        """지도는 키가 없으면 조용히 생략한다. 기능 저하는 허용하되 오류는 내지 않는다."""
        return (self.enable_places_map == "on"
                and self.google_maps_embed_api_key is not None)

    @property
    def highlights_enabled(self) -> bool:
        """하이라이트는 Tavily 로 찾고 YouTube API 로 검증한다. 둘 다 필요하다."""
        return (self.enable_highlights == "on" and self.web_search_enabled
                and self.youtube_kbo_api_key is not None)

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
