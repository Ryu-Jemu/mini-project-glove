from __future__ import annotations

from baseball.retriever import RRF_K, plan_query, rrf_fuse


def test_rrf_rewards_top_ranks_and_multiple_channels() -> None:
    channels = {"dense": ["a", "b", "c"], "bm25": ["b", "a"], "exact": ["c"]}
    weights = {"dense": 0.5, "bm25": 0.5, "exact": 1.0}
    scores = rrf_fuse(channels, weights)
    assert scores["a"] == 0.5 / (RRF_K + 1) + 0.5 / (RRF_K + 2)
    assert scores["c"] == 0.5 / (RRF_K + 3) + 1.0 / (RRF_K + 1)
    assert max(scores, key=scores.get) == "c"      # exact 가 rank 0 가중치를 받는다


def test_plan_query_extracts_rule_numbers_and_aliases() -> None:
    glossary = [{"term_en": "INFIELD FLY", "term_ko": "인필드 플라이", "aliases": ["인필드 플라이", "INFIELD FLY"]}]
    plan = plan_query("5.09 인필드 플라이가 뭐야", glossary)
    assert plan.exact_rule_nos == ["5.09"]
    assert "INFIELD FLY" in plan.bm25_extra

    assert plan_query("규칙5.09 알려줘", glossary).exact_rule_nos == ["5.09"]   # 한글 붙어도 탐지
    assert plan_query("3.141592 는?", glossary).exact_rule_nos == []            # 소수는 제외
