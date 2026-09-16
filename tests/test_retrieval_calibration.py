"""검색 보정 3건을 고정한다.

배경: 기본 규칙 질문("야구 경기 목적이 뭐야?")이 거부되던 원인이 여기 있었다.
  - abstain 이 dense 호출 실패를 '자료에 없음'으로 해석했다.
  - bm25_ratio 의 분모가 질문과 다른 척도라 그 조건절이 늘 참이었다.
  - 컨텍스트 하한이 단채널만 찾은 정답 청크를 잘라 냈다.
"""
from __future__ import annotations

import pytest

from baseball.lexical import LexicalIndex
from baseball.retriever import select_context, should_abstain


def _settings(**kw):
    from baseball.config import Settings

    return Settings(_env_file=None, **kw)


# --------------------------------------------------------------------------- abstain

def test_dense_failure_alone_does_not_abstain() -> None:
    """임베딩을 못 부른 것은 자료가 없다는 증거가 아니다(기본값)."""
    assert should_abstain(
        _settings(), has_exact=False, dense_failed=True, dense_top=0.0, bm25_ratio=0.0
    ) is False


def test_dense_failure_can_abstain_when_opted_in() -> None:
    assert should_abstain(
        _settings(abstain_on_dense_failure=True),
        has_exact=False, dense_failed=True, dense_top=0.0, bm25_ratio=0.0,
    ) is True


def test_weak_dense_and_weak_bm25_abstains() -> None:
    assert should_abstain(
        _settings(), has_exact=False, dense_failed=False, dense_top=0.1, bm25_ratio=0.01
    ) is True


@pytest.mark.parametrize(
    ("dense_top", "bm25_ratio"),
    [(0.9, 0.01), (0.1, 0.9)],
    ids=["dense-strong", "bm25-strong"],
)
def test_either_channel_strong_prevents_abstain(dense_top: float, bm25_ratio: float) -> None:
    assert should_abstain(
        _settings(), has_exact=False, dense_failed=False,
        dense_top=dense_top, bm25_ratio=bm25_ratio,
    ) is False


def test_exact_rule_hit_never_abstains() -> None:
    assert should_abstain(
        _settings(), has_exact=True, dense_failed=True, dense_top=0.0, bm25_ratio=0.0
    ) is False


# --------------------------------------------------------------------------- bm25 기준선

def _corpus() -> list[dict[str, object]]:
    bodies = [
        ("1.00 경기의 목적", "야구는 아홉 명의 선수로 구성된 두 팀이 겨루는 경기이다"),
        ("5.09 아웃", "타자가 친 공을 야수가 땅에 닿기 전에 잡으면 타자는 아웃이 된다"),
        ("용어 인필드 플라이", "내야에 높이 뜬 공으로 심판이 선언하면 타자는 자동으로 아웃이다"),
        ("7.01 경기의 종료", "정규 경기는 아홉 이닝을 치러야 하며 동점이면 연장에 들어간다"),
    ]
    return [
        {"id": f"c{i}", "content": f"{head}\n{body}", "tokens": 20}
        for i, (head, body) in enumerate(bodies)
    ]


def test_short_reference_is_on_the_scale_of_real_questions() -> None:
    """문장 기준선은 질문보다 훨씬 커서 bm25_ratio 가 임계에 닿지 못했다."""
    idx = LexicalIndex(_corpus())
    assert idx.ref_score_short < idx.ref_score

    hits = idx.search("야구 경기 목적이 뭐야?", 10)
    assert hits, "질문이 어떤 청크와도 매칭되지 않으면 이 검사는 의미가 없다"
    best = hits[0].score

    assert best / idx.ref_score < best / idx.ref_score_short   # 새 척도가 더 후하다


def test_reference_scores_survive_empty_corpus() -> None:
    idx = LexicalIndex([])
    assert idx.ref_score == 1.0 and idx.ref_score_short == 1.0


# --------------------------------------------------------------------------- 컨텍스트 하한

def _ranked(*scores: float) -> list[dict[str, object]]:
    return [{"id": f"d{i}", "tokens": 10, "score_fused": s} for i, s in enumerate(scores)]


def test_floor_no_longer_drops_single_channel_docs_near_the_top() -> None:
    """1위가 두 채널 합의로 두 배가 되면 하한이 나머지를 전부 자르던 상황."""
    ranked = _ranked(0.016393, 0.007143, 0.007000, 0.006000)   # 실측 RRF 값 모양
    docs, _ = select_context(ranked, k=6, settings=_settings(context_min_docs=3))
    assert [d["id"] for d in docs] == ["d0", "d1", "d2"]       # 상위 3건은 하한 면제

    old_behaviour, _ = select_context(ranked, k=6, settings=_settings(context_min_docs=1))
    assert [d["id"] for d in old_behaviour] == ["d0"]          # 예전엔 1위만 남았다


def test_floor_still_drops_clearly_irrelevant_tail() -> None:
    ranked = _ranked(1.0, 0.9, 0.8, 0.01)
    docs, _ = select_context(ranked, k=6, settings=_settings(context_min_docs=3))
    assert [d["id"] for d in docs] == ["d0", "d1", "d2"]       # 꼬리는 여전히 잘린다


def test_token_budget_and_k_still_bound_the_result() -> None:
    ranked = _ranked(1.0, 1.0, 1.0, 1.0, 1.0)
    assert len(select_context(ranked, k=2, settings=_settings())[0]) == 2

    docs, used = select_context(ranked, k=6, settings=_settings(context_max_tokens=25))
    assert len(docs) == 2 and used == 20                       # 10+10, 세 번째는 예산 초과
