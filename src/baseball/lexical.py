"""Kiwi 형태소 기반 BM25 인메모리 인덱스(Postgres는 한국어 텍스트 검색 설정이 없음)."""
from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

# 내용어 태그만 사용(조사·어미 제거)
CONTENT_TAGS = {
    "NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "MAG", "SL", "SN", "SH", "XR", "XPN", "XSN",
}
RULE_NO_RE = re.compile(r"(?<![\d.])([1-9]\.\d{2})(?![\d.])")
SUB_MARKER_RE = re.compile(r"[⒜-⒵⑴-⒇]")

_kiwi: Kiwi | None = None


def kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        # multi.dict 는 위키데이터 다어절 고유명사 사전(12MB)으로 규칙집에는 쓰임이 없다.
        # 끄면 상주 메모리가 약 65MB 줄어 무료 배포 한도에 여유가 생긴다.
        _kiwi = Kiwi(load_multi_dict=False)
    return _kiwi


def tokenize(text: str) -> list[str]:
    """형태소(내용어) + 규칙번호 원형 + 하위항목 기호."""
    norm = unicodedata.normalize("NFKC", text or "")
    tokens: list[str] = []
    for token in kiwi().tokenize(norm):
        if token.tag in CONTENT_TAGS and len(token.form) > 1 or token.tag in {"SL", "SN", "SH"}:
            tokens.append(token.form.lower())
    tokens.extend(RULE_NO_RE.findall(norm))
    tokens.extend(SUB_MARKER_RE.findall(norm))
    return tokens


@dataclass
class LexicalHit:
    chunk_id: str
    score: float


class LexicalIndex:
    """active 문서의 전체 청크로 1회 빌드(lifespan)."""

    def __init__(self, chunks: Sequence[dict[str, Any]]) -> None:
        self.chunks = list(chunks)
        self.ids = [c["id"] for c in self.chunks]
        self.corpus = [tokenize(c["content"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None
        self.ref_score = self._reference_score()

    def _reference_score(self, sample: int = 30) -> float:
        """규칙서 자체 문장으로 질의했을 때 최고 점수의 90퍼센타일(abstain 기준선)."""
        if not self.bm25 or not self.chunks:
            return 1.0
        step = max(1, len(self.chunks) // sample)
        tops: list[float] = []
        for c in self.chunks[::step][:sample]:
            body = c["content"].split("\n", 1)[-1][:120]
            scores = self.bm25.get_scores(tokenize(body))
            if len(scores):
                tops.append(float(max(scores)))
        if not tops:
            return 1.0
        tops.sort()
        idx = min(len(tops) - 1, int(len(tops) * 0.9))
        return max(tops[idx], 1e-6)

    def search(self, query: str, k: int = 20, extra_terms: Iterable[str] = ()) -> list[LexicalHit]:
        if not self.bm25:
            return []
        tokens = tokenize(query)
        for t in extra_terms:
            tokens.extend(tokenize(t))
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [LexicalHit(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]
