"""PDF 구조 파싱 불변식(2026 공식야구규칙 기준)."""
from __future__ import annotations

import re

import pytest

from baseball.pdf_parser import (
    PAGE_MARKER_RE,
    RH_NUM_RE,
    RH_TERM_RE,
    RH_TITLE_RE,
    RULE_HEADER_RE,
    TERM_RE,
    default_pdf_path,
    parse,
)

pytestmark = pytest.mark.skipif(not default_pdf_path().exists(), reason="규칙집 PDF 없음")


@pytest.fixture(scope="module")
def parsed():
    return parse(default_pdf_path())


def test_global_stats(parsed) -> None:
    _, stats = parsed
    assert stats.pages == 220
    assert stats.body_start == 25
    assert stats.rules == 85
    assert stats.monotonic is True
    assert stats.terms == 82
    assert stats.markers_stripped == 188
    assert stats.running_headers_stripped == 171


def test_rule_numbers_are_monotonic_and_cover_all_chapters(parsed) -> None:
    sections, _ = parsed
    rules = [s for s in sections if s.doc_type == "rule"]
    nums = [float(s.rule_no) for s in rules]
    assert nums == sorted(nums)
    chapters = {s.rule_chapter for s in rules}
    assert {f"{i}.00" for i in range(1, 10)} <= chapters


def test_titles_use_measured_values(parsed) -> None:
    """1.01 은 본문 첫 줄이므로 제목 없음, 5.09 의 실제 헤더 제목은 '아웃'."""
    rules = {s.rule_no: s for s in sections_of(parsed) if s.doc_type == "rule"}
    assert rules["1.01"].rule_title is None
    assert rules["5.09"].rule_title == "아웃"
    assert rules["1.00"].rule_title == "경기의 목적"


def test_terms_are_contiguous(parsed) -> None:
    terms = [s for s in sections_of(parsed) if s.doc_type == "term"]
    assert [t.term_no for t in terms] == list(range(1, 83))
    assert terms[0].term_en == "ADJUDGED"
    assert terms[-1].term_en == "WIND-UP POSITION"
    assert terms[39].term_en == "INFIELD FLY" and terms[39].term_ko == "인필드 플라이"


def test_front_summary_page(parsed) -> None:
    front = [s for s in sections_of(parsed) if s.doc_type == "front"]
    assert len(front) == 1 and front[0].page_start == 7


def test_no_running_header_or_marker_survives(parsed) -> None:
    for section in sections_of(parsed):
        for _, line in section.lines:
            assert not PAGE_MARKER_RE.match(line)
            assert not RH_NUM_RE.match(line)
            assert not RH_TITLE_RE.match(line)
            assert not RH_TERM_RE.match(line)


def test_terms_running_header_removed_from_body(parsed) -> None:
    """'40  용어의 정의' 형태의 러닝 헤더만 제거 대상(본문 상호참조 문구는 보존)."""
    leftovers = [
        line for s in sections_of(parsed) for _, line in s.lines if RH_TERM_RE.match(line)
    ]
    assert leftovers == []
    cross_refs = [
        line for s in sections_of(parsed) for _, line in s.lines if "용어의 정의 34" in line
    ]
    assert cross_refs, "본문의 '용어의 정의 34. 참조' 같은 상호참조는 남아 있어야 한다"


def test_figure_caption_is_not_a_rule_header(pages_sample) -> None:
    """앞부분 그림 캡션 '5.02 (C)' 는 본문 헤더로 잡히면 안 된다(공백 1칸 규칙)."""
    for page_no in ("7", "21"):
        for line in pages_sample[page_no].split("\n"):
            m = RULE_HEADER_RE.match(line)
            assert m is None or " " not in m.group(3)[:1]


def test_fixture_pages_are_stable(pages_sample) -> None:
    assert "1.00 경기의 목적" in pages_sample["25"]
    assert "2.00" in pages_sample["26"]
    assert any(TERM_RE.match(line) for line in pages_sample["193"].split("\n"))
    assert any(RULE_HEADER_RE.match(line) for line in pages_sample["25"].split("\n"))
    # 러닝 헤더(공백 2칸)는 본문 헤더(공백 1칸)와 구분된다
    assert any(RH_NUM_RE.match(line) or RH_TITLE_RE.match(line)
               for line in pages_sample["56"].split("\n"))


def sections_of(parsed):
    return parsed[0]
