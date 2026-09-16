from __future__ import annotations

import pytest

from baseball.chunker import SPLIT_THRESHOLD, build, glossary
from baseball.pdf_parser import default_pdf_path, parse

pytestmark = pytest.mark.skipif(not default_pdf_path().exists(), reason="규칙집 PDF 없음")


@pytest.fixture(scope="module")
def chunks():
    sections, _ = parse(default_pdf_path())
    return build(sections, index_version="test-v1", document_id="doc")


def test_token_budget(chunks) -> None:
    assert max(c.tokens for c in chunks) <= SPLIT_THRESHOLD


def test_ids_are_unique_and_deterministic(chunks) -> None:
    assert len({c.id for c in chunks}) == len(chunks)
    sections, _ = parse(default_pdf_path())
    again = build(sections, index_version="test-v1", document_id="doc")
    assert [c.id for c in again] == [c.id for c in chunks]


def test_every_chunk_has_page_and_rule_id(chunks) -> None:
    assert all(c.page_start > 0 and c.page_end >= c.page_start and c.rule_id for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_no_chapter_crossing(chunks) -> None:
    merged = [c for c in chunks if "~" in c.rule_id]
    for c in merged:
        left, right = c.rule_id.split("~")
        assert left.split(".")[0] == right.split(".")[0]


def test_sub_item_metadata(chunks) -> None:
    subs = [c for c in chunks if c.rule_no == "5.09" and c.sub_item]
    assert subs, "5.09 하위 항목 청크가 있어야 한다"
    first = subs[0]
    assert first.rule_id.startswith("5.09(")
    assert first.parent_id == "5.09"
    assert first.breadcrumb.startswith("[5.00 경기의 진행] 5.09 아웃")
    deep = [c for c in chunks if c.sub_item and len(c.sub_item) >= 2]
    assert deep and deep[0].rule_id.count("(") >= 2


def test_terms_and_glossary(chunks) -> None:
    g = glossary(chunks)
    assert len(g) == 82
    assert [x["term_no"] for x in g] == list(range(1, 83))
    entry = next(x for x in g if x["term_en"] == "INFIELD FLY")
    assert entry["term_ko"] == "인필드 플라이" and entry["rule_id"] == "DEF-40"


def test_breadcrumb_prefix_is_in_content(chunks) -> None:
    for c in chunks[:50]:
        assert c.content.startswith(c.breadcrumb)
