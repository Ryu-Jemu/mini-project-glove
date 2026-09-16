"""2026 공식야구규칙 PDF → 구조 인식 섹션. 줄바꿈은 절대 제거하지 않는다."""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import pymupdf

# --- 라인 정제 패턴(순서 중요) -------------------------------------------------
PAGE_MARKER_RE = re.compile(r"^\s*[․·.]\s*\d+\s*[․·.]\s*$")
RH_NUM_RE = re.compile(r"^\d\.\d{2}(~\d\.\d{2})?\s{2,}\S")            # '5.06  주루'
RH_TITLE_RE = re.compile(r"^\S.*\s{2,}\d\.\d{2}(~\d\.\d{2})?\s*$")    # '주루  5.06'
RH_TERM_RE = re.compile(
    r"^(?:\d+(?:~\d+)?\s{2,}용어의 정의|용어의 정의\s{2,}\d+(?:~\d+)?)\s*$"
)

# --- 구조 패턴 ---------------------------------------------------------------
BODY_ANCHOR_RE = re.compile(r"^1\.00 경기의 목적")
RULE_HEADER_RE = re.compile(r"^(\d)\.(\d{2}) (\S.*)$")                # 공백 정확히 1칸
TERMS_TITLE_RE = re.compile(r"^<용어의 정의>")
# 용어의 정의 뒤에 오는 부록(<야구 도량형> 등)·판권지는 용어 본문에 포함하지 않는다.
APPENDIX_RE = re.compile(r"^<(?!용어의 정의)[^>]+>")
TERM_RE = re.compile(r"^\s*(\d{1,3})\.\s*([A-Z][A-Za-z \-'’`/&.]+?)\s*\(([^)]+)\)\s*$")
CHANGE_SUMMARY_RE = re.compile(r"2026.*변경")

# 제목으로 채택하지 않을 어미(본문 첫 줄이 헤더 뒤에 붙는 경우 방지)
TITLE_BAD_TAIL_RE = re.compile(r"(니다|한다|된다|이다|는다|있다|없다)$")
TITLE_MAX_CHARS = 25

MIN_PAGE_CHARS = 50

DocType = Literal["front", "rule", "term"]


@dataclass(frozen=True)
class Page:
    number: int                 # 1-index
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class ParseStats:
    pages: int = 0
    body_start: int = 0
    rules: int = 0
    monotonic: bool = True
    terms: int = 0
    markers_stripped: int = 0
    running_headers_stripped: int = 0
    skipped_pages: list[int] = field(default_factory=list)
    term_gaps: list[int] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"pages={self.pages} body_start={self.body_start} rules={self.rules} "
            f"monotonic={self.monotonic} terms={self.terms} "
            f"markers_stripped={self.markers_stripped} "
            f"running_headers_stripped={self.running_headers_stripped}"
        )


@dataclass
class Section:
    doc_type: DocType
    lines: list[tuple[int, str]]          # (page_number, line)
    rule_no: str | None = None            # '5.09'
    rule_chapter: str | None = None       # '5.00'
    rule_title: str | None = None
    term_no: int | None = None
    term_en: str | None = None
    term_ko: str | None = None
    chapter_title: str | None = None
    header_page: int = 0

    @property
    def page_start(self) -> int:
        return self.lines[0][0] if self.lines else self.header_page

    @property
    def page_end(self) -> int:
        return self.lines[-1][0] if self.lines else self.header_page

    @property
    def text(self) -> str:
        return "\n".join(line for _, line in self.lines)


def _is_running_header(line: str) -> bool:
    return bool(RH_NUM_RE.match(line) or RH_TITLE_RE.match(line) or RH_TERM_RE.match(line))


def load_pages(pdf_path: str | Path, stats: ParseStats | None = None) -> tuple[list[Page], ParseStats]:
    """페이지별 텍스트를 읽어 러닝 헤더·페이지 마커만 제거한다(줄바꿈 보존)."""
    stats = stats or ParseStats()
    doc = pymupdf.open(str(pdf_path))
    pages: list[Page] = []
    stats.pages = doc.page_count
    for idx in range(doc.page_count):
        raw = doc[idx].get_text("text")
        kept: list[str] = []
        for line in raw.split("\n"):
            if PAGE_MARKER_RE.match(line):
                stats.markers_stripped += 1
                continue
            if _is_running_header(line):
                stats.running_headers_stripped += 1
                continue
            kept.append(line)
        page = Page(number=idx + 1, lines=tuple(kept))
        if len("".join(kept).strip()) < MIN_PAGE_CHARS:
            stats.skipped_pages.append(page.number)
        pages.append(page)
    doc.close()
    return pages, stats


def extract_front_table(pdf_path: str | Path, page_no: int = 7) -> list[str]:
    """p.7 '변경 요약' 은 표다. 평문화하면 변경 전/후 대응이 사라지므로 표로 읽어 문장으로 만든다."""
    try:
        doc = pymupdf.open(str(pdf_path))
        try:
            # find_tables() 는 stdout 으로 권고 메시지를 찍는다 → CLI 출력 오염 방지
            with contextlib.redirect_stdout(io.StringIO()):
                rows = doc[page_no - 1].find_tables().tables[0].extract()
        finally:
            doc.close()
    except Exception:
        return []
    lines: list[str] = []
    for row in rows[1:]:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        if len(cells) < 3 or not cells[0]:
            continue
        label, before, after = cells[0], cells[1], cells[2]
        if before in {"-", ""}:
            lines.append(f"{label}: {after}")
        elif after in {"-", ""}:
            lines.append(f"{label}: {before}")
        else:
            lines.append(f"{label}: {before} → {after}")
    return lines



def find_body_start(pages: Iterable[Page]) -> int:
    """'1.00 경기의 목적' 두 번째 등장 페이지(첫 등장은 목차)."""
    hits = [p.number for p in pages if any(BODY_ANCHOR_RE.match(l) for l in p.lines)]
    if len(hits) >= 2:
        return hits[1]
    if hits:
        return hits[0]
    raise ValueError("본문 시작 앵커('1.00 경기의 목적')를 찾지 못했습니다")


def find_terms_start(pages: Iterable[Page]) -> int:
    for p in pages:
        if any(TERMS_TITLE_RE.match(l) for l in p.lines):
            return p.number
    raise ValueError("'<용어의 정의>' 섹션을 찾지 못했습니다")


def _accept_title(candidate: str) -> str | None:
    t = candidate.strip()
    if not t or len(t) > TITLE_MAX_CHARS:
        return None
    if t.endswith((".", ",")) or TITLE_BAD_TAIL_RE.search(t):
        return None
    return t


def split(pages: list[Page], stats: ParseStats | None = None,
          pdf_path: str | Path | None = None) -> tuple[list[Section], ParseStats]:
    stats = stats or ParseStats()
    stats.pages = stats.pages or len(pages)
    body_start = find_body_start(pages)
    terms_start = find_terms_start(pages)
    stats.body_start = body_start
    by_number = {p.number: p for p in pages}
    sections: list[Section] = []

    # --- front: p.7 '2026년 변경 요약' --------------------------------------
    front_page = by_number.get(7)
    if front_page and any(CHANGE_SUMMARY_RE.search(l) for l in front_page.lines):
        table_lines = extract_front_table(pdf_path, 7) if pdf_path else []
        if table_lines:
            lines = [(7, "2026년 공식야구규칙 변경 요약사항")] + [(7, l) for l in table_lines]
        else:
            lines = [(7, l) for l in front_page.lines if l.strip()]
        if lines:
            sections.append(Section(doc_type="front", lines=lines, header_page=7))

    # --- rule: 본문 시작 ~ 용어의 정의 직전 ----------------------------------
    current: Section | None = None
    last_value = -1.0
    chapter_titles: dict[str, str] = {}
    for number in range(body_start, terms_start):
        page = by_number.get(number)
        if page is None:
            continue
        for line in page.lines:
            m = RULE_HEADER_RE.match(line)
            if m:
                rule_no = f"{m.group(1)}.{m.group(2)}"
                value = float(rule_no)
                if value >= last_value:  # 단조 증가만 헤더로 인정
                    last_value = value
                    title = _accept_title(m.group(3))
                    chapter = f"{m.group(1)}.00"
                    if rule_no.endswith(".00") and title:
                        chapter_titles[chapter] = title
                    current = Section(
                        doc_type="rule", lines=[], rule_no=rule_no,
                        rule_chapter=chapter, rule_title=title, header_page=number,
                    )
                    sections.append(current)
                    remainder = m.group(3).strip()
                    if title is None and remainder:
                        current.lines.append((number, remainder))
                    continue
            if current is not None and line.strip():
                current.lines.append((number, line))

    # --- term: 용어의 정의 --------------------------------------------------
    term: Section | None = None
    stop = False
    for number in range(terms_start, len(pages) + 1):
        if stop:
            break
        page = by_number.get(number)
        if page is None:
            continue
        for line in page.lines:
            if APPENDIX_RE.match(line):
                stop = True
                break
            m = TERM_RE.match(line)
            if m:
                term = Section(
                    doc_type="term", lines=[], term_no=int(m.group(1)),
                    term_en=m.group(2).strip(), term_ko=m.group(3).strip(),
                    header_page=number,
                )
                sections.append(term)
                continue
            if term is not None and line.strip():
                term.lines.append((number, line))

    # 본문이 없는 rule 섹션(= x.00 챕터 표제)은 제목을 본문으로 채워 보존한다.
    for sec in sections:
        if sec.doc_type == "rule" and not sec.lines:
            label = f"{sec.rule_no} {sec.rule_title}" if sec.rule_title else str(sec.rule_no)
            sec.lines = [(sec.header_page, label)]
    sections = [s for s in sections if s.lines]

    rules = [s for s in sections if s.doc_type == "rule"]
    terms = [s for s in sections if s.doc_type == "term"]
    stats.rules = len(rules)
    stats.terms = len(terms)
    nums = [float(s.rule_no) for s in rules if s.rule_no]
    stats.monotonic = all(a <= b for a, b in zip(nums, nums[1:]))
    expected = list(range(1, len(terms) + 1))
    stats.term_gaps = [t for t, e in zip([s.term_no for s in terms], expected) if t != e]
    # chapter title 역주입
    for s in rules:
        if s.rule_chapter:
            s.chapter_title = chapter_titles.get(s.rule_chapter)
    return sections, stats


def parse(pdf_path: str | Path) -> tuple[list[Section], ParseStats]:
    pages, stats = load_pages(pdf_path)
    return split(pages, stats, pdf_path=pdf_path)


def default_pdf_path() -> Path:
    from baseball.config import get_settings
    from baseball.storage import RULEBOOK_FILENAME

    return get_settings().base_dir / "data" / RULEBOOK_FILENAME


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m baseball.pdf_parser")
    ap.add_argument("command", choices=["stats", "dump"])
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--rule", default=None, help="dump: 특정 규칙 번호")
    args = ap.parse_args(argv)
    path = Path(args.pdf) if args.pdf else default_pdf_path()
    sections, stats = parse(path)
    if args.command == "stats":
        print(stats.line())
        if stats.skipped_pages:
            print(f"skipped_pages({len(stats.skipped_pages)}): {stats.skipped_pages}")
        if stats.term_gaps:
            print(f"term_gaps: {stats.term_gaps}")
        return 0
    for s in sections:
        if args.rule and s.rule_no != args.rule:
            continue
        head = s.rule_no or (f"DEF-{s.term_no}" if s.term_no else "FRONT")
        print(f"--- {head} title={s.rule_title!r} p.{s.page_start}-{s.page_end} ---")
        print(s.text[:600])
    return 0


if __name__ == "__main__":
    sys.exit(_main())
