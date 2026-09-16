"""테마 정본(.streamlit/config.toml) 과 앱 상수의 일관성."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".streamlit" / "config.toml"
APP = ROOT / "ui" / "app.py"
COLOR_RE = re.compile(r"^#[0-9A-F]{6}$")

pytestmark = pytest.mark.skipif(not CONFIG.exists(), reason=".streamlit/config.toml 없음")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def test_brand_red_is_canonical(cfg) -> None:
    assert cfg["theme"]["light"]["primaryColor"].upper() == "#C40037"
    assert cfg["theme"]["light"]["linkColor"].upper() == "#A50034"
    assert cfg["theme"]["dark"]["primaryColor"].upper() == "#D92D49"
    assert "#FD312E" not in CONFIG.read_text(encoding="utf-8")


def test_no_club_typeface(cfg) -> None:
    raw = CONFIG.read_text(encoding="utf-8").lower()
    assert "lgtwins" not in raw and "invincible" not in raw and "azurefd" not in raw


def test_no_colors_in_bare_theme(cfg) -> None:
    colour_keys = {"primaryColor", "backgroundColor", "secondaryBackgroundColor",
                   "textColor", "linkColor", "borderColor", "base"}
    assert set(cfg["theme"]) & colour_keys == set()


def test_theme_sections_and_fonts(cfg) -> None:
    lines = [l for l in CONFIG.read_text(encoding="utf-8").splitlines() if l.startswith("[theme")]
    assert sum(1 for l in lines if not l.startswith("[[")) == 5
    faces = cfg["theme"]["fontFaces"]
    assert len(faces) == 2
    assert all(f["family"] == "Pretendard" for f in faces)
    assert all(f["url"].startswith("https://cdn.jsdelivr.net/npm/pretendard@1.3.9/") for f in faces)
    for variant in ("light", "dark"):
        assert len(cfg["theme"][variant]["chartSequentialColors"]) == 10


def test_all_colors_are_uppercase_hex(cfg) -> None:
    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)
        elif isinstance(node, str) and node.startswith("#"):
            yield node
    assert all(COLOR_RE.match(c) for c in walk(cfg["theme"]))


def test_server_binds_loopback(cfg) -> None:
    """로컬 실행은 루프백에 묶되, 바인딩 지정을 config.toml 에 커밋하지는 않는다.

    Streamlit Community Cloud 는 server.address 를 덮어쓴다고 문서화하지 않았고
    server.address 는 실제 소켓 바인딩이므로, 파일에 남기면 배포본이 외부에서 닿지 않는다.
    설정 우선순위가 '명령행 > 환경변수 > 파일' 이므로 Makefile 플래그로 강제한다.
    """
    assert "address" not in cfg["server"]
    assert "port" not in cfg["server"]
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    ui_line = next(l for l in makefile.splitlines() if l.startswith("ui:"))
    assert "--server.address 127.0.0.1" in ui_line
    assert "--server.port 8501" in ui_line


def test_theme_py_matches_toml(cfg) -> None:
    import sys
    sys.path.insert(0, str(ROOT / "ui"))
    import theme  # noqa: E402

    assert theme.BRAND["light"]["primary"] == cfg["theme"]["light"]["primaryColor"]
    assert theme.BRAND["dark"]["primary"] == cfg["theme"]["dark"]["primaryColor"]
    for key, toml_key in [("bg", "backgroundColor"), ("surface", "secondaryBackgroundColor"),
                          ("text", "textColor"), ("link", "linkColor")]:
        assert theme.BRAND["light"][key] == cfg["theme"]["light"][toml_key]
        assert theme.BRAND["dark"][key] == cfg["theme"]["dark"][toml_key]


def test_app_title_constant() -> None:
    """모듈 import 없이 소스 텍스트로 단언(Streamlit bare 모드 부작용 회피)."""
    match = re.search(r'^APP_TITLE = "(.+)"$', APP.read_text(encoding="utf-8"), re.M)
    assert match and match.group(1) == "KBO 야구 규칙 도우미"
    for artboard in ["Main.dc.html", "Sidebar.dc.html"]:
        text = (ROOT / "design" / "canvas" / artboard).read_text(encoding="utf-8")
        assert "KBO 야구 규칙 도우미" in text


def test_rulebook_url_points_at_a_branch_that_has_the_pdf() -> None:
    """주소에 브랜치 이름이 박혀 있다.

    한동안 main 을 가리켰는데 그 브랜치는 PDF 가 없는 별개 프로젝트라 사이드바의
    "원문 PDF 열기" 가 404 였다. 아무도 이걸 잡지 못했다.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
    import service

    assert "/blob/develop/" in service.RULEBOOK_URL
    assert service.RULEBOOK_URL.endswith(".pdf")
    assert "/blob/main/" not in service.RULEBOOK_URL


def test_rulebook_url_is_overridable_by_secret(monkeypatch) -> None:
    """배포 브랜치가 바뀌면 소스를 고치지 않고 시크릿으로 덮는다."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
    import service

    monkeypatch.setenv("RULEBOOK_PDF_URL", "https://example.com/a.pdf")
    assert service.document_url() == "https://example.com/a.pdf"
    monkeypatch.delenv("RULEBOOK_PDF_URL")
    assert service.document_url() == service.RULEBOOK_URL


def test_rulebook_pdf_url_reaches_the_deploy(monkeypatch) -> None:
    """시크릿 화이트리스트에 없으면 Cloud 에서 조용히 무시된다."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
    import service

    assert "RULEBOOK_PDF_URL" in service._SECRET_KEYS


# --------------------------------------------------------------------------- UI 불변식

def test_ui_never_uses_raw_html() -> None:
    """st.html 은 sanitize 된다. unsafe_allow_html·components.html 은 그렇지 않다.

    영상·지도·맛집 카드를 붙이면서 이 선을 넘고 싶어지는데, 1.63 에는 st.video·
    st.iframe·st.container(border=True) 가 1급 위젯으로 있어 넘을 이유가 없다.
    """
    from pathlib import Path

    for path in sorted(Path("ui").glob("*.py")):
        src = path.read_text(encoding="utf-8")
        assert "unsafe_allow_html" not in src, path
        assert "components.html" not in src, path


def test_new_settings_are_in_the_deployment_whitelist() -> None:
    """ui/service.py 의 _SECRET_KEYS 에 없는 env 는 Cloud 배포본에 전달되지 않는다.

    로컬에서는 절대 드러나지 않고 배포본에서만 조용히 꺼지므로 테스트로 막는다.
    """
    import re
    from pathlib import Path

    src = Path("ui/service.py").read_text(encoding="utf-8")
    block = re.search(r"_SECRET_KEYS = \((.*?)\n\)", src, re.S)
    assert block is not None
    listed = set(re.findall(r'"([A-Z0-9_]+)"', block.group(1)))
    required = {
        "ENABLE_SCHEDULE_TOOL", "ENABLE_PLACES", "ENABLE_PLACES_MAP", "ENABLE_HIGHLIGHTS",
        "GOOGLE_MAPS_EMBED_API_KEY", "YOUTUBE_KBO_API_KEY", "KBO_SCHEDULE_LOOKBACK_DAYS",
        "PLACES_MAX_RESULTS", "PLACES_MIN_SCORE", "PLACES_CACHE_TTL_SECONDS",
        "YOUTUBE_HTTP_TIMEOUT_SECONDS", "YOUTUBE_CACHE_TTL_SECONDS", "HIGHLIGHT_MAX_VIDEOS",
    }
    assert required <= listed, sorted(required - listed)


def test_chat_is_rendered_inside_its_tab() -> None:
    """tab_chat 이 선언만 되고 쓰이지 않아 '대화' 탭이 비어 있던 적이 있다."""
    from pathlib import Path

    src = Path("ui/app.py").read_text(encoding="utf-8")
    assert "with tab_chat:" in src


def test_answer_is_written_outside_the_status_box() -> None:
    """status 상자 안에 쓰면 질문한 그 턴에는 답변도 영상도 보이지 않는다."""
    from pathlib import Path

    src = Path("ui/app.py").read_text(encoding="utf-8").splitlines()
    opened = next(i for i, l in enumerate(src) if "st.status(" in l)
    written = next(i for i, l in enumerate(src) if "st.write_stream(" in l)
    # with 블록이 아니어야 하고, 들여쓰기가 더 깊어지지 않아야 한다
    assert "with st.status(" not in "\n".join(src)
    lead = lambda s: len(s) - len(s.lstrip())
    assert lead(src[written]) <= lead(src[opened])


def test_chat_input_stays_in_the_main_container() -> None:
    """탭·컬럼 안에 넣으면 streamlit 이 position="inline" 으로 그린다.

    하단 고정이 풀려 답변이 그 아래에 쌓이고, 입력창이 화면 위로 밀려 사라진 것처럼
    보인다. 근거: streamlit/elements/widgets/chat.py 의
    "Use bottom position if chat input is within the main container".
    실제로 '대화' 탭을 고치면서 한 번 깨뜨렸다.
    """
    from pathlib import Path

    for line in Path("ui/app.py").read_text(encoding="utf-8").splitlines():
        if "st.chat_input(" in line:
            assert not line.startswith((" ", "\t")), f"들여쓰기된 위치에 있다: {line!r}"
            break
    else:
        raise AssertionError("st.chat_input 을 찾지 못했다")


def test_evidence_is_not_shown_twice() -> None:
    """맛집 네 건이 카드로 한 번, 근거 자료 목록으로 또 한 번 나오던 것을 막는다."""
    from pathlib import Path

    src = Path("ui/app.py").read_text(encoding="utf-8")
    assert "shown.add(\"place\")" in src
    assert "s.get(\"kind\") not in shown" in src


def test_snippets_are_flattened_before_rendering() -> None:
    """st.caption 은 마크다운을 렌더한다. 블로그 제목(#)·표(|)가 그대로 들어가면
    카드 안에 거대한 제목과 빈 표가 생긴다."""
    from pathlib import Path

    src = Path("ui/app.py").read_text(encoding="utf-8")
    assert "_plain(" in src and "_MD_NOISE" in src
    assert "p[\"snippet\"][:120]" not in src          # 잘라내기만 하던 옛 코드
