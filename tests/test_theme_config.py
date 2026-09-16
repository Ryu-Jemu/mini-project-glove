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
