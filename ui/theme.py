"""Streamlit 테마 보조 — 색상 정본은 .streamlit/config.toml 이고 여기서는 칩·알림만 그린다."""
from __future__ import annotations

from typing import Literal

import streamlit as st

APP_TITLE = "KBO 야구 규칙 도우미"

BRAND: dict[str, dict[str, str]] = {
    "light": {"primary": "#C40037", "primary_hover": "#A50034", "on_primary": "#FFFFFF",
              "bg": "#FFFFFF", "surface": "#F4F4F4", "text": "#131313", "muted": "#666666",
              "border": "#E2E2E2", "border_interactive": "#888888", "link": "#A50034",
              "focus": "#C40037", "error_bg": "#FFEBEA", "error_text": "#860F26",
              "chip_snapshot": "#6D6E71"},
    "dark": {"primary": "#D92D49", "primary_hover": "#EB5E69", "on_primary": "#FFFFFF",
             "bg": "#131313", "surface": "#222222", "text": "#F1F1F1", "muted": "#A7A9AC",
             "border": "#333333", "border_interactive": "#727171", "link": "#EB5E69",
             "focus": "#FF9FA1", "error_bg": "#3D1A1C", "error_text": "#FFBEBE",
             "chip_snapshot": "#A7A9AC"},
}

CHIP_PREFIX = {"rule": "규칙집 · ", "snapshot": "스냅샷 · ", "web": "웹 · ", "kbo": "KBO · "}


def theme_mode() -> Literal["light", "dark"]:
    mode = getattr(getattr(st.context, "theme", None), "type", None)
    return mode if mode in {"light", "dark"} else "light"


def build_css(c: dict[str, str]) -> str:
    return f"""<style>
.st-key-sources-panel .tw-chip {{
  display:inline-flex; align-items:center; gap:4px; margin:0 6px 6px 0;
  border:1px solid {c['border_interactive']}; border-radius:999px;
  padding:3px 11px; font-size:0.78rem; line-height:1.5;
}}
.st-key-sources-panel .tw-chip--rule {{ border-color:{c['primary']}; color:{c['primary']}; }}
.st-key-sources-panel .tw-chip--snapshot {{ border-color:{c['chip_snapshot']}; color:{c['chip_snapshot']}; }}
.st-key-sources-panel .tw-chip--web {{ border-style:dashed; color:{c['muted']}; }}
.st-key-sources-panel .tw-chip--kbo {{ border-color:{c['link']}; color:{c['link']}; }}
.st-key-partial-notice {{
  background:{c['error_bg']}; color:{c['error_text']};
  border:1px solid {c['error_text']}; border-radius:10px; padding:10px 14px; font-size:0.86rem;
}}
.st-key-app-footer {{ color:{c['muted']}; font-size:0.8rem; }}
.stButton>button:focus-visible,
[data-testid="stChatInput"] textarea:focus-visible {{
  outline:2px solid {c['focus']}; outline-offset:2px;
}}
</style>"""


def inject_theme(mode: Literal["light", "dark"] | None = None) -> None:
    st.html(build_css(BRAND[mode or theme_mode()]))


def chip(kind: str, label: str) -> str:
    prefix = CHIP_PREFIX.get(kind, "")
    safe = (label or "").replace("<", "&lt;").replace(">", "&gt;")
    return f'<span class="tw-chip tw-chip--{kind}">{prefix}{safe}</span>'
