"""팔레트 대비비(WCAG) — 네트워크·API 키 불필요."""
from __future__ import annotations

import pytest


def _luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = channel(r), channel(g), channel(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: str, bg: str) -> float:
    a, b = _luminance(fg), _luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


PAIRS = [
    ("light body", "#131313", "#FFFFFF", 4.5),
    ("light body on surface", "#131313", "#F4F4F4", 4.5),
    ("light muted", "#666666", "#FFFFFF", 4.5),
    ("light link", "#A50034", "#FFFFFF", 4.5),
    ("primary label", "#FFFFFF", "#C40037", 4.5),
    ("hover label", "#FFFFFF", "#A50034", 4.5),
    ("primary chip vs page", "#C40037", "#FFFFFF", 3.0),
    ("light input border", "#888888", "#FFFFFF", 3.0),
    ("dark body", "#F1F1F1", "#131313", 4.5),
    ("dark body on surface", "#F1F1F1", "#222222", 4.5),
    ("dark muted", "#A7A9AC", "#131313", 4.5),
    ("dark link", "#EB5E69", "#131313", 4.5),
    ("dark link on surface", "#EB5E69", "#222222", 4.5),
    ("dark primary label", "#FFFFFF", "#D92D49", 4.5),
    ("dark primary vs page", "#D92D49", "#131313", 3.0),
    ("dark input border", "#727171", "#131313", 3.0),
    ("light error", "#860F26", "#FFEBEA", 4.5),
    ("light warn", "#624000", "#FCEEDB", 4.5),
    ("light success", "#00581E", "#E3F6E5", 4.5),
    ("dark error", "#FFBEBE", "#3D1A1C", 4.5),
]


@pytest.mark.parametrize("name, fg, bg, minimum", PAIRS, ids=[p[0] for p in PAIRS])
def test_wcag_contrast(name: str, fg: str, bg: str, minimum: float) -> None:
    ratio = contrast_ratio(fg, bg)
    assert ratio >= minimum, f"{name}: {ratio:.2f} < {minimum}"


def test_brand_red_is_not_used_as_dark_text() -> None:
    """#C40037 은 다크 배경에서 본문·링크로 쓰면 안 된다(3.01:1)."""
    assert contrast_ratio("#C40037", "#131313") < 4.5
