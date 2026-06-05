"""Configure matplotlib for Chinese labels on Linux / Windows / macOS."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_SELECTED_FONT: Optional[str] = None

# Bundled font (optional): place NotoSansSC-Regular.otf here
_BUNDLED_FONT = Path(__file__).resolve().parent / "fonts" / "NotoSansSC-Regular.otf"

# System font paths (first match wins)
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
    "C:/Windows/Fonts/simhei.ttf",  # SimHei
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    str(_BUNDLED_FONT),
]


def _register_font(path: Path) -> str:
    import matplotlib.font_manager as fm

    fm.fontManager.addfont(str(path))
    return fm.FontProperties(fname=str(path)).get_name()


def configure_matplotlib_chinese(*, verbose: bool = False) -> Optional[str]:
    """
    Register a CJK-capable font and set matplotlib rcParams.
    Returns the font family name, or None if no suitable font was found.
    """
    global _CONFIGURED, _SELECTED_FONT
    if _CONFIGURED:
        return _SELECTED_FONT

    import matplotlib.pyplot as plt

    for candidate in _FONT_CANDIDATES:
        path = Path(candidate)
        if not path.is_file():
            continue
        try:
            name = _register_font(path)
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            _SELECTED_FONT = name
            _CONFIGURED = True
            if verbose:
                print(f"[plot] 中文字体: {name} ({path})")
            return name
        except OSError:
            continue

    # Legacy names only (often missing on Linux)
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    _CONFIGURED = True
    warnings.warn(
        "[plot] 未找到可用的中文字体文件，图表中文可能显示为方框。"
        " 请安装: sudo apt-get install -y fonts-noto-cjk"
        " 或将 NotoSansSC-Regular.otf 放到 HMARL/rescuenet/fonts/",
        stacklevel=2,
    )
    return None


# Apply on import so any `import matplotlib.pyplot` after rescuenet.plot_curves works.
configure_matplotlib_chinese()
