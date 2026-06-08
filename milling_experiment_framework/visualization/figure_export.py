from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


_ORIGINAL_FIGURE_SAVEFIG: Any | None = None
_PATCHED = False
_SAVING_DUAL = False


def enable_dual_figure_export() -> None:
    """Patch matplotlib so future PNG saves also emit publication SVG files."""
    global _ORIGINAL_FIGURE_SAVEFIG, _PATCHED
    if _PATCHED:
        return

    _ORIGINAL_FIGURE_SAVEFIG = Figure.savefig

    def savefig_with_svg(self: Figure, fname: Any, *args: Any, **kwargs: Any) -> Any:
        global _SAVING_DUAL
        result = _ORIGINAL_FIGURE_SAVEFIG(self, fname, *args, **kwargs)
        if _SAVING_DUAL or not _should_auto_save_svg(fname, kwargs):
            return result
        svg_path = Path(fname).with_suffix(".svg")
        svg_kwargs = _svg_kwargs(kwargs)
        _SAVING_DUAL = True
        try:
            with _publication_svg_rc():
                _ORIGINAL_FIGURE_SAVEFIG(self, svg_path, *args, **svg_kwargs)
        finally:
            _SAVING_DUAL = False
        return result

    Figure.savefig = savefig_with_svg
    _PATCHED = True


def save_figure_dual(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    svg_dpi: int = 600,
    bbox_inches: str | None = "tight",
    facecolor: str = "white",
    **kwargs: Any,
) -> tuple[Path, Path]:
    """Save a matplotlib figure as publication-ready PNG and SVG files.

    SVG is vector output, but ``svg_dpi`` still matters for any rasterized
    artists embedded in the SVG.
    """
    png_path = Path(path).with_suffix(".png")
    svg_path = png_path.with_suffix(".svg")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    png_kwargs = {
        "dpi": int(dpi),
        "bbox_inches": bbox_inches,
        "facecolor": facecolor,
        **kwargs,
    }
    svg_kwargs = {
        "format": "svg",
        "dpi": max(int(svg_dpi), int(dpi)),
        "bbox_inches": bbox_inches,
        "facecolor": facecolor,
        **kwargs,
    }

    original = _ORIGINAL_FIGURE_SAVEFIG or Figure.savefig
    original(fig, png_path, **png_kwargs)
    with _publication_svg_rc():
        original(fig, svg_path, **svg_kwargs)
    return png_path, svg_path


def _should_auto_save_svg(fname: Any, kwargs: dict[str, Any]) -> bool:
    if not isinstance(fname, (str, Path)):
        return False
    path = Path(fname)
    explicit_format = kwargs.get("format")
    if explicit_format is not None and str(explicit_format).lower() != "png":
        return False
    return path.suffix.lower() == ".png"


def _svg_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    svg_kwargs = dict(kwargs)
    dpi = svg_kwargs.get("dpi", 300)
    try:
        svg_dpi = max(600, int(dpi))
    except Exception:
        svg_dpi = 600
    svg_kwargs["format"] = "svg"
    svg_kwargs["dpi"] = svg_dpi
    return svg_kwargs


@contextmanager
def _publication_svg_rc():
    import matplotlib as mpl

    previous = {
        "svg.fonttype": mpl.rcParams.get("svg.fonttype"),
        "svg.image_inline": mpl.rcParams.get("svg.image_inline"),
    }
    try:
        mpl.rcParams["svg.fonttype"] = "none"
        mpl.rcParams["svg.image_inline"] = True
        yield
    finally:
        for key, value in previous.items():
            mpl.rcParams[key] = value
