"""Tests for the optional guided-tour feature (cortex.webgl.make_static(tour=True))."""
import cortex.webgl  # noqa: F401  (ensures cortex.webgl is importable)
from cortex.webgl import serve
from cortex.webgl.FallbackLoader import FallbackLoader


def _render_base_template(tour):
    """Render the base webgl template with the given `tour` flag (no subject needed)."""
    loader = FallbackLoader([serve.cwd])
    tpl = loader.load("template.html")
    html = tpl.generate(
        title="test",
        leapmotion=False,
        python_interface=False,
        tour=tour,
        colormaps=[("RdBu_r", "colormaps/RdBu_r.png")],
        default_cmap="RdBu_r",
    )
    return html.decode("utf-8") if isinstance(html, bytes) else html


def test_tour_include_present_when_enabled():
    html = _render_base_template(True)
    assert "resources/js/tour.js" in html
    assert "resources/css/tour.css" in html


def test_tour_include_absent_by_default():
    html = _render_base_template(False)
    assert "resources/js/tour.js" not in html
    assert "resources/css/tour.css" not in html
