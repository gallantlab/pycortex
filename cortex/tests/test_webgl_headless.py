"""Tests for headless WebGL rendering across data types, angles, surfaces, and panels.

These tests exercise the WebGL viewer through the headless Chromium browser,
verifying that all supported data types, camera angles, surface morphing
states, and predefined panel layouts render correctly without a display server.

All tests are skipped if playwright is not installed.
"""

import os
import time

import numpy as np
import pytest

import cortex
import cortex.export
from cortex.export.save_views import (
    angle_view_params,
    default_view_params,
    unfold_view_params,
)
from cortex.tests.testing_utils import has_playwright

pytestmark = pytest.mark.skipif(
    not has_playwright, reason="playwright and chromium are required"
)

subj = "S1"
xfmname = "fullhead"
nverts = 304380
volshape = (31, 100, 100)

ALL_PANEL_PRESETS = {
    name: getattr(cortex.export, name)
    for name in sorted(dir(cortex.export))
    if name.startswith("params_")
}


def make_dataview(dtype_name):
    """Return a Dataview instance for the given type name."""
    np.random.seed(0)
    if dtype_name == "Volume":
        return cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    elif dtype_name == "Vertex":
        return cortex.Vertex(np.random.randn(nverts), subj)
    elif dtype_name == "VolumeRGB":
        r, g, b = [np.random.randn(*volshape) for _ in range(3)]
        return cortex.VolumeRGB(r, g, b, subj, xfmname)
    elif dtype_name == "VertexRGB":
        r, g, b = [np.random.randn(nverts) for _ in range(3)]
        return cortex.VertexRGB(r, g, b, subj)
    elif dtype_name == "Volume2D":
        a1, a2 = np.random.randn(*volshape), np.random.randn(*volshape)
        return cortex.Volume2D(a1, a2, subject=subj, xfmname=xfmname)
    elif dtype_name == "Vertex2D":
        a1, a2 = np.random.randn(nverts), np.random.randn(nverts)
        return cortex.Vertex2D(a1, a2, subject=subj)
    else:
        raise ValueError(f"Unknown dtype_name: {dtype_name}")


def _wait_for_file(path, timeout=30):
    """Poll until file exists and has nonzero size, raise after timeout."""
    for _ in range(int(timeout / 0.1)):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        time.sleep(0.1)
    raise RuntimeError(f"File {path!r} not written within {timeout}s")


# ---------------------------------------------------------------------------
# Group 1: Data type smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_name",
    ["Volume", "Vertex", "VolumeRGB", "VertexRGB", "Volume2D", "Vertex2D"],
)
def test_datatype_renders(dtype_name, tmp_path):
    """Each data type should render in the headless viewer without errors."""
    vol = make_dataview(dtype_name)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        time.sleep(10)
        outfile = str(tmp_path / "test.png")
        handle.getImage(outfile, (512, 384))
        _wait_for_file(outfile)
        assert os.path.isfile(outfile)
        assert os.path.getsize(outfile) > 0
        # No uncaught JS errors
        pageerrors = [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"


# ---------------------------------------------------------------------------
# Group 2: All predefined camera angles
# ---------------------------------------------------------------------------


class TestAllAngles:
    """Test all predefined camera angles render correctly.

    Uses a single headless browser session for all angles.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _setup_viewer(self, tmp_path_factory):
        vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        cls = type(self)
        cls.tmp_dir = tmp_path_factory.mktemp("angles")
        with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
            time.sleep(10)
            cls.handle = handle
            yield

    @pytest.mark.parametrize("angle_name", list(angle_view_params.keys()))
    def test_angle(self, angle_name):
        handle = type(self).handle
        view_params = {**default_view_params, **angle_view_params[angle_name]}
        if angle_name == "flatmap":
            view_params.update(unfold_view_params["flatmap"])
        else:
            view_params.update(unfold_view_params["inflated"])
        handle._set_view(**view_params)
        time.sleep(1)
        outfile = str(type(self).tmp_dir / f"{angle_name}.png")
        handle.getImage(outfile, (512, 384))
        _wait_for_file(outfile)
        assert os.path.isfile(outfile)
        assert os.path.getsize(outfile) > 1000, "Image too small — may be blank"


# ---------------------------------------------------------------------------
# Group 3: All surface types
# ---------------------------------------------------------------------------


class TestAllSurfaces:
    """Test all surface morph states render correctly.

    Uses a single headless browser session for all surfaces.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _setup_viewer(self, tmp_path_factory):
        vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        cls = type(self)
        cls.tmp_dir = tmp_path_factory.mktemp("surfaces")
        with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
            time.sleep(10)
            cls.handle = handle
            yield

    @pytest.mark.parametrize("surface_name", list(unfold_view_params.keys()))
    def test_surface(self, surface_name):
        handle = type(self).handle
        view_params = {
            **default_view_params,
            **angle_view_params["lateral_pivot"],
            **unfold_view_params[surface_name],
        }
        handle._set_view(**view_params)
        time.sleep(1)
        outfile = str(type(self).tmp_dir / f"{surface_name}.png")
        handle.getImage(outfile, (512, 384))
        _wait_for_file(outfile)
        assert os.path.isfile(outfile)
        assert os.path.getsize(outfile) > 1000, "Image too small — may be blank"


# ---------------------------------------------------------------------------
# Group 4: Predefined panel layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", list(ALL_PANEL_PRESETS.keys()))
def test_panel_preset(preset_name, tmp_path):
    """Each predefined panel layout should render without errors."""
    import matplotlib.pyplot as plt

    preset = ALL_PANEL_PRESETS[preset_name]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    save_name = str(tmp_path / f"{preset_name}.png")
    fig = cortex.export.plot_panels(
        vol,
        panels=preset["panels"],
        figsize=preset.get("figsize", (16, 9)),
        windowsize=(1024, 768),
        save_name=save_name,
        sleep=10,
        viewer_params={},
        headless=True,
    )
    assert fig is not None
    assert os.path.isfile(save_name)
    assert os.path.getsize(save_name) > 0
    plt.close(fig)


# ---------------------------------------------------------------------------
# Group 5: _capture_view roundtrip
# ---------------------------------------------------------------------------


def test_capture_view_roundtrip():
    """Setting view parameters and capturing them back should match."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        time.sleep(10)
        target_params = {
            "camera.azimuth": 90,
            "camera.altitude": 90,
        }
        handle._set_view(**target_params)
        time.sleep(2)
        captured = handle._capture_view()
        for key, expected in target_params.items():
            assert captured[key] == pytest.approx(
                expected, abs=1.0
            ), f"{key}: expected {expected}, got {captured[key]}"


# ---------------------------------------------------------------------------
# Group 6: Overlay visibility
# ---------------------------------------------------------------------------


def test_overlay_visibility_changes_image(tmp_path):
    """Rendering with and without overlays should produce different images."""
    from PIL import Image

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    view = {
        **default_view_params,
        **angle_view_params["lateral_pivot"],
        **unfold_view_params["inflated"],
    }

    # Render WITH overlays
    f1 = str(tmp_path / "with_overlay.png")
    with cortex.export.headless_viewer(
        vol, viewer_params=dict(overlays_visible=["rois"])
    ) as handle:
        time.sleep(10)
        handle._set_view(**view)
        time.sleep(1)
        handle.getImage(f1, (512, 384))
        _wait_for_file(f1)

    # Render WITHOUT overlays
    f2 = str(tmp_path / "without_overlay.png")
    with cortex.export.headless_viewer(
        vol, viewer_params=dict(overlays_visible=[])
    ) as handle:
        time.sleep(10)
        handle._set_view(**view)
        time.sleep(1)
        handle.getImage(f2, (512, 384))
        _wait_for_file(f2)

    img1 = np.array(Image.open(f1))
    img2 = np.array(Image.open(f2))
    assert not np.array_equal(img1, img2), "Images with/without overlays should differ"


# ---------------------------------------------------------------------------
# Group 7: Vertex NaN-mask regression tests (#612, #626)
# ---------------------------------------------------------------------------


def _count_red_pixels(png_path):
    """Count strongly red-dominant pixels (R - max(G, B) > 50)."""
    from PIL import Image

    rgb = np.array(Image.open(png_path))[..., :3].astype(int)
    return int((rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) > 50).sum())


def test_vertex_no_nan_renders_data(tmp_path):
    """A NaN-free Vertex must render visibly, not fall through to transparent.

    Regression test for #626: prior to the fix, the surface_vertex shader's
    nanmask attribute defaulted to zeros when the Python data had no NaNs,
    causing every vertex to be discarded and the brain to render with only
    the grayscale curvature underlay.
    """
    np.random.seed(0)
    # Constant high values + chromatic colormap so colored pixels are
    # easily distinguishable from the grayscale curvature underlay.
    data = np.full(nverts, 5.0)
    vtx = cortex.Vertex(data, subj, vmin=0, vmax=1, cmap="Reds")

    view = {
        **default_view_params,
        **angle_view_params["lateral_pivot"],
        **unfold_view_params["inflated"],
    }

    with cortex.export.headless_viewer(vtx, viewer_params={}) as handle:
        time.sleep(10)
        handle._set_view(**view)
        time.sleep(1)
        outfile = str(tmp_path / "vtx.png")
        handle.getImage(outfile, (512, 384))
        _wait_for_file(outfile)

        n_red = _count_red_pixels(outfile)
        assert n_red > 1000, (
            f"Vertex data does not appear to be rendering "
            f"(only {n_red} red-dominant pixels). "
            "Surface may be falling through to grayscale curvature (#626)."
        )


def test_vertex_with_nan_renders_partial(tmp_path):
    """A Vertex with some NaN values still renders the non-NaN portion (#612).

    Sanity check that the per-vertex NaN mask path keeps working: half-NaN
    data should render strictly fewer red pixels than fully-valid data, but
    still meaningfully more than zero.
    """
    np.random.seed(0)

    full = np.full(nverts, 5.0)
    half_nan = full.copy()
    half_nan[: nverts // 2] = np.nan

    view = {
        **default_view_params,
        **angle_view_params["lateral_pivot"],
        **unfold_view_params["inflated"],
    }

    def render(data, name):
        vtx = cortex.Vertex(data, subj, vmin=0, vmax=1, cmap="Reds")
        with cortex.export.headless_viewer(vtx, viewer_params={}) as handle:
            time.sleep(10)
            handle._set_view(**view)
            time.sleep(1)
            outfile = str(tmp_path / f"{name}.png")
            handle.getImage(outfile, (512, 384))
            _wait_for_file(outfile)
            return _count_red_pixels(outfile)

    n_full = render(full, "full")
    n_half = render(half_nan, "half_nan")

    assert n_full > 1000, "Fully-valid Vertex should render visibly"
    assert (
        n_half > 100
    ), "Half-NaN Vertex should still render the non-NaN half (#612 regression)"
    assert n_half < n_full, (
        f"Expected half-NaN render ({n_half} red px) to have fewer red "
        f"pixels than fully-valid render ({n_full} red px)"
    )


# ---------------------------------------------------------------------------
# Group 8: addData dataset switching
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="addData relies on _convert_dataset closure from show(), "
    "which is not available in the headless code path",
    raises=NameError,
)
def test_addData_no_crash():
    """Adding a second dataset to an open viewer should not crash."""
    vol1 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    vol2 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol1, viewer_params={}) as handle:
        time.sleep(10)
        handle.addData(second=vol2)
        time.sleep(2)
        pageerrors = [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors after addData: {pageerrors}"


# ---------------------------------------------------------------------------
# Group 8: show_multi (multi-viewer page)
# ---------------------------------------------------------------------------


def _free_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _MultiViewerHandle:
    """Spin up a show_multi server + Playwright page; clean up on exit."""

    def __init__(self, views, layout, yoke):
        self.views = views
        self.layout = layout
        self.yoke = yoke
        self.errors = []

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        port = _free_port()
        # autoclose=False so the server outlives the client disconnect when
        # Playwright reloads or navigates away during the test.
        self.server = cortex.webgl.show_multi(
            self.views,
            layout=self.layout,
            yoke=self.yoke,
            port=port,
            open_browser=False,
            autoclose=False,
            display_url=False,
            title="pytest multi viewer",
        )
        # Give the surface CTM endpoints a moment to come up.
        time.sleep(2)
        self._pw = sync_playwright().start()
        # swiftshader gives us a software WebGL context that survives multiple
        # contexts on a single headless process.
        self._browser = self._pw.chromium.launch(
            headless=True, args=["--no-sandbox", "--use-gl=swiftshader"]
        )
        self.page = self._browser.new_page(viewport={"width": 1200, "height": 700})
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.goto(
            "http://localhost:%d/" % port,
            wait_until="networkidle",
            timeout=45000,
        )
        # Wait for the JS-side viewers array to populate and CTMs to load.
        self.page.wait_for_function(
            "() => window.viewers && window.viewers.length === %d" % len(self.views),
            timeout=30000,
        )
        time.sleep(4)
        return self

    def __exit__(self, *exc):
        try:
            self._browser.close()
        finally:
            self._pw.stop()
            self.server.stop()


def test_show_multi_smoke():
    """show_multi(2 vols) renders two viewers, two canvases, no JS errors."""
    np.random.seed(0)
    v1 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    v2 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with _MultiViewerHandle([v1, v2], layout=(1, 2), yoke=False) as h:
        canvases = h.page.evaluate("document.querySelectorAll('canvas').length")
        n_viewers = h.page.evaluate("window.viewers.length")
        assert n_viewers == 2
        assert canvases >= 2  # at least one per viewer
        # No uncaught JS errors during construction.
        assert h.errors == [], f"JS pageerrors: {h.errors}"


def test_show_multi_yoke_lockstep():
    """With yoke ON, dispatching change on viewer 0 syncs viewer 1 azimuth + mix."""
    np.random.seed(0)
    v1 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    v2 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with _MultiViewerHandle([v1, v2], layout=(1, 2), yoke=True) as h:
        result = h.page.evaluate(
            """() => {
                window.viewers[1].controls.setAzimuth(45);
                window.viewers[0].controls.setAzimuth(200);
                window.viewers[0].controls.dispatchEvent({type:'change'});
                window.viewers[0].controls.setMix(0.5);
                window.viewers[0].controls.dispatchEvent({type:'change'});
                return {
                    az0: window.viewers[0].controls.azimuth,
                    az1: window.viewers[1].controls.azimuth,
                    mix0: window.viewers[0].controls.mix,
                    mix1: window.viewers[1].controls.mix,
                };
            }"""
        )
        assert result["az0"] == result["az1"]
        assert result["mix0"] == result["mix1"]
        assert h.errors == [], f"JS pageerrors: {h.errors}"


def test_show_multi_unyoked_independent():
    """With yoke OFF, change on viewer 0 does not affect viewer 1."""
    np.random.seed(0)
    v1 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    v2 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with _MultiViewerHandle([v1, v2], layout=(1, 2), yoke=False) as h:
        result = h.page.evaluate(
            """() => {
                window.viewers[1].controls.setAzimuth(45);
                window.viewers[0].controls.setAzimuth(200);
                window.viewers[0].controls.dispatchEvent({type:'change'});
                return {
                    az0: window.viewers[0].controls.azimuth,
                    az1: window.viewers[1].controls.azimuth,
                };
            }"""
        )
        assert result["az0"] == 200
        assert result["az1"] == 45
        assert h.errors == [], f"JS pageerrors: {h.errors}"
