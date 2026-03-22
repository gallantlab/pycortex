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


def _wait_for_file(path, timeout=20):
    """Poll until file exists, raise after timeout."""
    for _ in range(int(timeout / 0.1)):
        if os.path.exists(path):
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
# Group 7: addData dataset switching
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
