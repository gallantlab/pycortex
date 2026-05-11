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
        handle._set_view(**view)
        time.sleep(1)
        handle.getImage(f1, (512, 384))
        _wait_for_file(f1)

    # Render WITHOUT overlays
    f2 = str(tmp_path / "without_overlay.png")
    with cortex.export.headless_viewer(
        vol, viewer_params=dict(overlays_visible=[])
    ) as handle:
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
# Group 8: VertexRGB alpha attenuation regression test (#631)
# ---------------------------------------------------------------------------


def test_vertexrgb_alpha_zero_renders_curvature_only(tmp_path):
    """VertexRGB with α=0 must render the curvature underlay, not bright color.

    Regression test for #631: prior to the fix, the WebGL fragment shader's
    premultiplied-alpha composite formula (gl_FragColor = vColor + (1-α)·bg)
    consumed un-premultiplied RGB bytes from VertexRGB.vertices, so α=0 left
    the foreground color fully opaque and clipped toward white instead of
    falling through to the gray curvature.

    With the fix, RGB is premultiplied at the WebGL serialization step
    (cortex/webgl/data.py), so packaged vColor.rgb=0 when α=0, and the
    shader produces pure curvature gray.
    """
    from PIL import Image

    rng = np.random.default_rng(631)
    # Bright, saturated colors -- if the bug returns these will leak through
    # as red/green/blue pixels. With the fix and α=0, only neutral (curvature)
    # gray pixels should remain in the brain region.
    r = rng.uniform(0.7, 1.0, nverts).astype(np.float32)
    g = rng.uniform(0.0, 0.3, nverts).astype(np.float32)
    b = rng.uniform(0.0, 0.3, nverts).astype(np.float32)
    alpha = np.zeros(nverts, dtype=np.float32)

    vrgb = cortex.VertexRGB(
        r,
        g,
        b,
        subj,
        alpha=cortex.Vertex(alpha, subj, vmin=0, vmax=1),
    )

    view = {
        **default_view_params,
        **angle_view_params["lateral_pivot"],
        **unfold_view_params["inflated"],
    }
    with cortex.export.headless_viewer(vrgb, viewer_params={}) as handle:
        handle._set_view(**view)
        time.sleep(1)
        outfile = str(tmp_path / "alpha_zero.png")
        handle.getImage(outfile, (512, 384))
        _wait_for_file(outfile)

        rgb = np.array(Image.open(outfile))[..., :3].astype(int)
        # Count strongly red-dominant pixels: with the bug, α=0 lets the
        # bright reds through and we'd see thousands of them. With the fix,
        # the brain renders curvature gray (R≈G≈B) and red-dominant pixels
        # fall to near zero (a handful from anti-aliased ROI overlays).
        n_red = int((rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) > 50).sum())
        assert n_red < 500, (
            f"VertexRGB with α=0 produced {n_red} red-dominant pixels; "
            "expected near-zero. The shader composite is consuming "
            "un-premultiplied RGB (issue #631)."
        )


def test_volumergb_alpha_half_renders_correct_blend(tmp_path):
    """VolumeRGB with α=0.5 must blend halfway, not double-attenuate.

    Companion regression to test_vertexrgb_alpha_zero_renders_curvature_only
    (#631). VolumeRGB ships through the PNG texture path: Three.js sets
    ``tex.premultiplyAlpha = true`` on upload, so the texture is premultiplied
    once by WebGL itself. Package therefore must NOT premultiply on the
    Python side -- if it does, the shader sees double-attenuated RGB.

    α=0 won't catch that bug (0·anything = 0), so we use α=0.5 with bright
    uniform red. With curvature contribution included, observed shader
    output for the brain region is:
      - correct (single premult by JS): median R ≈ 145-160
      - bug (double premult: Py + JS):  median R ≈ 90-105
    Threshold at 125 sits in the middle of the gap and tolerates 20+ LSB
    of boundary/interpolation noise on either side.
    """
    from PIL import Image

    # Uniform saturated red over the whole volume, half transparent. Wrap in
    # explicit Volume(vmin=0, vmax=1) so the .volume property doesn't
    # auto-normalize a constant array to NaN.
    r = cortex.Volume(
        np.full(volshape, 1.0, dtype=np.float32), subj, xfmname, vmin=0, vmax=1
    )
    g = cortex.Volume(
        np.full(volshape, 0.0, dtype=np.float32), subj, xfmname, vmin=0, vmax=1
    )
    b = cortex.Volume(
        np.full(volshape, 0.0, dtype=np.float32), subj, xfmname, vmin=0, vmax=1
    )
    alpha = cortex.Volume(
        np.full(volshape, 0.5, dtype=np.float32), subj, xfmname, vmin=0, vmax=1
    )
    vrgb = cortex.VolumeRGB(r, g, b, subj, xfmname, alpha=alpha)

    view = {
        **default_view_params,
        **angle_view_params["lateral_pivot"],
        **unfold_view_params["inflated"],
    }
    with cortex.export.headless_viewer(vrgb, viewer_params={}) as handle:
        handle._set_view(**view)
        time.sleep(1)
        outfile = str(tmp_path / "volumergb_alpha_half.png")
        handle.getImage(outfile, (512, 384))
        _wait_for_file(outfile)

        rgb = np.array(Image.open(outfile))[..., :3].astype(int)
        # Brain-region pixels are red-dominant under both correct and buggy
        # paths, but their R intensity differs. Pick the strongly red
        # pixels (R clearly > G,B) and check median R.
        red_mask = rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) > 30
        assert red_mask.sum() > 1000, (
            "Expected a large red-dominant region for half-transparent red "
            f"VolumeRGB; got only {red_mask.sum()} pixels. Did the brain render?"
        )
        median_r = float(np.median(rgb[red_mask, 0]))
        # Discriminator: correct path produces ~145-160, double-premult
        # produces ~90-105. Threshold at 125 sits in the middle.
        assert median_r > 125, (
            f"VolumeRGB α=0.5 brain pixels have median R={median_r:.0f}; "
            "expected ~150. R<125 indicates Package is double-premultiplying "
            "VolumeRGB (issue #631 regression)."
        )


def test_vertex2d_alpha_half_renders_correct_blend(tmp_path):
    """Vertex2D with α=0.5 must blend halfway, not over-attenuate the bg.

    Companion regression to the issue #631 fix on the colormap-texture
    path. The 2D dataview ships dim1 / dim2 as separate scalar maps and
    the LUT lookup happens on the GPU via
    ``texture2D(colormap, vec2(dim1_norm, dim2_norm))``. The shader's
    composite (shaderlib.js:851) uses the premultiplied-over formula
    ``vColor + (1-α)·bg``, so the colormap texture itself must be
    premultiplied on upload (``tex.premultiplyAlpha = true`` in
    mriview.js). Without that, alpha-bearing colormaps like
    ``RdBu_r_alpha`` produce ``R + (1-α)·bg`` -- where the foreground
    is added on top of a partially-attenuated curvature -- instead of
    the correct ``α·R + (1-α)·bg``.

    α=0 doesn't catch this bug because most alpha colormaps store
    ``(0, 0, 0, 0)`` at the transparent end of the LUT (so neither the
    buggy nor the correct shader produces foreground there). At α=0.5
    the LUT stores its full RGB with α=127, and the difference between
    buggy and correct composites is maximal in the brain region.

    Empirical pixel stats for RdBu_r_alpha at data=+1, alpha=0.5,
    inflated lateral_pivot view, default viewer params (S1):

      - correct (premultiplied): red_dom median R ≈ 93 (25/50/75 = 80/93/110)
      - buggy (un-premultiplied): red_dom median R ≈ 129 (25/50/75 = 105/129/149)

    Threshold at 115 sits between the two distributions.
    """
    from PIL import Image

    # data=+1 puts every vertex at the deep red end of RdBu_r_alpha,
    # alpha=0.5 puts every vertex at mid-α (LUT row ~128).
    data = np.full(nverts, 1.0, dtype=np.float32)
    alpha = np.full(nverts, 0.5, dtype=np.float32)

    vtx2d = cortex.Vertex2D(
        data, alpha, subj,
        cmap="RdBu_r_alpha",
        vmin=-1, vmax=1,
        vmin2=0, vmax2=1,
    )

    view = {
        **default_view_params,
        **angle_view_params["lateral_pivot"],
        **unfold_view_params["inflated"],
    }
    # The cmap <img> elements decode asynchronously in Chromium. If the first
    # render frame happens before the LUT image has decoded, three.js skips
    # the texImage2D upload and the data layer renders against a 1×1 black
    # texture (R==G==B everywhere -- looks like the curvature underlay).
    # Retry _set_view + getImage until we observe a colored data layer
    # (some pixels with R clearly > G or B, or vice-versa). We then run the
    # premultiplication discriminator on that frame.
    with cortex.export.headless_viewer(vtx2d, viewer_params={}) as handle:
        # viewer.loaded already resolved by the context manager; a short
        # extra pause covers the gap before the cmap <img> decodes. The
        # retry loop below is the real guard for slow decodes.
        time.sleep(2)
        rgb = None
        outfile = None
        for attempt in range(6):
            handle._set_view(**view)
            time.sleep(3)
            # Use a fresh filename each retry so we never read a partial PNG
            # left over from a prior iteration (getImage writes async).
            outfile = str(tmp_path / f"vertex2d_alpha_half_{attempt}.png")
            handle.getImage(outfile, (512, 384))
            _wait_for_file(outfile)
            # Give the PNG writer a moment to finish flushing.
            time.sleep(1)
            try:
                rgb = np.array(Image.open(outfile))[..., :3].astype(int)
            except Exception:
                continue
            # "Colored" = at least some pixels deviate strongly from R==G==B.
            # In a curvature-only (cmap-unbound) frame all brain pixels have
            # R==G==B exactly; any non-zero count of channel-divergent pixels
            # means the cmap texture is bound.
            channel_spread = np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(
                rgb[..., 1] - rgb[..., 2]
            )
            if (channel_spread > 5).sum() > 1000:
                break
        else:
            pytest.skip(
                "Cmap texture never bound in headless Chromium across 6 "
                "render retries; can't discriminate fix vs bug."
            )

        # Both fix and bug produce a red-dominant brain region (the bug
        # doesn't zero the foreground, just over-brightens it). The
        # discriminator is the *median R intensity* of those red-dominant
        # pixels: the buggy un-premultiplied path adds the full R on top of
        # half the curvature, biasing R upward; the correct premultiplied
        # path attenuates R by α before adding curvature.
        #
        # First, the brain must render as a red-dominant region (this is
        # also satisfied by the bug, but if even this fails the cmap is
        # unbound and we can't discriminate).
        red_mask = rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) > 20
        assert red_mask.sum() > 1000, (
            f"Vertex2D α=0.5 deep-red rendered only {red_mask.sum()} "
            "red-dominant pixels. Check that the cmap LUT bound and the "
            "data layer rendered at all."
        )
        median_r = float(np.median(rgb[red_mask, 0]))
        assert median_r < 115, (
            f"Vertex2D α=0.5 brain pixels have median R={median_r:.0f}; "
            "expected ≈93 (correct), saw ≥115 which is in the buggy range "
            "(~129). The colormap texture is being sampled straight-alpha "
            "while the shader applies a premultiplied composite -- check "
            "mriview.js cmap texture premultiplyAlpha."
        )


# ---------------------------------------------------------------------------
# Group 9: addData dataset switching
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
        handle.addData(second=vol2)
        time.sleep(2)
        pageerrors = [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors after addData: {pageerrors}"


# ---------------------------------------------------------------------------
# Group 10: Manual visual A/B comparison across all alpha-bearing dataviews
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("RUN_VISUAL_COMPARISON"),
    reason="Manual visual comparison; set RUN_VISUAL_COMPARISON=1 to run.",
)
def test_visual_comparison_alpha_dataviews(tmp_path):
    """Render all 6 dataview types via quickshow + webgl, side-by-side.

    Skipped by default — set ``RUN_VISUAL_COMPARISON=1`` to run. Builds a
    grid where each row is one dataview type (Volume, Vertex, Volume2D,
    Vertex2D, VolumeRGB, VertexRGB) and the two columns are the matplotlib
    (``cortex.quickshow``) reference vs the headless WebGL flatmap render.
    Used as a manual smoke check that the alpha-blend fix
    (``Package``-side premultiply for VertexRGB + cmap-LUT
    ``premultiplyAlpha=true`` for the 2D-cmap path) keeps both viewers in
    visual agreement across every alpha-encoding pattern.

    Plain Volume / Vertex have no native per-element alpha (pycortex's
    bundled ``*_alpha`` colormaps are all 2D and only apply to the 2D
    dataview types), so those two rows act as a no-alpha baseline. The
    other four rows exercise alpha: Volume2D / Vertex2D via the 2D-alpha
    cmap ``RdBu_r_alpha``, VolumeRGB / VertexRGB via the ``alpha=`` kwarg.

    Renders are intentionally low-resolution (quickshow ``height=256``,
    webgl ``size=(512, 384)``) so the final composite PNG stays small.
    Both viewers run with no labels, no ROIs, and curvature underlay on.

    The composite PNG is written under ``tmp_path`` and the absolute path
    is printed at the end of the test so the file is easy to open.
    """
    import matplotlib.pyplot as plt

    import cortex.polyutils

    # ------- Synthesize data and alpha maps (mirrors plot_data_with_alpha.py) -

    # Volumetric
    zz, yy, xx = np.mgrid[0:31, 0:100, 0:100]
    data_vol = (xx - 50) / 50.0  # ~ [-1, 1]
    center = np.array([15, 50, 50])
    sigma_v = 25.0
    dist2 = (
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    accuracy_vol = np.exp(-dist2 / (2 * sigma_v**2))  # [0, 1] bump
    red_vol = np.clip(xx / 99.0, 0, 1)
    green_vol = np.clip(yy / 99.0, 0, 1)
    blue_vol = np.clip(zz / 30.0, 0, 1)

    # Surface (vertex) — encode by spatial coordinate, not vertex index
    surfs = [
        cortex.polyutils.Surface(*d)
        for d in cortex.db.get_surf(subj, "fiducial")
    ]
    num_verts = [s.pts.shape[0] for s in surfs]
    pts = np.vstack([surfs[0].pts, surfs[1].pts])
    y_centered = pts[:, 1] - pts[:, 1].mean()
    data_vtx = y_centered / np.abs(y_centered).max()  # [-1, 1]
    xyz_norm = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0))

    def _bump(surf, seed, sigma):
        d = np.linalg.norm(surf.pts - surf.pts[seed], axis=1)
        return np.exp(-(d**2) / (2 * sigma**2))

    accuracy_vtx = np.hstack(
        [
            _bump(surfs[0], num_verts[0] // 2, sigma=40.0),
            _bump(surfs[1], num_verts[1] // 2, sigma=40.0),
        ]
    )

    # ------- Build the six dataviews ----------------------------------------
    # Volume / Vertex have no native per-element alpha — pycortex's bundled
    # `*_alpha` colormaps are all 2D LUTs and only apply to Volume2D /
    # Vertex2D. So plain Volume / Vertex use a non-alpha cmap (`viridis`)
    # and serve as the no-alpha baseline; Volume2D / Vertex2D pair data
    # against accuracy via the 2D-alpha cmap `RdBu_r_alpha`; VolumeRGB /
    # VertexRGB use the native `alpha=` kwarg.

    cmap_plain = "viridis"
    cmap_2d = "RdBu_r_alpha"

    dataviews = [
        (
            "Volume",
            cortex.Volume(
                data_vol, subj, xfmname,
                cmap=cmap_plain, vmin=-1, vmax=1,
            ),
        ),
        (
            "Vertex",
            cortex.Vertex(
                data_vtx, subj,
                cmap=cmap_plain, vmin=-1, vmax=1,
            ),
        ),
        (
            "Volume2D",
            cortex.Volume2D(
                data_vol, accuracy_vol, subj, xfmname,
                cmap=cmap_2d,
                vmin=-1, vmax=1, vmin2=0, vmax2=1,
            ),
        ),
        (
            "Vertex2D",
            cortex.Vertex2D(
                data_vtx, accuracy_vtx, subj,
                cmap=cmap_2d,
                vmin=-1, vmax=1, vmin2=0, vmax2=1,
            ),
        ),
        (
            "VolumeRGB",
            cortex.VolumeRGB(
                cortex.Volume(red_vol, subj, xfmname, vmin=0, vmax=1),
                cortex.Volume(green_vol, subj, xfmname, vmin=0, vmax=1),
                cortex.Volume(blue_vol, subj, xfmname, vmin=0, vmax=1),
                subj, xfmname,
                alpha=cortex.Volume(accuracy_vol, subj, xfmname, vmin=0, vmax=1),
            ),
        ),
        (
            "VertexRGB",
            cortex.VertexRGB(
                cortex.Vertex(xyz_norm[:, 0], subj, vmin=0, vmax=1),
                cortex.Vertex(xyz_norm[:, 1], subj, vmin=0, vmax=1),
                cortex.Vertex(xyz_norm[:, 2], subj, vmin=0, vmax=1),
                subj,
                alpha=cortex.Vertex(accuracy_vtx, subj, vmin=0, vmax=1),
            ),
        ),
    ]

    # ------- Render each dataview through both paths ------------------------
    # Each WebGL render spins up its own headless browser via plot_panels;
    # six sequential launches × ~15s sleep = ~90s+ end to end. That's fine
    # for a manual A/B and avoids the broken `addData` path on headless.

    n = len(dataviews)
    fig, axes = plt.subplots(n, 2, figsize=(7, 2.2 * n))

    flatmap_panel = [
        {
            "extent": [0.0, 0.0, 1.0, 1.0],
            "view": {"angle": "flatmap", "surface": "flatmap"},
        }
    ]

    for row, (name, view) in enumerate(dataviews):
        # quickshow → low-res PNG
        qs_path = tmp_path / f"qs_{name}.png"
        qs_fig = cortex.quickshow(
            view,
            with_curvature=True,
            with_rois=False,
            with_labels=False,
            with_colorbar=False,
            with_sulci=False,
            with_borders=False,
            height=256,
        )
        qs_fig.savefig(qs_path, bbox_inches="tight", pad_inches=0, dpi=80)
        plt.close(qs_fig)

        # webgl → trimmed flatmap PNG via plot_panels (single flatmap panel)
        wg_path = str(tmp_path / f"wg_{name}.png")
        wg_fig = cortex.export.plot_panels(
            view,
            panels=flatmap_panel,
            figsize=(6, 3),
            windowsize=(512, 384),
            save_name=wg_path,
            sleep=10,
            viewer_params=dict(labels_visible=[], overlays_visible=[]),
            headless=True,
        )
        plt.close(wg_fig)

        ax_qs, ax_wg = axes[row]
        ax_qs.imshow(plt.imread(qs_path))
        ax_qs.set_title(f"{name} — quickshow", fontsize=9)
        ax_qs.axis("off")
        ax_wg.imshow(plt.imread(wg_path))
        ax_wg.set_title(f"{name} — webgl (flatmap)", fontsize=9)
        ax_wg.axis("off")

    fig.suptitle(
        "Alpha-bearing dataviews: quickshow vs WebGL", fontsize=11,
    )
    fig.tight_layout()
    out_path = tmp_path / "alpha_dataview_comparison.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"\nVisual comparison saved to:\n  {out_path}\n")
    assert out_path.exists()
    assert out_path.stat().st_size > 0
