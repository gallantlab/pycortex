"""Tests for headless WebGL rendering across data types, angles, surfaces, and panels.

These tests exercise the WebGL viewer through the headless Chromium browser,
verifying that all supported data types, camera angles, surface morphing
states, and predefined panel layouts render correctly without a display server.

All tests are skipped if playwright is not installed.
"""

import json
import os
import time
import urllib.request

import numpy as np
import pytest

import cortex
import cortex.export
from cortex.export.save_views import (
    angle_view_params,
    default_view_params,
    unfold_view_params,
)
from cortex.tests.testing_utils import has_playwright, wait_for_file

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


def _assert_no_browser_failures(handle):
    """Fail on uncaught JS exceptions, or on WebGL reporting its own failure.

    ``[pageerror]`` covers uncaught exceptions; ``filter_webgl_failures`` covers
    a shader that compiled but failed to *link*, which three.js reports only on
    console.error, so nothing raises and the render comes back blank.

    ``browser_errors`` is current to within ``EVENT_POLL_INTERVAL``, so it does
    not matter whether this is called inside the ``with`` block.

    Not the only defense: a driver that silently links an over-allocating shader
    reports nothing, which is what ``_assert_not_blank`` is for. Neither covers
    a shader variant no test renders.

    Known limitation: ``browser_errors`` is cumulative and never cleared, so
    with a handle shared across tests (``TestAddData``) one transient failure
    fails every later test in the class too. Misattributed, not missed; a
    watermark index from the previous call would isolate it.
    """
    from cortex.export.headless import filter_webgl_failures

    errors = handle._pw_thread.browser_errors
    pageerrors = [e for e in errors if "[pageerror]" in e]
    assert not pageerrors, f"JS errors: {pageerrors}"
    failures = filter_webgl_failures(errors)
    assert not failures, f"WebGL reported a failure: {failures}"


def _assert_not_blank(path):
    """Fail if the render came out as a single flat color.

    A shader that fails to link, or geometry that never reached the GPU, leaves
    only the background -- otherwise indistinguishable from success, since the
    png is written and nothing raises.
    """
    from PIL import Image

    rgb = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3).astype(np.uint32)
    # Packed to one int per pixel; np.unique on a structured view costs ~16x more.
    ncolors = len(np.unique((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]))
    assert ncolors > 10, (
        f"{path} has only {ncolors} distinct color(s); the brain was probably "
        "never drawn."
    )


# ---------------------------------------------------------------------------
# Group 1: Data type smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_name",
    [
        "Volume",
        "Vertex",
        "VolumeRGB",
        "VertexRGB",
        "Volume2D",
        # gh-714: the Vertex2D flatmap shader fails to link, so the render comes
        # back blank and three.js reports it on console.error. Strict, so a
        # render that starts succeeding reports an XPASS.
        pytest.param(
            "Vertex2D",
            marks=pytest.mark.xfail(
                strict=True, reason="gh-714: Vertex2D shader fails to link"
            ),
        ),
    ],
)
def test_datatype_renders(dtype_name, tmp_path):
    """Each data type should render in the headless viewer without errors."""
    vol = make_dataview(dtype_name)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        outfile = str(tmp_path / "test.png")
        handle.getImage(outfile, (512, 384))
        wait_for_file(outfile)
        assert os.path.isfile(outfile)
        assert os.path.getsize(outfile) > 0

    # Browser errors first, since one may explain a blank file.
    _assert_no_browser_failures(handle)
    _assert_not_blank(outfile)


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
        wait_for_file(outfile)
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
        wait_for_file(outfile)
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
        wait_for_file(f1)

    # Render WITHOUT overlays
    f2 = str(tmp_path / "without_overlay.png")
    with cortex.export.headless_viewer(
        vol, viewer_params=dict(overlays_visible=[])
    ) as handle:
        handle._set_view(**view)
        time.sleep(1)
        handle.getImage(f2, (512, 384))
        wait_for_file(f2)

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
        wait_for_file(outfile)

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
            wait_for_file(outfile)
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
        wait_for_file(outfile)

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
        wait_for_file(outfile)

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
            wait_for_file(outfile)
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
# Group 8b: opacity slider for vertex data (#684)
# ---------------------------------------------------------------------------


def test_vertex_opacity_slider_fades_data(tmp_path):
    """The ``opacity`` control must fade Vertex data into the curvature.

    Regression test for #684: the ``surface_vertex`` fragment shader declared
    the ``dataAlpha`` uniform (driven by the dat.GUI opacity slider and the
    ``o`` shortcut) but never used it, so opacity changes were a no-op for
    Vertex / Vertex2D / VertexRGB data while working for Volume data.

    Renders a constant, saturated-red Vertex at opacity 1 → 0 → 1 within one
    viewer session and checks that the red pixels disappear at 0 and come
    back at 1.
    """
    image_size = (512, 384)
    n_pixels = image_size[0] * image_size[1]
    # At this lateral/inflated view the brain fills ~30% of the frame and,
    # with a saturated colormap at full opacity, ~27% of the frame is
    # red-dominant (measured). Require a third of that so small view or
    # lighting changes don't cause flakiness, while staying far above what a
    # curvature-only render produces (0%).
    min_visible_data_fraction = 0.10
    # At opacity 0 the data layer must be gone: allow up to 1% of the visible
    # count for anti-aliased edges (measured: 0).
    max_hidden_to_visible_ratio = 0.01
    # Toggling back to opacity 1 must reproduce the original render.
    restore_tolerance = 0.05

    data = np.full(nverts, 5.0)
    vtx = cortex.Vertex(data, subj, vmin=0, vmax=1, cmap="Reds")

    view = {
        **default_view_params,
        **angle_view_params["lateral_pivot"],
        **unfold_view_params["inflated"],
    }

    def render(handle, name):
        outfile = str(tmp_path / f"{name}.png")
        time.sleep(1)
        handle.getImage(outfile, image_size)
        wait_for_file(outfile)
        return _count_red_pixels(outfile)

    # No ROI/sulci overlays or labels: their anti-aliased colored edges would
    # otherwise add stray red-dominant pixels.
    with cortex.export.headless_viewer(
        vtx, viewer_params=dict(overlays_visible=[], labels_visible=[])
    ) as handle:
        handle._set_view(**view)
        n_full = render(handle, "opacity_1")

        handle._set_view(**{"surface.{subject}.opacity": 0.0})
        n_zero = render(handle, "opacity_0")

        handle._set_view(**{"surface.{subject}.opacity": 1.0})
        n_restored = render(handle, "opacity_1_again")

    assert n_full >= min_visible_data_fraction * n_pixels, (
        f"Vertex data at opacity 1 should cover at least "
        f"{min_visible_data_fraction:.0%} of the frame; got {n_full} of "
        f"{n_pixels} pixels ({n_full / n_pixels:.1%})"
    )
    assert n_zero <= max_hidden_to_visible_ratio * n_full, (
        f"Vertex data at opacity 0 still renders {n_zero} red-dominant pixels "
        f"({n_zero / n_full:.1%} of the {n_full} visible at opacity 1); "
        "the surface_vertex shader is ignoring dataAlpha (#684)."
    )
    assert abs(n_restored - n_full) <= restore_tolerance * n_full, (
        f"Restoring opacity 1 should reproduce the original render within "
        f"{restore_tolerance:.0%}; got {n_restored} vs {n_full} red-dominant "
        "pixels"
    )


# ---------------------------------------------------------------------------
# Group 9: addData dataset switching
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def _addData_viewer():
    """A single headless viewer shared by the ``TestAddData`` sequence."""
    np.random.seed(1)
    vol1 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol1, viewer_params={}) as handle:
        yield handle
    # A final check after teardown, which flushes anything still queued.
    _assert_no_browser_failures(handle)


def _served_metadata(handle):
    """Return the dataset metadata embedded in the served ``mixer.html``.

    ``show()`` regenerates the page from the same ``metadata`` dict that
    ``addData`` merges into, so this is how we check that a reload of the
    viewer would show everything that has been added so far.
    """
    url = "http://localhost:%d/mixer.html" % handle.server.port
    with urllib.request.urlopen(url, timeout=30) as resp:
        page = resp.read().decode("utf-8")
    marker = "dataset.fromJSON("
    start = page.index(marker) + len(marker)
    return json.JSONDecoder().raw_decode(page, start)[0]


def _fetch(handle, path):
    """GET ``path`` from the viewer's tornado server, returning the body."""
    url = "http://localhost:%d%s" % (handle.server.port, path)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def _active_name(handle):
    """Name of the dataview the viewer currently displays.

    ``handle.active.name`` cannot be used: ``JSProxy.name`` is the javascript
    path of the proxy itself, so the dataview's own name has to be read out of
    the queried attributes.
    """
    return handle.active.attrs["name"][1]


def _image_array(handle, outfile, size=(512, 384)):
    """Render the current view to ``outfile`` and return it as an array."""
    from PIL import Image

    handle.getImage(outfile, size)
    wait_for_file(outfile)
    return np.asarray(Image.open(outfile).convert("RGB"), dtype=np.int16)


class TestAddData:
    """``JSMixer.addData`` pushes new data into an already running viewer.

    The tests share one browser session and run in definition order: each one
    builds on the dataviews added by the previous ones.
    """

    def test_adds_dataview(self, _addData_viewer):
        """Adding a dataview registers it and makes it the active one."""
        handle = _addData_viewer
        vol2 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        handle.addData(second=vol2)
        time.sleep(2)

        _assert_no_browser_failures(handle)

        # "data" is the name webshow gives to a bare Dataview.
        assert set(handle.dataviews.attrs) == {"data", "second"}
        # As on the initial page load, the viewer switches to the new data.
        assert _active_name(handle) == "second"

    def test_updates_served_metadata(self, _addData_viewer):
        """The added dataview survives a page reload, images included."""
        handle = _addData_viewer
        metadata = _served_metadata(handle)
        assert [view["name"] for view in metadata["views"]] == ["data", "second"]

        # Every brain referenced by the (old and new) dataviews must still be
        # served by the DataHandler.
        assert len(metadata["images"]) == 2
        for name, urls in metadata["images"].items():
            assert name in metadata["data"]
            for url in urls:
                assert _fetch(handle, url)[1:4] == b"PNG"

    def test_changes_rendered_image(self, _addData_viewer, tmp_path):
        """Switching between the old and the new dataview changes the render."""
        handle = _addData_viewer
        added = _image_array(handle, str(tmp_path / "second.png"))

        handle.setData("data")
        time.sleep(2)
        original = _image_array(handle, str(tmp_path / "data.png"))

        assert _active_name(handle) == "data"
        assert np.abs(added - original).mean() > 1, (
            "The render did not change when switching to the dataview added "
            "by addData; the new data was probably never loaded."
        )

    def test_replaces_existing_name(self, _addData_viewer):
        """Re-adding a name replaces it instead of duplicating it."""
        handle = _addData_viewer
        previous_brains = set(_served_metadata(handle)["images"])
        vol3 = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        handle.addData(second=vol3)
        time.sleep(2)

        metadata = _served_metadata(handle)
        names = [view["name"] for view in metadata["views"]]
        assert names == ["data", "second"]
        assert set(handle.dataviews.attrs) == {"data", "second"}
        assert _active_name(handle) == "second"

        # The images of the replaced dataview are dropped rather than piling
        # up in the server on every refresh.
        assert len(metadata["images"]) == 2
        assert metadata["views"][1]["data"][0] not in previous_brains

        _assert_no_browser_failures(handle)

    def test_rejects_unknown_subject(self, _addData_viewer):
        """Surfaces cannot be added to a running viewer, so neither can subjects."""
        handle = _addData_viewer
        other = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        # The subject only has to differ from the one the viewer was started
        # with; it never reaches the database because the check comes first.
        other.subject = "not_a_loaded_subject"
        with pytest.raises(ValueError, match="not_a_loaded_subject"):
            handle.addData(third=other)

        # The rejected dataview must not have leaked into the viewer state.
        metadata = _served_metadata(handle)
        assert [view["name"] for view in metadata["views"]] == ["data", "second"]
        assert len(metadata["images"]) == 2


def test_addData_vertex_data(tmp_path):
    """Vertex data added at runtime is reordered to match the CTM surfaces.

    Vertex data is uploaded as a raw vertex attribute array, so it has to go
    through ``Package.reorder`` with the same ctm files the viewer was built
    with. Getting this wrong renders a scrambled (but non-empty) brain, so the
    check here is that the render changes and that no JS error is raised.
    """
    np.random.seed(2)
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    vertex = cortex.Vertex(np.random.randn(nverts), subj)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        before = _image_array(handle, str(tmp_path / "before.png"))

        handle.addData(vertexdata=vertex)
        time.sleep(3)

        assert _active_name(handle) == "vertexdata"
        after = _image_array(handle, str(tmp_path / "after.png"))
        assert np.abs(after - before).mean() > 1

        # Vertex data is served as a raw .npy blob rather than a PNG mosaic.
        metadata = _served_metadata(handle)
        assert [view["name"] for view in metadata["views"]] == ["data", "vertexdata"]
        vertex_name = metadata["views"][1]["data"][0]
        assert "mosaic" not in metadata["data"][vertex_name]
        assert _fetch(handle, metadata["images"][vertex_name][0])[1:6] == b"NUMPY"

        _assert_no_browser_failures(handle)


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



# ---------------------------------------------------------------------------
# Group 11: Saved views and the animation GUI
# ---------------------------------------------------------------------------


def _js_attrs(handle, path):
    """Read a javascript object's properties, with values for the scalar ones.

    ``send(method="get", ...)`` cannot be used for this: for a non-object
    property, ``Websock.prototype.get`` returns the property *name* rather than
    its value (that is what makes the "set" method work). ``query`` is the
    accessor that carries values, and is what ``JSProxy.attrs`` uses.
    """
    resp = handle.send(method="query", params=[path])
    assert isinstance(resp, list) and resp and isinstance(resp[0], dict), resp
    return resp[0]


def _js_value(handle, path):
    """Read one scalar javascript property, e.g. viewopts.movie_post.token."""
    parent, _, name = path.rpartition(".")
    entry = _js_attrs(handle, parent)[name]
    assert len(entry) > 1, f"{path} is not a scalar: {entry}"
    return entry[1]


def test_retrieve_new_views_roundtrip():
    """Views saved through the GUI come back to python via the handle."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        assert handle.retrieve_new_views() == {}

        target = {"camera.azimuth": 90, "camera.altitude": 90}
        handle._set_view(**target)
        time.sleep(2)

        # What the "save view" button calls.
        handle.send(method="run",
                    params=["window.viewer.saveNewView", ["from_gui"]])

        views = handle.retrieve_new_views()
        assert set(views) == {"from_gui"}
        saved = views["from_gui"]
        for key, expected in target.items():
            assert saved[key] == pytest.approx(expected, abs=1.0)

        # Keys keep the {subject} placeholder, so the view stays interchangeable
        # with what _capture_view writes and with saved views/*.json files.
        assert "surface.{subject}.unfold" in saved

        # The javascript capture must be a subset of the python one; otherwise
        # _set_view would reject keys coming back out of the browser.
        captured = handle._capture_view()
        assert set(saved) <= set(captured), set(saved) - set(captured)

        # And it must round-trip back in without complaint.
        handle._set_view(**saved)

        pageerrors = [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"


def test_save_new_views_writes_to_the_filestore():
    """save_new_views stores GUI views and promotes them out of new_views."""
    viewdir = os.path.join(cortex.db.filestore, subj, "views")
    names = ["_pytest_saved_a", "_pytest saved b"]
    written = []

    try:
        vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
            handle._set_view(**{"camera.azimuth": 90})
            time.sleep(2)
            for name in names:
                handle.send(method="run",
                            params=["window.viewer.saveNewView", [name]])
            assert set(handle.retrieve_new_views()) == set(names)

            paths = handle.save_new_views()
            written = list(paths.values())
            assert set(paths) == set(names)

            for name in names:
                path = os.path.join(viewdir, name + ".json")
                assert paths[name] == path
                assert os.path.isfile(path)
                with open(path) as fp:
                    stored = json.load(fp)
                # Stored in _capture_view's format, so it feeds straight back in.
                assert "surface.{subject}.unfold" in stored
                assert stored["camera.azimuth"] == pytest.approx(90, abs=1.0)
                handle._set_view(**stored)

            # Saved views are no longer "new": they move into the views menu.
            assert handle.retrieve_new_views() == {}
            buttons = _js_attrs(
                handle, "window.viewer.ui._desc.camera._desc.views._desc")
            for name in names:
                assert name in buttons

            # A second save is a no-op rather than a rewrite, since there is
            # nothing left to promote.
            assert handle.save_new_views() == {}

            # Re-saving under an existing name needs is_overwrite.
            handle.send(method="run",
                        params=["window.viewer.saveNewView", [names[0]]])
            with pytest.raises(IOError):
                handle.save_new_views()
            assert handle.save_new_views(is_overwrite=True) == {
                names[0]: os.path.join(viewdir, names[0] + ".json")}

            # A name that would escape the views directory is refused outright.
            handle.send(method="run",
                        params=["window.viewer.saveNewView", ["../_pytest_evil"]])
            with pytest.raises(ValueError):
                handle.save_new_views()
            assert not os.path.exists(
                os.path.join(cortex.db.filestore, subj, "_pytest_evil.json"))

            with pytest.raises(KeyError):
                handle.save_new_views(names=["_pytest_no_such_view"])

            pageerrors = [e for e in handle._pw_thread.browser_errors
                          if "[pageerror]" in e]
            assert len(pageerrors) == 0, f"JS errors: {pageerrors}"
    finally:
        # The S1 filestore is checked in; leave nothing behind.
        for path in set(written) | {os.path.join(viewdir, n + ".json")
                                    for n in names}:
            if os.path.exists(path):
                os.remove(path)


def test_saved_views_are_loaded_into_the_viewer():
    """views/*.json for the displayed subject reach the browser and the menu."""
    from cortex.export.save_views import default_view_params

    viewdir = os.path.join(cortex.db.filestore, subj, "views")
    os.makedirs(viewdir, exist_ok=True)
    name = "_pytest_tmp_view"
    viewfile = os.path.join(viewdir, name + ".json")
    with open(viewfile, "w") as fp:
        json.dump(dict(default_view_params), fp)

    try:
        vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
            # Only the displayed subject's views are shipped to the browser.
            assert set(_js_attrs(handle, "window.viewopts.saved_views")) == {subj}
            assert name in _js_attrs(
                handle, "window.viewopts.saved_views.%s" % subj)

            # ... and each one becomes a button under camera > views.
            buttons = _js_attrs(
                handle, "window.viewer.ui._desc.camera._desc.views._desc")
            assert name in buttons

            # Clicking it applies the view.
            handle._set_view(**{"camera.azimuth": 10})
            time.sleep(1)
            handle.send(method="run", params=[
                "window.viewer.ui._desc.camera._desc.views._desc"
                ".%s.action" % name, []])
            time.sleep(2)
            assert handle.ui.get("camera.azimuth")[0] == pytest.approx(
                default_view_params["camera.azimuth"], abs=1.0)
    finally:
        os.remove(viewfile)


def _post(url, **fields):
    """POST form fields, returning the HTTP status (including error statuses)."""
    import urllib.error
    import urllib.parse

    data = urllib.parse.urlencode(fields).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


# 1x1 transparent png, as the browser would send it
_TINY_PNG = ("data:image/png;base64,"
             "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8"
             "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def test_movie_handler_rejects_bad_requests():
    """The frame-render endpoint refuses bad tokens, names, and escaping paths."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        url = f"http://localhost:{handle.server.port}/movie"
        token = _js_value(handle, "window.viewopts.movie_post.token")
        assert isinstance(token, str) and len(token) > 0

        assert _post(url, token="wrong", name="f", frame=0, png=_TINY_PNG) == 403
        assert _post(url, name="f", frame=0, png=_TINY_PNG) == 403
        assert _post(url, token=token, dir="../..", name="f", frame=0,
                     png=_TINY_PNG) == 403
        assert _post(url, token=token, dir="/etc", name="f", frame=0,
                     png=_TINY_PNG) == 403
        assert _post(url, token=token, name="../evil", frame=0,
                     png=_TINY_PNG) == 400
        assert _post(url, token=token, name="f", frame="nope",
                     png=_TINY_PNG) == 400
        assert _post(url, token=token, name="f", frame=0, png="garbage") == 400


def test_movie_handler_writes_frames(tmp_path):
    """A well-formed request lands as <movie_dir>/<dir>/<name>_<frame>.png."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(
            vol, viewer_params=dict(movie_dir=str(tmp_path))) as handle:
        url = f"http://localhost:{handle.server.port}/movie"
        token = _js_value(handle, "window.viewopts.movie_post.token")
        assert _js_value(handle, "window.viewopts.movie_post.root") == str(
            os.path.realpath(tmp_path))

        assert _post(url, token=token, dir="frames", name="brainmovie",
                     frame=7, png=_TINY_PNG) == 200

        out = tmp_path / "frames" / "brainmovie_00007.png"
        assert out.exists()
        assert out.stat().st_size > 0


def test_static_viewer_has_views_but_no_render_target(tmp_path):
    """A static export carries saved views, but nowhere to write frames."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    outpath = str(tmp_path / "static")
    cortex.webgl.make_static(outpath, vol, html_embed=False, copy_ctmfiles=False)

    with open(os.path.join(outpath, "index.html")) as fp:
        html = fp.read()
    assert "viewtools.js" in html
    assert "saved_views" in html
    # No python behind a static viewer, so the animation panel must not offer
    # to render frames to disk.
    assert "movie_post" not in html


# ---------------------------------------------------------------------------
# Group 12: Smoothed animation trajectories
# ---------------------------------------------------------------------------

# Keyframes exercising every kind of channel at once: an angle that crosses the
# 0/360 wrap, a plain scalar, a vector, a discrete property, a boolean and a
# string -- and a different interpolation mode on each keyframe.
SMOOTHING_KEYFRAMES = [
    {"frame": 0, "interpolation": "Bezier",
     "camera.azimuth": 300.0, "camera.altitude": 10.0,
     "camera.target": [0.0, 0.0, 0.0], "surface.S1.layers": 1,
     "surface.S1.dither": False, "surface.S1.sampler": "nearest"},
    {"frame": 10, "interpolation": "CubicHermite",
     "camera.azimuth": 40.0, "camera.altitude": 90.0,
     "camera.target": [10.0, 20.0, 30.0], "surface.S1.layers": 4,
     "surface.S1.dither": True, "surface.S1.sampler": "trilinear"},
    {"frame": 20, "interpolation": "BezierInHoldOut",
     "camera.azimuth": 140.0, "camera.altitude": 30.0,
     "camera.target": [5.0, 5.0, 5.0], "surface.S1.layers": 2,
     "surface.S1.dither": False, "surface.S1.sampler": "nearest"},
    {"frame": 30, "interpolation": "Linear",
     "camera.azimuth": 200.0, "camera.altitude": 55.0,
     "camera.target": [1.0, 2.0, 3.0], "surface.S1.layers": 3,
     "surface.S1.dither": True, "surface.S1.sampler": "trilinear"},
]


def _assert_views_match(expected, actual, tol=1e-6):
    """Compare two view dicts property by property."""
    assert set(expected) == set(actual), set(expected) ^ set(actual)
    for prop, want in expected.items():
        got = actual[prop]
        if isinstance(want, list):
            assert got == pytest.approx(want, abs=tol), prop
        elif isinstance(want, bool) or not isinstance(want, (int, float)):
            assert got == want, prop
        else:
            assert got == pytest.approx(want, abs=tol), prop


def test_interpolation_js_is_loaded_with_all_eight_modes():
    """The browser knows the same modes python does."""
    from cortex.webgl.interpolation import Interpolation

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        modes = _js_attrs(handle, "window.jsplot.interpolation.Interpolation")
        assert set(modes) == {mode.value for mode in Interpolation}
        assert _js_value(handle, "window.jsplot.interpolation.DEFAULT_MODE") == \
            Interpolation.Bezier.value


def test_browser_and_python_interpolate_identically():
    """The whole point of keeping two implementations: they must agree.

    An animation built in the panel is played back in javascript but rendered
    to disk through _get_anim_seq in python, so any divergence would show up as
    a movie that does not match its preview.
    """
    from cortex.webgl.interpolation import build_channels, evaluate

    frames = [i * 0.5 for i in range(61)]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        from_js = handle.send(method="run", params=[
            "window.jsplot.viewtools.viewsAt", [SMOOTHING_KEYFRAMES, frames]])
        assert isinstance(from_js, list) and len(from_js) == len(frames), from_js

        channels = build_channels(SMOOTHING_KEYFRAMES, time_key="frame")
        for frame, js_view in zip(frames, from_js):
            _assert_views_match(evaluate(channels, frame), js_view)

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"


def test_animation_panel_defaults_to_bezier():
    """Opening the panel sets up per-keyframe smoothing state."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        # What the "create animation" button calls. The key has a space in it,
        # which the dotted-path walker in python_interface.js handles fine.
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.create animation.action", []])
        time.sleep(1)

        state = _js_attrs(handle, "window.viewer._anim")
        assert "mode" in state, state
        assert _js_value(handle, "window.viewer._anim.mode") == "Bezier"


def test_get_anim_seq_linear_path_is_unchanged():
    """The pairwise easings still produce exactly what they always did."""
    keyframes = [
        {"time": 0.0, "camera.azimuth": 10.0, "camera.altitude": 20.0},
        {"time": 1.0, "camera.azimuth": 90.0, "camera.altitude": 60.0},
        {"time": 2.0, "camera.azimuth": 170.0, "camera.altitude": 40.0},
    ]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        seq = handle._get_anim_seq([dict(k) for k in keyframes], fps=30,
                                   interpolation="linear")
        # 30 frames per second over two seconds, plus the closing frame.
        assert len(seq) == 61
        assert "time" not in seq[0]
        assert seq[0]["camera.azimuth"] == pytest.approx(10.0)
        assert seq[30]["camera.azimuth"] == pytest.approx(90.0)
        assert seq[-1]["camera.azimuth"] == pytest.approx(170.0)
        # Straight lines between the keyframes: the quarter point is halfway
        # from the first keyframe to the second.
        assert seq[15]["camera.azimuth"] == pytest.approx(50.0)
        assert seq[15]["camera.altitude"] == pytest.approx(40.0)


def test_get_anim_seq_all_linear_modes_reproduce_the_legacy_path():
    """'linear' and a list of Linear keyframes are the same curve.

    This ties the two code paths together: whatever the smoothed path does to
    frame times and property handling, it has to land on the old answer when
    every keyframe is linear.
    """
    keyframes = [
        {"time": 0.0, "camera.altitude": 20.0, "camera.target": [0.0, 0.0, 0.0]},
        {"time": 0.7, "camera.altitude": 60.0, "camera.target": [3.0, 6.0, 9.0]},
        {"time": 2.0, "camera.altitude": 40.0, "camera.target": [1.0, 1.0, 1.0]},
    ]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        legacy = handle._get_anim_seq([dict(k) for k in keyframes], fps=30,
                                      interpolation="linear")
        smoothed = handle._get_anim_seq([dict(k) for k in keyframes], fps=30,
                                        interpolation="Linear")
        assert len(legacy) == len(smoothed)
        for want, got in zip(legacy, smoothed):
            _assert_views_match(want, got)


def test_get_anim_seq_honours_per_keyframe_modes():
    """A keyframe's own mode takes over, and selects the smoothed path."""
    keyframes = [
        {"time": 0.0, "camera.altitude": 0.0,
         "interpolation": "BezierInHoldOut"},
        {"time": 1.0, "camera.altitude": 40.0, "interpolation": "Linear"},
        {"time": 2.0, "camera.altitude": 80.0, "interpolation": "Linear"},
    ]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        # interpolation defaults to 'linear', but the keyframes override it.
        seq = handle._get_anim_seq([dict(k) for k in keyframes], fps=30)
        assert len(seq) == 61
        # Held across the first second...
        assert seq[15]["camera.altitude"] == pytest.approx(0.0)
        assert seq[29]["camera.altitude"] == pytest.approx(0.0)
        # ... then linear to the end.
        assert seq[30]["camera.altitude"] == pytest.approx(40.0)
        assert seq[45]["camera.altitude"] == pytest.approx(60.0)
        assert seq[-1]["camera.altitude"] == pytest.approx(80.0)
        assert "interpolation" not in seq[0]


def test_get_anim_seq_rejects_mixing_an_easing_with_keyframe_modes():
    """smoothstep eases a segment and has no per-keyframe equivalent."""
    keyframes = [
        {"time": 0.0, "camera.altitude": 0.0, "interpolation": "Bezier"},
        {"time": 1.0, "camera.altitude": 40.0, "interpolation": "Bezier"},
    ]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        with pytest.raises(ValueError, match="whole segment"):
            handle._get_anim_seq([dict(k) for k in keyframes], fps=30,
                                 interpolation="smoothstep")
        with pytest.raises(ValueError, match="Unknown interpolation"):
            handle._get_anim_seq([dict(k) for k in keyframes], fps=30,
                                 interpolation="wobble")


def test_static_viewer_ships_the_interpolation_module(tmp_path):
    """Smoothing is pure browser-side, so static exports get it too.

    Only the script tag is checked here: make_static does not copy the
    resources tree, it is either inlined by htmlembed or served alongside.
    """
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    outpath = str(tmp_path / "static")
    cortex.webgl.make_static(outpath, vol, html_embed=False, copy_ctmfiles=False)

    with open(os.path.join(outpath, "index.html")) as fp:
        html = fp.read()
    assert "interpolation.js" in html
    # It has to come before viewtools.js, which uses it at panel-open time.
    assert html.index("interpolation.js") < html.index("viewtools.js")


# ---------------------------------------------------------------------------
# Group 13: Default views every subject gets
# ---------------------------------------------------------------------------


def test_default_views_reach_the_browser_and_the_menu():
    """Every subject gets the standard orientations without saving anything."""
    from cortex.export.save_views import default_subject_views
    from cortex.webgl.view import _has_flatmap

    expected = set(default_subject_views(_has_flatmap(subj)))
    assert "dorsal" in expected and "lateral_left_inflated" in expected

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        shipped = _js_attrs(handle, "window.viewopts.saved_views.%s" % subj)
        assert expected <= set(shipped), expected - set(shipped)

        buttons = _js_attrs(
            handle, "window.viewer.ui._desc.camera._desc.views._desc")
        assert expected <= set(buttons), expected - set(buttons)


def test_clicking_a_default_view_applies_it():
    """The buttons are wired, not just present."""
    from cortex.export.save_views import default_subject_views
    from cortex.webgl.view import _has_flatmap

    views = default_subject_views(_has_flatmap(subj))
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        # Somewhere that is not the view we are about to ask for.
        handle._set_view(**{"camera.azimuth": 10, "camera.altitude": 45})
        time.sleep(1)

        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.views._desc"
            ".lateral_left.action", []])
        time.sleep(2)

        want = views["lateral_left"]
        assert handle.ui.get("camera.azimuth")[0] == pytest.approx(
            want["camera.azimuth"], abs=1.0)
        assert handle.ui.get("camera.altitude")[0] == pytest.approx(
            want["camera.altitude"], abs=1.0)


def test_a_saved_view_overrides_the_default_of_the_same_name():
    """A views/dorsal.json in the filestore wins over the built-in dorsal."""
    from cortex.export.save_views import default_subject_views

    viewdir = os.path.join(cortex.db.filestore, subj, "views")
    os.makedirs(viewdir, exist_ok=True)
    viewfile = os.path.join(viewdir, "dorsal.json")
    assert not os.path.exists(viewfile), (
        "%s already exists; this test would overwrite it" % viewfile)

    builtin = default_subject_views()["dorsal"]
    mine = dict(default_view_params)
    mine["camera.azimuth"] = 123.0
    mine["camera.altitude"] = 47.0
    assert mine["camera.azimuth"] != builtin["camera.azimuth"]

    with open(viewfile, "w") as fp:
        json.dump(mine, fp)
    try:
        vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
            shipped = _js_attrs(
                handle, "window.viewopts.saved_views.%s.dorsal" % subj)
            assert "camera.azimuth" in shipped

            handle.send(method="run", params=[
                "window.viewer.ui._desc.camera._desc.views._desc"
                ".dorsal.action", []])
            time.sleep(2)
            assert handle.ui.get("camera.azimuth")[0] == pytest.approx(
                mine["camera.azimuth"], abs=1.0)

            # The other defaults are untouched by the override.
            buttons = _js_attrs(
                handle, "window.viewer.ui._desc.camera._desc.views._desc")
            assert "ventral" in buttons and "lateral_right" in buttons
    finally:
        os.remove(viewfile)


def test_quickflat_size_reaches_the_browser():
    """The animation panel's flat-render hint is computed in python."""
    from cortex.webgl.view import _quickflat_size

    expected = _quickflat_size(subj)
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        assert subj in _js_attrs(handle, "window.viewopts.quickflat_size")
        # .slice() hands back a plain array, which survives the JSON round trip
        # that a bare property read does not.
        shipped = handle.send(method="run", params=[
            "window.viewopts.quickflat_size.%s.slice" % subj, []])
        assert shipped == expected

    if expected is not None:
        width, height = expected
        assert height == 1024
        assert width > 0


def test_quickflat_size_matches_a_real_quickflat_png(tmp_path):
    """The hint has to be the size make_png actually writes, not near it."""
    from PIL import Image

    from cortex.webgl.view import _quickflat_size

    expected = _quickflat_size(subj)
    if expected is None:
        pytest.skip("%s has no flat surface" % subj)

    out = str(tmp_path / "flat.png")
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    cortex.quickflat.make_png(out, vol, with_rois=False, with_labels=False,
                              with_colorbar=False)
    assert list(Image.open(out).size) == expected
