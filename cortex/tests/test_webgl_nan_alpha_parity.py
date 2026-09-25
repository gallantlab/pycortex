"""quickflat (matplotlib) and the WebGL viewer must agree on NaN and alpha.

For a set of dataviews covering every NaN / alpha pattern, render the flatmap
with ``cortex.quickshow`` and with the headless WebGL viewer and compare the
fraction of the brain that shows data (red-dominant pixels over all non-white
pixels). Pixel-exact comparison is not possible (different rasterizers,
resolutions, curvature rendering), but the fraction of transparent cortex is,
within a tolerance.

Skipped if playwright is not installed.
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

subj, xfmname, volshape = "S1", "fullhead", (31, 100, 100)
FLAT = {
    **default_view_params,
    **angle_view_params["flatmap"],
    **unfold_view_params["flatmap"],
}
TOL = 0.12  # absolute tolerance on the visible-data fraction


def _fractions(path):
    """(fraction of brain pixels showing data, number of brain pixels)."""
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    rgb = np.asarray(Image.alpha_composite(bg, im).convert("RGB")).astype(int)
    brain = rgb.min(axis=2) < 235  # anything not (near) white
    red = (rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])) > 40
    n_brain = int(brain.sum())
    return (int((red & brain).sum()) / max(n_brain, 1)), n_brain


def _quickshow(view, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = cortex.quickshow(
        view, with_curvature=True, with_rois=False, with_labels=False,
        with_colorbar=False, with_sulci=False, with_borders=False, height=256,
    )
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=80)
    plt.close(fig)
    return _fractions(path)


def _webgl(view, path):
    with cortex.export.headless_viewer(
        view, viewer_params=dict(labels_visible=[], overlays_visible=[])
    ) as handle:
        handle._set_view(**FLAT)
        time.sleep(3)
        handle.getImage(path, (1024, 768))
        for _ in range(300):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                break
            time.sleep(0.1)
        time.sleep(0.3)
        errors = [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]
    assert not errors, errors
    return _fractions(path)


def _cases():
    zz, yy, xx = np.mgrid[0 : volshape[0], 0 : volshape[1], 0 : volshape[2]]
    post = yy < 35  # posterior slab
    pts = cortex.db.get_surf(subj, "fiducial", merge=True)[0]
    nv = pts.shape[0]
    vpost = pts[:, 1] < np.percentile(pts[:, 1], 35)
    mask = cortex.db.get_mask(subj, xfmname, "thick")

    def V(d, **kw):
        return cortex.Volume(d, subj, xfmname, **kw)

    def X(d, **kw):
        return cortex.Vertex(d, subj, **kw)

    ones_v, ones_x = np.ones(volshape), np.ones(nv)
    nan_v = ones_v.copy(); nan_v[post] = np.nan
    nan_x = ones_x.copy(); nan_x[vpost] = np.nan
    a0_v = (~post).astype(float)
    a0_x = (~vpost).astype(float)
    kw2d = dict(cmap="RdBu_r_alpha", vmin=-1, vmax=1, vmin2=0, vmax2=1)
    zeros_v, zeros_x = np.zeros(volshape), np.zeros(nv)

    return {
        "volume_nan": V(nan_v * 5, cmap="Reds", vmin=0, vmax=1),
        "vertex_nan": X(nan_x * 5, cmap="Reds", vmin=0, vmax=1),
        "volume_masked_nan": V((nan_v * 5)[mask], cmap="Reds", vmin=0, vmax=1),
        "volume2d_nan_dim1": cortex.Volume2D(nan_v, ones_v, subj, xfmname, **kw2d),
        "volume2d_nan_dim2": cortex.Volume2D(ones_v, nan_v, subj, xfmname, **kw2d),
        "vertex2d_nan_dim2": cortex.Vertex2D(ones_x, nan_x, subj, **kw2d),
        "volume2d_alpha0": cortex.Volume2D(ones_v, ones_v, subj, xfmname, alpha=a0_v, **kw2d),
        "vertex2d_alpha0": cortex.Vertex2D(ones_x, ones_x, subj, alpha=a0_x, **kw2d),
        "vertex2d_alpha_nan": cortex.Vertex2D(
            ones_x, ones_x, subj, alpha=np.where(vpost, np.nan, 1.0), **kw2d
        ),
        "volumergb_nan_channel": cortex.VolumeRGB(
            V(nan_v, vmin=0, vmax=1), V(zeros_v, vmin=0, vmax=1),
            V(zeros_v, vmin=0, vmax=1), subj, xfmname,
        ),
        "volumergb_masked_alpha": cortex.VolumeRGB(
            V(nan_v, vmin=0, vmax=1), V(zeros_v, vmin=0, vmax=1),
            V(zeros_v, vmin=0, vmax=1), subj, xfmname,
            alpha=V(np.full(mask.sum(), 1.0), vmin=0, vmax=1),
        ),
        "volumergb_nan_in_alpha": cortex.VolumeRGB(
            V(ones_v, vmin=0, vmax=1), V(zeros_v, vmin=0, vmax=1),
            V(zeros_v, vmin=0, vmax=1), subj, xfmname,
            alpha=np.where(post, np.nan, 1.0),
        ),
        "volumergb_color_voxels_nan": cortex.VolumeRGB(
            nan_v, zeros_v, zeros_v, subj, xfmname, vmin=0, vmax=1,
        ),
        "vertexrgb_alpha0": cortex.VertexRGB(
            X(ones_x, vmin=0, vmax=1), X(zeros_x, vmin=0, vmax=1),
            X(zeros_x, vmin=0, vmax=1), subj, alpha=a0_x,
        ),
        "vertexrgb_nan_in_alpha": cortex.VertexRGB(
            X(ones_x, vmin=0, vmax=1), X(zeros_x, vmin=0, vmax=1),
            X(zeros_x, vmin=0, vmax=1), subj, alpha=np.where(vpost, np.nan, 1.0),
        ),
    }


@pytest.fixture(scope="module")
def cases():
    return _cases()


@pytest.mark.parametrize("name", [
    "volume_nan", "vertex_nan", "volume_masked_nan",
    "volume2d_nan_dim1", "volume2d_nan_dim2", "vertex2d_nan_dim2",
    "volume2d_alpha0", "vertex2d_alpha0", "vertex2d_alpha_nan",
    "volumergb_nan_channel", "volumergb_masked_alpha", "volumergb_nan_in_alpha",
    "volumergb_color_voxels_nan", "vertexrgb_alpha0", "vertexrgb_nan_in_alpha",
])
def test_visible_fraction_matches(name, cases, tmp_path):
    view = cases[name]
    f_qs, n_qs = _quickshow(view, str(tmp_path / ("qs_%s.png" % name)))
    f_wg, n_wg = _webgl(view, str(tmp_path / ("wg_%s.png" % name)))
    assert n_qs > 1000 and n_wg > 1000, "brain not found in one of the renders"
    # every case hides roughly the posterior third: neither fully shown nor hidden
    assert 0.25 < f_qs < 0.9, "quickshow fraction %.2f" % f_qs
    assert 0.25 < f_wg < 0.9, "webgl fraction %.2f" % f_wg
    assert abs(f_qs - f_wg) < TOL, (
        "%s: quickshow shows data on %.0f%% of the cortex, WebGL on %.0f%%"
        % (name, 100 * f_qs, 100 * f_wg)
    )


def test_multilayer_nanmean_toggle(tmp_path):
    """With several layers, NaN voxels are left out of the average (like
    quickflat's ``nanmean=True``) unless the surface's ``nanmean`` toggle is
    off, in which case one NaN at any depth makes the fragment transparent."""
    zz, yy, xx = np.mgrid[0 : volshape[0], 0 : volshape[1], 0 : volshape[2]]
    d = np.full(volshape, 5.0)
    d[(xx + yy + zz) % 3 == 0] = np.nan  # a third of the voxels, scattered
    vol = cortex.Volume(d, subj, xfmname, cmap="Reds", vmin=0, vmax=1)
    view = {
        **default_view_params,
        **angle_view_params["lateral_pivot"],
        **unfold_view_params["inflated"],
    }

    def _red(path):
        from PIL import Image

        rgb = np.asarray(Image.open(path).convert("RGB")).astype(int)
        return int((rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) > 50).sum())

    counts = {}
    with cortex.export.headless_viewer(
        vol, viewer_params=dict(labels_visible=[], overlays_visible=[])
    ) as handle:
        handle._set_view(**view)
        time.sleep(2)
        for layers, nanmean in [(1, True), (8, True), (8, False)]:
            handle.ui.set("surface.%s.layers" % subj, layers)
            handle.ui.set("surface.%s.nanmean" % subj, nanmean)
            time.sleep(2.5)
            path = str(tmp_path / ("layers%d_nanmean%s.png" % (layers, nanmean)))
            handle.getImage(path, (512, 384))
            for _ in range(300):
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    break
                time.sleep(0.1)
            time.sleep(0.3)
            counts[(layers, nanmean)] = _red(path)
        errors = [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]
    assert not errors, errors
    assert counts[(1, True)] > 5000
    # nanmean: averaging over the valid layers shows at least as much cortex
    assert counts[(8, True)] >= 0.9 * counts[(1, True)], counts
    # toggle off: any NaN among the 8 layers hides the fragment
    assert counts[(8, False)] < 0.6 * counts[(8, True)], counts


def test_multilayer_nanmean_toggle_rgb(tmp_path):
    """Same as above for RGB data: NaN became alpha 0 in the texture, so with
    ``nanmean`` fully transparent layer samples are left out of the average."""
    zz, yy, xx = np.mgrid[0 : volshape[0], 0 : volshape[1], 0 : volshape[2]]
    r = np.ones(volshape)
    r[(xx + yy + zz) % 3 == 0] = np.nan
    zeros = np.zeros(volshape)
    vol = cortex.VolumeRGB(
        cortex.Volume(r, subj, xfmname, vmin=0, vmax=1),
        cortex.Volume(zeros, subj, xfmname, vmin=0, vmax=1),
        cortex.Volume(zeros, subj, xfmname, vmin=0, vmax=1), subj, xfmname,
    )
    view = {
        **default_view_params,
        **angle_view_params["lateral_pivot"],
        **unfold_view_params["inflated"],
    }

    def _red(path):
        """Total redness: sum of R - max(G, B) over red-dominant pixels, so
        that a partially transparent red (alpha-weighted average) scores
        lower than an opaque one covering the same pixels."""
        from PIL import Image

        rgb = np.asarray(Image.open(path).convert("RGB")).astype(int)
        redness = rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])
        return int(redness[redness > 50].sum())

    counts = {}
    with cortex.export.headless_viewer(
        vol, viewer_params=dict(labels_visible=[], overlays_visible=[])
    ) as handle:
        handle._set_view(**view)
        time.sleep(2)
        for layers, nanmean in [(1, True), (8, True), (8, False)]:
            handle.ui.set("surface.%s.layers" % subj, layers)
            handle.ui.set("surface.%s.nanmean" % subj, nanmean)
            time.sleep(2.5)
            path = str(tmp_path / ("rgb_layers%d_nanmean%s.png" % (layers, nanmean)))
            handle.getImage(path, (512, 384))
            for _ in range(300):
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    break
                time.sleep(0.1)
            time.sleep(0.3)
            counts[(layers, nanmean)] = _red(path)
        errors = [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]
    assert not errors, errors
    assert counts[(1, True)] > 5000 * 50
    assert counts[(8, True)] >= 0.9 * counts[(1, True)], counts
    # alpha-weighted average: the NaN layers dilute the red (about a third)
    assert counts[(8, False)] < 0.85 * counts[(8, True)], counts


def test_vertex_movie_nan_in_next_frame_is_transparent(tmp_path):
    """Between two frames the vertex shader mixes frame f and f+1. A vertex
    that is NaN in f+1 (replaced by 0 in the GPU buffer) must be masked while
    interpolating, not fade towards a fake 0."""
    nverts = cortex.db.get_surf(subj, "fiducial", merge=True)[0].shape[0]
    nl = cortex.db.get_surf(subj, "fiducial")[0][0].shape[0]
    movie = np.full((2, nverts), 5.0)
    movie[1, :nl] = np.nan  # left hemisphere undefined in frame 1 only
    vtx = cortex.Vertex(movie, subj, cmap="Reds", vmin=0, vmax=1)
    view = {
        **default_view_params,
        **angle_view_params["lateral_pivot"],
        **unfold_view_params["inflated"],
    }

    def _red(path):
        from PIL import Image

        rgb = np.asarray(Image.open(path).convert("RGB")).astype(int)
        return int((rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) > 50).sum())

    counts = {}
    with cortex.export.headless_viewer(
        vtx, viewer_params=dict(labels_visible=[], overlays_visible=[])
    ) as handle:
        handle._set_view(**view)
        time.sleep(2)
        for frame in (0.0, 0.5, 1.0):
            handle.setFrame(frame)
            time.sleep(2)
            path = str(tmp_path / ("frame_%.1f.png" % frame))
            handle.getImage(path, (512, 384))
            for _ in range(300):
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    break
                time.sleep(0.1)
            time.sleep(0.3)
            counts[frame] = _red(path)
        errors = [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]
    assert not errors, errors
    assert counts[1.0] < 0.8 * counts[0.0], counts  # left hemisphere hidden in frame 1
    # while blending towards frame 1 the NaN vertices are already masked
    assert abs(counts[0.5] - counts[1.0]) <= 0.1 * counts[1.0], counts
