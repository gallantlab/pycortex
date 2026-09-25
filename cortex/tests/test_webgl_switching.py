"""NaN and alpha state must not leak between datasets in the WebGL viewer.

One headless viewer is loaded with several dataviews that differ only in where
they are NaN / transparent. After every ``setData`` switch the rendered image
must match the image obtained when that dataview is shown alone in a fresh
viewer; otherwise a NaN mask, an alpha map or an RGB alpha channel leaked from
the previously displayed dataset.

All dataviews render *red* where visible (``Reds`` colormap at a constant high
value, red RGB channels, or the red corner of ``RdBu_r_alpha``), so "visible
data" is simply the number of red-dominant pixels.

All tests are skipped if playwright is not installed.
"""
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
VIEW = {
    **default_view_params,
    **angle_view_params["lateral_pivot"],
    **unfold_view_params["inflated"],
}
VIEWER_PARAMS = dict(labels_visible=[], overlays_visible=[])
RTOL = 0.05  # relative tolerance on red-pixel counts


def _count_red(path):
    from PIL import Image

    rgb = np.asarray(Image.open(path).convert("RGB")).astype(int)
    return int((rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) > 50).sum())


def _render(handle, path, size=(512, 384)):
    import os

    handle.getImage(path, size)
    for _ in range(300):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("image not written: %s" % path)
    time.sleep(0.3)
    return _count_red(path)


def _pageerrors(handle):
    return [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]


def make_views():
    """Dataviews that are red where visible; half of them hide one half.

    Every dataview uses distinct data: two views sharing byte-identical data
    get the same content-hash name and the package would serve them twice.
    """
    pts_left = cortex.db.get_surf(subj, "fiducial")[0][0]
    nl = pts_left.shape[0]
    nv = cortex.db.get_surf(subj, "fiducial", merge=True)[0].shape[0]
    left = np.arange(nv) < nl  # left hemisphere vertices
    zz, yy, xx = np.mgrid[0 : volshape[0], 0 : volshape[1], 0 : volshape[2]]
    half_vox = xx < volshape[2] // 2

    def V(data, **kw):
        return cortex.Volume(data, subj, xfmname, **kw)

    def X(data, **kw):
        return cortex.Vertex(data, subj, **kw)

    views = {}
    # scalar vertex / volume (Reds, constant high value -> pure red)
    d = np.full(nv, 5.0); d[left] = np.nan
    views["vtx_nan"] = X(d, cmap="Reds", vmin=0, vmax=1)
    views["vtx_full"] = X(np.full(nv, 4.9), cmap="Reds", vmin=0, vmax=1)
    d = np.full(volshape, 5.0); d[half_vox] = np.nan
    views["vol_nan"] = V(d, cmap="Reds", vmin=0, vmax=1)
    views["vol_full"] = V(np.full(volshape, 4.9), cmap="Reds", vmin=0, vmax=1)
    # RGB with alpha channel
    views["vtxrgb_a0"] = cortex.VertexRGB(
        X(np.ones(nv), vmin=0, vmax=1), X(np.zeros(nv), vmin=0, vmax=1),
        X(np.zeros(nv), vmin=0, vmax=1), subj,
        alpha=X((~left).astype(float), vmin=0, vmax=1),
    )
    views["vtxrgb_full"] = cortex.VertexRGB(
        X(np.full(nv, 0.99), vmin=0, vmax=1), X(np.zeros(nv), vmin=0, vmax=1),
        X(np.zeros(nv), vmin=0, vmax=1), subj,
    )
    views["volrgb_a0"] = cortex.VolumeRGB(
        V(np.ones(volshape), vmin=0, vmax=1), V(np.zeros(volshape), vmin=0, vmax=1),
        V(np.zeros(volshape), vmin=0, vmax=1), subj, xfmname,
        alpha=V((~half_vox).astype(float), vmin=0, vmax=1),
    )
    views["volrgb_full"] = cortex.VolumeRGB(
        V(np.full(volshape, 0.99), vmin=0, vmax=1), V(np.zeros(volshape), vmin=0, vmax=1),
        V(np.zeros(volshape), vmin=0, vmax=1), subj, xfmname,
    )
    # 2D views: dim1 = 1 (red corner of RdBu_r_alpha), dim2 = 1 (opaque)
    kw2d = dict(cmap="RdBu_r_alpha", vmin=-1, vmax=1, vmin2=0, vmax2=1)
    views["vtx2d_alpha"] = cortex.Vertex2D(
        np.ones(nv), np.ones(nv), subj, alpha=(~left).astype(float), **kw2d
    )
    d2 = np.full(nv, 0.99); d2[left] = np.nan
    views["vtx2d_nan_dim2"] = cortex.Vertex2D(np.ones(nv), d2, subj, **kw2d)
    views["vtx2d_full"] = cortex.Vertex2D(np.full(nv, 0.98), np.ones(nv), subj, **kw2d)
    d2 = np.full(volshape, 0.99); d2[half_vox] = np.nan
    views["vol2d_nan_dim2"] = cortex.Volume2D(np.ones(volshape), d2, subj, xfmname, **kw2d)
    views["vol2d_alpha"] = cortex.Volume2D(
        np.ones(volshape), np.full(volshape, 0.98), subj, xfmname,
        alpha=(~half_vox).astype(float), **kw2d
    )
    return views


SEQUENCES = {
    "vertex_nan": ["vtx_nan", "vtx_full", "vtx_nan"],
    "vertexrgb_alpha": ["vtxrgb_a0", "vtxrgb_full", "vtxrgb_a0"],
    "vertex_scalar_vs_rgb": ["vtx_nan", "vtxrgb_full", "vtx_full", "vtxrgb_a0", "vtx_full"],
    "volume_nan": ["vol_nan", "vol_full", "vol_nan"],
    "volumergb_alpha": ["volrgb_a0", "volrgb_full", "volrgb_a0"],
    "volume_vs_vertex": ["vol_nan", "vtx_full", "vol_full", "vtx_nan", "vol_full"],
    "twod_alpha_and_nan": [
        "vtx2d_alpha", "vtx_full", "vtx2d_nan_dim2", "vtx2d_full", "vtx2d_alpha",
        "vol2d_alpha", "vol_full", "vol2d_nan_dim2", "vol2d_alpha",
    ],
}


class TestSwitching:
    """One multi-dataset viewer; per-dataset baselines from single viewers."""

    @pytest.fixture(autouse=True, scope="class")
    def _viewer(self, tmp_path_factory):
        cls = type(self)
        cls.views = make_views()
        cls.tmp = tmp_path_factory.mktemp("switching")
        cls.baseline = {}
        with cortex.export.headless_viewer(
            cortex.Dataset(**cls.views), viewer_params=VIEWER_PARAMS
        ) as handle:
            cls.handle = handle
            handle._set_view(**VIEW)
            time.sleep(2)
            yield

    @classmethod
    def _baseline(cls, name):
        if name not in cls.baseline:
            with cortex.export.headless_viewer(
                cls.views[name], viewer_params=VIEWER_PARAMS
            ) as handle:
                handle._set_view(**VIEW)
                time.sleep(2)
                cls.baseline[name] = _render(
                    handle, str(cls.tmp / ("baseline_%s.png" % name))
                )
        return cls.baseline[name]

    def _switch_and_count(self, name, tag):
        handle = type(self).handle
        handle.setData(name)
        time.sleep(2.5)
        return _render(handle, str(type(self).tmp / ("%s_%s.png" % (tag, name))))

    @pytest.mark.parametrize("sequence", sorted(SEQUENCES))
    def test_sequence(self, sequence):
        handle = type(self).handle
        errors_before = len(_pageerrors(handle))
        for step, name in enumerate(SEQUENCES[sequence]):
            count = self._switch_and_count(name, "%s_%d" % (sequence, step))
            expected = self._baseline(name)
            assert expected > 500, "baseline for %s renders nothing" % name
            assert abs(count - expected) <= RTOL * expected, (
                "%s step %d: %s rendered %d red pixels after %s, expected %d "
                "(NaN/alpha state leaked from the previous dataset?)"
                % (
                    sequence, step, name, count,
                    SEQUENCES[sequence][step - 1] if step else "load", expected,
                )
            )
        assert len(_pageerrors(handle)) == errors_before, _pageerrors(handle)

    def test_hidden_half_is_really_hidden(self):
        """Sanity check of the metric: the half-NaN / half-transparent
        dataviews show clearly fewer red pixels than their full versions."""
        for hidden, full in [
            ("vtx_nan", "vtx_full"), ("vtxrgb_a0", "vtxrgb_full"),
            ("vol_nan", "vol_full"), ("volrgb_a0", "volrgb_full"),
            ("vtx2d_alpha", "vtx2d_full"), ("vtx2d_nan_dim2", "vtx2d_full"),
        ]:
            assert self._baseline(hidden) < 0.8 * self._baseline(full), (hidden, full)


def test_addData_does_not_leak_nan_or_alpha(tmp_path):
    """Data pushed into a running viewer must not inherit the previous
    dataset's NaN mask or alpha."""
    views = make_views()
    # Reference for the RGB view shown on its own (shading differs between a
    # colormapped and an RGB view, so RGB is only compared with RGB).
    with cortex.export.headless_viewer(views["vtxrgb_full"], viewer_params=VIEWER_PARAMS) as handle:
        handle._set_view(**VIEW)
        time.sleep(2)
        n_rgb_full_alone = _render(handle, str(tmp_path / "rgb_full_alone.png"))

    with cortex.export.headless_viewer(views["vtx_nan"], viewer_params=VIEWER_PARAMS) as handle:
        handle._set_view(**VIEW)
        time.sleep(2)
        n_nan = _render(handle, str(tmp_path / "nan.png"))

        handle.addData(full=views["vtx_full"])
        time.sleep(3)
        n_full = _render(handle, str(tmp_path / "full.png"))

        handle.addData(rgb_a0=views["vtxrgb_a0"])
        time.sleep(3)
        n_a0 = _render(handle, str(tmp_path / "rgb_a0.png"))

        handle.addData(rgb_full=views["vtxrgb_full"])
        time.sleep(3)
        n_rgb_full = _render(handle, str(tmp_path / "rgb_full.png"))

        assert not _pageerrors(handle), _pageerrors(handle)

    assert n_full > 1.5 * n_nan, "NaN mask leaked into data added with addData"
    assert abs(n_rgb_full - n_rgb_full_alone) <= RTOL * n_rgb_full_alone, (
        "RGB alpha (or a NaN mask) leaked into data added with addData"
    )
    assert n_a0 < 0.8 * n_rgb_full, "alpha=0 half is not hidden after addData"
