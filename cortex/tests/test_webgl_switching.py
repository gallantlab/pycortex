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
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import numpy.typing as npt
import pytest

import cortex
import cortex.export
from cortex.dataset import Dataview
from cortex.export.save_views import (
    angle_view_params,
    default_view_params,
    unfold_view_params,
    ViewParams,
)
from cortex.tests.testing_utils import (
    count_red_pixels,
    has_playwright,
    page_errors,
    render,
    set_view,
    wait_active,
)

pytestmark = pytest.mark.skipif(
    not has_playwright, reason="playwright and chromium are required"
)

subj, xfmname, volshape = "S1", "fullhead", (31, 100, 100)
VIEW: ViewParams = {
    **default_view_params,
    **angle_view_params["lateral_pivot"],
    **unfold_view_params["inflated"],
}
VIEWER_PARAMS = dict(labels_visible=[], overlays_visible=[])
RTOL = 0.05  # relative tolerance on red-pixel counts


def _render(handle: Any, path: str) -> int:
    render(handle, path)
    return count_red_pixels(path)


@pytest.fixture(scope="module")
def views() -> dict[str, Dataview]:
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

    def V(data: npt.NDArray, **kw: Any) -> cortex.Volume:
        return cortex.Volume(data, subj, xfmname, **kw)

    def X(data: npt.NDArray, **kw: Any) -> cortex.Vertex:
        return cortex.Vertex(data, subj, **kw)

    views: dict[str, Dataview] = {}
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
    kw2d: dict[str, Any] = dict(cmap="RdBu_r_alpha", vmin=-1, vmax=1, vmin2=0, vmax2=1)
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


@pytest.fixture(scope="module")
def baseline(
    views: dict[str, Dataview], tmp_path_factory: pytest.TempPathFactory
) -> Callable[[str], int]:
    """Red-pixel count of each dataview shown alone in a fresh viewer.

    A fresh viewer per dataview is the expensive part of this module, so the
    counts are cached and shared by every test in it.
    """
    tmp = tmp_path_factory.mktemp("baselines")
    counts: dict[str, int] = {}

    def get(name: str) -> int:
        if name not in counts:
            with cortex.export.headless_viewer(
                views[name], viewer_params=VIEWER_PARAMS
            ) as handle:
                set_view(handle, VIEW)
                counts[name] = _render(handle, str(tmp / ("baseline_%s.png" % name)))
        return counts[name]

    return get


class TestSwitching:
    """One multi-dataset viewer; per-dataset baselines from single viewers."""

    tmp: Path
    handle: Any

    @pytest.fixture(autouse=True, scope="class")
    def _viewer(
        self, views: dict[str, Dataview], tmp_path_factory: pytest.TempPathFactory
    ) -> Iterator[None]:
        cls = type(self)
        cls.tmp = tmp_path_factory.mktemp("switching")
        with cortex.export.headless_viewer(
            cortex.Dataset(**views), viewer_params=VIEWER_PARAMS
        ) as handle:
            cls.handle = handle
            set_view(handle, VIEW)
            yield

    def _switch_and_count(self, name: str, tag: str) -> int:
        handle = type(self).handle
        handle.setData(name)
        wait_active(handle, name)
        return _render(handle, str(type(self).tmp / ("%s_%s.png" % (tag, name))))

    @pytest.mark.parametrize("sequence", sorted(SEQUENCES))
    def test_sequence(self, sequence: str, baseline: Callable[[str], int]) -> None:
        handle = type(self).handle
        errors_before = len(page_errors(handle))
        for step, name in enumerate(SEQUENCES[sequence]):
            count = self._switch_and_count(name, "%s_%d" % (sequence, step))
            expected = baseline(name)
            assert expected > 500, "baseline for %s renders nothing" % name
            assert abs(count - expected) <= RTOL * expected, (
                "%s step %d: %s rendered %d red pixels after %s, expected %d "
                "(NaN/alpha state leaked from the previous dataset?)"
                % (
                    sequence, step, name, count,
                    SEQUENCES[sequence][step - 1] if step else "load", expected,
                )
            )
        assert len(page_errors(handle)) == errors_before, page_errors(handle)

    def test_hidden_half_is_really_hidden(self, baseline: Callable[[str], int]) -> None:
        """Sanity check of the metric: the half-NaN / half-transparent
        dataviews show clearly fewer red pixels than their full versions."""
        for hidden, full in [
            ("vtx_nan", "vtx_full"), ("vtxrgb_a0", "vtxrgb_full"),
            ("vol_nan", "vol_full"), ("volrgb_a0", "volrgb_full"),
            ("vtx2d_alpha", "vtx2d_full"), ("vtx2d_nan_dim2", "vtx2d_full"),
        ]:
            assert baseline(hidden) < 0.8 * baseline(full), (hidden, full)


def test_addData_does_not_leak_nan_or_alpha(
    tmp_path: Path, views: dict[str, Dataview], baseline: Callable[[str], int]
) -> None:
    """Data pushed into a running viewer must not inherit the previous
    dataset's NaN mask or alpha."""
    # Reference for the RGB view shown on its own (shading differs between a
    # colormapped and an RGB view, so RGB is only compared with RGB).
    n_rgb_full_alone = baseline("vtxrgb_full")

    with cortex.export.headless_viewer(views["vtx_nan"], viewer_params=VIEWER_PARAMS) as handle:
        set_view(handle, VIEW)
        n_nan = _render(handle, str(tmp_path / "nan.png"))

        handle.addData(full=views["vtx_full"])
        wait_active(handle, "full")
        n_full = _render(handle, str(tmp_path / "full.png"))

        handle.addData(rgb_a0=views["vtxrgb_a0"])
        wait_active(handle, "rgb_a0")
        n_a0 = _render(handle, str(tmp_path / "rgb_a0.png"))

        handle.addData(rgb_full=views["vtxrgb_full"])
        wait_active(handle, "rgb_full")
        n_rgb_full = _render(handle, str(tmp_path / "rgb_full.png"))

        assert not page_errors(handle), page_errors(handle)

    assert n_full > 1.5 * n_nan, "NaN mask leaked into data added with addData"
    assert abs(n_rgb_full - n_rgb_full_alone) <= RTOL * n_rgb_full_alone, (
        "RGB alpha (or a NaN mask) leaked into data added with addData"
    )
    assert n_a0 < 0.8 * n_rgb_full, "alpha=0 half is not hidden after addData"
