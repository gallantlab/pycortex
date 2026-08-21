"""NaN / alpha handling must be identical across quickflat, the WebGL data
package and the RGB conversions (browser-free tests).

Rule: a NaN anywhere at a voxel/vertex -- in either dimension of a 2D view, in
any RGB channel, or in the alpha map itself -- renders as alpha 0. Where there
is no NaN, the alpha (2D ``*_alpha`` colormap, ``alpha=`` kwarg, RGB alpha
channel) is honored.
"""
import json
import os
import tempfile
import warnings

import numpy as np
import pytest

import cortex
from cortex import dataset
from cortex.testing_utils import has_installed
from cortex.webgl.data import Package
from cortex.webgl.serve import NPEncode

subj, xfmname, volshape = "S1", "fullhead", (31, 100, 100)
no_inkscape = not has_installed("inkscape")


def _nverts():
    return cortex.db.get_surf(subj, "fiducial", merge=True)[0].shape[0]


def _vol_grid():
    zz, yy, xx = np.mgrid[0 : volshape[0], 0 : volshape[1], 0 : volshape[2]]
    return zz, yy, xx


def _make_2d(kind, d1, d2, alpha=None, **kw):
    kw.setdefault("cmap", "RdBu_r_alpha")
    kw.update(vmin=-1, vmax=1, vmin2=0, vmax2=1)
    if kind == "Volume2D":
        return cortex.Volume2D(d1, d2, subj, xfmname, alpha=alpha, **kw)
    return cortex.Vertex2D(d1, d2, subj, alpha=alpha, **kw)


def _rgba(view):
    """uint8 RGBA array of the quickflat/raw representation, time axis dropped."""
    if isinstance(view, (cortex.Volume2D, cortex.VolumeRGB)):
        arr = view.volume if isinstance(view, cortex.VolumeRGB) else view.raw.volume
    else:
        arr = view.vertices if isinstance(view, cortex.VertexRGB) else view.raw.vertices
    return arr[0]


# ---------------------------------------------------------------------------
# Volume2D / Vertex2D: NaN in either dim or in alpha -> 0; alpha= honored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["Volume2D", "Vertex2D"])
@pytest.mark.parametrize("nan_in", ["dim1", "dim2", "alpha"])
def test_2d_nan_anywhere_is_transparent(kind, nan_in):
    rng = np.random.default_rng(0)
    if kind == "Volume2D":
        shape = volshape
        region = _vol_grid()[1] < 35
    else:
        shape = (_nverts(),)
        region = np.arange(shape[0]) < shape[0] // 3
    d1 = rng.uniform(-1, 1, shape)
    d2 = np.ones(shape)
    alpha = rng.uniform(0.2, 1, shape)
    {"dim1": d1, "dim2": d2, "alpha": alpha}[nan_in][region] = np.nan

    rgba = _rgba(_make_2d(kind, d1, d2, alpha=alpha))
    assert rgba[region][..., 3].max() == 0, "NaN must give alpha 0"
    assert rgba[~region][..., 3].min() > 0, "non-NaN must keep some alpha"


@pytest.mark.parametrize("kind", ["Volume2D", "Vertex2D"])
def test_2d_alpha_kwarg_multiplies_colormap_alpha(kind):
    rng = np.random.default_rng(1)
    shape = volshape if kind == "Volume2D" else (_nverts(),)
    d1 = rng.uniform(-1, 1, shape)
    d2 = rng.uniform(0, 1, shape)
    user = rng.uniform(0, 1, shape)

    a_cmap = _rgba(_make_2d(kind, d1, d2)).astype(float)[..., 3]
    a_both = _rgba(_make_2d(kind, d1, d2, alpha=user))[..., 3]
    expected = np.round(a_cmap * user).astype(np.uint8)
    np.testing.assert_array_equal(a_both, expected)

    # alpha=1 everywhere is a no-op
    a_one = _rgba(_make_2d(kind, d1, d2, alpha=np.ones(shape)))[..., 3]
    np.testing.assert_array_equal(a_one, a_cmap.astype(np.uint8))


def test_2d_alpha_accepts_volume_with_own_range():
    rng = np.random.default_rng(2)
    d1 = rng.uniform(-1, 1, volshape)
    d2 = np.ones(volshape)
    acc = rng.uniform(0, 10, volshape)  # e.g. a "confidence" in [0, 10]
    v = _make_2d(
        "Volume2D", d1, d2, alpha=cortex.Volume(acc, subj, xfmname, vmin=0, vmax=10)
    )
    a = _rgba(v)[..., 3].astype(float)
    a_cmap = _rgba(_make_2d("Volume2D", d1, d2)).astype(float)[..., 3]
    np.testing.assert_array_equal(a, np.round(a_cmap * acc / 10.0))


def test_2d_alpha_not_in_attrs_and_json_serializable():
    """The alpha map used to be stuffed into ``attrs`` as an ndarray, which
    crashed the WebGL viewer (500: ndarray is not JSON serializable)."""
    rng = np.random.default_rng(3)
    v = _make_2d(
        "Volume2D",
        rng.uniform(-1, 1, volshape),
        np.ones(volshape),
        alpha=rng.uniform(0, 1, volshape),
    )
    assert not any(isinstance(x, np.ndarray) for x in v.attrs.values())
    assert isinstance(v.alpha, cortex.Volume)
    js = v.to_json()
    json.dumps(js, cls=NPEncode)
    json.dumps(js)  # plain encoder, as used by the mixer.html handler
    assert js["alpha"] == [v.alpha.name]

    v_noalpha = _make_2d("Volume2D", rng.uniform(-1, 1, volshape), np.ones(volshape))
    assert "alpha" not in v_noalpha.to_json()


def test_2d_alpha_rejects_other_subject_or_xfm():
    rng = np.random.default_rng(4)
    d1 = rng.uniform(-1, 1, volshape)
    bad = cortex.Volume(rng.uniform(0, 1, volshape), subj, xfmname)
    bad.subject = "not_S1"
    with pytest.raises(ValueError):
        _make_2d("Volume2D", d1, np.ones(volshape), alpha=bad)


def test_package_ships_2d_alpha_as_float_brain():
    rng = np.random.default_rng(5)
    alpha = rng.uniform(0, 1, volshape)
    alpha[0] = np.nan
    v = _make_2d("Volume2D", rng.uniform(-1, 1, volshape), np.ones(volshape), alpha=alpha)
    pkg = Package(dataset.Dataset(v=v))
    assert len(pkg.brains) == 3
    assert all(b["raw"] is False for b in pkg.brains.values())
    meta = pkg.metadata()
    assert meta["views"][0]["alpha"] == [v.alpha.name]
    assert v.alpha.name in meta["images"]


def test_2d_to_json_keeps_zero_bounds():
    """``vmin=0``/``vmax=0`` must not fall back to the auto range
    (truthiness bug; same class as 5482c8bf)."""
    rng = np.random.default_rng(6)
    dim1 = cortex.Volume(rng.uniform(-1, 1, volshape), subj, xfmname)  # auto range
    dim2 = cortex.Volume(rng.uniform(-1, 1, volshape), subj, xfmname)
    v = cortex.Volume2D(dim1, dim2, vmin=0, vmax=1, vmin2=-1, vmax2=0)
    js = v.to_json()
    assert js["vmin"][0] == [0, -1]
    assert js["vmax"][0] == [1, 0]


def test_2d_alpha_hdf_roundtrip():
    rng = np.random.default_rng(7)
    alpha = rng.uniform(0, 1, volshape)
    v = _make_2d("Volume2D", rng.uniform(-1, 1, volshape), np.ones(volshape), alpha=alpha)
    tf = tempfile.NamedTemporaryFile(suffix=".hdf", delete=False)
    tf.close()
    os.unlink(tf.name)
    try:
        dataset.Dataset(twod=v).save(tf.name)
        loaded = cortex.load(tf.name)
        assert isinstance(loaded.twod.alpha, cortex.Volume)
        np.testing.assert_allclose(loaded.twod.alpha.data, alpha, atol=1e-6)
        np.testing.assert_array_equal(_rgba(loaded.twod), _rgba(v))
    finally:
        if os.path.exists(tf.name):
            os.unlink(tf.name)


# ---------------------------------------------------------------------------
# VolumeRGB / VertexRGB
# ---------------------------------------------------------------------------


def test_volumergb_masked_alpha_nan_channel_is_transparent():
    """A masked (linear) alpha Volume used to be written through a temporary
    (``alpha.volume[mask] = vmin``), so NaN voxels stayed opaque."""
    rng = np.random.default_rng(8)
    mask = cortex.db.get_mask(subj, xfmname, "thick")
    zz, yy, xx = _vol_grid()
    r = rng.uniform(0, 1, volshape)
    r[yy < 35] = np.nan
    alpha_lin = cortex.Volume(np.full(mask.sum(), 0.8), subj, xfmname, vmin=0, vmax=1)
    rgb = cortex.VolumeRGB(
        cortex.Volume(r, subj, xfmname, vmin=0, vmax=1),
        cortex.Volume(rng.uniform(0, 1, volshape), subj, xfmname, vmin=0, vmax=1),
        cortex.Volume(rng.uniform(0, 1, volshape), subj, xfmname, vmin=0, vmax=1),
        subj,
        xfmname,
        alpha=alpha_lin,
    )
    a = _rgba(rgb)[..., 3]
    assert a[np.isnan(r) & mask].max() == 0
    assert a[~np.isnan(r) & mask].min() == 204  # round(0.8 * 255)
    # the user's alpha object is untouched
    assert np.all(alpha_lin.data == 0.8)


@pytest.mark.parametrize("cls", ["VertexRGB", "VolumeRGB"])
def test_rgb_multiframe_nan_masks_per_frame(cls):
    """Regression for #629: multi-frame data + NaN raised IndexError because the
    auto alpha was single-frame while the NaN mask was (T, ...)."""
    rng = np.random.default_rng(9)
    T = 3
    shape = (T, _nverts()) if cls == "VertexRGB" else (T,) + volshape
    r, g, b = (rng.uniform(0, 1, shape) for _ in range(3))
    idx = (slice(None),) + (0,) * (len(shape) - 2) + (slice(0, 50),)
    r[(1,) + idx[1:]] = np.nan  # frame 1 only
    if cls == "VertexRGB":
        rgb = cortex.VertexRGB(r, g, b, subj)
        arr = rgb.vertices
    else:
        rgb = cortex.VolumeRGB(r, g, b, subj, xfmname)
        arr = rgb.volume
    assert arr.shape[0] == T
    nan_here = np.isnan(r)
    assert arr[..., 3][nan_here].max() == 0
    assert arr[..., 3][~nan_here].min() > 0
    # a user-supplied single-frame alpha is broadcast, not rejected
    if cls == "VertexRGB":
        rgb.alpha = np.full(shape[1:], 0.5)
        arr = rgb.vertices
    else:
        rgb.alpha = np.full(shape[1:], 0.5)
        arr = rgb.volume
    assert arr[..., 3][nan_here].max() == 0
    assert arr[..., 3][~nan_here].min() == 127  # int(0.5 * 255)


@pytest.mark.parametrize("cls", ["VertexRGB", "VolumeRGB"])
def test_rgb_nan_in_alpha_is_transparent(cls):
    rng = np.random.default_rng(10)
    shape = (_nverts(),) if cls == "VertexRGB" else volshape
    alpha = rng.uniform(0.5, 1, shape)
    region = np.arange(shape[0]) < shape[0] // 2
    alpha[region] = np.nan
    r, g, b = (rng.uniform(0, 1, shape) for _ in range(3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "invalid value in cast" warnings
        if cls == "VertexRGB":
            arr = cortex.VertexRGB(r, g, b, subj, alpha=alpha).vertices[0]
        else:
            arr = cortex.VolumeRGB(r, g, b, subj, xfmname, alpha=alpha).volume[0]
    assert arr[region][..., 3].max() == 0
    assert arr[~region][..., 3].min() > 0


def test_color_voxels_does_not_mutate_caller_alpha():
    rng = np.random.default_rng(11)
    r = rng.uniform(0, 1, volshape)
    r[0] = np.nan
    alpha = np.ones(volshape)
    rgb = cortex.VolumeRGB(
        r, rng.uniform(0, 1, volshape), rng.uniform(0, 1, volshape), subj, xfmname,
        vmin=0, vmax=1, alpha=alpha,  # vmin/vmax -> color_voxels path
    )
    assert np.all(alpha == 1.0)
    a = rgb.volume[0][..., 3]
    assert a[0].max() == 0
    assert a[1:].min() == 255


# ---------------------------------------------------------------------------
# quickflat
# ---------------------------------------------------------------------------


def test_make_flatmap_image_rgb_averages_premultiplied():
    """Transparent (alpha 0 / NaN) voxels must not darken neighbouring pixels.

    quickflat averages RGBA over the voxels that fall into a pixel (cortical
    thickness); the WebGL viewer does this in premultiplied space. Averaging
    straight RGBA gave dark halos around transparent regions in quickflat only.
    A voxel checkerboard of alpha 0/1 makes almost every pixel a mix.
    """
    zz, yy, xx = _vol_grid()
    checker = ((xx + yy + zz) % 2).astype(bool)
    # Opaque voxels are pure red; the others are NaN in the red channel (so
    # they become transparent, and black in the uint8 conversion). Straight
    # averaging mixes that black into neighbouring pixels (R ~ 128).
    r = np.where(checker, 1.0, np.nan).astype(np.float32)
    red = cortex.VolumeRGB(
        cortex.Volume(r, subj, xfmname, vmin=0, vmax=1),
        cortex.Volume(np.zeros(volshape, np.float32), subj, xfmname, vmin=0, vmax=1),
        cortex.Volume(np.zeros(volshape, np.float32), subj, xfmname, vmin=0, vmax=1),
        subj,
        xfmname,
    )
    img, _ = cortex.quickflat.utils.make_flatmap_image(red)
    a = img[..., 3]
    partial = (a > 0) & (a < 255)  # pixels straddling the alpha edge
    assert partial.sum() > 100
    # Color must stay pure bright red regardless of partial coverage
    assert img[partial][:, 0].min() >= 250
    assert img[partial][:, 1:3].max() <= 5
    # and fully opaque pixels are unchanged
    if (a == 255).any():
        assert img[a == 255][:, 0].min() == 255


def test_make_flatmap_image_volume_nan_transparent_when_masked():
    mask = cortex.db.get_mask(subj, xfmname, "thick")
    data = np.ones(mask.sum())
    data[: data.size // 2] = np.nan
    vol = cortex.Volume(data, subj, xfmname, vmin=0, vmax=1)
    img, _ = cortex.quickflat.utils.make_flatmap_image(vol, nanmean=True)
    assert np.nanmin(img) == 1
    assert np.isnan(img).any()


@pytest.mark.skipif(no_inkscape, reason="Inkscape required")
def test_make_svg_scalar_with_nan():
    """``make_svg`` indexed ``arr[..., 3]`` on a 2-D scalar image."""
    data = np.random.default_rng(12).uniform(0, 1, volshape)
    data[:, :35] = np.nan
    vol = cortex.Volume(data, subj, xfmname, vmin=0, vmax=1, cmap="viridis")
    tf = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    tf.close()
    try:
        cortex.quickflat.make_svg(tf.name, vol, with_labels=False)
        assert os.path.getsize(tf.name) > 0
    finally:
        os.unlink(tf.name)
