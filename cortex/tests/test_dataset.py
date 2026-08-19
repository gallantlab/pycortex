import os
import subprocess
import sys
import tempfile

import cortex
import numpy as np
import pytest

from cortex import db, dataset
from cortex.testing_utils import has_installed

subj, xfmname, nverts, volshape = "S1", "fullhead", 304380, (31, 100, 100)

no_inkscape = not has_installed("inkscape")


def test_braindata():
    vol = np.random.randn(*volshape)
    tf = tempfile.TemporaryFile(suffix=".png")
    mask = db.get_mask(subj, xfmname, "thick")

    data = dataset.Volume(vol, subj, xfmname, cmap="RdBu_r", vmin=0, vmax=1)
    # quickflat.make_png(tf, data)
    mdata = data.masked["thick"]
    assert len(mdata.data) == mask.sum()
    assert np.allclose(mdata.volume[:, mask], mdata.data)


def test_dataset():
    vol = np.random.randn(*volshape)
    stack = (np.ones(volshape[::-1]) * np.linspace(0, 1, volshape[0])).T
    mask = db.get_mask(subj, xfmname, "thick")

    ds = dataset.Dataset(randvol=(vol, subj, xfmname), stack=(stack, subj, xfmname))
    ds.append(thickstack=ds.stack.masked["thick"])
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds.save(tf.name)

    ds = dataset.Dataset.from_file(tf.name)
    assert len(ds["thickstack"].data) == mask.sum()
    assert np.allclose(ds["stack"].data[mask], ds["thickstack"].data)


def test_findmask():
    vol = np.random.rand(10, *volshape)
    mask = db.get_mask(subj, xfmname, "thin")
    ds = dataset.Volume(vol[:, mask], subj, xfmname)
    assert np.allclose(ds.volume[:, mask], vol[:, mask])


def test_rgb():
    red, green, blue, alpha = [np.random.randn(*volshape) for _ in range(4)]

    rgb = dataset.VolumeRGB(red, green, blue, subj, xfmname)
    assert rgb.volume.shape == tuple([1] + list(volshape) + [4])
    assert rgb.volume.dtype == np.uint8
    assert rgb.volume[..., 3].max() > 0

    rgba = dataset.VolumeRGB(red, green, blue, subj, xfmname, alpha=alpha)
    assert rgba.volume.shape == tuple([1] + list(volshape) + [4])

    data = dataset.Volume.random(subj, xfmname)
    assert data.raw.volume.shape == tuple([1] + list(volshape) + [4])
    data.raw.to_json()

    red, green, blue, alpha = [np.random.randn(nverts) for _ in range(4)]

    rgb = dataset.VertexRGB(red, green, blue, subj)
    assert rgb.vertices.shape == (1, nverts, 4)
    assert rgb.vertices.dtype == np.uint8
    assert rgb.vertices[..., 3].max() > 0

    rgba = dataset.VertexRGB(red, green, blue, subj, alpha=alpha)
    assert rgba.vertices.shape == (1, nverts, 4)

    data = dataset.Vertex.random(subj)
    assert data.raw.vertices.shape == (1, nverts, 4)
    data.raw.to_json()


def test_vertexrgb_shared_range():
    """VertexRGB should support shared_range like VolumeRGB."""
    red, green, blue = [np.random.randn(nverts) for _ in range(3)]
    rgb = dataset.VertexRGB(red, green, blue, subj, autorange='shared')
    assert rgb.vertices.shape == (1, nverts, 4)
    assert rgb.vertices.dtype == np.uint8

    # With explicit shared_vmin/shared_vmax
    rgb = dataset.VertexRGB(
        red, green, blue, subj, vmin=0, vmax=1
    )
    assert rgb.vertices.shape == (1, nverts, 4)
    assert rgb.vertices.dtype == np.uint8


def test_vertexrgb_custom_colors():
    """VertexRGB should support custom channel colors like VolumeRGB."""
    red, green, blue = [np.random.randn(nverts) for _ in range(3)]
    rgb = dataset.VertexRGB(
        red,
        green,
        blue,
        subj,
        channel1color=dataset.Colors.RoseRed,
        channel2color=dataset.Colors.LimeGreen,
        channel3color=dataset.Colors.SkyBlue,
    )
    assert rgb.vertices.shape == (1, nverts, 4)
    assert rgb.vertices.dtype == np.uint8


def test_rgb_rejects_unknown_kwargs():
    """VolumeRGB and VertexRGB should reject unknown keyword arguments."""
    red, green, blue = [np.random.randn(nverts) for _ in range(3)]
    with pytest.raises(TypeError):
        dataset.VertexRGB(red, green, blue, subj, bogus_kwarg=True)  # type: ignore

    red, green, blue = [np.random.randn(*volshape) for _ in range(3)]
    with pytest.raises(TypeError):
        dataset.VolumeRGB(red, green, blue, subj, xfmname, bogus_kwarg=True)  # type: ignore


def test_volumergb_shared_range():
    """VolumeRGB with shared_range should still work after refactor."""
    red, green, blue = [np.random.randn(*volshape) for _ in range(3)]
    rgb = dataset.VolumeRGB(
        red, green, blue, subj, xfmname, vmin=0, vmax=1
    )
    assert rgb.volume.shape == tuple([1] + list(volshape) + [4])
    assert rgb.volume.dtype == np.uint8


def test_volumergb_custom_colors():
    """VolumeRGB with custom colors should still work after refactor."""
    red, green, blue = [np.random.randn(*volshape) for _ in range(3)]
    rgb = dataset.VolumeRGB(
        red,
        green,
        blue,
        subj,
        xfmname,
        channel1color=dataset.Colors.RoseRed,
        channel2color=dataset.Colors.LimeGreen,
        channel3color=dataset.Colors.SkyBlue,
    )
    assert rgb.volume.shape == tuple([1] + list(volshape) + [4])
    assert rgb.volume.dtype == np.uint8


def test_2D():
    d1 = cortex.Volume.random(subj, xfmname)
    d2 = cortex.Volume.random(subj, xfmname).masked["thick"]
    twod = cortex.Volume2D(d1, d2)
    cortex.Volume2D(
        d1.data, d2.data, subject=subj, xfmname=xfmname, vmin=0, vmax=2, vmin2=1
    )
    twod.to_json()


def test_braindata_hash():
    d = cortex.Volume.random(subj, xfmname)
    hash(d)


def test_dataset_save():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    mrand = np.random.randn(2, *volshape)
    rand = np.random.randn(*volshape)
    ds = cortex.Dataset(test=(mrand, subj, xfmname))
    ds.append(twod=cortex.Volume2D(rand, rand, subj, xfmname))
    ds.append(rgb=cortex.VolumeRGB(rand, rand, rand, subj, xfmname))
    ds.append(vert=cortex.Vertex.random(subj))
    ds.save(tf.name)

    ds = cortex.load(tf.name)
    assert isinstance(ds.test, cortex.Volume)
    assert ds.test.data.shape == mrand.shape
    assert isinstance(ds.twod, cortex.Volume2D)
    assert ds.twod.dim1.data.shape == rand.shape
    assert ds.twod.dim2.data.shape == rand.shape
    assert ds.rgb.volume.shape == tuple([1] + list(volshape) + [4])
    assert isinstance(ds.vert, cortex.Vertex)


def test_mask_save():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds = cortex.Dataset(test=(np.random.randn(*volshape), subj, xfmname))
    ds.append(masked=ds.test.masked["thin"])
    data = ds.masked.data
    ds.save(tf.name)

    ds = cortex.load(tf.name)
    assert ds.masked.shape == volshape
    assert np.allclose(ds.masked.data, data)


def test_overwrite():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds = cortex.Dataset(test=(np.random.randn(*volshape), subj, xfmname))
    ds.save(tf.name)

    ds.save()
    assert ds.test.data.shape == volshape


def test_pack():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds = cortex.Dataset(test=(np.random.randn(*volshape), subj, xfmname))
    ds.save(tf.name, pack=True)

    ds = cortex.load(tf.name)
    pts, polys = cortex.db.get_surf(subj, "fiducial", "lh")
    dpts, dpolys = ds.get_surf(subj, "fiducial", "lh")
    assert np.allclose(pts, dpts)

    overlay_db = cortex.db.get_overlay(subj, None, modify_svg_file=False)
    rois_db = overlay_db.rois.labels.elements.keys()
    # keep the temporary file object in memory to avoid the file being deleted
    temp_file = ds.get_overlay(subj, "rois")
    overlay_ds = cortex.db.get_overlay(subj, temp_file.name, modify_svg_file=False)
    rois_ds = overlay_ds.rois.labels.elements.keys()
    assert rois_db == rois_ds

    xfm = cortex.db.get_xfm(subj, xfmname)
    assert np.allclose(xfm.xfm, ds.get_xfm(subj, xfmname).xfm)


def test_map():
    dv = cortex.Volume.random(subj, xfmname)
    dv.map("nearest")


def test_convertraw():
    ds = cortex.Dataset(test=(np.random.randn(*volshape), subj, xfmname))
    ds.test.raw


def test_vertexdata_copy():
    vd = cortex.Vertex(np.random.randn(nverts), subj)
    vdcopy = vd.copy(vd.data)
    assert np.allclose(vd.data, vdcopy.data)


def test_vertexdata_set():
    vd = cortex.Vertex(np.random.randn(nverts), subj)
    newdata = np.random.randn(nverts)
    vd.data = newdata
    assert np.allclose(newdata, vd.data)


def test_vertexdata_index():
    vd = cortex.Vertex(np.random.randn(10, nverts), subj)
    assert np.allclose(vd[0].data, vd.data[0])


def test_vertex_rgb_movie():
    r = g = b = np.random.randn(nverts)
    rgb = cortex.VertexRGB(r, g, b, subj)


def test_volumedata_copy():
    v = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    vc = v.copy(v.data)
    assert np.allclose(v.data, vc.data)


def test_volumedata_copy_with_custom_mask():
    mask = cortex.get_cortical_mask(subj, xfmname, "thick")
    mask[16] = True
    nmask = mask.sum()
    data = np.random.randn(nmask)
    v = cortex.Volume(data, subj, xfmname, mask=mask)
    vc = v.copy(v.data)
    assert np.allclose(v.data, vc.data)


@pytest.mark.skipif(no_inkscape, reason="Inkscape required")
def test_int64_in_dataviewrgb():
    data = np.arange(np.prod(volshape)).reshape(volshape, order="C")
    view = cortex.VolumeRGB(data, data + 1, data + 2, subject=subj, xfmname=xfmname)
    cortex.quickshow(view)

    data = np.arange(nverts)
    view = cortex.VertexRGB(data, data + 1, data + 2, subject=subj)
    cortex.quickshow(view)


@pytest.mark.skipif(no_inkscape, reason="Inkscape required")
def test_vmin_none_in_dataview2d():
    data = np.arange(np.prod(volshape)).reshape(volshape, order="C")
    view = cortex.Volume2D(data, data + 1, subject=subj, xfmname=xfmname)
    cortex.quickshow(view)

    data = np.arange(nverts)
    view = cortex.Vertex2D(data, data + 1, subject=subj)
    cortex.quickshow(view)


def test_dataset_operators():
    vol = cortex.Volume.random(subj, xfmname)
    array = np.random.randn(*volshape)

    assert np.allclose(vol.data + array, (vol + array).data)
    assert np.allclose(vol.data - array, (vol - array).data)
    assert np.allclose(vol.data * array, (vol * array).data)
    assert np.allclose(vol.data // array, (vol // array).data)  # floordiv
    assert np.allclose(vol.data / array, (vol / array).data)  # truediv
    # numpy doesn't like fractional powers of negative numbers
    assert np.allclose(vol.data**array, (vol**array).data, equal_nan=True)
    assert np.allclose(-vol.data, (-vol).data)
    assert np.allclose(abs(vol.data), abs(vol).data)


def test_blend_curvature():
    view = cortex.Vertex.empty(subj)
    alpha = np.linspace(0, 1, view.data.size).reshape(view.data.shape)

    # blend_curvature is deprecated; the warning should fire on every call.
    with pytest.warns(DeprecationWarning, match="blend_curvature is deprecated"):
        view_rgb: cortex.VertexRGB = view.blend_curvature(alpha)
    with pytest.warns(DeprecationWarning):
        view_rgb = view.blend_curvature(alpha > 0.3)
    # test that it returns a VertexRGB
    assert isinstance(view_rgb, cortex.VertexRGB)

    # test on Vertex2D
    view_2d = cortex.Vertex2D(view_rgb.red.data, view_rgb.green.data, subj)
    with pytest.warns(DeprecationWarning):
        view_rgb = view_2d.blend_curvature(alpha)

    # test on VertexRGB
    with pytest.warns(DeprecationWarning):
        view_rgb_new = view_rgb.blend_curvature(alpha)
    # test that it returns a different VertexRGB
    assert not np.allclose(view_rgb.red.data, view_rgb_new.red.data)
    # test that it returns a VertexRGB with same values when alpha is ones
    with pytest.warns(DeprecationWarning):
        view_rgb_new = view_rgb.blend_curvature(np.ones_like(alpha))
    assert np.allclose(view_rgb.red.data, view_rgb_new.red.data)


def test_get_cmapdict():
    red, green, blue = [np.random.randn(*volshape) for _ in range(3)]
    view = cortex.Volume2D(red, green, subject=subj, xfmname=xfmname)

    # test that it returns a dict with correct keys
    cmapdict = view.get_cmapdict()
    assert "cmap" in cmapdict and "vmin" in cmapdict and "vmax" in cmapdict

    # Calling it twice should not try to register the cmap twice to matplotlib
    view.get_cmapdict()

    # VolumeRGB should return an empty dict
    view = cortex.VolumeRGB(red, green, blue, subject=subj, xfmname=xfmname)
    cmapdict = view.get_cmapdict()
    assert "cmap" not in cmapdict


def test_warn_non_perceptually_uniform_2D_cmap():
    data0, data1 = [np.random.randn(*volshape) for _ in range(2)]
    view = cortex.Volume2D(
        data0, data1, subject=subj, xfmname=xfmname, cmap="RdBu_covar"
    )
    with pytest.warns(UserWarning):
        cortex.quickshow(view)


def test_nan_transparent_vertex_raw():
    """NaN values in Vertex.raw should have alpha=0 (transparent)."""
    data = np.random.randn(nverts)
    nan_indices = [0, 10, 100, nverts - 1]
    data[nan_indices] = np.nan

    vtx = dataset.Vertex(data, subj, vmin=-2, vmax=2, cmap="RdBu_r")
    raw = vtx.raw

    # Default alpha: NaN positions should have alpha=0
    vertices = raw.vertices  # (1, nverts, 4)
    for idx in nan_indices:
        assert vertices[0, idx, 3] == 0, (
            f"NaN vertex {idx} should have alpha=0, got {vertices[0, idx, 3]}"
        )

    # Non-NaN positions should have non-zero alpha
    non_nan_idx = 1
    assert not np.isnan(data[non_nan_idx])
    assert vertices[0, non_nan_idx, 3] > 0


def test_nan_transparent_vertex_raw_alpha_override():
    """NaN values should remain transparent even when user overrides alpha."""
    data = np.random.randn(nverts)
    nan_indices = [0, 10, 100, nverts - 1]
    data[nan_indices] = np.nan

    vtx = dataset.Vertex(data, subj, vmin=-2, vmax=2, cmap="RdBu_r")
    raw = vtx.raw

    # Override alpha with all-opaque values
    alpha = np.ones(nverts) * 0.8
    raw.alpha = alpha

    vertices = raw.vertices  # (1, nverts, 4)
    for idx in nan_indices:
        assert vertices[0, idx, 3] == 0, (
            f"NaN vertex {idx} should have alpha=0 after override, got {vertices[0, idx, 3]}"
        )

    # Non-NaN positions should reflect the user's alpha
    non_nan_idx = 1
    assert not np.isnan(data[non_nan_idx])
    assert vertices[0, non_nan_idx, 3] > 0


def test_nan_transparent_volume_raw():
    """NaN values in Volume.raw should have alpha=0 (transparent)."""
    data = np.random.randn(*volshape)
    data[0, 0, 0] = np.nan
    data[10, 50, 50] = np.nan

    vol = dataset.Volume(data, subj, xfmname, vmin=-2, vmax=2, cmap="RdBu_r")
    raw = vol.raw

    # Default alpha: NaN positions should have alpha=0
    volume = raw.volume  # (1, z, y, x, 4)
    assert volume[0, 0, 0, 0, 3] == 0
    assert volume[0, 10, 50, 50, 3] == 0

    # Non-NaN positions should have non-zero alpha
    assert not np.isnan(data[15, 50, 50])
    assert volume[0, 15, 50, 50, 3] > 0


def test_nan_transparent_volume_raw_alpha_override():
    """NaN values should remain transparent even when user overrides alpha."""
    data = np.random.randn(*volshape)
    data[0, 0, 0] = np.nan
    data[10, 50, 50] = np.nan

    vol = dataset.Volume(data, subj, xfmname, vmin=-2, vmax=2, cmap="RdBu_r")
    raw = vol.raw

    # Override alpha with all-opaque values
    alpha = np.ones(volshape) * 0.8
    raw.alpha = alpha

    volume = raw.volume  # (1, z, y, x, 4)
    assert volume[0, 0, 0, 0, 3] == 0, (
        f"NaN voxel should have alpha=0 after override, got {volume[0, 0, 0, 0, 3]}"
    )
    assert volume[0, 10, 50, 50, 3] == 0

    # Non-NaN positions should reflect the user's alpha
    assert not np.isnan(data[15, 50, 50])
    assert volume[0, 15, 50, 50, 3] > 0


@pytest.mark.parametrize(
    "submodule",
    [
        "cortex.dataset._hdf",
        "cortex.dataset._space",
        "cortex.dataset.braindata",
        "cortex.dataset.dataset",
        "cortex.dataset.view2D",
        "cortex.dataset.viewRGB",
        "cortex.dataset.views",
    ],
)
def test_submodule_can_be_imported_first(submodule):
    """Importing any submodule before the package must not hit a partial module.

    ``views`` breaks its circular dependency on ``viewRGB``/``view2D`` with
    deferred imports at the bottom of its own module, so import order inside
    ``cortex/dataset/__init__.py`` is load-bearing: anything reaching
    ``view2D``/``viewRGB`` before ``views`` has finished sees a partially
    initialised module. Cheap to get wrong, and invisible to a type checker.
    """
    code = "import %s; import cortex.dataset" % submodule
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        "importing %s first broke the package:\n%s" % (submodule, result.stderr)
    )


def test_alias_isinstance_semantics_are_unchanged():
    """The pre-restructure names must narrow exactly as they used to.

    ``BrainData``/``VolumeData``/``VertexData`` are now aliases for
    ``ScalarView``/``Volume``/``Vertex``. That is only safe because
    ``isinstance(x, VolumeData)`` was never true for ``Volume2D`` or
    ``VolumeRGB`` -- they did not inherit it.
    """
    from cortex.dataset.braindata import BrainData, VertexData, VolumeData

    vol = cortex.Volume(np.zeros(volshape), subj, xfmname)
    vtx = cortex.Vertex(np.zeros(nverts), subj)
    v2d = cortex.Volume2D(np.zeros(volshape), np.ones(volshape), subj, xfmname)
    vrgb = cortex.VolumeRGB(
        np.zeros(volshape), np.zeros(volshape), np.zeros(volshape), subj, xfmname
    )

    assert isinstance(vol, VolumeData) and isinstance(vol, BrainData)
    assert isinstance(vtx, VertexData) and isinstance(vtx, BrainData)
    assert not isinstance(vol, VertexData)
    # the crux: the composites are not scalar data
    assert not isinstance(v2d, VolumeData)
    assert not isinstance(vrgb, VolumeData)
    assert not isinstance(v2d, BrainData)
    assert not isinstance(vrgb, BrainData)
    # and the composite bases still narrow
    assert isinstance(v2d, dataset.Dataview2D)
    assert isinstance(vrgb, dataset.DataviewRGB)
    assert isinstance(vol, dataset.Dataview)


def test_alpha_read_does_not_mutate_the_caller():
    """Reading ``.alpha`` used to write into a caller-supplied Volume/Vertex."""
    alpha = cortex.Volume(np.ones(volshape), subj, xfmname, vmin=0, vmax=1)
    before = alpha.data.copy()
    view = cortex.VolumeRGB(
        np.zeros(volshape),
        np.zeros(volshape),
        np.zeros(volshape),
        subj,
        xfmname,
        alpha=alpha,
    )
    for _ in range(3):
        view.alpha
    view.volume
    assert np.array_equal(alpha.data, before)
    # and it is memoized, so repeated reads are the same object
    assert view.alpha is view.alpha
    # ...but the setter invalidates the memo
    first = view.alpha
    view.alpha = np.ones(volshape)
    assert view.alpha is not first


def test_composite_views_have_a_working_copy():
    """The 2D and RGB families had no working ``copy()`` at all.

    ``Dataview.copy`` splatted ``cmap=``/``vmin=``/``vmax=`` into
    ``self.__class__(...)``, which their constructors do not accept.
    """
    v2d = cortex.Volume2D(np.zeros(volshape), np.ones(volshape), subj, xfmname)
    vrgb = cortex.VolumeRGB(
        np.zeros(volshape), np.zeros(volshape), np.zeros(volshape), subj, xfmname
    )
    assert isinstance(v2d.copy(), cortex.Volume2D)
    assert isinstance(vrgb.copy(), cortex.VolumeRGB)


def test_vertex2d_survives_an_hdf_round_trip():
    """Vertex2D used to be silently dropped by Dataset.from_file.

    ``_from_hdf_view`` indexed ``xfmname[0]`` unconditionally, but slot 7 of the
    view record is null for surface data, and ``from_file`` swallows exceptions
    via ``traceback.print_exc``.
    """
    import tempfile as _tempfile

    ds = cortex.Dataset(
        v2d=cortex.Vertex2D(np.zeros(nverts), np.ones(nverts), subj),
        vol2d=cortex.Volume2D(np.zeros(volshape), np.ones(volshape), subj, xfmname),
    )
    with _tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "roundtrip.hdf")
        ds.save(path)
        ds.h5.close()
        loaded = cortex.Dataset.from_file(path)
        try:
            assert set(loaded.views) == {"v2d", "vol2d"}
            assert isinstance(loaded.views["v2d"], cortex.Vertex2D)
            assert isinstance(loaded.views["vol2d"], cortex.Volume2D)
        finally:
            loaded.h5.close()


def test_each_view_subclasses_exactly_one_spatial_interface():
    """The renderers fork on volumetric-vs-surface; the fork must be total.

    ``Volume2D`` used to satisfy neither, because it had no ``.volume`` -- every
    consumer special-cased it and reached for ``.raw.volume`` itself, while
    ``Vertex2D`` already had the symmetric delegating property.
    """
    from cortex.dataset.views import SurfaceView, VolumetricView

    ones_v, ones_x = np.ones(volshape), np.ones(nverts)
    views = {
        "Volume": cortex.Volume(ones_v, subj, xfmname),
        "Vertex": cortex.Vertex(ones_x, subj),
        "Volume2D": cortex.Volume2D(ones_v, ones_v * 2, subj, xfmname),
        "Vertex2D": cortex.Vertex2D(ones_x, ones_x * 2, subj),
        "VolumeRGB": cortex.VolumeRGB(ones_v, ones_v, ones_v, subj, xfmname),
        "VertexRGB": cortex.VertexRGB(ones_x, ones_x, ones_x, subj),
    }
    for name, view in views.items():
        volumetric = isinstance(view, VolumetricView)
        surface = isinstance(view, SurfaceView)
        assert volumetric != surface, (
            "%s subclasses %s spatial interface"
            % (name, "both" if volumetric else "neither")
        )
        assert volumetric is name.startswith("Volume"), name
        # and the interface each claims actually works
        if volumetric:
            assert view.volume.ndim >= 4, name
            assert isinstance(view.xfmname, str), name
        else:
            assert view.vertices.ndim >= 2, name


def test_colormapped_matches_the_hasattr_it_replaces():
    from cortex.dataset.view2D import Dataview2D
    from cortex.dataset.views import ScalarView

    COLORMAPPED = (ScalarView, Dataview2D)

    ones_v, ones_x = np.ones(volshape), np.ones(nverts)
    for name, view in {
        "Volume": cortex.Volume(ones_v, subj, xfmname),
        "Vertex": cortex.Vertex(ones_x, subj),
        "Volume2D": cortex.Volume2D(ones_v, ones_v * 2, subj, xfmname),
        "Vertex2D": cortex.Vertex2D(ones_x, ones_x * 2, subj),
        "VolumeRGB": cortex.VolumeRGB(ones_v, ones_v, ones_v, subj, xfmname),
        "VertexRGB": cortex.VertexRGB(ones_x, ones_x, ones_x, subj),
    }.items():
        # The RGB views have no cmap at all; the 2D ones do, which is why a
        # scalar-only test would not be equivalent to the old hasattr.
        assert isinstance(view, COLORMAPPED) == hasattr(view, "cmap"), name
        assert isinstance(view, COLORMAPPED) is not name.endswith("RGB"), name


def test_volume2d_volume_mirrors_vertex2d_vertices():
    """The property added to make the volumetric/surface fork total."""
    v2d = cortex.Volume2D(np.zeros(volshape), np.ones(volshape), subj, xfmname)
    x2d = cortex.Vertex2D(np.zeros(nverts), np.ones(nverts), subj)
    assert np.array_equal(v2d.volume, v2d.raw.volume)
    assert np.array_equal(x2d.vertices, x2d.raw.vertices)
    # uint8 RGBA, like VolumeRGB.volume rather than Volume.volume
    assert v2d.volume.dtype == np.uint8
    assert v2d.volume.shape[-1] == 4


def test_as_renderable_rejects_a_view_with_neither_interface():
    """``as_renderable`` is the one boundary; it must reject, not defer.

    Unlike an enumeration of known classes it accepts a view in *any* space, so
    long as it exposes the interface.
    """
    from cortex.dataset.views import as_renderable

    vol = cortex.Volume(np.ones(volshape), subj, xfmname)
    assert as_renderable(vol) is vol

    class Unrenderable(dataset.Dataview):
        """A view exposing neither a volume nor vertices."""

        @property
        def space(self):
            raise NotImplementedError

        @property
        def raw(self):
            raise NotImplementedError

        def uniques(self, collapse=False):
            raise NotImplementedError

        def _write_hdf(self, h5, name="data"):
            raise NotImplementedError

    with pytest.raises(TypeError, match="is not renderable"):
        as_renderable(Unrenderable())


def test_as_renderable_accepts_a_third_party_view_that_inherits_a_spatial_interface():
    """Still open, but by explicit opt-in rather than structurally.

    A view in a space this package has never heard of is renderable as soon as it
    inherits whichever spatial interface describes how its data is sampled -- no entry
    and no union to edit.

    Note what it has to implement: ``spatial_data`` and nothing else about its
    array. ``vertices`` is the surface interface's name for that same array and
    arrives concrete, so a third-party view does not publish it twice.
    """
    from cortex.dataset.views import as_renderable
    from cortex.dataset.views import SurfaceView

    class ThirdPartyView(SurfaceView):
        @property
        def space(self):
            raise NotImplementedError

        @property
        def subject(self):
            return subj

        _array = np.zeros((1, 10))

        @property
        def spatial_data(self):
            return self._array

        @property
        def raw(self):
            raise NotImplementedError

        def uniques(self, collapse=False):
            raise NotImplementedError

        def _write_hdf(self, h5, name="data"):
            raise NotImplementedError

    view = ThirdPartyView()
    assert as_renderable(view) is view
    assert isinstance(view, SurfaceView)
    assert view.vertices is view.spatial_data


def test_as_renderable_rejects_a_lookalike_that_merely_has_the_attributes():
    """The soundness a runtime_checkable Protocol could not provide.

    A Protocol's isinstance tests only for the *presence* of the member names, so
    this class would satisfy one. A nominal base class rejects it.
    """
    from cortex.dataset.views import as_renderable
    from cortex.dataset.views import SurfaceView, VolumetricView

    class Lookalike(dataset.Dataview):
        """Every name a spatial interface wants, none of the meanings."""

        subject = 42
        xfmname = None
        volume = "not an array"
        vertices = "not an array either"

        @property
        def space(self):
            raise NotImplementedError

        @property
        def raw(self):
            raise NotImplementedError

        def uniques(self, collapse=False):
            raise NotImplementedError

        def _write_hdf(self, h5, name="data"):
            raise NotImplementedError

    fake = Lookalike()
    assert not isinstance(fake, VolumetricView)
    assert not isinstance(fake, SurfaceView)
    with pytest.raises(TypeError, match="is not renderable"):
        as_renderable(fake)


def test_which_arrays_a_2d_view_colormaps_is_the_spaces_call():
    """``BrainSpace.align``, not fifteen lines of mask logic inside Volume2D.raw.

    Two arrays flattened under the same mask already line up and are far smaller,
    so they can be colormapped as they are stored; under different masks -- or with
    one flattened and one not -- position *i* means different voxels in each, and
    only the unmasked volumes are comparable. That is volumetric knowledge, and
    ``Vertex2D.raw`` existed as a two-line copy of the same method purely because
    it had none of it.
    """
    from cortex.dataset.view2D import Dataview2D, Vertex2D, Volume2D
    from cortex.dataset.views import Vertex, Volume

    # one implementation, on the column; the subclasses only narrow the type
    assert "raw" in vars(Dataview2D)
    for cls in (Volume2D, Vertex2D):
        assert cls.raw.fget.__doc__ and "arrowing" in cls.raw.fget.__doc__, cls

    m1 = db.get_mask(subj, xfmname, "thick")
    m2 = db.get_mask(subj, xfmname, "thin")
    masked = Volume(np.random.randn(m1.sum()), subj, xfmname, mask=m1)
    masked_b = Volume(np.random.randn(m1.sum()), subj, xfmname, mask=m1)
    other_mask = Volume(np.random.randn(m2.sum()), subj, xfmname, mask=m2)
    full = Volume(np.random.randn(*volshape), subj, xfmname)

    # same mask: the stored arrays, which are flat and much smaller
    first, second = Volume2D(masked, masked_b).space.align(masked, masked_b)
    assert first.shape == second.shape == masked.data.shape
    assert np.array_equal(first, masked.data)

    # anything else volumetric: the unmasked volumes
    for a, b in ((masked, other_mask), (masked, full), (full, full)):
        first, second = Volume2D(a, b).space.align(a, b)
        assert first.shape == second.shape == (1,) + volshape

    # a surface space has no masks, so it always aligns the stored arrays
    vtx = Vertex(np.random.randn(nverts), subj)
    first, second = Vertex2D(vtx, vtx).space.align(vtx, vtx)
    assert first.shape == second.shape == (nverts,)

    # and views the space cannot align at all say so, rather than colormapping
    # two arrays that do not correspond
    retino = Volume(
        np.random.randn(*db.get_xfm(subj, "retinotopy").shape), subj, "retinotopy"
    )
    with pytest.raises(ValueError, match="same xfmname"):
        Volume2D(full, retino).raw


def _third_space_family():
    """Build a third space's three view classes exactly as INHERITANCE.md prescribes.

    Returns ``(MySpace, MyView, MyView2D, MyViewRGB, nthings)``. Kept as a helper so
    the skeleton in the doc has exactly one executable counterpart; if the skeleton
    changes, this changes with it.
    """
    from cortex.dataset._space import BrainSpace, SpaceViews
    from cortex.dataset.view2D import Dataview2D, _resolve_2d_channels
    from cortex.dataset.viewRGB import Colors, DataviewRGB, _resolve_rgb_channels
    from cortex.dataset.views import RenderableView, ScalarView

    nthings = 10

    class MySpatial(RenderableView):
        """A third spatial interface: sampled by index, through no transform."""

    class MySpace(BrainSpace):
        hdf_key = "myspace"
        spec_keys = ("myarg",)

        def __init__(self, subject, myarg="a"):
            super().__init__(subject)
            self._myarg = myarg

        @property
        def myarg(self):
            return self._myarg

        @property
        def xfmname(self):
            return None

        def coerce(self, data):
            if data is None:
                data = np.zeros((nthings,))
            if data.shape[-1] != nthings:
                raise ValueError("bad length")
            return data

        def is_movie(self, data):
            return data.ndim > 1

        @property
        def spatial_shape(self):
            return (nthings,)

        def wrap(self, data, **kw):
            return MyView(data, self.subject, self.myarg, **kw)

        def wrap_rgb(self, r, g, b, a=None, **kw):
            return MyViewRGB(r, g, b, self.subject, self.myarg, a, **kw)

        def to_json(self):
            return {}

        def write_hdf_attrs(self, h5, node):
            node.attrs["myarg"] = self.myarg

        @classmethod
        def from_hdf(cls, attrs, *, subject, xfmname, mask):
            if "myarg" not in attrs:
                return None
            return cls(subject, attrs["myarg"])

        @classmethod
        def views(cls):
            return SpaceViews(scalar=MyView, twod=MyView2D, rgb=MyViewRGB)

    class MyView(ScalarView, MySpatial):
        """A scalar view in MySpace."""

        def __init__(self, data, subject, myarg, cmap=None, vmin=None, vmax=None,
                     description="", state=None, **kwargs):
            super().__init__(data, MySpace(subject, myarg), cmap=cmap, vmin=vmin,
                             vmax=vmax, description=description, state=state,
                             **kwargs)
            self._resolve_percentiles()

        # The doc's skeleton also carries a bare `_space: MySpace` annotation
        # here. Omitted: it is a static-only device with no runtime effect, and
        # inside an untyped test helper mypy would only emit an
        # annotation-unchecked note for it.
        @property
        def space(self):
            return self._space

        @property
        def spatial_data(self):
            return self.data if self.movie else self.data[np.newaxis]

        @property
        def raw(self):
            return self._build_raw()

        @classmethod
        def empty(cls, subject, myarg, value=0, **kwargs):
            shape = MySpace(subject, myarg).template_shape
            return cls(cls._sample(shape, value), subject, myarg, **kwargs)

        @classmethod
        def random(cls, subject, myarg, **kwargs):
            shape = MySpace(subject, myarg).template_shape
            return cls(cls._sample(shape, None), subject, myarg, **kwargs)

        def __repr__(self):
            return "<my data for (%s)>" % self.subject

        @property
        def nthings(self):
            return self.space.spatial_shape[0]

    class MyView2D(Dataview2D[MyView], MySpatial):
        """Two MyViews, jointly colormapped."""

        def __init__(self, dim1, dim2, subject=None, myarg=None, description="",
                     cmap=None, vmin=None, vmax=None, vmin2=None, vmax2=None,
                     **kwargs):
            chan1, chan2 = _resolve_2d_channels(
                dim1, dim2, channel_cls=MyView, space_cls=MySpace, subject=subject,
                spec={"myarg": myarg}, ranges=((vmin, vmax), (vmin2, vmax2)))
            super().__init__(chan1, chan2, description=description, cmap=cmap,
                             vmin=vmin, vmax=vmax, vmin2=vmin2, vmax2=vmax2,
                             **kwargs)

        @property
        def raw(self):
            return super().raw

        def __repr__(self):
            return "<2D my data for (%s)>" % self.dim1.subject

    class MyViewRGB(DataviewRGB[MyView], MySpatial):
        """Three MyView channels plus alpha."""

        def __init__(self, channel1, channel2, channel3, subject=None, myarg=None,
                     alpha=None, description="", state=None,
                     channel1color=Colors.Red, channel2color=Colors.Green,
                     channel3color=Colors.Blue, max_color_value=None,
                     max_color_saturation=1.0, vmin=None, vmax=None,
                     autorange="individual", priority=1):
            red, green, blue, resolved_alpha = _resolve_rgb_channels(
                (channel1, channel2, channel3), channel_cls=MyView,
                space_cls=MySpace, subject=subject, spec={"myarg": myarg},
                colors=(channel1color, channel2color, channel3color),
                max_color_value=max_color_value,
                max_color_saturation=max_color_saturation,
                vmin=vmin, vmax=vmax, autorange=autorange, alpha=alpha)
            super().__init__(red, green, blue, alpha=resolved_alpha, subject=subject,
                             description=description, state=state, priority=priority)

        @property
        def space(self):
            return self.red.space

        def __repr__(self):
            return "<RGB my data for (%s)>" % self.subject

    return MySpace, MyView, MyView2D, MyViewRGB, nthings


def test_the_documented_skeleton_for_a_new_space_actually_works():
    """Every claim the INHERITANCE.md skeleton makes, exercised.

    The doc tells a reader to copy three class skeletons and fill them in. This is
    that, filled in against a synthetic ten-element space, so the doc cannot rot
    into describing an extension point that no longer exists.
    """
    from cortex.dataset.views import as_renderable

    MySpace, MyView, MyView2D, MyViewRGB, nthings = _third_space_family()
    arr = np.random.randn(nthings)

    # --- the scalar view
    view = MyView(arr, subj, "a")
    assert repr(view) == "<my data for (S1)>"
    assert view.subject == subj and view.nthings == nthings
    assert view.space.xfmname is None
    assert view.spatial_data.shape == (1, nthings)          # frame axis added
    assert MyView(np.random.randn(3, nthings), subj, "a").spatial_data.shape == (3, nthings)
    assert view.name.startswith("__") and len(view.name) == 18
    assert np.allclose((view + 1).data, arr + 1)            # inherited operators
    assert view.vmin is not None and view.vmax is not None  # _resolve_percentiles

    # --- empty / random, which read template_shape off the space
    assert np.all(MyView.empty(subj, "a", 2).data == 2)
    assert MyView.random(subj, "a").data.shape == (nthings,)

    # --- raw, which goes out through space.wrap_rgb and comes back concrete
    raw = view.raw
    assert isinstance(raw, MyViewRGB)
    assert raw.spatial_data.shape == (1, nthings, 4)
    assert raw.spatial_data.dtype == np.uint8

    # --- the 2D view, from raw arrays and from built views
    twod = MyView2D(arr, arr, subj, "a")
    assert repr(twod) == "<2D my data for (S1)>"
    assert twod.spatial_data.shape == (1, nthings, 4)
    assert isinstance(twod.raw, MyViewRGB)
    assert isinstance(MyView2D(view, view).dim1, MyView)
    # the shared resolver enforces the space's spec keys
    with pytest.raises(TypeError, match="myarg"):
        MyView2D(arr, arr, subj)

    # --- the RGB view, single frame and movie
    rgb = MyViewRGB(arr, arr, arr, subj, "a")
    assert repr(rgb) == "<RGB my data for (S1)>"
    assert rgb.spatial_data.shape == (1, nthings, 4)
    movie = MyViewRGB(*([np.random.randn(4, nthings)] * 3), subj, "a")
    assert movie.spatial_data.shape == (4, nthings, 4)

    # --- serialization, with the space contributing no layout keys of its own
    assert set(view.to_json(simple=True)) == {"name", "subject", "min", "max"}
    assert "data" in view.to_json(simple=False)
    assert [type(u).__name__ for u in twod.uniques()] == ["MyView", "MyView"]
    assert [type(u).__name__ for u in rgb.uniques(collapse=True)] == ["MyViewRGB"]

    # --- and it is renderable without being either built-in spatial kind
    from cortex.dataset.views import SurfaceView, VolumetricView

    assert as_renderable(view) is view
    assert not isinstance(view, (VolumetricView, SurfaceView))


def test_a_third_space_is_registered_behind_the_catch_all():
    """Known bug, pinned so the fix has something to flip.

    ``register_space`` appends, and ``SurfaceSpace.from_hdf`` accepts any node
    without a transform, so a space registered by a third party is never reached.
    See "Known bug: a third space cannot be read back from HDF" in INHERITANCE.md.
    """
    from cortex.dataset import _space as space_mod

    order = [c.__name__ for c in space_mod.registered_spaces()]
    assert order == ["VolumeSpace", "SurfaceSpace"]

    MySpace = _third_space_family()[0]
    saved = list(space_mod._SPACES)
    try:
        space_mod.register_space(MySpace)
        after = [c.__name__ for c in space_mod.registered_spaces()]
        # the defect: the catch-all still precedes the new space
        assert after.index("SurfaceSpace") < after.index("MySpace")
        # and the catch-all really does claim a node the new space wrote
        claimed = None
        for cls in space_mod.registered_spaces():
            claimed = cls.from_hdf({"myarg": "a"}, subject=subj, xfmname=None, mask=None)
            if claimed is not None:
                break
        assert type(claimed).__name__ == "SurfaceSpace"
    finally:
        space_mod._SPACES[:] = saved


def test_a_space_declares_what_identifies_it():
    """``spec_keys`` plus ``from_spec``, rather than a lambda per constructor.

    Volume2D, Vertex2D, VolumeRGB and VertexRGB each used to be handed a lambda
    naming their space class with ``_require`` applied to each of its arguments,
    *and* a dict of those same keys to validate channel objects against. "A
    volumetric space is subject plus xfmname, and both are mandatory" was written
    into four view constructors instead of into VolumeSpace.
    """
    from cortex.dataset._space import BrainSpace, SurfaceSpace, VolumeSpace

    assert BrainSpace.spec_keys == ()
    assert VolumeSpace.spec_keys == ("xfmname",)
    assert SurfaceSpace.spec_keys == ()

    space = VolumeSpace.from_spec(subj, xfmname=xfmname)
    assert isinstance(space, VolumeSpace)
    assert space.subject == subj and space.xfmname == xfmname
    assert isinstance(SurfaceSpace.from_spec(subj), SurfaceSpace)

    # every spec key is mandatory, and so is the subject
    with pytest.raises(TypeError, match="xfmname"):
        VolumeSpace.from_spec(subj, xfmname=None)
    with pytest.raises(TypeError, match="Subject"):
        VolumeSpace.from_spec(None, xfmname=xfmname)
    with pytest.raises(TypeError, match="Subject"):
        SurfaceSpace.from_spec(None)


def test_both_composite_columns_share_one_channel_resolver():
    """The same four checks, in the same order, for 2D and RGB alike.

    They were written out twice with different wording. The messages now name the
    offending argument, which is the only reason the resolver is told what its
    caller calls them.
    """
    from cortex.dataset._space import VolumeSpace
    from cortex.dataset.views import Volume, _resolve_channels

    vol = Volume(np.random.randn(*volshape), subj, xfmname)
    common = dict(channel_cls=Volume, space_cls=VolumeSpace, spec={"xfmname": xfmname})

    # views in: the space comes from the first channel, and they are handed back
    space, views = _resolve_channels(
        [vol, vol], subject=None, argnames=("dim1", "dim2"), **common
    )
    assert space is vol.space and views == [vol, vol]

    # arrays in: no views, and the space is built from the spec
    arr = np.random.randn(*volshape)
    space, views = _resolve_channels(
        [arr, arr, arr],
        subject=subj,
        argnames=("channel1", "channel2", "channel3"),
        **common,
    )
    assert views is None
    assert isinstance(space, VolumeSpace) and space.xfmname == xfmname

    # mixing is rejected in both directions, and says which argument
    for channels, offender in (([vol, arr], "dim2"), ([arr, vol], "dim2")):
        with pytest.raises(TypeError, match=offender):
            _resolve_channels(
                channels, subject=None, argnames=("dim1", "dim2"), **common
            )


def test_xfmname_comes_from_the_space_not_from_a_channel():
    """One implementation of ``xfmname`` for every volumetric view.

    It was abstract and answered three times: ``self.space.xfmname`` on Volume,
    ``self.dim1.xfmname`` on Volume2D, ``self.red.xfmname`` on VolumeRGB -- the
    last two reaching through a channel for a value the channel reads off the
    space they all share.
    """
    from cortex.dataset.view2D import Volume2D
    from cortex.dataset.viewRGB import VolumeRGB
    from cortex.dataset.views import Volume, VolumetricView

    assert "xfmname" not in VolumetricView.__abstractmethods__
    for cls in (Volume, Volume2D, VolumeRGB):
        assert "xfmname" not in vars(cls), cls

    vol = Volume(np.random.randn(*volshape), subj, xfmname)
    for view in (vol, Volume2D(vol, vol), VolumeRGB(vol, vol, vol)):
        assert view.xfmname == view.space.xfmname == xfmname, type(view)


def test_the_spatial_array_is_implemented_once_per_view():
    """``spatial_data`` is the abstract member; ``volume``/``vertices`` are aliases.

    It used to be the other way round, which cost one property per space per
    column: ``VolumeRGB.volume`` and ``VertexRGB.vertices`` were the same
    ``_rgba_stack()`` call under two names, as were ``Volume2D.volume`` and
    ``Vertex2D.vertices`` over ``raw``. Guard the direction, because re-abstracting
    ``volume``/``vertices`` would silently reintroduce all four.
    """
    from cortex.dataset.view2D import Dataview2D
    from cortex.dataset.viewRGB import DataviewRGB
    from cortex.dataset.views import (
        RenderableView,
        SurfaceView,
        Volume,
        VolumetricView,
    )

    assert "spatial_data" in RenderableView.__abstractmethods__
    for iface, alias in ((VolumetricView, "volume"), (SurfaceView, "vertices")):
        assert alias in vars(iface), alias
        assert alias not in iface.__abstractmethods__, alias

    # each column class implements it exactly once, on the column and not on
    # either of its two spatial subclasses
    for column in (DataviewRGB, Dataview2D):
        assert "spatial_data" in vars(column), column
        for sub in column.__subclasses__():
            assert "spatial_data" not in vars(sub), sub

    # and the scalar column, whose two views differ in how they do it, is where
    # the per-space implementations legitimately live
    assert "spatial_data" in vars(Volume)


def test_a_spatial_subclass_must_implement_its_interface():
    """Abstract members mean a forgotten accessor fails at construction."""
    from cortex.dataset.views import VolumetricView

    class Incomplete(VolumetricView):
        @property
        def space(self):
            raise NotImplementedError

        @property
        def subject(self):
            return subj

        @property
        def raw(self):
            raise NotImplementedError

        def uniques(self, collapse=False):
            raise NotImplementedError

        def _write_hdf(self, h5, name="data"):
            raise NotImplementedError

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()


def test_static_only_protocols_refuse_isinstance():
    """The presence-only check cannot creep back in.

    The spatial interfaces answer "what kind of view is this?" at runtime, so they
    sound. The Protocols answer "what does this function need?", which is a purely
    static question -- so they are deliberately not @runtime_checkable, and
    isinstance against them is a TypeError rather than a hasattr sweep.
    """
    from typing import Any

    from cortex.dataset.views import HasSubject

    vol = cortex.Volume(np.ones(volshape), subj, xfmname)
    # The guard has two halves. Statically, mypy rejects `isinstance(vol,
    # HasSubject)` outright ("Only @runtime_checkable protocols can be used with
    # instance and class checks") -- so this goes through a variable to get past
    # the checker and exercise the *runtime* half.
    proto: Any = HasSubject
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(vol, proto)


def test_static_only_protocols_are_still_satisfied_structurally():
    """They are real contracts, just checked statically. Verify the shape holds."""
    from cortex.dataset.views import HasSubject

    vol = cortex.Volume(np.ones(volshape), subj, xfmname)
    # what HasSubject promises, on every view kind
    for view in (
        vol,
        cortex.Vertex(np.ones(nverts), subj),
        cortex.Volume2D(np.ones(volshape), np.ones(volshape) * 2, subj, xfmname),
        cortex.VertexRGB(np.ones(nverts), np.ones(nverts), np.ones(nverts), subj),
    ):
        assert isinstance(view.subject, str)

    # and the composite helpers now ask for nothing more than that
    import inspect

    from cortex.quickflat import composite

    for fn in (composite.add_curvature, composite.add_rois,
               composite.add_sulci, composite.add_custom):
        annot = inspect.signature(fn).parameters["dataview"].annotation
        assert "HasSubject" in str(annot), (fn.__name__, annot)


def test_protocol_implementations_are_declared_not_merely_structural():
    """Every protocol a view satisfies is claimed explicitly in its bases.

    Explicit inheritance of a Protocol makes the claim visible on the class *and*
    machine-checked -- a subclass that failed to provide a member would be
    abstract and uninstantiable. It does not re-enable ``isinstance``, which still
    requires ``@runtime_checkable``; that is checked separately.
    """
    from cortex.dataset.views import (
        Dataview,
        HasSubject,
        SurfaceView,
        VolumetricView,
    )

    assert HasSubject in Dataview.__bases__

    # ...and the HasSubject reachable from every export path is that same class,
    # not a same-shaped copy. Two structurally identical Protocols type-check
    # interchangeably, so a duplicate declaration is invisible to mypy and to the
    # assertion above -- `cortex.dataset.HasSubject` was once a second class that
    # no view actually inherited. Only object identity catches that.
    assert cortex.dataset.HasSubject is HasSubject

    # blend_curvature is defined once, on the spatial interface, not per class
    assert "blend_curvature" in SurfaceView.__dict__
    assert not any(
        "blend_curvature" in getattr(cortex, n).__dict__
        for n in ("Vertex", "Vertex2D", "VertexRGB")
    )

    expected = {
        "Volume": (VolumetricView, False),
        "Volume2D": (VolumetricView, False),
        "VolumeRGB": (VolumetricView, False),
        "Vertex": (SurfaceView, True),
        "Vertex2D": (SurfaceView, True),
        "VertexRGB": (SurfaceView, True),
    }
    for name, (spatial, blends) in expected.items():
        cls = getattr(cortex, name)
        mro = cls.__mro__
        # every view claims HasSubject, via Dataview
        assert HasSubject in mro, name
        # exactly one spatial interface
        assert spatial in mro, name
        other = SurfaceView if spatial is VolumetricView else VolumetricView
        assert other not in mro, name
        # curvature blending is a surface-only contract, inherited from the interface
        assert hasattr(cls, "blend_curvature") is blends, name


def test_a_third_spatial_kind_needs_no_change_to_the_renderer():
    """Nothing branches on which spatial kind a view is, so a new one just works.

    ``make_flatmap_image`` used to fork on ``isinstance(braindata, SurfaceView)``
    with an ``else`` that silently assumed exactly two kinds existed -- a third
    would have taken the volumetric path and then failed on ``.xfmname``. It now
    reads ``space.xfmname`` (what to sample through) and ``spatial_data`` (what to
    sample), so a spatial kind it has never heard of is drawn by the same code.
    """
    import inspect

    from cortex.dataset._space import BrainSpace
    from cortex.dataset.views import as_renderable
    from cortex.dataset.views import RenderableView, SurfaceView, VolumetricView
    from cortex.quickflat import utils as qutils

    class ThirdSpace(BrainSpace):
        """A space that is neither volumetric nor surface-shaped."""

        hdf_key = "third"

        @property
        def xfmname(self):
            return None          # sampled without a transform

        def coerce(self, data):
            return np.zeros((1, 4)) if data is None else data

        def is_movie(self, data):
            return data.ndim > 1

        @property
        def spatial_shape(self):
            return (4,)

        def wrap(self, data, **kwargs):
            raise NotImplementedError

        def wrap_rgb(self, red, green, blue, alpha=None, **kwargs):
            raise NotImplementedError

        def to_json(self):
            return {}

        def write_hdf_attrs(self, h5, node):
            return None

        @classmethod
        def from_hdf(cls, attrs, *, subject, xfmname, mask):
            return None

        @classmethod
        def views(cls):
            raise NotImplementedError

    class ThirdSpatialKind(RenderableView):
        """A spatial kind this package has never heard of."""

        @property
        def space(self):
            return self._space

        def __init__(self):
            self._space = ThirdSpace(subj)

        @property
        def spatial_data(self):
            return np.zeros((1, 4))

        @property
        def raw(self):
            raise NotImplementedError

        def uniques(self, collapse=False):
            raise NotImplementedError

        def _write_hdf(self, h5, name="data"):
            raise NotImplementedError

    view = ThirdSpatialKind()
    # it is renderable without being either built-in spatial interface
    assert as_renderable(view) is view
    assert not isinstance(view, (VolumetricView, SurfaceView))
    # and the two facts the renderer needs are both available
    assert view.space.xfmname is None
    assert view.spatial_data.shape == (1, 4)
    # the renderer no longer asks which spatial kind it holds. (It still has
    # isinstance checks for np.ma.MaskedArray -- about the array, not the space.)
    src = inspect.getsource(qutils.make_flatmap_image)
    for banned in ("SurfaceView", "VolumetricView", "Volume2D", "Vertex2D"):
        assert banned not in src, banned
    assert "spatial_data" in src and "space.xfmname" in src


def test_uniques_yields_packables():
    """``uniques()`` promises ``Packable``, and every element really is one.

    The annotation used to be ``Iterator[Dataview]``, which is wider than the
    truth and lacks ``name`` -- the one member every consumer reaches for first.
    ``webgl.data.Package`` type-checked only because the list it built was
    ``Any``.
    """
    from cortex.dataset import Packable

    vol = cortex.Volume.random(subj, xfmname)
    vtx = cortex.Vertex.random(subj)
    views = [
        vol,
        vtx,
        cortex.Volume2D(vol, vol),
        cortex.Vertex2D(vtx, vtx),
        cortex.VolumeRGB(vol, vol, vol),
        cortex.VertexRGB(vtx, vtx, vtx),
    ]
    for view in views:
        for collapse in (True, False):
            produced = list(view.uniques(collapse=collapse))
            assert produced, type(view).__name__
            for item in produced:
                assert isinstance(item, Packable), (type(view).__name__, type(item))
                # the whole point: it has an addressable name
                assert item.name.startswith("__")

    # A 2D view is deliberately *not* packable: it owns no array of its own, only
    # the two channels it decomposes into, so it has no content-addressed name.
    for cls in (cortex.Volume2D, cortex.Vertex2D):
        assert not issubclass(cls, Packable), cls.__name__
        assert not hasattr(cls, "name"), cls.__name__

    # Both columns claim it in their bases rather than happening to satisfy it.
    assert Packable in cortex.dataset.ScalarView.__bases__
    assert Packable in cortex.dataset.DataviewRGB.__bases__


def test_packable_name_is_not_hoisted_onto_the_spatial_interface():
    """``name`` must keep hashing the *stored* array, not ``spatial_data``.

    For an RGB view the two coincide, which makes hoisting ``name`` onto
    ``RenderableView`` look free. It is not: a masked ``Volume`` stores a flat
    array while ``spatial_data`` is the unmasked 3-D one, so unifying them would
    silently rename every existing HDF node.
    """
    from cortex.dataset._hdf import _hash

    mask = cortex.db.get_mask(subj, xfmname, "thick")
    vol = cortex.Volume(np.random.randn(mask.sum()).astype(np.float32), subj, xfmname,
                        mask=mask)

    assert vol.name == "__%s" % _hash(vol.data)[:16]
    assert vol.name != "__%s" % _hash(vol.spatial_data)[:16]

    # ...whereas for RGB the stored array *is* the sampled one, which is why the
    # two RGB classes could share one implementation.
    rgb = cortex.VolumeRGB(vol, vol, vol)
    assert rgb.name == "__%s" % _hash(rgb.spatial_data)[:16]


def test_empty_and_random_take_their_shape_from_the_space():
    """``empty``/``random`` ask the space, so a new space gets both for free.

    ``Volume`` used to call ``db.get_xfm(...).shape`` and ``Vertex``
    ``SurfaceSpace(...).nverts`` -- the same question, asked two ways, in four
    methods. Both now read ``BrainSpace.template_shape``.
    """
    from cortex.dataset._space import SurfaceSpace, VolumeSpace

    vspace, sspace = VolumeSpace(subj, xfmname), SurfaceSpace(subj)
    assert vspace.template_shape == cortex.db.get_xfm(subj, xfmname).shape
    assert sspace.template_shape == (sspace.nverts,)

    for view, shape in ((cortex.Volume.empty(subj, xfmname), volshape),
                        (cortex.Vertex.empty(subj), (nverts,))):
        assert view.data.shape == shape
        assert view.data.dtype == np.float64  # np.ones()*value, not np.full
        assert (view.data == 0).all()

    assert (cortex.Volume.empty(subj, xfmname, value=3).data == 3).all()
    assert (cortex.Vertex.empty(subj, value=3).data == 3).all()
    assert cortex.Volume.random(subj, xfmname).data.shape == volshape
    assert cortex.Vertex.random(subj).data.shape == (nverts,)

    # kwargs still reach the constructor
    assert cortex.Vertex.empty(subj, cmap="hot").cmap == "hot"
    assert cortex.Volume.random(subj, xfmname, cmap="hot").cmap == "hot"


def test_template_shape_is_distinct_from_spatial_shape():
    """The two answer different questions; conflating them breaks ``empty``.

    ``spatial_shape`` describes an array already bound by ``coerce``, so a fresh
    ``VolumeSpace`` reports ``()`` for it -- which is exactly why ``empty`` cannot
    use it. They coincide for a surface, whose vertex count is known as soon as
    the space exists.
    """
    from cortex.dataset._space import SurfaceSpace, VolumeSpace

    fresh = VolumeSpace(subj, xfmname)
    assert fresh.spatial_shape == ()
    assert fresh.template_shape == volshape
    fresh.coerce(np.random.randn(*volshape))
    assert fresh.spatial_shape == volshape

    sspace = SurfaceSpace(subj)
    assert sspace.spatial_shape == sspace.template_shape == (nverts,)

    # A masked volume reports the *unmasked* geometry, since that is what a
    # caller building a fresh array needs.
    assert VolumeSpace(subj, xfmname, mask="thick").template_shape == volshape


def llen_of(view):
    return view.space.llen


def test_space_supplies_its_own_wire_layout_keys():
    """``to_json`` is one implementation; the space adds what the browser needs.

    ``Volume`` and ``Vertex`` each overrode ``to_json`` to bolt on ``shape`` and
    ``split``/``frames`` -- the same question (how does the browser unpack this
    array?) answered per space. These keys are read by
    ``webgl/resources/js/dataset.js``, so they are a hard interface.
    """
    from cortex.dataset.views import ScalarView

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    vtx = cortex.Vertex(np.random.randn(nverts), subj)
    vtx_movie = cortex.Vertex(np.random.randn(3, nverts), subj)

    # only ScalarView implements it now
    assert "to_json" in ScalarView.__dict__
    assert "to_json" not in cortex.Volume.__dict__
    assert "to_json" not in cortex.Vertex.__dict__

    assert vol.to_json(simple=True)["shape"] == volshape
    assert "split" not in vol.to_json(simple=True)

    vj = vtx.to_json(simple=True)
    assert vj["split"] == vtx.llen and vj["frames"] == 1
    assert vtx_movie.to_json(simple=True)["frames"] == 3
    assert "shape" not in vj

    # the non-simple form still carries the space's own description
    assert "xfm" in vol.to_json()
    assert "xfm" not in vtx.to_json()

    # The default is concrete and contributes nothing, so a space that needs no
    # unpacking hint inherits it. Called unbound on purpose, to reach the base
    # rather than SurfaceSpace's override -- and to execute its body, which once
    # raised NameError because DataviewJSON is a TYPE_CHECKING-only import there.
    from cortex.dataset._space import BrainSpace
    assert "describe_layout" in BrainSpace.__dict__
    assert BrainSpace.describe_layout(vtx.space, vtx.data) == {}
    assert vtx.space.describe_layout(vtx.data) == {"split": llen_of(vtx), "frames": 1}


def test_hemisphere_split_lives_on_the_space():
    """One rule for where the hemispheres meet, used by both surface columns.

    ``Vertex.left/right`` branched on ``self.movie`` and ``VertexRGB.left/right``
    reached through ``self.red.llen`` for the boundary -- four properties encoding
    one piece of space geometry.
    """
    vtx = cortex.Vertex(np.random.randn(nverts), subj)
    movie = cortex.Vertex(np.random.randn(3, nverts), subj)
    rgb = cortex.VertexRGB(vtx, vtx, vtx)
    llen = vtx.space.llen

    # the vertex axis is axis 0 only for a plain array; second otherwise, which
    # covers (t, v) and the (frames, v, 4) an RGB view ships
    assert vtx.left.shape == (llen,)
    assert movie.left.shape == (3, llen)
    assert rgb.left.shape == (1, llen, 4)
    for view in (vtx, movie, rgb):
        assert view.left.shape[-1 if view is not rgb else 1] + \
               view.right.shape[-1 if view is not rgb else 1] == vtx.nverts

    # slices, not copies
    assert vtx.left.base is not None

    # and RGB no longer consults a channel for geometry
    import inspect
    src = inspect.getsource(type(rgb).left.fget)
    assert "red.llen" not in src and "split_hemispheres" in src


def test_rgb_default_alpha_tracks_the_channels_frame_count():
    """A synthesised alpha must have as many frames as the channels.

    ``VertexRGB._default_alpha`` sized itself from ``vertices.shape[1]``, the
    vertex count alone, so a movie got a one-frame alpha against the channels'
    ``t`` and ``_rgba_stack`` raised ``ValueError`` on the ragged array --
    surface RGB movies were unconstructible. ``VolumeRGB`` never had the bug,
    because ``volume`` carries the frame axis.
    """
    for cls, chan, nchan in (
        (cortex.VertexRGB, cortex.Vertex(np.random.randn(4, nverts), subj), nverts),
        (cortex.VolumeRGB, cortex.Volume(np.random.randn(4, *volshape), subj, xfmname), None),
    ):
        rgb = cls(chan, chan, chan)
        assert rgb.alpha.data.shape[0] == 4, cls.__name__
        assert rgb.alpha.data.shape == chan.data.shape, cls.__name__
        stacked = rgb.vertices if cls is cortex.VertexRGB else rgb.volume
        assert stacked.shape[0] == 4 and stacked.shape[-1] == 4, cls.__name__
        assert rgb.to_json(simple=True).get("frames", 4) == 4, cls.__name__

    # a one-frame view keeps the shape it always had, since that shape feeds the
    # content hash used as the HDF node name
    still = cortex.Vertex(np.random.randn(nverts), subj)
    assert cortex.VertexRGB(still, still, still).alpha.data.shape == (nverts,)

    # NaNs still mark only their own frame transparent
    d = np.random.randn(3, nverts)
    d[1, :50] = np.nan
    movie = cortex.VertexRGB(*([cortex.Vertex(d, subj)] * 3))
    assert (movie.vertices[1, :50, 3] == 0).all()
    assert (movie.vertices[0, :, 3] != 0).all()
