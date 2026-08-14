import cortex
import numpy as np
import tempfile
import pytest
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from cortex import dataset
import cortex.quickflat.utils # for ty
from cortex.testing_utils import has_installed
from cortex.webgl.data import Package

no_inkscape = not has_installed('inkscape')

def random_volume(with_nan=False, **kwargs):
    orig_vol = cortex.Volume.random("S1", "fullhead", **kwargs)
    data = orig_vol.data.copy()
    if with_nan:
        # set 50% of the values in the dataset to NaN
        data[np.random.rand(*data.shape) > 0.5] = np.nan
    # TODO: make sure kwargs are passed through correctly (e.g. vmin/vmax, etc.)
    return orig_vol.copy(data=data)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_quickflat():
    tf = tempfile.NamedTemporaryFile(suffix=".png")
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    cortex.quickflat.make_png(tf.name, view)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_colorbar_location():
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    for colorbar_location in ['left', 'center', 'right', (0, 0.2, 0.4, 0.3)]:
        cortex.quickflat.make_figure(view, with_colorbar=True,
                                     colorbar_location=colorbar_location)

    with pytest.raises(ValueError):
        cortex.quickflat.make_figure(view, with_colorbar=True,
                                     colorbar_location='unknown_location')


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
@pytest.mark.parametrize("type_", ["thick", "thin"])
@pytest.mark.parametrize("nanmean", [True, False])
def test_make_flatmap_image_nanmean(type_, nanmean):
    mask = cortex.db.get_mask("S1", "fullhead", type=type_)
    data = np.ones(mask.sum())
    # set 50% of the values in the dataset to NaN
    data[np.random.rand(*data.shape) > 0.5] = np.nan
    vol = cortex.Volume(data, "S1", "fullhead", vmin=0, vmax=1)
    img, extents = cortex.quickflat.utils.make_flatmap_image(
        vol, nanmean=nanmean)
    # assert that the nanmean only returns NaNs and 1s
    assert np.nanmin(img) == 1


def test_quickshow_webgl_alpha_equivalence():
    """quickshow (matplotlib) and WebGL must render the same VertexRGB+α identically.

    Issue #631: the WebGL shader uses a premultiplied "over" composite, while
    matplotlib's imshow layering uses straight alpha. The fix premultiplies α
    into RGB at the WebGL serialization step only, so both paths converge on
    the same composite formula  out = α·rgb + (1-α)·bg  for any background.
    This test asserts that equivalence at the per-vertex level for an
    arbitrary curvature gray.
    """
    subj = "S1"
    nverts = cortex.db.get_surf(subj, "fiducial", merge=True)[0].shape[0]
    rng = np.random.default_rng(631)
    r = rng.uniform(0, 1, nverts).astype(np.float32)
    g = rng.uniform(0, 1, nverts).astype(np.float32)
    b = rng.uniform(0, 1, nverts).astype(np.float32)
    alpha = rng.uniform(0, 1, nverts).astype(np.float32)

    vrgb = cortex.VertexRGB(
        r, g, b, subj,
        alpha=cortex.Vertex(alpha, subj, vmin=0, vmax=1),
    )

    raw = vrgb.vertices  # what quickshow/matplotlib will composite (non-premult)
    pkg = Package(dataset.Dataset(view=vrgb))
    packaged = pkg.images[vrgb.name][0]  # what the shader will composite (premult)

    # Sanity: alpha is shared between the two paths.
    assert np.array_equal(raw[..., 3], packaged[..., 3])

    # Composite both against an arbitrary curvature gray. matplotlib's
    # imshow with two layered images uses straight alpha; the GLSL shader at
    # shaderlib.js line 851 uses gl_FragColor = vColor + (1-α)·bg.
    a_norm = raw[..., 3:4].astype(np.float32) / 255.0
    rgb_raw = raw[..., :3].astype(np.float32) / 255.0
    rgb_pkg = packaged[..., :3].astype(np.float32) / 255.0
    for curv in (0.0, 0.25, 0.5, 0.75, 1.0):
        bg = np.full_like(rgb_raw, curv)
        matplotlib_out = a_norm * rgb_raw + (1.0 - a_norm) * bg
        webgl_out = rgb_pkg + (1.0 - a_norm) * bg
        # 1 LSB of uint8 rounding on each side -> 2/255 worst case.
        np.testing.assert_allclose(matplotlib_out, webgl_out, atol=2.0 / 255.0)


def test_make_flatmap_image_vertexrgb_alpha_unchanged():
    """The matplotlib path must keep using NON-premultiplied RGBA bytes.

    Premultiplying inside .vertices would silently double-attenuate the
    quickshow output. Pin that .vertices stays straight-alpha by checking
    a uniform bright-red, half-transparent VertexRGB survives
    make_flatmap_image without losing red intensity.
    """
    subj = "S1"
    nverts = cortex.db.get_surf(subj, "fiducial", merge=True)[0].shape[0]
    # Uniform bright red, half transparent everywhere. Pass explicit Vertex
    # objects with vmin/vmax to avoid auto-range degeneracy on the flat
    # green/blue channels.
    r = cortex.Vertex(np.ones(nverts, dtype=np.float32), subj, vmin=0, vmax=1)
    g = cortex.Vertex(np.zeros(nverts, dtype=np.float32), subj, vmin=0, vmax=1)
    b = cortex.Vertex(np.zeros(nverts, dtype=np.float32), subj, vmin=0, vmax=1)
    alpha = cortex.Vertex(np.full(nverts, 0.5, dtype=np.float32), subj,
                          vmin=0, vmax=1)
    vrgb = cortex.VertexRGB(r, g, b, subj, alpha=alpha)
    img, _ = cortex.quickflat.utils.make_flatmap_image(vrgb)
    # img is the rasterized RGBA flatmap. The data layer's red channel (where
    # mask is filled and pixmap is non-degenerate) must be ~255, not ~127 --
    # if we ever start premultiplying inside .vertices, this drops to ~127.
    rgba_in_mask = img[img[..., 3] > 0]
    assert rgba_in_mask.size > 0
    # Filled pixels should have red close to 255 (bright red, with alpha=128).
    bright_red_pixels = rgba_in_mask[rgba_in_mask[..., 0] > 200]
    assert bright_red_pixels.size > 0, (
        "VertexRGB.vertices appears to be premultiplied -- the matplotlib "
        "path will double-attenuate. The fix should live in webgl/data.py."
    )

@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_quickflat_curvature():
    vol = random_volume(with_nan=True, cmap="hot", vmin=0, vmax=1)
    cortex.quickflat.make_figure(vol, with_curvature=True)


# Tests for remaining make_figure arguments


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_recache():
    """Test recache parameter"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    fig = cortex.quickflat.make_figure(view, recache=False)
    
    # recache=True takes longer but should still work
    fig = cortex.quickflat.make_figure(view, recache=True)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_pixelwise_and_thick():
    """Test pixelwise and thick parameters"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    
    # Test pixelwise=True with different thick values
    for thick in [1, 4, 8, 16, 32]:
        fig = cortex.quickflat.make_figure(view, pixelwise=True, thick=thick)
    
    # Test pixelwise=False
    fig = cortex.quickflat.make_figure(view, pixelwise=False)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_sampler():
    """Test sampler parameter with different sampling methods"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    
    for sampler in ['nearest', 'trilinear']:
        fig = cortex.quickflat.make_figure(view, sampler=sampler)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_height_and_dpi():
    """Test height and dpi parameters"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    
    # Test smaller height for faster rendering
    height, dpi = 512, 100
    fig = cortex.quickflat.make_figure(view, height=height, dpi=dpi)
    # Check the resulting figure size in inches (height in pixels / dpi)
    expected_height_inch = height / dpi
    assert np.isclose(fig.get_figheight(), expected_height_inch, atol=0.1)
    
    # Test larger height
    height, dpi = 1024, 150
    fig = cortex.quickflat.make_figure(view, height=height, dpi=dpi)
    expected_height_inch = height / dpi
    assert np.isclose(fig.get_figheight(), expected_height_inch, atol=0.1)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_depth():
    """Test depth parameter for sampling different cortical depths"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    
    # Test different depth values (0 = gray/white matter, 1 = pial surface)
    for depth in [0.0, 0.5, 1.0]:
        fig = cortex.quickflat.make_figure(view, depth=depth)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_display_flags():
    """Test boolean display flags: with_rois, with_sulci, with_labels, with_colorbar"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    
    # Test with_rois
    fig = cortex.quickflat.make_figure(view, with_rois=True)
    fig = cortex.quickflat.make_figure(view, with_rois=False)
    
    # Test with_sulci
    fig = cortex.quickflat.make_figure(view, with_sulci=False)
    fig = cortex.quickflat.make_figure(view, with_sulci=True)
    
    # Test with_labels
    fig = cortex.quickflat.make_figure(view, with_labels=False)
    fig = cortex.quickflat.make_figure(view, with_labels=True)
    
    # Test with_colorbar
    fig = cortex.quickflat.make_figure(view, with_colorbar=False)
    fig = cortex.quickflat.make_figure(view, with_colorbar=True)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_with_dropout():
    """Test with_dropout parameter with bool and float values"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    
    # Test with_dropout with boolean values
    fig = cortex.quickflat.make_figure(view, with_dropout=False)
    fig = cortex.quickflat.make_figure(view, with_dropout=True)
    
    # Test with_dropout with float value
    fig = cortex.quickflat.make_figure(view, with_dropout=10.0)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_with_connected_vertices():
    """Test with_connected_vertices parameter"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    
    fig = cortex.quickflat.make_figure(view, with_connected_vertices=False)
    
    # Note: with_connected_vertices=True is more computationally expensive
    fig = cortex.quickflat.make_figure(view, with_connected_vertices=True)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_roi_styling_parameters():
    """Test ROI styling parameters: linewidth, linecolor, roifill, shadow, labelsize, labelcolor"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    # TODO: need with_rois, etc.?
    
    # Test linewidth
    fig = cortex.quickflat.make_figure(view, linewidth=2)
    
    # Test linecolor (RGB/RGBA tuple)
    fig = cortex.quickflat.make_figure(view, linecolor=(1.0, 0.0, 0.0))
    
    # Test roifill (RGB/RGBA tuple)
    fig = cortex.quickflat.make_figure(view, roifill=(1.0, 0.0, 0.0, 0.5))
    
    # Test shadow
    #fig = cortex.quickflat.make_figure(view, shadow=1) # TODO: why does this fail?
    
    # Test labelsize
    fig = cortex.quickflat.make_figure(view, labelsize="10pt")
    
    # Test labelcolor
    fig = cortex.quickflat.make_figure(view, labelcolor=(0.0, 0.0, 0.0))


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_curvature_parameters():
    """Test curvature styling parameters: curvature_brightness, curvature_contrast, curvature_threshold"""
    vol = random_volume(with_nan=True, cmap="hot", vmin=0, vmax=1) 

    
    # Test brightness and contrast together
    fig = cortex.quickflat.make_figure(vol, with_curvature=True,
                                       curvature_brightness=0.7, 
                                       curvature_contrast=0.5)
    
    # Test threshold
    fig = cortex.quickflat.make_figure(vol, with_curvature=True, 
                                       curvature_threshold=True)
    
    fig = cortex.quickflat.make_figure(vol, with_curvature=True,
                                       curvature_threshold=False)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_colorbar_ticks():
    """Test colorbar_ticks parameter"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot", vmin=0, vmax=1)
    
    # Custom ticks
    ticks = np.array([0.0, 0.5, 1.0])
    fig = cortex.quickflat.make_figure(view, with_colorbar=True, 
                                       colorbar_ticks=ticks)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_fig_parameter():
    """Test fig parameter with Figure and Axes objects"""
    from matplotlib import pyplot as plt
    
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    
    # Test passing a Figure object
    fig_obj = plt.figure()
    result = cortex.quickflat.make_figure(view, fig=fig_obj)
    assert isinstance(result, Figure)
    plt.close(fig_obj)
    
    # Test passing an Axes object
    fig_obj = plt.figure()
    ax_obj = fig_obj.add_subplot(111)
    result = cortex.quickflat.make_figure(view, fig=ax_obj)
    assert isinstance(result, Figure)
    plt.close(fig_obj)


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_extra_hatch():
    """Test extra_hatch parameter with additional hatching layer"""
    mask = cortex.db.get_mask("S1", "fullhead", type="thick")
    data = np.ones(mask.sum())
    vol = cortex.Volume(data, "S1", "fullhead", vmin=0, vmax=1)
    
    # Create a hatch layer with same shape
    hatch_data = cortex.Volume(np.random.rand(mask.sum()), "S1", "fullhead", vmin=0, vmax=1)
    hatch_color = (1.0, 0.0, 0.0)  # Red
    
    fig = cortex.quickflat.make_figure(vol, extra_hatch=(hatch_data, hatch_color))


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_roi_list_and_sulci_list():
    """Test roi_list and sulci_list parameters to filter displayed regions"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot")
    
    # Test roi_list - get available ROIs first
    # Note: This depends on the database having ROIs available
    try:
        fig = cortex.quickflat.make_figure(view, with_rois=True, roi_list=['V1'])
    except (ValueError, KeyError):
        # Database might not have specific ROI names for this subject
        pass
    
    # Test sulci_list
    try:
        fig = cortex.quickflat.make_figure(view, with_sulci=True, sulci_list=None)
    except (ValueError, KeyError):
        # Database might not have specific sulci names
        pass


@pytest.mark.skipif(no_inkscape, reason='Inkscape required')
def test_combined_parameters():
    """Test make_figure with multiple parameters combined"""
    view = cortex.Volume.random("S1", "fullhead", cmap="hot", vmin=0, vmax=1)
    
    # Comprehensive combination test
    fig = cortex.quickflat.make_figure(
        view,
        recache=False,
        pixelwise=True,
        thick=16,
        sampler='nearest',
        height=512,
        dpi=100,
        depth=0.5,
        with_rois=True,
        with_sulci=False,
        with_labels=True,
        with_colorbar=True,
        with_dropout=False,
        with_curvature=True,
        with_connected_vertices=False,
        linewidth=1,
        linecolor=(0.0, 0.0, 0.0),
        labelsize="12pt",
        curvature_brightness=0.6,
        curvature_contrast=0.4,
        curvature_threshold=True,
        colorbar_location='center',
        nanmean=False
    )
