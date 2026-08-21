import cortex
import numpy as np
import tempfile
import pytest

from cortex import dataset
from cortex.testing_utils import has_installed
from cortex.webgl.data import Package

no_inkscape = not has_installed('inkscape')


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
