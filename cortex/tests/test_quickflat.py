import cortex
import numpy as np
import tempfile
import pytest

from cortex.testing_utils import has_installed

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
