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
		cortex.quickflat.make_figure(
			view, with_colorbar=True, colorbar_location=colorbar_location)

	with pytest.raises(ValueError):
		cortex.quickflat.make_figure(
			view, with_colorbar=True, colorbar_location='unknown_location')
