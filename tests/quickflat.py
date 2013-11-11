import cortex
import numpy as np
import tempfile

def test_quickflat():
	tf = tempfile.NamedTemporaryFile(suffix=".png")
	view = cortex.DataView((np.random.randn(31, 100, 100), "S1", "fullhead"), cmap="hot")
	cortex.quickflat.make_png(tf.name, view)
