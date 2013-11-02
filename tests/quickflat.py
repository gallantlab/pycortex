import cortex
import numpy as np
import tempfile

def test_quickflat():
	tf = tempfile.NamedTemporaryFile(suffix=".png")
	view = cortex.DataView((np.random.randn(32, 100, 100), "AH", "AH_huth"), cmap="hot")
	cortex.quickflat.make_png(tf.name, view)