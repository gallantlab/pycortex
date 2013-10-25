import cortex
import numpy as np
def test_braindata_hash():
	d = cortex.VolumeData(np.random.randn(32, 100, 100), "AH", "AH_huth")
	hash(d)

def test_dataset_save():
	import tempfile
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(test=(np.random.randn(32, 100, 100), "AH", "AH_huth"))
	ds.save(tf.name)
	
	ds = cortex.openFile(tf.name)
	assert isinstance(ds.test, cortex.DataView)
	assert ds.test.data.shape == (32, 100, 100)

def test_overwrite():
	import tempfile
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(test=(np.random.randn(32, 100, 100), "AH", "AH_huth"))
	ds.save(tf.name)
	
	ds.save()
	assert ds.test.data.shape == (32, 100, 100)