import cortex
import tempfile
import numpy as np
def test_braindata_hash():
	d = cortex.VolumeData(np.random.randn(32, 100, 100), "AH", "AH_huth")
	hash(d)

def test_dataset_save():
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(test=(np.random.randn(32, 100, 100), "AH", "AH_huth"))
	ds.save(tf.name)
	
	ds = cortex.openFile(tf.name)
	assert isinstance(ds.test, cortex.DataView)
	assert ds.test.data.shape == (32, 100, 100)

def test_mask_save():
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(test=(np.random.randn(32, 100, 100), "AH", "AH_huth"))
	ds.append(masked=ds.test.data.masked['thin'])
	data = ds.masked.data.data
	ds.save(tf.name)

	ds = cortex.openFile(tf.name)
	assert ds.masked.data.shape == (32, 100, 100)
	assert np.allclose(ds.masked.data.data, data)

def test_overwrite():
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(test=(np.random.randn(32, 100, 100), "AH", "AH_huth"))
	ds.save(tf.name)
	
	ds.save()
	assert ds.test.data.shape == (32, 100, 100)

def test_view_multi():
	ds1 = cortex.VolumeData(np.random.randn(32, 100, 100), "AH", "AH_huth")
	ds2 = cortex.VolumeData(np.random.randn(32, 100, 100), "AH", "AH_huth")
	ds3 = cortex.VolumeData(np.random.randn(32, 100, 100), "AH", "AH_huth")

	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(view1=[(ds1, ds2), (ds1,ds3)], view2=[(ds1, ds3)], view3 = ds3)
	ds.save(tf.name)

	ds = cortex.openFile(tf.name)
	assert len(ds.view1.data) == 2
	assert len(ds.view2.data) == 1
	assert isinstance(ds.view3.data, cortex.VolumeData)
	assert ds.view3.vmin == ds3.data.min()

def test_pack():
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(test=(np.random.randn(32, 100, 100), "AH", "AH_huth"))
	ds.save(tf.name, pack=True)

	ds = cortex.openFile(tf.name)
	pts, polys = cortex.surfs.getSurf("AH", "fiducial", "lh")
	dpts, dpolys = ds.getSurf("AH", "fiducial", "lh")
	assert np.allclose(pts, dpts)

	rois = cortex.surfs.getOverlay("AH", "rois")
	assert rois.rois.keys() == ds.getOverlay("AH", "rois").rois.keys()

	xfm = cortex.surfs.getXfm("AH", "AH_huth")
	assert np.allclose(xfm.xfm, ds.getXfm("AH", "AH_huth").xfm)