import cortex
import tempfile
import numpy as np

from cortex import db, dataset

subj, xfmname, nverts = "S1", "fullhead", 304380

def test_braindata():
	vol = np.random.randn(31, 100, 100)
	tf = tempfile.TemporaryFile(suffix='.png')
	mask = db.get_mask(subj, xfmname, "thick")

	data = dataset.DataView((vol, subj, xfmname), cmap='RdBu_r', vmin=0, vmax=1)
	# quickflat.make_png(tf, data)
	mdata = data.copy(data.data.masked['thick'])
	assert len(mdata.data.data) == mask.sum()
	assert np.allclose(mdata.data.volume[mask], mdata.data.data)

def test_dataset():
	vol = np.random.randn(31, 100, 100)
	stack = (np.ones((100, 100, 31))*np.linspace(0, 1, 31)).T
	raw = (np.random.rand(10, 31, 100, 100, 3)*256).astype(np.uint8)
	mask = db.get_mask(subj, xfmname, "thick")

	ds = dataset.Dataset(randvol=(vol, subj, xfmname), stack=(stack, subj, xfmname))
	ds.append(thickstack=ds.stack.copy(ds.stack.data.masked['thick']))
	ds.append(raw=dataset.VolumeData(raw, subj, xfmname).masked['thin'])
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds.save(tf.name)

	ds = dataset.Dataset.from_file(tf.name)
	assert len(ds['thickstack'].data.data) == mask.sum()
	assert np.allclose(ds['stack'].data.data[mask], ds['thickstack'].data.data)
	assert ds['raw'].data.volume.shape == (10, 31, 100, 100, 4)
	return ds

def test_findmask():
	vol = (np.random.rand(10, 31, 100, 100, 3)*256).astype(np.uint8)
	mask = db.get_mask(subj, xfmname, "thin")
	ds = dataset.VolumeData(vol[:, mask], subj, xfmname)
	assert np.allclose(ds.volume[:, mask, :3], vol[:, mask])
	return ds

def test_rgb():
	vol = (np.random.rand(31, 100, 100, 3)*256).astype(np.uint8)

	ds = dataset.VolumeData(vol, subj, xfmname, "thick")
	dsm = ds.masked['thick']
	assert dsm.volume.shape == (31, 100, 100, 4)
	return dsm

def test_movie():
	vol = (np.random.rand(10, 31, 100, 100, 3)*256).astype(np.uint8)

	ds = dataset.VolumeData(vol, subj, xfmname)
	dsm = ds.masked['thick']
	assert dsm.volume.shape == (10, 31, 100, 100, 4)
	assert np.allclose(dsm.data[:,:,:3], vol[:,dsm.mask])

def test_braindata_hash():
	d = cortex.VolumeData(np.random.randn(31, 100, 100), subj, xfmname)
	hash(d)

def test_dataset_save():
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(test=(np.random.randn(31, 100, 100), subj, xfmname))
	ds.save(tf.name)
	
	ds = cortex.openFile(tf.name)
	assert isinstance(ds.test, cortex.DataView)
	assert ds.test.data.shape == (31, 100, 100)

def test_mask_save():
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(test=(np.random.randn(31, 100, 100), subj, xfmname))
	ds.append(masked=ds.test.data.masked['thin'])
	data = ds.masked.data.data
	ds.save(tf.name)

	ds = cortex.openFile(tf.name)
	assert ds.masked.data.shape == (31, 100, 100)
	assert np.allclose(ds.masked.data.data, data)

def test_overwrite():
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(test=(np.random.randn(31, 100, 100), subj, xfmname))
	ds.save(tf.name)
	
	ds.save()
	assert ds.test.data.shape == (31, 100, 100)

def test_view_multi():
	ds1 = cortex.VolumeData(np.random.randn(31, 100, 100), subj, xfmname)
	ds2 = cortex.VolumeData(np.random.randn(31, 100, 100), subj, xfmname)
	ds3 = cortex.VolumeData(np.random.randn(31, 100, 100), subj, xfmname)

	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(view1=[(ds1, ds2)], view2=[(ds1, ds3)], view3 = ds3)
	ds.save(tf.name)

	ds = cortex.openFile(tf.name)
	assert len(ds.view1.data) == 1
	assert len(ds.view2.data) == 1
	assert isinstance(ds.view3.data, cortex.VolumeData)

def test_pack():
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds = cortex.Dataset(test=(np.random.randn(31, 100, 100), subj, xfmname))
	ds.save(tf.name, pack=True)

	ds = cortex.openFile(tf.name)
	pts, polys = cortex.db.get_surf(subj, "fiducial", "lh")
	dpts, dpolys = ds.get_surf(subj, "fiducial", "lh")
	assert np.allclose(pts, dpts)

	rois = cortex.db.get_overlay(subj, "rois")
	# Dataset.get_overlay returns a file handle, not an ROIpack ?
	#assert rois.rois.keys() == ds.get_overlay(subj, "rois").rois.keys()

	xfm = cortex.db.get_xfm(subj, xfmname)
	assert np.allclose(xfm.xfm, ds.get_xfm(subj, xfmname).xfm)

def test_convertraw():
	ds = cortex.Dataset(test=(np.random.randn(31, 100, 100), subj, xfmname))
	ds.test.raw

def test_vertexdata_copy():
	vd = cortex.VertexData(np.random.randn(nverts), subj)
	vdcopy = vd.copy()
	assert np.allclose(vd.data, vdcopy.data)

def test_vertexdata_set():
	vd = cortex.VertexData(np.random.randn(nverts), subj)
	newdata = np.random.randn(nverts)
	vd.data = newdata
	assert np.allclose(newdata, vd.data)