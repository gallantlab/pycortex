import tempfile
import numpy as np
from .. import dataset
from .. import quickflat
from ..db import surfs

subj, xfm = "AH", "AH_huth"

def test_braindata():
	vol = np.random.randn(32, 100, 100)
	tf = tempfile.TemporaryFile(suffix='.png')
	mask = surfs.getMask(subj, xfm, "thick")

	data = dataset.BrainData(vol, subj, xfm, cmap='RdBu_r', vmin=0, vmax=1)
	# quickflat.make_png(tf, data)
	mdata = data.masked['thick']
	assert len(mdata.data) == mask.sum()
	assert np.allclose(mdata.volume[mask], mdata.data)

def test_dataset():
	vol = np.random.randn(32, 100, 100)
	stack = (np.ones((100, 100, 32))*np.linspace(0, 1, 32)).T
	raw = (np.random.rand(10, 32, 100, 100, 3)*256).astype(np.uint8)
	mask = surfs.getMask(subj, xfm, "thick")

	ds = dataset.Dataset(randvol=(vol, subj, xfm), stack=(stack, subj, xfm))
	ds.append(thickstack=ds.stack.masked['thick'])
	ds.append(raw=dataset.BrainData(raw, subj, xfm).masked['thin'])
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds.save(tf.name)

	ds = dataset.Dataset.from_file(tf.name)
	assert len(ds['thickstack'].data) == mask.sum()
	assert np.allclose(ds['stack'].data[mask], ds['thickstack'].data)
	assert ds['raw'].volume.shape == (10, 32, 100, 100, 4)
	return ds

def test_findmask():
	vol = (np.random.rand(10, 32, 100, 100, 3)*256).astype(np.uint8)
	mask = surfs.getMask(subj, xfm, "thin")
	ds = dataset.BrainData(vol[:, mask], subj, xfm)
	assert np.allclose(ds.volume[:, mask, :3], vol[:, mask])
	return ds

def test_rgb():
	vol = (np.random.rand(32, 100, 100, 3)*256).astype(np.uint8)

	ds = dataset.BrainData(vol, subj, xfm, "thick")
	dsm = ds.masked['thick']
	assert dsm.volume.shape == (32, 100, 100, 4)
	return dsm

def test_movie():
	vol = (np.random.rand(10, 32, 100, 100, 3)*256).astype(np.uint8)

	ds = dataset.BrainData(vol, subj, xfm)
	dsm = ds.masked['thick']
	assert dsm.volume.shape == (10, 32, 100, 100, 4)
	assert np.allclose(dsm.data, vol[:,dsm.mask])