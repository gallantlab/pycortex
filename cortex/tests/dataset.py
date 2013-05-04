import tempfile
import numpy as np
from .. import dataset
from .. import quickflat
from ..db import surfs

def test_braindata():
	vol = np.random.randn(32, 100, 100)
	tf = tempfile.TemporaryFile(suffix='.png')
	mask = surfs.getMask("AH", "AH_huth", "thick")

	data = dataset.BrainData(vol, "AH", "AH_huth", cmap='RdBu_r', vmin=0, vmax=1)
	# quickflat.make_png(tf, data)
	mdata = data.masked['thick']
	assert len(mdata.data) == mask.sum()
	assert np.allclose(mdata.volume[mask], mdata.data)

def test_dataset():
	vol = np.random.randn(32, 100, 100)
	stack = (np.ones((100, 100, 32))*np.linspace(0, 1, 32)).T
	mask = surfs.getMask("AH", "AH_huth", "thick")

	ds = dataset.Dataset(randvol=(vol, "AH", "AH_huth"), stack=(stack, "AH", "AH_huth"))
	ds.append(thickstack=ds.stack.masked['thick'])
	tf = tempfile.NamedTemporaryFile(suffix=".hdf")
	ds.save(tf.name)

	ds = dataset.Dataset.from_file(tf.name)
	assert len(ds['thickstack'].data) == mask.sum()
	assert np.allclose(ds['stack'].data[mask], ds['thickstack'].data)
	return ds