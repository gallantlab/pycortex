import cortex
import tempfile
import numpy as np

from cortex import db, dataset

subj, xfmname, nverts = "S1", "fullhead", 304380

def test_braindata():
    vol = np.random.randn(31, 100, 100)
    tf = tempfile.TemporaryFile(suffix='.png')
    mask = db.get_mask(subj, xfmname, "thick")

    data = dataset.Volume(vol, subj, xfmname, cmap='RdBu_r', vmin=0, vmax=1)
    # quickflat.make_png(tf, data)
    mdata = data.masked['thick']
    assert len(mdata.data) == mask.sum()
    assert np.allclose(mdata.volume[:, mask], mdata.data)

def test_dataset():
    vol = np.random.randn(31, 100, 100)
    stack = (np.ones((100, 100, 31))*np.linspace(0, 1, 31)).T
    mask = db.get_mask(subj, xfmname, "thick")

    ds = dataset.Dataset(randvol=(vol, subj, xfmname), stack=(stack, subj, xfmname))
    ds.append(thickstack=ds.stack.masked['thick'])
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds.save(tf.name)

    ds = dataset.Dataset.from_file(tf.name)
    assert len(ds['thickstack'].data) == mask.sum()
    assert np.allclose(ds['stack'].data[mask], ds['thickstack'].data)
    return ds

def test_findmask():
    vol = np.random.rand(10, 31, 100, 100)
    mask = db.get_mask(subj, xfmname, "thin")
    ds = dataset.Volume(vol[:, mask], subj, xfmname)
    assert np.allclose(ds.volume[:, mask], vol[:, mask])
    return ds

def test_rgb():
    red, green, blue, alpha = [np.random.randn(31, 100, 100) for _ in range(4)]

    rgb = dataset.VolumeRGB(red, green, blue, subj, xfmname)
    assert rgb.volume.shape == (1, 31, 100, 100, 4)
    assert rgb.volume.dtype == np.uint8

    rgba = dataset.VolumeRGB(red, green, blue, subj, xfmname, alpha=alpha)
    assert rgba.volume.shape == (1, 31, 100, 100, 4)

def test_braindata_hash():
    d = cortex.Volume.random(subj, xfmname)
    hash(d)

def test_dataset_save():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds = cortex.Dataset(test=(np.random.randn(31, 100, 100), subj, xfmname))
    ds.save(tf.name)
    
    ds = cortex.openFile(tf.name)
    assert isinstance(ds.test, cortex.Volume)
    assert ds.test.data.shape == (31, 100, 100)

def test_mask_save():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds = cortex.Dataset(test=(np.random.randn(31, 100, 100), subj, xfmname))
    ds.append(masked=ds.test.masked['thin'])
    data = ds.masked.data
    ds.save(tf.name)

    ds = cortex.openFile(tf.name)
    assert ds.masked.shape == (31, 100, 100)
    assert np.allclose(ds.masked.data, data)

def test_overwrite():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds = cortex.Dataset(test=(np.random.randn(31, 100, 100), subj, xfmname))
    ds.save(tf.name)
    
    ds.save()
    assert ds.test.data.shape == (31, 100, 100)

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
    assert rois.rois.keys() == ds.get_overlay(subj, "rois").rois.keys()

    xfm = cortex.db.get_xfm(subj, xfmname)
    assert np.allclose(xfm.xfm, ds.get_xfm(subj, xfmname).xfm)

"""
def test_convertraw():
    ds = cortex.Dataset(test=(np.random.randn(31, 100, 100), subj, xfmname))
    ds.test.raw
"""

def test_vertexdata_copy():
    vd = cortex.Vertex(np.random.randn(nverts), subj)
    vdcopy = vd.copy()
    assert np.allclose(vd.data, vdcopy.data)

def test_vertexdata_set():
    vd = cortex.Vertex(np.random.randn(nverts), subj)
    newdata = np.random.randn(nverts)
    vd.data = newdata
    assert np.allclose(newdata, vd.data)