import cortex
import tempfile
import numpy as np

from cortex import db, dataset

subj, xfmname, nverts, volshape = "S1", "fullhead", 304380, (31,100,100)

def test_braindata():
    vol = np.random.randn(*volshape)
    tf = tempfile.TemporaryFile(suffix='.png')
    mask = db.get_mask(subj, xfmname, "thick")

    data = dataset.Volume(vol, subj, xfmname, cmap='RdBu_r', vmin=0, vmax=1)
    # quickflat.make_png(tf, data)
    mdata = data.masked['thick']
    assert len(mdata.data) == mask.sum()
    assert np.allclose(mdata.volume[:, mask], mdata.data)

def test_dataset():
    vol = np.random.randn(*volshape)
    stack = (np.ones(volshape[::-1])*np.linspace(0, 1, volshape[0])).T
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
    vol = np.random.rand(10, *volshape)
    mask = db.get_mask(subj, xfmname, "thin")
    ds = dataset.Volume(vol[:, mask], subj, xfmname)
    assert np.allclose(ds.volume[:, mask], vol[:, mask])
    return ds

def test_rgb():
    red, green, blue, alpha = [np.random.randn(*volshape) for _ in range(4)]

    rgb = dataset.VolumeRGB(red, green, blue, subj, xfmname)
    assert rgb.volume.shape == tuple([1] + list(volshape) + [4])
    assert rgb.volume.dtype == np.uint8

    rgba = dataset.VolumeRGB(red, green, blue, subj, xfmname, alpha=alpha)
    assert rgba.volume.shape == tuple([1] + list(volshape) + [4])

    data = dataset.Volume.random(subj, xfmname)
    assert data.raw.volume.shape == tuple([1] + list(volshape) + [4])
    data.raw.to_json()

    red, green, blue, alpha = [np.random.randn(nverts) for _ in range(4)]

    rgb = dataset.VertexRGB(red, green, blue, subj)
    assert rgb.vertices.shape == (1, nverts, 4)
    assert rgb.vertices.dtype == np.uint8

    rgba = dataset.VertexRGB(red, green, blue, subj, alpha=alpha)
    assert rgba.vertices.shape == (1, nverts, 4)

    data = dataset.Vertex.random(subj)
    assert data.raw.vertices.shape == (1, nverts, 4)
    data.raw.to_json()

def test_2D():
    d1 = cortex.Volume.random(subj, xfmname)
    d2 = cortex.Volume.random(subj, xfmname).masked['thick']
    twod = cortex.Volume2D(d1, d2)
    cortex.Volume2D(d1.data, d2.data, subject=subj, xfmname=xfmname, vmin=0, vmax=2, vmin2=1)
    twod.to_json()
    return twod

def test_braindata_hash():
    d = cortex.Volume.random(subj, xfmname)
    hash(d)

def test_dataset_save():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    mrand = np.random.randn(2, *volshape)
    rand = np.random.randn(*volshape)
    ds = cortex.Dataset(test=(mrand, subj, xfmname))
    ds.append(twod=cortex.Volume2D(rand, rand, subj, xfmname))
    ds.append(rgb =cortex.VolumeRGB(rand, rand, rand, subj, xfmname))
    ds.append(vert=cortex.Vertex.random(subj))
    ds.save(tf.name)
    
    ds = cortex.load(tf.name)
    assert isinstance(ds.test, cortex.Volume)
    assert ds.test.data.shape == mrand.shape
    assert isinstance(ds.twod, cortex.Volume2D)
    assert ds.twod.dim1.data.shape == rand.shape
    assert ds.twod.dim2.data.shape == rand.shape
    assert ds.rgb.volume.shape == tuple([1] + list(volshape) + [4])
    assert isinstance(ds.vert, cortex.Vertex)

def test_mask_save():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds = cortex.Dataset(test=(np.random.randn(*volshape), subj, xfmname))
    ds.append(masked=ds.test.masked['thin'])
    data = ds.masked.data
    ds.save(tf.name)

    ds = cortex.load(tf.name)
    assert ds.masked.shape == volshape
    assert np.allclose(ds.masked.data, data)

def test_overwrite():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds = cortex.Dataset(test=(np.random.randn(*volshape), subj, xfmname))
    ds.save(tf.name)
    
    ds.save()
    assert ds.test.data.shape == volshape

def test_pack():
    tf = tempfile.NamedTemporaryFile(suffix=".hdf")
    ds = cortex.Dataset(test=(np.random.randn(*volshape), subj, xfmname))
    ds.save(tf.name, pack=True)

    ds = cortex.load(tf.name)
    pts, polys = cortex.db.get_surf(subj, "fiducial", "lh")
    dpts, dpolys = ds.get_surf(subj, "fiducial", "lh")
    assert np.allclose(pts, dpts)

    rois = cortex.db.get_overlay(subj, "rois")
    # Dataset.get_overlay returns a file handle, not an ROIpack ?
    #assert rois.rois.keys() == ds.get_overlay(subj, "rois").rois.keys()

    xfm = cortex.db.get_xfm(subj, xfmname)
    assert np.allclose(xfm.xfm, ds.get_xfm(subj, xfmname).xfm)

def test_map():
    dv = cortex.Volume.random(subj, xfmname)
    dv.map("nearest")


def test_convertraw():
    ds = cortex.Dataset(test=(np.random.randn(*volshape), subj, xfmname))
    ds.test.raw

def test_vertexdata_copy():
    vd = cortex.Vertex(np.random.randn(nverts), subj)
    vdcopy = vd.copy(vd.data)
    assert np.allclose(vd.data, vdcopy.data)

def test_vertexdata_set():
    vd = cortex.Vertex(np.random.randn(nverts), subj)
    newdata = np.random.randn(nverts)
    vd.data = newdata
    assert np.allclose(newdata, vd.data)

def test_vertexdata_index():
    vd = cortex.Vertex(np.random.randn(10, nverts), subj)
    assert np.allclose(vd[0].data, vd.data[0])


def test_vertex_rgb_movie():
    r = g = b = np.random.randn(nverts)
    rgb = cortex.VertexRGB(r, g, b, subj)

    
def test_volumedata_copy():
    v = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    vc = v.copy(v.data)
    assert np.allclose(v.data, vc.data)

def test_volumedata_copy_with_custom_mask():
    mask = cortex.get_cortical_mask(subj, xfmname, "thick")
    mask[16] = True
    nmask = mask.sum()
    data = np.random.randn(nmask)
    v = cortex.Volume(data, subj, xfmname, mask=mask)
    vc = v.copy(v.data)
    assert np.allclose(v.data, vc.data)
