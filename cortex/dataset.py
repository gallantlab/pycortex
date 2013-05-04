"""Module for maintaining brain data and their masks

HDF5 format:

/subjects/
    s1/
        transforms/
            xfm1/
                mat[4,4]
                masks/
                    thin[z,y,x]
                    thick[z,y,x]
            xfm2/
                mat[4,4]
                masks/
                    thin[z,y,x]
        surfaces
            fiducial
                lh[n,3]
                rh[n,3]
            inflated
                lh[n,3]
                rh[n,3]
/datasets/
    ds1
    ds2
"""
import time

import numpy as np
import tables

from .db import surfs
from . import volume
from . import utils

class Dataset(object):
    def __init__(self, **kwargs):
        self.datasets = {}
        for name, ds in kwargs.items():
            if isinstance(ds, BrainData):
                self.datasets[name] = ds
            else:
                self.datasets[name] = BrainData(*ds)

    @classmethod
    def from_file(cls, filename):
        datasets = dict()
        h5 = tables.openFile(filename)
        for node in h5.walkNodes("/datasets/"):
            if not isinstance(node, tables.Group):
                datasets[node.name] = BrainData.from_file(h5, name=node.name)
        h5.close()
        return cls(**datasets)

    def append(self, **kwargs):
        for name, data in kwargs.items():
            if isinstance(data, BrainData):
                self.datasets[name] = data
            else:
                self.datasets[name] = BrainData(*data)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif attr in self.datasets:
            return self.datasets[attr]

        raise AttributeError

    def __getitem__(self, item):
        return self.datasets[item]

    def __repr__(self):
        return "<Dataset with names [%s]>"%(', '.join(self.datasets.keys()))

    def __dir__(self):
        return self.__dict__.keys() + self.datasets.keys()

    def save(self, filename, pack=False):
        h5 = tables.openFile(filename, "w")
        _hdf_init(h5)
        for name, ds in self.datasets.items():
            ds.save(h5, name=name)
        if pack:
            raise NotImplementedError

        h5.close()

class BrainData(object):
    def __init__(self, data, subject, xfmname, mask=None, **kwargs):
        """Three possible variables: raw, volume, movie, vertex. Enumerated with size:
        raw volume movie: (t, z, y, x, c)
        raw volume image: (z, y, x, c)
        reg volume movie: (t, z, y, x)
        reg volume image: (z, y, x)
        raw linear movie: (t, v, c)
        reg linear movie: (t, v)
        raw linear image: (v, c)
        reg linear image: (v,)
        """
        self.data = data
        self.subject = subject
        self.xfmname = xfmname
        
        self._check_size(mask)
        self.attrs = kwargs

        self.masked = Masker(self)

    def _check_size(self, mask):
        self.raw = self.data.dtype == np.uint8
        if self.data.ndim == 5:
            if not self.raw:
                raise ValueError("Invalid data shape")
            self.linear = False
            self.movie = True
        elif self.data.ndim == 4:
            self.linear = False
            self.movie = not self.raw
        elif self.data.ndim == 3:
            self.linear = self.movie = self.raw
        elif self.data.ndim == 2:
            self.linear = True
            self.movie = not self.raw
        elif self.data.ndim == 1:
            self.linear = True
            self.movie = False
        else:
            raise ValueError("Invalid data shape")

        self.vertex = self.xfmname is None
        if self.vertex and not self.linear:
            raise ValueError('Vertex data must be linear!')

        if self.linear:
            #try to guess mask type
            if mask is None and not self.vertex:
                nvox = self.data.shape[-1]
                if self.raw:
                    nvox = self.data.shape[-2]
                self.masktype, self.mask = _find_mask(nvox, self.subject, self.xfmname)
            elif isinstance(mask, str):
                self.mask = surfs.getMask(self.subject, self.xfmname, mask)
                self.masktype = mask

    def __repr__(self):
        maskstr = ""
        if self.linear:
            maskstr = ", %s mask"%self.masktype
        return "<Data for (%s,%s)%s>"%(self.subject, self.xfmname, maskstr)

    @classmethod
    def from_file(cls, filename, name="data"):
        if isinstance(filename, str):
            fname, ext = os.path.splitext(filename)
            if ext in (".hdf", ".h5"):
                h5 = tables.openFile(filename)
                node = h5.getNode("/datasets", name)
                data, attrs = _hdf_read(node)
                h5.close()
                return cls(data, **attrs)
        elif isinstance(filename, tables.File):
            node = filename.getNode("/datasets", name)
            data, attrs = _hdf_read(node)
            return cls(data, **attrs)

    @property
    def volume(self):
        if not self.linear:
            return self.data
        return volume.unmask(self.mask, self.data)

    def save(self, filename, name="data"):
        if isinstance(filename, str):
            fname, ext = os.path.splitext(filename)
            if ext in (".hdf", ".h5"):
                h5 = tables.openFile(filename, "a")
                _hdf_init(h5)
                self._write_hdf(h5, name=name)
                h5.close()
        elif isinstance(filename, tables.File):
            self._write_hdf(filename, name=name)

    def _write_hdf(self, h5, name="data"):
        node = _hdf_write(h5, self.data, name=name)
        node.attrs.subject = self.subject
        node.attrs.xfmname = self.xfmname
        if self.linear:
            node.attrs.mask = self.masktype
        for name, value in self.attrs.items():
            node.attrs[name] = value

class Masker(object):
    def __init__(self, ds):
        self.ds = ds

        self.data = None
        if ds.linear:
            self.data = ds.data

    def __getitem__(self, masktype):
        s, x = self.ds.subject, self.ds.xfmname
        mask = surfs.getMask(s, x, masktype)
        if self.ds.movie:
            return BrainData(self.ds.volume[:,mask], s, x, mask=masktype)
        return BrainData(self.ds.volume[mask], s, x, mask=masktype)

def _find_mask(nvox, subject, xfmname):
    import re
    import glob
    import nibabel
    files = surfs.getFiles(subject)['masks'].format(xfmname=xfmname, type="*")
    for fname in glob.glob(files):
        nib = nibabel.load(fname)
        mask = nib.get_data() != 0
        if nvox == np.sum(mask):
            name = re.compile(r'([^_]+)_([\w]+)_([\w]+).nii.gz').search(fname)
            return name.group(3), mask

    raise ValueError('Cannot find a valid mask')

def _hdf_init(h5):
    try:
        h5.getNode("/datasets")
    except tables.NoSuchNodeError:
        h5.createGroup("/","datasets")
    try:
        h5.getNode("/subjects")
    except tables.NoSuchNodeError:
        h5.createGroup("/", "subjects")

def _hdf_write(h5, data, name="data", group="/datasets"):
    atom = tables.Atom.from_dtype(data.dtype)
    filt = tables.filters.Filters(complevel=9, complib='blosc', shuffle=True)
    try:
        ds = h5.getNode("%s/%s"%(group, name))
        ds[:] = data
    except tables.NoSuchNodeError:
        ds = h5.createCArray("%s"%group, name, atom, data.shape, filters=filt)
        ds[:] = data
    except ValueError:
        h5.removeNode("%s/%s"%(group, name))
        ds = h5.createCArray("%s"%group, name, atom, data.shape, filters=filt)
        ds[:] = data
        
    return ds

def _hdf_read(node):
    names = set(node.attrs._v_attrnames)
    names -= set(['CLASS', 'TITLE', 'VERSION'])
    attrs = dict((name, node.attrs[name]) for name in names)
    return node[:], attrs