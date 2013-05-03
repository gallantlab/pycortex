"""Module for maintaining brain data and their masks"""
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
        self.h5 = tables.openFile(filename)


    def __getattr__(self, attr):
        if attr in self.datasets:
            return self.datasets[attr]
        raise AttributeError

    def save(self, filename=None):
        pass

    def import_packed(self):
        pass

    def pack_data(self):
        pass

class BrainData(object):
    def __init__(self, data, subject, xfmname, mask=None, cmap=None, vmin=None, vmax=None):
        """Three possible variables: raw, volume, movie. Enumerated with size:
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
        self.masktype = mask
        self.cmap = cmap
        self.min = vmin
        self.max = vmax

        self._mtime = None
        self._check_size()

    def _check_size(self):
        self.raw = self.data.dtype == np.uint8
        if data.ndim == 5:
            if not self.raw:
                raise ValueError("Invalid data shape")
            self.linear = False
            self.movie = True
        elif data.ndim == 4:
            self.linear = False
            self.movie = not self.raw
        elif data.ndim == 3:
            self.linear = self.movie = self.raw
        elif data.ndim == 2:
            self.linear = True
            self.movie = not self.raw
        elif data.ndim == 1:
            self.linear = True
            self.movie = False
        else:
            raise ValueError("Invalid data shape")

        if self.linear:
            #try to guess mask type
            if self.masktype is None:
                self.mask, self.masktype = _find_mask(self.data.shape[-1])
            elif isinstance(self.masktype, str):
                self.mask = surfs.getMask(self.subject, self.xfmname, self.masktype)
            elif isinstance(self.masktype, np.ndarray):
                self.mask = self.masktype
                self.masktype = None

    @classmethod
    def from_file(cls, filename, dataname="data"):
        pass

    @property
    def volume(self):
        if not self.linear:
            return self.data
        return volume.unmask(self.mask, data)

    def save(self, filename, name="data"):
        if isinstance(filename, str):
            fname, ext = os.path.splitext(filename)
            if ext in (".hdf", ".h5"):
                import tables
                h5 = tables.openFile(filename, "a")


class Masker(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, masktype):
        pass


def _find_mask(nvox):
    import re
    import glob
    import nibabel
    files = surfs.getFiles(self.subject)['masks'].format(xfmname=self.xfmname, type="*")
    for fname in glob.glob(files):
        nib = nibabel.load(fname)
        mask = nib.get_data() != 0
        if nvox == np.sum(mask):
            name = re.compile(r'([^_]+)_([\w]+)_([\w]+).nii.gz').search(fname)
            return name.group(3), mask

    raise ValueError('Cannot find a valid mask')

def _hdf_init(h5):
    try:
        h5.getNode("/dataset")
    except tables.NoSuchNodeError:
        h5.createGroup("/","datasets")
    try:
        h5.getNode("/subjects")
    except tables.NoSuchNodeError:
        h5.createGroup("/", "subjects")

def _hdf_write(h5, data, name="data"):
    atom = tables.Atom.from_dtype(data.dtype)
    filt = tables.filters.Filter(complevel=9, complib='blosc', shuffle=True)
    try:
        ds = h5.getNode("/dataset/%s"%name)
        ds[:] = data
    except tables.NoSuchNodeError:
        ds = h5.createCArray("/dataset", name, atom, data.shape, filters=filt)
        ds[:] = data
    except ValueError:
        h5.removeNode("/dataset/%s"%name)
        ds = h5.createCArray("/dataset", name, atom, data.shape, filters=filt)
        ds[:] = data
        
    return ds