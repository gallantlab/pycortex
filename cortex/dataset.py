"""Module for maintaining brain data and their masks"""
import numpy as np

from .db import surfs
from . import volume
from . import utils

class Dataset(object):
    def __init__(self, **kwargs):
        self.datasets = kwargs

    @classmethod
    def from_file(cls, filename):
        pass

    def __getattr__(self, attr):
        if attr in self.datasets:
            return self.datasets[attr]
        raise AttributeError

    def save(self, filename):
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
        self.mask = mask
        self.cmap = cmap
        self.min = vmin
        self.max = vmax

        self.raw = data.dtype == np.uint8
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
            self.masktype = mask

    @classmethod
    def fromfile(cls, filename, dataname="data"):
        pass

    @property
    def volume(self):
        if not self.linear:
            return self.data
        return volume.unmask(self.mask, data)

    def save(self, filename, name="data", savemask=True):
        _, ext = os.path.splitext(filename)
        if ext == ".hdf":
            import tables
            h5 = tables.openFile(filename, "a")
            atom = tables.Atom.from_dtype(np.dtype())
            ds = h5.createCArray("/", name, )

class Masker(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, masktype):
        pass
