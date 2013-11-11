import hashlib
import numpy as np
import h5py

from ..db import surfs

class BrainData(object):
    def __init__(self, data):
        if isinstance(data, str):
            import nibabel
            nib = nibabel.load(data)
            data = nib.get_data().T
        self._data = data

    @property
    def data(self):
        if isinstance(self._data, h5py.Dataset):
            return self._data.value
        return self._data

    @property
    def name(self):
        '''Name of this BrainData, according to its hash'''
        return "__%s"%_hash(self.data)[:16]

    def copy(self):
        raise NotImplementedError("Copy not supported for BrainData, use VolumeData or VertexData")

    def exp(self):
        """Copy of this object with data exponentiated.
        """
        return self.copy(np.exp(self.data))

    def __hash__(self):
        return hash(_hash(self.data))

    def _write_hdf(self, h5, name=None):
        if name is None:
            name = self.name

        dgrp = h5.require_group("/data")
        if name in dgrp:
            #don't need to update anything, since it's saved already
            return h5.get("/data/%s"%name)

        node = _hdf_write(h5, self.data, name=name)
        node.attrs['subject'] = self.subject
        return node

    @staticmethod
    def from_hdf(dataset, node):
        subj = node.attrs['subject']
        if "xfmname" in node.attrs:
            xfmname = node.attrs['xfmname']
            mask = None
            if "mask" in node.attrs:
                try:
                    surfs.getMask(subj, xfmname, node.attrs['mask'])
                    mask = node.attrs['mask']
                except IOError:
                    mask = dataset.getMask(subj, xfmname, node.attrs['mask'])
            return VolumeData(node, subj, xfmname, mask=mask)
        else:
            return VertexData(node, subj)

    @classmethod
    def add_numpy_methods(cls):
        """Adds numpy operator methods (+, -, etc.) to this class to allow
        simple manipulation of the data, e.g. with VolumeData v:
        v + 1 # Returns new VolumeData with 1 added to data
        v ** 2 # Returns new VolumeData with data squared
        """
        # Binary operations
        npops = ["__add__", "__sub__", "__mul__", "__div__", "__pow__",
                 "__neg__", "__abs__"]

        def make_opfun(op): # function nesting creates closure containing op
            def opfun(self, *args):
                return self.copy(getattr(self.data, op)(*args))
            return opfun
        
        for op in npops:
            opfun = make_opfun(op)
            opfun.__name__ = op
            setattr(cls, opfun.__name__, opfun)

BrainData.add_numpy_methods()

class VolumeData(BrainData):
    def __init__(self, data, subject, xfmname, mask=None):
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
        super(VolumeData, self).__init__(data)
        try:
            basestring
        except NameError:
            subject = subject if isinstance(subject, str) else subject.decode('utf-8')
            xfmname = xfmname if isinstance(xfmname, str) else xfmname.decode('utf-8')
        self.subject = subject
        self.xfmname = xfmname

        self._check_size(mask)
        self.masked = _masker(self)

    def copy(self, data=None):
        """Copies this VolumeData.
        """
        if data is None:
            data = self.data
        return VolumeData(data, self.subject, self.xfmname, mask=self._mask)

    def to_json(self):
        xfm = surfs.getXfm(self.subject, self.xfmname, 'coord').xfm
        return dict(
            data=self.name,
            subject=self.subject, 
            xfm=list(np.array(xfm).ravel()),
            movie=self.movie,
            raw=self.raw,
            shape=self.shape,
            min=float(self.data.min()),
            max=float(self.data.max()),
        )

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

        if self.linear:
            if mask is None:
                #try to guess mask type
                nvox = self.data.shape[-2 if self.raw else -1]
                if self.raw:
                    nvox = self.data.shape[-2]
                self._mask, self.mask = _find_mask(nvox, self.subject, self.xfmname)
            elif isinstance(mask, str):
                self.mask = surfs.getMask(self.subject, self.xfmname, mask)
                self._mask = mask
            elif isinstance(mask, np.ndarray):
                self.mask = mask
                self._mask = mask

            self.shape = self.mask.shape
        else:
            self._mask = None
            shape = self.data.shape
            if self.movie:
                shape = shape[1:]
            if self.raw:
                shape = shape[:-1]
            xfm = surfs.getXfm(self.subject, self.xfmname)
            if xfm.shape != shape:
                raise ValueError("Volumetric data must be same shape as reference for transform")
            self.shape = shape

    def map(self, projection="nearest"):
        """Convert this VolumeData into a VertexData using the given sampler
        """
        from .. import utils
        mapper = utils.get_mapper(self.subject, self.xfmname, projection)
        data = mapper(self)
        return data

    def __repr__(self):
        maskstr = "volumetric"
        if self.linear:
            name = self._mask
            if isinstance(self._mask, np.ndarray):
                name = "custom"
            maskstr = "%s masked"%name
        if self.raw:
            maskstr += " raw"
        if self.movie:
            maskstr += " movie"
        maskstr = maskstr[0].upper()+maskstr[1:]
        return "<%s data for (%s, %s)>"%(maskstr, self.subject, self.xfmname)

    @property
    def volume(self):
        """Standardizes the VolumeData, ensuring that masked data are unmasked"""
        from .. import volume
        if self.linear:
            data = volume.unmask(self.mask, self.data[:])
        else:
            data = self.data[:]

        if self.raw and data.shape[-1] == 3:
            #stack the alpha dimension
            shape = data.shape[:3]+(1,)
            if self.movie:
                shape = data.shape[:4]+(1,)
            alpha = 255*np.ones(shape).astype(np.uint8)
            data = np.concatenate([data, alpha], axis=-1)

        return data

    def save(self, filename, name=None):
        """Save the dataset into an hdf file with the provided name
        """
        if isinstance(filename, str):
            fname, ext = os.path.splitext(filename)
            if ext in (".hdf", ".h5"):
                h5 = h5py.File(filename, "a")
                self._write_hdf(h5, name=name)
                h5.close()
            else:
                raise TypeError('Unknown file type')
        elif isinstance(filename, h5py.Group):
            self._write_hdf(filename, name=name)

    def _write_hdf(self, h5, name=None):
        node = super(VolumeData, self)._write_hdf(h5, name=name)
        node.attrs['xfmname'] = self.xfmname

        #write the mask into the file, as necessary
        if self._mask is not None:
            mask = self._mask
            if isinstance(self._mask, np.ndarray):
                mgrp = "/subjects/{subj}/transforms/{xfm}/masks/"
                mgrp = mgrp.format(subj=self.subject, xfm=self.xfmname)
                mname = "__%s" % _hash(self._mask)[:8]
                _hdf_write(h5, self._mask, name=mname, group=mgrp)
                mask = mname

            node.attrs['mask'] = mask

        return node


class VertexData(VolumeData):
    def __init__(self, data, subject):
        """Represents `data` at each vertex on a `subject`s cortex.
        `data` shape possibilities:

        raw linear movie: (t, v, c)
        reg linear movie: (t, v)
        raw linear image: (v, c)
        reg linear image: (v,)

        where t is the number of time points, c is colors (i.e. RGB), and v is the
        number of vertices (either in both hemispheres or one hemisphere).
        """
        try:
            basestring
        except NameError:
            subject = subject if isinstance(subject, str) else subject.decode('utf-8')
        self.subject = subject

        left, right = surfs.getSurf(self.subject, "fiducial")
        self.llen = len(left[0])
        self.rlen = len(right[0])
        self._set_data(data)

    def _set_data(self, data):
        """Stores data for this VertexData. Also sets flags if `data` appears to
        be in 'movie' or 'raw' format. See __init__ for `data` shape possibilities.
        """
        self._data = data
        
        self.movie = False
        self.raw = data.dtype == np.uint8
        if data.ndim == 3:
            self.movie = True
            if not self.raw:
                raise ValueError('Invalid data shape')
        elif data.ndim == 2:
            self.movie = not self.raw

        self.nverts = self.data.shape[-2 if self.raw else -1]
        if self.llen == self.nverts:
            # Just data for left hemisphere
            self.hem = "left"
            rshape = list(self.data.shape)
            rshape[1 if self.movie else 0] = self.rlen
            self._data = np.hstack([self.data, np.zeros(rshape, dtype=self.data.dtype)])
        elif self.rlen == self.nverts:
            # Just data for right hemisphere
            self.hem = "right"
            lshape = list(self.data.shape)
            lshape[1 if self.movie else 0] = self.llen
            self._data = np.hstack([np.zeros(lshape, dtype=self.data.dtype), self.data])
        elif self.llen + self.rlen == self.nverts:
            # Data for both hemispheres
            self.hem = "both"
        else:
            raise ValueError('Invalid number of vertices for subject')

    def copy(self, data=None):
        """Copies this VertexData. Uses __new__ to avoid expensive initialization that
        involves loading the surface from disk. Can also be used to cheaply create a
        new VertexData object for a subject with new `data`, if supplied.
        """
        newvd = self.__class__.__new__(self.__class__)
        newvd.subject = self.subject
        newvd.attrs = self.attrs
        newvd.llen = self.llen
        newvd.rlen = self.rlen
        
        if newdata is None:
            newvd._set_data(self.data)
        else:
            newvd._set_data(data)

        return newvd

    def _check_size(self):
        raise NotImplementedError

    def volume(self, xfmname, projection='nearest', **kwargs):
        import warnings
        warnings.warn('Inverse mapping cannot be accurate')
        from .. import utils
        mapper = utils.get_mapper(self.subject, xfmname, projection)
        return mapper.backwards(self, **kwargs)

    def __repr__(self):
        maskstr = ''
        if self.raw:
            maskstr += " raw"
        if self.movie:
            maskstr += " movie"
        return "<%s vertex data for %s>"%(maskstr, self.subject)

    def __getitem__(self, idx):
        if not self.movie:
            raise TypeError("Cannot index non-movie data")
        
        #return VertexData(self.data[idx], self.subject, **self.attrs)
        return self.copy(self.data[idx])

    @property
    def vertices(self):
        if self.raw and self.data.shape[-1] < 4:
            shape = (1,)+self.data.shape[::-1][1:]
            return np.vstack([self.data.T, 255*np.ones(shape, dtype=np.uint8)]).T
        return self.data

    @property
    def left(self):
        if self.movie:
            return self.vertices[:,:self.llen]
        else:
            return self.vertices[:self.llen]

    @property
    def right(self):
        if self.movie:
            return self.vertices[:,self.llen:]
        else:
            return self.vertices[self.llen:]

def _find_mask(nvox, subject, xfmname):
    import os
    import re
    import glob
    import nibabel
    files = surfs.getFiles(subject)['masks'].format(xfmname=xfmname, type="*")
    for fname in glob.glob(files):
        nib = nibabel.load(fname)
        mask = nib.get_data().T != 0
        if nvox == np.sum(mask):
            fname = os.path.split(fname)[1]
            name = re.compile(r'mask_([\w]+).nii.gz').search(fname)
            return name.group(1), mask

    raise ValueError('Cannot find a valid mask')


class _masker(object):
    def __init__(self, ds):
        self.ds = ds

        self.data = None
        if ds.linear:
            self.data = ds.data

    def __getitem__(self, masktype):
        s, x = self.ds.subject, self.ds.xfmname
        mask = surfs.getMask(s, x, masktype)
        if self.ds.movie:
            return VolumeData(self.ds.volume[:,mask], s, x, mask=masktype)
        return VolumeData(self.ds.volume[mask], s, x, mask=masktype)

def _hash(array):
    '''A simple numpy hash function'''
    return hashlib.sha1(array.astype(np.uint8)).hexdigest()

def _hdf_write(h5, data, name="data", group="/data"):
    try:
        node = h5.require_dataset("%s/%s"%(group, name), data.shape, data.dtype, exact=True)
    except TypeError:
        del h5[group][name]
        node = h5.create_dataset("%s/%s"%(group, name), data.shape, data.dtype, exact=True)

    node[:] = data
    return node