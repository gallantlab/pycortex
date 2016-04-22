import hashlib
import numpy as np
import h5py

from ..database import db

class BrainData(object):
    def __init__(self, data, subject, **kwargs):
        if isinstance(data, str):
            import nibabel
            nib = nibabel.load(data)
            data = nib.get_data().T
        self._data = data
        try:
            basestring
        except NameError:
            subject = subject if isinstance(subject, str) else subject.decode('utf-8')
        self.subject = subject
        super(BrainData, self).__init__(**kwargs)

    @property
    def data(self):
        if isinstance(self._data, h5py.Dataset):
            return self._data.value
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def name(self):
        '''Name of this BrainData, according to its hash'''
        return "__%s"%_hash(self.data)[:16]

    def exp(self):
        """Copy of this object with data exponentiated.
        """
        return self.copy(np.exp(self.data))

    def uniques(self, collapse=False):
        yield self

    def __hash__(self):
        return hash(_hash(self.data))

    def _write_hdf(self, h5, name=None):
        if name is None:
            name = self.name

        dgrp = h5.require_group("/data")
        if name in dgrp and "__%s"%_hash(dgrp[name].value)[:16] == name:
            #don't need to update anything, since it's the same data
            return h5.get("/data/%s"%name)

        node = _hdf_write(h5, self.data, name=name)
        node.attrs['subject'] = self.subject
        return node

    def to_json(self, simple=False):
        sdict = super(BrainData, self).to_json(simple=simple)
        if simple:
            sdict.update(dict(name=self.name,
                subject=self.subject,
                min=float(np.nan_to_num(self.data).min()), 
                max=float(np.nan_to_num(self.data).max()),
                ))
        return sdict

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
    def __init__(self, data, subject, xfmname, mask=None, **kwargs):
        """Three possible variables: volume, movie, vertex. Enumerated with size:
        volume movie: (t, z, y, x)
        volume image: (z, y, x)
        linear movie: (t, v)
        linear image: (v,)
        """
        if self.__class__ == VolumeData:
            raise TypeError('Cannot directly instantiate VolumeData objects')
        super(VolumeData, self).__init__(data, subject, **kwargs)
        try:
            basestring
        except NameError:
            xfmname = xfmname if isinstance(xfmname, str) else xfmname.decode('utf-8')
        self.xfmname = xfmname

        self._check_size(mask)
        self.masked = _masker(self)

    def to_json(self, simple=False):
        if simple:
            sdict = super(VolumeData, self).to_json(simple=simple)
            sdict["shape"] = self.shape
            return sdict
        
        xfm = db.get_xfm(self.subject, self.xfmname, 'coord').xfm
        sdict = dict(xfm=[list(np.array(xfm).ravel())], data=[self.name])
        sdict.update(super(VolumeData, self).to_json())
        return sdict

    @classmethod
    def empty(cls, subject, xfmname, value=0, **kwargs):
        xfm = db.get_xfm(subject, xfmname)
        shape = xfm.shape
        return cls(np.ones(shape)*value, subject, xfmname, **kwargs)

    @classmethod
    def random(cls, subject, xfmname, **kwargs):
        xfm = db.get_xfm(subject, xfmname)
        shape = xfm.shape
        return cls(np.random.randn(*shape), subject, xfmname, **kwargs)

    def _check_size(self, mask):
        if self.data.ndim not in (1, 2, 3, 4):
            raise ValueError("Invalid data shape")
        
        self.linear = self.data.ndim in (1, 2)
        self.movie = self.data.ndim in (2, 4)

        if self.linear:
            #Guess the mask
            if mask is None:
                nvox = self.data.shape[-1]
                self._mask, self.mask = _find_mask(nvox, self.subject, self.xfmname)
            elif isinstance(mask, str):
                self.mask = db.get_mask(self.subject, self.xfmname, mask)
                self._mask = mask
            elif isinstance(mask, np.ndarray):
                self.mask = mask > 0
                self._mask = mask > 0

            self.shape = self.mask.shape
        else:
            self._mask = None
            shape = self.data.shape
            if self.movie:
                shape = shape[1:]
            xfm = db.get_xfm(self.subject, self.xfmname)
            if xfm.shape != shape:
                raise ValueError("Volumetric data (shape %s) is not the same shape as reference for transform (shape %s)" % (str(shape), str(xfm.shape)))
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
        if self.movie:
            maskstr += " movie"
        maskstr = maskstr[0].upper()+maskstr[1:]
        return "<%s data for (%s, %s)>"%(maskstr, self.subject, self.xfmname)

    def copy(self, data):
        return super(VolumeData, self).copy(data, self.subject, self.xfmname, mask=self._mask)

    @property
    def volume(self):
        """Standardizes the VolumeData, ensuring that masked data are unmasked"""
        from .. import volume
        if self.linear:
            data = volume.unmask(self.mask, self.data[:])
        else:
            data = self.data[:]

        if not self.movie:
            data = data[np.newaxis]

        return data

    def save(self, filename, name=None):
        """Save the dataset into an hdf file with the provided name
        """
        import os
        if isinstance(filename, str):
            fname, ext = os.path.splitext(filename)
            if ext in (".hdf", ".h5",".hf5"):
                h5 = h5py.File(filename, "a")
                self._write_hdf(h5, name=name)
                h5.close()
            else:
                raise TypeError('Unknown file type')
        elif isinstance(filename, h5py.Group):
            self._write_hdf(filename, name=name)

    def _write_hdf(self, h5, name=None):
        node = super(VolumeData, self)._write_hdf(h5, name=name)
        
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

    def save_nii(self, filename):
        """Save as a nifti file at the given filename. Nifti headers are
        copied from the reference nifti file.
        """
        xfm = db.get_xfm(self.subject, self.xfmname)
        affine = xfm.reference.get_affine()
        import nibabel
        new_nii = nibabel.Nifti1Image(self.volume.T, affine)
        nibabel.save(new_nii, filename)

class VertexData(BrainData):
    def __init__(self, data, subject, **kwargs):
        """Represents `data` at each vertex on a `subject`s cortex.
        `data` shape possibilities:

        reg linear movie: (t, v)
        reg linear image: (v,)
        None: creates zero-filled VertexData

        where t is the number of time points, c is colors (i.e. RGB), and v is the
        number of vertices (either in both hemispheres or one hemisphere).
        """
        if self.__class__ == VertexData:
            raise TypeError('Cannot directly instantiate VertexData objects')
        super(VertexData, self).__init__(data, subject, **kwargs)
        try:
            left, right = db.get_surf(self.subject, "wm")
        except IOError:
            left, right = db.get_surf(self.subject, "fiducial")
        self.llen = len(left[0])
        self.rlen = len(right[0])
        self._set_data(data)

    @classmethod
    def empty(cls, subject, value=0, **kwargs):
        try:
            left, right = db.get_surf(subject, "wm")
        except IOError:
            left, right = db.get_surf(subject, "fiducial")
        nverts = len(left[0]) + len(right[0])
        return cls(np.ones((nverts,))*value, subject, **kwargs)

    @classmethod
    def random(cls, subject, **kwargs):
        try:
            left, right = db.get_surf(subject, "wm")
        except IOError:
            left, right = db.get_surf(subject, "fiducial")
        nverts = len(left[0]) + len(right[0])
        return cls(np.random.randn(nverts), subject, **kwargs)

    def _set_data(self, data):
        """Stores data for this VertexData. Also sets flags if `data` appears to
        be in 'movie' or 'raw' format. See __init__ for `data` shape possibilities.
        """
        if data is None:
            data = np.zeros((self.llen + self.rlen,))
        
        self._data = data
        self.movie = self.data.ndim > 1
        self.nverts = self.data.shape[-1]
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
            raise ValueError('Invalid number of vertices for subject (given %d, should be %d for left hem, %d for right hem, or %d for both)' % (self.nverts, self.llen, self.rlen, self.llen+self.rlen))

    def copy(self, data):
        return super(VertexData, self).copy(data, self.subject)

    def volume(self, xfmname, projection='nearest', **kwargs):
        import warnings
        warnings.warn('Inverse mapping cannot be accurate')
        from .. import utils
        mapper = utils.get_mapper(self.subject, xfmname, projection)
        return mapper.backwards(self, **kwargs)

    def __repr__(self):
        maskstr = ""
        if self.movie:
            maskstr = "movie "
        return "<Vertex %sdata for %s>"%(maskstr, self.subject)

    def __getitem__(self, idx):
        if not self.movie:
            raise TypeError("Cannot index non-movie data")
        
        #return VertexData(self.data[idx], self.subject, **self.attrs)
        return self.copy(self.data[idx])

    def to_json(self, simple=False):
        if simple:
            sdict = dict(split=self.llen, frames=self.vertices.shape[0])
            sdict.update(super(VertexData, self).to_json(simple=simple))
            return sdict
            
        sdict = dict(data=[self.name])
        sdict.update(super(VertexData, self).to_json())
        return sdict

    @property
    def vertices(self):
        verts = self.data
        if not self.movie:
            verts = verts[np.newaxis]
        return verts

    @property
    def left(self):
        if self.movie:
            return self.data[:,:self.llen]
        else:
            return self.data[:self.llen]

    @property
    def right(self):
        if self.movie:
            return self.data[:,self.llen:]
        else:
            return self.data[self.llen:]

def _find_mask(nvox, subject, xfmname):
    import os
    import re
    import glob
    import nibabel
    files = db.get_paths(subject)['masks'].format(xfmname=xfmname, type="*")
    for fname in glob.glob(files):
        nib = nibabel.load(fname)
        mask = nib.get_data().T != 0
        if nvox == np.sum(mask):
            fname = os.path.split(fname)[1]
            name = re.compile(r'mask_([\w]+).nii.gz').search(fname)
            return name.group(1), mask

    raise ValueError('Cannot find a valid mask')


class _masker(object):
    def __init__(self, dv):
        self.dv = dv

        self.data = None
        if dv.linear:
            self.data = dv.data

    def __getitem__(self, masktype):
        try:
            mask = db.get_mask(self.dv.subject, self.dv.xfmname, masktype)
            return self.dv.copy(self.dv.volume[:,mask].squeeze())
        except:
            self.dv.copy(self.dv.volume[:, mask].squeeze())

def _hash(array):
    '''A simple numpy hash function'''
    return hashlib.sha1(array.tostring()).hexdigest()

def _hdf_write(h5, data, name="data", group="/data"):
    try:
        node = h5.require_dataset("%s/%s"%(group, name), data.shape, data.dtype, exact=True)
    except TypeError:
        del h5[group][name]
        node = h5.create_dataset("%s/%s"%(group, name), data.shape, data.dtype, exact=True)

    node[:] = data
    return node
