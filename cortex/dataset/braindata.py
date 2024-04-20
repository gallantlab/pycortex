import hashlib
from copy import deepcopy

import h5py
import numpy as np

from ..database import db


class BrainData(object):
    """
    Abstract base class for brain data.

    Parameters
    ----------
    data : ndarray or str
        The data array (size depends on specific use case) or path to file 
        readable by nibabel.
    subject : str
        Subject identifier. Must exist in the pycortex database.
    """
    def __init__(self, data, subject, **kwargs):
        if isinstance(data, str):
            import nibabel
            nib = nibabel.load(data)
            data = nib.get_fdata().T
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
            return self._data[()]
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def name(self):
        """Name of this BrainData, computed from hash of data.
        TODO:WHAT IS THIS USEFUL FOR
        """
        return "__%s"%_hash(self.data)[:16]

    def exp(self):
        """Return copy of this brain data with data exponentiated.
        """
        return self.copy(np.exp(self.data))

    def uniques(self, collapse=False):
        """TODO: WHAT IS THIS
        """
        yield self

    def __hash__(self):
        return hash(_hash(self.data))

    def _write_hdf(self, h5, name=None):
        if name is None:
            name = self.name
        dgrp = h5.require_group("/data")
        
        if name in dgrp and "__%s" % _hash(dgrp[name][()])[:16] == name:
            #don't need to update anything, since it's the same data
            return h5.get("/data/%s"%name)

        node = _hdf_write(h5, self.data, name=name)
        node.attrs['subject'] = self.subject
        return node

    def to_json(self, simple=False):
        """Creates JSON description of this brain data.
        """
        sdict = super(BrainData, self).to_json(simple=simple)
        if simple:
            sdict.update(dict(name=self.name,
                subject=self.subject,
                min=float(np.nan_to_num(self.data).min()), 
                max=float(np.nan_to_num(self.data).max()),
                ))
        return sdict

    @classmethod
    def _add_numpy_methods(cls):
        """Adds numpy operator methods (+, -, etc.) to this class to allow
        simple manipulation of the data, e.g. with VolumeData v:
        v + 1 # Returns new VolumeData with 1 added to data
        v ** 2 # Returns new VolumeData with data squared
        """
        # Binary operations
        npops = ["__add__", "__sub__", "__mul__", "__floordiv__", "__truediv__",
                 "__div__", "__pow__", "__neg__", "__abs__"]

        def make_opfun(op): # function nesting creates closure containing op
            def opfun(self, *args):
                return self.copy(getattr(self.data, op)(*args))
            return opfun

        for op in npops:
            opfun = make_opfun(op)
            opfun.__name__ = op
            setattr(cls, opfun.__name__, opfun)

BrainData._add_numpy_methods()

class VolumeData(BrainData):
    """
    Abstract base class for all volumetric brain data.

    Parameters
    ----------
    data : ndarray
        The data. Can be 3D with shape (z,y,x), 1D with shape (v,) for masked data,
        4D with shape (t,z,y,x), or 2D with shape (t,v). For masked data, if the
        size of the given array matches any of the existing masks in the database,
        that mask will automatically be loaded. If it does not, an error will be 
        raised.
    subject : str
        Subject identifier. Must exist in the pycortex database.
    xfmname : str
        Transform name. Must exist in the pycortex database.
    mask : ndarray, optional
        Binary 3D array with shape (z,y,x) showing which voxels are selected.
        If masked data is given, the mask will automatically be loaded if it 
        exists in the pycortex database.
    **kwargs
        Other keyword arguments are passed to superclass inits.
    """
    def __init__(self, data, subject, xfmname, mask=None, **kwargs):
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
        """Creates JSON description of this brain data.
        """
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
        """
        Create a constant-valued VolumeData for the given subject and xfmname.
        Often useful for testing purposes.

        Parameters
        ----------
        subject : str
            Subject identifier. Must exist in the pycortex database.
        xfmname : str
            Transform name. Must exist in the pycortex database.
        value : float, optional
            Value that the VolumeData will be filled with.
        **kwargs
            Other keyword arguments are passed to the init function for this 
            class.

        Returns
        -------
        VolumeData subclass
            A VolumeData subclass object whose data is constant, equal to value.
        """
        xfm = db.get_xfm(subject, xfmname)
        shape = xfm.shape
        return cls(np.ones(shape)*value, subject, xfmname, **kwargs)

    @classmethod
    def random(cls, subject, xfmname, **kwargs):
        """
        Create a random-valued VolumeData for the given subject and xfmname.
        Random values are from gaussian distribution with mean 0, s.d. 1.
        Often useful for testing purposes.

        Parameters
        ----------
        subject : str
            Subject identifier. Must exist in the pycortex database.
        xfmname : str
            Transform name. Must exist in the pycortex database.
        **kwargs
            Other keyword arguments are passed to the init function for this 
            class.

        Returns
        -------
        VolumeData subclass
            A VolumeData subclass object whose data is random.
        """
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
            elif isinstance(mask, np.ndarray):
                self.mask = mask > 0
                self._mask = mask > 0
            else:
                self.mask = db.get_mask(self.subject, self.xfmname, mask)
                self._mask = mask

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
        """Convert this VolumeData into VertexData using the given projection 
        method.

        Parameters
        ----------
        projection : str, optional
            Type of projection to use. Default: nearest.

        Returns
        -------
        VertexData subclass
            Vertex valued version of this VolumeData.
        """
        from cortex import utils
        mapper = utils.get_mapper(self.subject, self.xfmname, projection)
        data = mapper(self)
        # Note: this is OK, because VolumeRGB and Volume2D objects (which
        # have different requirements for vmin, vmax, cmap) do not inherit
        # from VolumeData, and so do not have this method.
        data.vmin = self.vmin
        data.vmax = self.vmax
        data.cmap = self.cmap
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
        """Returns a 3D or 4D volume for this VolumeData, automatically unmasking
        masked data.
        """
        from cortex import volume
        if self.linear:
            data = volume.unmask(self.mask, self.data[:])
        else:
            data = self.data[:]

        if not self.movie:
            data = data[np.newaxis]

        return data

    def save(self, filename, name=None):
        """Save the dataset into the hdf file `filename` with the provided name.
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
        copied from the reference image for this VolumeData's transform.
        """
        xfm = db.get_xfm(self.subject, self.xfmname)
        affine = xfm.reference.affine
        import nibabel
        new_nii = nibabel.Nifti1Image(self.volume.T, affine)
        nibabel.save(new_nii, filename)

class VertexData(BrainData):
    """
    Abstract base class for all vertex-wise brain data (i.e. in surface space).

    Parameters
    ----------
    data : ndarray
        The data. Can be 1D with shape (v,), or 2D with shape (t,v). Here, v can
        be the number of vertices in both hemispheres, or the number of vertices
        in either one of the hemispheres. In that case, the data for the other 
        hemisphere will be filled with zeros.
    subject : str
        Subject identifier. Must exist in the pycortex database.
    **kwargs
        Other keyword arguments are passed to the superclass init function.
    """
    def __init__(self, data, subject, **kwargs):
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
        """
        Create a constant-valued VertexData for the given subject.
        Often useful for testing purposes.

        Parameters
        ----------
        subject : str
            Subject identifier. Must exist in the pycortex database.
        value : float, optional
            Value that the VertexData will be filled with.
        **kwargs
            Other keyword arguments are passed to the init function for this 
            class.

        Returns
        -------
        VertexData subclass
            A VertexData subclass object whose data is constant, equal to value.
        """
        try:
            left, right = db.get_surf(subject, "wm")
        except IOError:
            left, right = db.get_surf(subject, "fiducial")
        nverts = len(left[0]) + len(right[0])
        return cls(np.ones((nverts,))*value, subject, **kwargs)

    @classmethod
    def random(cls, subject, **kwargs):
        """
        Create a random-valued VertexData for the given subject.
        Random values are from gaussian distribution with mean 0, s.d. 1.
        Often useful for testing purposes.

        Parameters
        ----------
        subject : str
            Subject identifier. Must exist in the pycortex database.
        **kwargs
            Other keyword arguments are passed to the init function for this 
            class.

        Returns
        -------
        VertexData subclass
            A VertexData subclass object with random data.
        """
        try:
            left, right = db.get_surf(subject, "wm")
        except IOError:
            left, right = db.get_surf(subject, "fiducial")
        nverts = len(left[0]) + len(right[0])
        return cls(np.random.randn(nverts), subject, **kwargs)

    def _set_data(self, data):
        """
        Stores data for this VertexData. Also sets flags if `data` appears to
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
        """
        Return a new VertexData object for the same subject but with data
        replaced by the given `data`. 

        This is useful for efficiently creating many VertexData objects, since 
        it doesn't require reloading the surfaces from the database to check 
        numbers of vertices, etc.
        """
        return super(VertexData, self).copy(data, self.subject)

    def volume(self, xfmname, projection='nearest', **kwargs):
        """
        Map this VertexData back to volume space, creating a VolumeData object.
        This uses the `mapper.backwards` function, which is not particularly
        accurate.

        Parameters
        ----------
        xfmname : str
            Transform name for the volume space that this vertex data will be 
            projected into. Must exist in the pycortex database.
        projection : str, optional
            The type of projection method to use. See the docs for `mapper` for
            possibilities. Default: nearest.
        **kwargs 
            Other keyword args are passed to the `mapper.backwards` function.

        Returns
        -------
        VolumeData
            Volume containing the back-projected vertex data.
        """
        import warnings
        warnings.warn('Inverse mapping cannot be accurate')
        from cortex import utils
        mapper = utils.get_mapper(self.subject, xfmname, projection)
        return mapper.backwards(self, **kwargs)

    def __repr__(self):
        maskstr = ""
        if self.movie:
            maskstr = "movie "
        return "<Vertex %sdata for %s>"%(maskstr, self.subject)

    def __getitem__(self, idx):
        """Get the VertexData for the given time index. Only works for movie (2D)
        vertex data.
        """
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
        """Data for only the left hemisphere vertices.
        """
        if self.movie:
            return self.data[:,:self.llen]
        else:
            return self.data[:self.llen]

    @property
    def right(self):
        """Data for only the right hemisphere vertices.
        """
        if self.movie:
            return self.data[:,self.llen:]
        else:
            return self.data[self.llen:]

    def blend_curvature(self, alpha, threshold=0, brightness=0.5,
                        contrast=0.25, smooth=20):
        """Blend the data with a curvature map depending on a transparency map.
        
        Vertex objects cannot use transparency as Volume objects. This method
        is a hack to mimic the transparency of Volume objects, blending the
        Vertex data with a curvature map. This method returns a VertexRGB
        object, and the colormap parameters (vmin, vmax, cmap, ...) of the
        original Vertex object cannot be changed later on.

        Parameters
        ----------
        alpha : array of shape (n_vertices, )
            Transparency map.
        threshold : float
            Threshold for the curvature map.
        brightness : float
            Brightness of the curvature map.
        contrast : float
            Contrast of the curvature map.
        smooth : float
            Smoothness of the curvature map.
        
        Returns
        -------
        blended : VertexRGB object
            The original map blended with a curvature map.
        """
        from .views import Vertex
        # prepare curvature map
        curvature = db.get_surfinfo(self.subject, smooth=smooth).data
        curvature = (curvature > threshold).astype("float")
        curvature = curvature * contrast + brightness
        curvature_raw = Vertex(curvature, self.subject, vmin=0, vmax=1,
                               cmap="gray").raw

        # prepare alpha map
        alpha = np.clip(alpha.astype("float"), 0, 1)

        # blend original map with curvature map
        blended = deepcopy(self.raw)  # copy because VertexRGB.raw returns self
        blended.red.data = blended.red.data * alpha + (1 - alpha) * curvature_raw.red.data
        blended.green.data = blended.green.data * alpha + (1 - alpha) * curvature_raw.green.data
        blended.blue.data = blended.blue.data * alpha + (1 - alpha) * curvature_raw.blue.data
        blended.red.data = blended.red.data.astype("uint8")
        blended.green.data = blended.green.data.astype("uint8")
        blended.blue.data = blended.blue.data.astype("uint8")

        return blended


def _find_mask(nvox, subject, xfmname):
    import glob
    import os
    import re

    import nibabel
    files = db.get_paths(subject)['masks'].format(xfmname=xfmname, type="*")
    for fname in glob.glob(files):
        nib = nibabel.load(fname)
        mask = nib.get_fdata().T != 0
        if nvox == np.sum(mask):
            fname = os.path.split(fname)[1]
            name = re.compile(r'mask_(.+).nii.gz').search(fname)
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
