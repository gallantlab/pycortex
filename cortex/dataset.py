"""Module for maintaining brain data and their masks

Three basic classes, with child classes:
  1. Dataset
  2. BrainData
    a. VolumeData
    b. VertexData
  3. View
    a. DataView

Dataset holds a collection of View and BrainData objects. It provides a thin
wrapper around h5py to store data. Datasets will store all View and BrainData
objects into the h5py file, reconstituting each when requested.


"""
import os
import hashlib
import tempfile
import numpy as np
import h5py

from .db import surfs
from .xfm import Transform
from . import volume
from . import utils

class Dataset(object):
    def __init__(self, **kwargs):
        self.subjects = {}
        self.views = {}
        self.data = {}

        self.append(**kwargs)

    def append(self, **kwargs):
        for name, data in kwargs.items():
            norm = normalize(data)

            if isinstance(norm, BrainData):
                self.data[hash(norm)] = norm
                self.views[name] = DataView(norm)
            elif isinstance(norm, Dataset):
                self.views.update(norm.views)
                self.data.update(norm.data)

        return self

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif attr in self.views:
            return self.views[attr]

        raise AttributeError

    def __getitem__(self, item):
        return self.views[item]

    def __iter__(self):
        for name, dv in sorted(self.views.items(), key=lambda x: x[1].priority):
            yield name, dv

    def __repr__(self):
        views = sorted(self.views.items(), key=lambda x: x[1].priority)
        return "<Dataset with views [%s]>"%(', '.join([n for n, d in views]))

    def __len__(self):
        return len(self.views)

    def __dir__(self):
        return list(self.__dict__.keys()) + list(self.views.keys())

    @classmethod
    def from_file(cls, filename):
        ds = cls()
        views = set()
        ds.h5 = h5py.File(filename)
        for name, node in ds.h5['data']:
            data = BrainData.from_file(ds, node)
            self.data[hash(data)] = data

        for name, node in ds.h5['views']:
            pass

    def save(self, filename, pack=False):
        raise NotImplementedError

    def getSurf(self, subject, type, hemi='both', merge=False, nudge=False):
        if hemi == 'both':
            left = self.getSurf(subject, type, "lh")
            right = self.getSurf(subject, type, "rh")
            if merge:
                pts = np.vstack([left[0], right[0]])
                polys = np.vstack([left[1], right[1]+len(left[0])])
                return pts, polys

            return left, right
        try:
            group = self.h5['subjects'][subject]['surfaces'][type][hemi]
            pts, polys = group.pts[:], group.polys[:]
            if nudge:
                if hemi == 'lh':
                    pts[:,0] -= pts[:,0].min()
                else:
                    pts[:,0] -= pts[:,0].max()
            return pts, polys
        except KeyError:
            raise IOError('Subject not found in package')

    def getXfm(self, subject, xfmname):
        try:
            group = self.h5['subjects'][subject]['transforms'][xfmname]
            return Transform(group.xfm[:], group.attrs.shape)
        except KeyError:
            raise IOError('Transform not found in package')

    def getMask(self, subject, xfmname, maskname):
        try:
            group = self.h5['subjects'][subject]['transforms'][xfmname]['masks']
            return group[maskname]
        except KeyError:
            raise IOError('Mask not found in package')

    def getOverlay(self, subject, type='rois', **kwargs):
        try:
            group = self.h5['subjects'][subject]
            if type == "rois":
                import tempfile
                tf = tempfile.NamedTemporaryFile()
                tf.write(group['rois'].value)
                tf.seek(0)
                return tf
        except KeyError:
            raise IOError('Overlay not found in package')

        raise TypeError('Unknown overlay type')

    def prepend(self, prefix):
        ds = dict()
        for name, data in self:
            ds[prefix+name] = data

        return Dataset(**ds)

class BrainData(object):
    def __init__(self, data):
        if isinstance(data, str):
            import nibabel
            nib = nibabel.load(data)
            data = nib.get_data().T
        self.data = data

    def exp(self):
        """Copy of this object with data exponentiated.
        """
        return self.copy(np.exp(self.data))

    def __hash__(self):
        return hashlib.sha1(self.data.view(np.uint8)).hexdigest()

    def _write_hdf(self, h5, name="data"):
        node = _hdf_write(h5['data'], name, self.data)
        node.attrs['subject'] = self.subject
        for name, value in self.attrs.items():
            node.attrs[name] = value
        return node

    @staticmethod
    def from_hdf(dataset, node):
        subj = node.attrs['subject']
        if "xfmname" in node.attrs:
            xfmname = node.attrs['xfmname']
            mask = dataset.getMask(node.attrs['mask'])
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

    def copy(self, newdata=None):
        """Copies this VolumeData.
        """
        if newdata is None:
            return VolumeData(self.data, self.subject, self.xfmname, mask=self._mask, **self.attrs)
        else:
            return VolumeData(newdata, self.subject, self.xfmname, mask=self._mask, **self.attrs)

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
        mapper = utils.get_mapper(self.subject, self.xfmname, projection)
        data = mapper(self)
        data.attrs['projection'] = (self.xfmname, projection)
        return data

    def __repr__(self):
        maskstr = "volumetric"
        if self.linear:
            maskstr = "%s masked"%self.masktype
        if self.raw:
            maskstr += " raw"
        if self.movie:
            maskstr += " movie"
        maskstr = maskstr[0].upper()+maskstr[1:]
        return "<%s data for (%s, %s)>"%(maskstr, self.subject, self.xfmname)

    def __getitem__(self, idx):
        if not self.movie:
            raise ValueError('Cannot index non-movie data')
        return VolumeData(self.data[idx], self.subject, self.xfmname, mask=self._mask, **self.attrs)

    @property
    def volume(self):
        """Standardizes the VolumeData, ensuring that masked data are unmasked"""
        if self.linear:
            data = volume.unmask(self.mask, self.data[:])
        else:
            data = self.data[:]

        if self.raw and data.shape[-1] == 3:
            #stack the alpha dimension
            alpha = 255*np.ones(data.shape[:3]+(1,)).astype(np.uint8)
            data = np.concatenate([data, alpha], axis=-1)

        return data

    def save(self, filename, name="data"):
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

    def _write_hdf(self, h5, name="data"):
        node = super(VolumeData, self)._write_hdf(h5, name=name)

        #write the mask into the file, as necessary
        mask = self._mask
        if isinstance(self._mask, np.ndarray):
            mgrp = h5.file['subjects'][self.subject]['transforms'][self.xfmname]['masks']
            mname = "__%s" % hashlib.sha1(self._mask.view(np.uint8)).hexdigest()[:8]
            _hdf_write(mgrp, mname, self._mask)
            mask = mname

        node.attrs['xfmname'] = self.xfmname
        node.attrs['mask'] = mask

        return node


class VertexData(VolumeData):
    def __init__(self, data, subject):
        """Vertex Data possibilities

        raw linear movie: (t, v, c)
        reg linear movie: (t, v)
        raw linear image: (v, c)
        reg linear image: (v,)

        where t is the number of time points, c is colors (i.e. RGB), and v is the
        number of vertices (either in both hemispheres or one hemisphere)
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
        """Sets the data of this VertexData. Also sets flags if the data appears to
        be in 'movie' or 'raw' format. See __init__ for data shape possibilities.
        """
        self.data = data
        
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
            self.data = np.hstack([self.data, np.zeros(rshape, dtype=self.data.dtype)])
        elif self.rlen == self.nverts:
            # Just data for right hemisphere
            self.hem = "right"
            lshape = list(self.data.shape)
            lshape[1 if self.movie else 0] = self.llen
            self.data = np.hstack([np.zeros(lshape, dtype=self.data.dtype), self.data])
        elif self.llen + self.rlen == self.nverts:
            # Data for both hemispheres
            self.hem = "both"
        else:
            raise ValueError('Invalid number of vertices for subject')

    def copy(self, newdata=None):
        """Copies this VertexData. Uses __new__ to avoid expensive initialization.
        """
        newvd = self.__class__.__new__(self.__class__)
        newvd.subject = self.subject
        newvd.attrs = self.attrs
        newvd.llen = self.llen
        newvd.rlen = self.rlen
        
        if newdata is None:
            newvd._set_data(self.data)
        else:
            newvd._set_data(newdata)

        return newvd

    def _check_size(self):
        raise NotImplementedError

    def volume(self, xfmname, projection='nearest', **kwargs):
        import warnings
        warnings.warn('Inverse mapping cannot be accurate')
        from . import utils
        mapper = utils.get_mapper(self.subject, xfmname, projection)
        return mapper.backwards(self, **kwargs)

    def __repr__(self):
        maskstr = ''
        if 'projection' in self.attrs:
            maskstr = '(%s, %s) mapped'%self.attrs['projection']

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

class View(object):
    indices = ('data', 'description', 'cmap', 'vmin', 'vmax' 'state', 'animation')
    def __init__(self, cmap=None, vmin=None, vmax=None, state=None):
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.state = state
        self.priority = 0

    def __call__(self, data):
        return DataView(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, state=self.state)

class DataView(View):
    def __init__(self, data, cmap=None, vmin=None, vmax=None, description="", **kwargs):
        self.data = data
        self.description = description
        super(DataView, self).__init__(cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    def _write_hdf(self, h5, idx=None):
        views = h5.get("/views")
        datas = h5.get("/data")
        if idx is None:
            ds.resize(len(ds)+1, axis=0)
        ds[idx, 0] = json.dumps(self.data)
        ds[idx, 1] = self.description
        ds[idx, 2] = self.cmap
        ds[idx, 3] = json.dumps(self.vmin)
        ds[idx, 4] = json.dumps(self.vmax)
        ds[idx, 5] = json.dumps(self.state)

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

def normalize(data):
    if isinstance(data, (Dataset, View)):
        return data
    elif isinstance(data, dict):
        return Dataset(**data)
    elif isinstance(data, str):
        return Dataset.from_file(data)
    elif isinstance(data, tuple):
        if len(data) == 3:
            return DataView(VolumeData(*data))
        else:
            return DataView(VertexData(*data))

    raise TypeError('Unknown input type')

def _from_file(filename, name="data"):
    import tables
    if isinstance(filename, str):
        fname, ext = os.path.splitext(filename)
        if ext in (".hdf", ".h5"):
            h5 = tables.openFile(filename)
            node = h5.getNode("/datasets", name)
            data, attrs = _hdf_read(node)
            h5.close()
            if 'xfmname' in attrs:
                return VolumeData(data, **attrs)
            return VertexData(data, **attrs)
    elif isinstance(filename, tables.File):
        node = filename.getNode("/datasets", name)
        data, attrs = _hdf_read(node)
        if 'xfmname' in attrs:
            return VolumeData(data, **attrs)
        return VertexData(data, **attrs)

    raise TypeError('Unknown file type')

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

def _hdf_init(h5):
    if h5.get("/data") is None:
        h5.create_group("/data")
    if h5.get("/subjects") is None:
        h5.create_group("/subjects")
    if h5.get("/views") is None:
        h5.create_dataset("/views", (0,len(View.indices)), maxshape=(0, 10), 
            dtype=h5py.special_dtype(vlen=unicode)


def _hdf_write(h5, data, name="data", group="/data"):
    atom = tables.Atom.from_dtype(data.dtype)
    filt = tables.filters.Filters(complevel=9, complib='blosc', shuffle=True)
    create = False
    try:
        ds = h5.getNode("%s/%s"%(group, name))
        ds[:] = data
    except tables.NoSuchNodeError:
        create = True
    except ValueError:
        h5.removeNode("%s/%s"%(group, name))
        create = True

    if create:
        ds = h5.createCArray(group, name, atom, data.shape, filters=filt, createparents=True)
        ds[:] = data
        
    return ds

def _hdf_read(node):
    names = set(node.attrs._v_attrnames)
    names -= set(['CLASS', 'TITLE', 'VERSION'])
    attrs = dict((name, node.attrs[name]) for name in names)
    return node[:], attrs

def _pack_subjs(h5, subjects):
    for subject in subjects:
        rois = surfs.getOverlay(subject, type='rois')
        h5.createArray("/subjects", "rois", rois.toxml(pretty=False))
        surfaces = surfs.getFiles(subject)['surfs']
        for surf in surfaces.keys():
            for hemi in ("lh", "rh"):
                pts, polys = surfs.getSurf(subject, surf, hemi)
                group = "/subjects/%s/surfaces/%s/%s"%(subject, surf, hemi)
                _hdf_write(h5, pts, "pts", group)
                _hdf_write(h5, polys, "polys", group)

def _pack_xfms(h5, xfms):
    for subj, xfmname in xfms:
        xfm = surfs.getXfm(subj, xfmname, 'coord')
        group = "/subjects/%s/transforms/%s"%(subj, xfmname)
        node = _hdf_write(h5, np.array(xfm.xfm), "xfm", group)
        node.attrs.shape = xfm.shape

def _pack_masks(h5, masks):
    for subj, xfm, maskname in masks:
        mask = surfs.getMask(subj, xfm, maskname)
        group = "/subjects/%s/transforms/%s/masks"%(subj, xfm)
        _hdf_write(h5, mask, maskname, group)
