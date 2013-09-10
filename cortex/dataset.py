"""Module for maintaining brain data and their masks
"""
import os
import numpy as np

from .db import surfs
from .xfm import Transform
from . import volume
from . import utils

class Dataset(object):
    def __init__(self, **kwargs):
        self.subjects = {}
        self.datasets = {}
        self.views = []
        for name, data in kwargs.items():
            norm = normalize(data)
            if isinstance(norm, (VolumeData, VertexData)):
                self.datasets[name] = norm
            elif isinstance(norm, Dataset):
                self.datasets.update(norm.datasets)

    @classmethod
    def from_file(cls, filename):
        import tables
        datasets = dict()
        h5 = tables.openFile(filename)
        for node in h5.walkNodes("/datasets/"):
            if not isinstance(node, tables.Group):
                datasets[node.name] = _from_file(h5, name=node.name)

        ds = cls(**datasets)
        if len(h5.root.subjects._v_children.keys()):
            ds.subjects = h5.root.subjects
        else:
            h5.close()
        return ds

    def append(self, **kwargs):
        for name, data in kwargs.items():
            if isinstance(data, VolumeData):
                self.datasets[name] = data
            else:
                self.datasets[name] = VolumeData(*data)

        return self

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif attr in self.datasets:
            return self.datasets[attr]

        raise AttributeError

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.datasets[self.datasets.keys()[item]]
        return self.datasets[item]

    def __iter__(self):
        for name, ds in sorted(self.datasets.items(), key=lambda x: x[1].priority):
            yield name, ds

    def __repr__(self):
        datasets = sorted(self.datasets.items(), key=lambda x: x[1].priority)
        return "<Dataset with names [%s]>"%(', '.join([n for n, d in datasets]))

    def __len__(self):
        return len(self.datasets)

    def __add__(self, other):
        if isinstance(other, dict):
            other = Dataset(**other)

        if not isinstance(other, Dataset):
            return NotImplemented

        entries = self.datasets.copy()
        entries.update(other.datasets)
        return Dataset(**entries)


    def __dir__(self):
        return list(self.__dict__.keys()) + list(self.datasets.keys())

    def save(self, filename, pack=False):
        import tables
        h5 = tables.openFile(filename, "a")
        _hdf_init(h5)
        for name, data in self.datasets.items():
            data.save(h5, name=name)

        if pack:
            subjs = set()
            xfms = set()
            masks = set()
            for name, data in self.datasets.items():
                subjs.add(data.subject)
                xfms.add((data.subject, data.xfmname))
                if data.linear:
                    masks.add((data.subject, data.xfmname, data.masktype))

            _pack_subjs(h5, subjs)
            _pack_xfms(h5, xfms)
            _pack_masks(h5, masks)

        h5.close()

    def getSurf(self, subject, type, hemi='both', merge=False, nudge=False):
        import tables
        if hemi == 'both':
            left = self.getSurf(subject, type, "lh")
            right = self.getSurf(subject, type, "rh")
            if merge:
                pts = np.vstack([left[0], right[0]])
                polys = np.vstack([left[1], right[1]+len(left[0])])
                return pts, polys

            return left, right
        try:
            node = getattr(getattr(getattr(self.subjects, subject).surfaces, type), hemi)
            pts, polys = node.pts[:], node.polys[:]
            if nudge:
                if hemi == 'lh':
                    pts[:,0] -= pts[:,0].min()
                else:
                    pts[:,0] -= pts[:,0].max()
            return pts, polys
        except tables.NoSuchNodeError:
            raise IOError('Subject not found in package')

    def getXfm(self, subject, xfmname):
        import tables
        try:
            node = getattr(getattr(self.subjects, subject).transforms, xfmname)
            return Transform(node.xfm[:], node.attrs.shape)
        except tables.NoSuchNodeError:
            raise IOError('Transform not found in package')

    def getMask(self, subject, xfmname, maskname):
        import tables
        try:
            node = getattr(getattr(self.subjects, subject).transforms, xfmname).masks
            return getattr(node, maskname)[:]
        except tables.NoSuchNodeError:
            raise IOError('Mask not found in package')

    def getOverlay(self, subject, type='rois', **kwargs):
        import tables
        try:
            node = getattr(getattr(self.subjects, subject), type)
            if type == "rois":
                import tempfile
                tf = tempfile.NamedTemporaryFile()
                tf.write(node.read())
                tf.seek(0)
                return tf
        except tables.NoSuchNodeError:
            raise IOError('Overlay not found in package')

        raise TypeError('Unknown overlay type')

    def prepend(self, prefix):
        ds = dict()
        for name, data in self:
            ds[prefix+name] = data

        return Dataset(**ds)

class VolumeData(object):
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
        if isinstance(data, str):
            import nibabel
            nib = nibabel.load(data)
            data = nib.get_data().T

        self.data = data
        try:
            basestring
        except NameError:
            subject = subject if isinstance(subject, str) else subject.decode('utf-8')
            xfmname = xfmname if isinstance(xfmname, str) else xfmname.decode('utf-8')
        self.subject = subject
        self.xfmname = xfmname
        self.attrs = kwargs
        
        self._check_size(mask)
        self.masked = Masker(self)

        #self.add_numpy_methods()

    def copy(self, newdata=None):
        """Copies this VolumeData.
        """
        if newdata is None:
            return VolumeData(self.data, self.subject, self.xfmname, **self.attrs)
        else:
            return VolumeData(newdata, self.subject, self.xfmname, **self.attrs)

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
                self.masktype, self.mask = _find_mask(nvox, self.subject, self.xfmname)
            elif isinstance(mask, str):
                self.mask = surfs.getMask(self.subject, self.xfmname, mask)
                self.masktype = mask
            elif isinstance(mask, np.ndarray):
                self.mask = mask
                self.masktype = "user-supplied"

            self.shape = self.mask.shape
        else:
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
        return VolumeData(self.data[idx], self.subject, self.xfmname, **self.attrs)

    @property
    def volume(self):
        """Standardizes the VolumeData, ensuring that masked data are unmasked"""
        if self.linear:
            data = volume.unmask(self.mask, self.data)
        else:
            data = self.data

        if self.raw and data.shape[-1] == 3:
            #stack the alpha dimension
            alpha = 255*np.ones(data.shape[:3]+(1,)).astype(np.uint8)
            data = np.concatenate([data, alpha], axis=-1)

        return data

    @property
    def priority(self):
        """Sets the priority of this VolumeData in a dataset.
        """
        if 'priority' in self.attrs:
            return self.attrs['priority']
        return 1000

    @priority.setter
    def priority(self, val):
        self.attrs['priority'] = val

    def save(self, filename, name="data"):
        """Save the dataset into an hdf file with the provided name
        """
        import tables
        if isinstance(filename, str):
            fname, ext = os.path.splitext(filename)
            if ext in (".hdf", ".h5"):
                h5 = tables.openFile(filename, "a")
                _hdf_init(h5)
                self._write_hdf(h5, name=name)
                h5.close()
            else:
                raise TypeError('Unknown file type')
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

    def exp(self):
        """Copy of this object with data exponentiated.
        """
        return self.copy(np.exp(self.data))
    
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

VolumeData.add_numpy_methods()


class VertexData(VolumeData):
    def __init__(self, data, subject, **kwargs):
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
        self.attrs = kwargs
        
        left, right = surfs.getSurf(self.subject, "fiducial")
        self.llen = len(left[0])
        self.rlen = len(right[0])
        self._set_data(data)

    def _set_data(self, data):
        """Sets the data of this VertexData. Also sets flags if the data appears to
        be in 'movie' or 'raw' format. See __init__ for data shape possibilities.
        """
        # If no data is given, initialize to empty floats
        if data is None:
            data = np.zeros((self.llen + self.rlen,))

        # Store the data
        self.data = data

        # Figure out what kind of data it is
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

    def _write_hdf(self, h5, name="data"):
        node = _hdf_write(h5, self.data, name=name)
        node.attrs.subject = self.subject
        for name, value in self.attrs.items():
            node.attrs[name] = value

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
            return VolumeData(self.ds.volume[:,mask], s, x, mask=masktype)
        return VolumeData(self.ds.volume[mask], s, x, mask=masktype)

def normalize(data):
    if isinstance(data, (VolumeData, VertexData, Dataset)):
        return data
    elif isinstance(data, dict):
        return Dataset(**data)
    elif isinstance(data, str):
        return Dataset.from_file(data)
    elif isinstance(data, tuple):
        if len(data) == 3:
            return VolumeData(*data)
        else:
            return VertexData(*data)

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
    import tables
    try:
        h5.getNode("/datasets")
    except tables.NoSuchNodeError:
        h5.createGroup("/","datasets")
    try:
        h5.getNode("/subjects")
    except tables.NoSuchNodeError:
        h5.createGroup("/", "subjects")


def _hdf_write(h5, data, name="data", group="/datasets"):
    import tables
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
