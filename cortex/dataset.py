"""Module for maintaining brain data and their masks

HDF5 format:

/subjects/
    s1/
        transforms/
            xfm1/
                xfm[4,4]
                masks/
                    thin[z,y,x]
                    thick[z,y,x]
            xfm2/
                xfm[4,4]
                masks/
                    thin[z,y,x]
        surfaces
            fiducial
                lh
                    pts[n,3]
                    polys[m,3]
                rh
                    pts[n,3]
                    polys[m,3]
/datasets/
    ds1
    ds2
"""
import numpy as np

from .db import surfs
from .xfm import Transform
from . import volume
from . import utils

class Dataset(object):
    def __init__(self, **kwargs):
        self.subjects = {}
        self.datasets = {}
        for name, data in kwargs.items():
            if isinstance(data, BrainData):
                self.datasets[name] = data
            else:
                self.datasets[name] = BrainData(*data)

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
        if isinstance(item, int):
            return self.datasets[self.datasets.keys()[item]]
        return self.datasets[item]

    def __iter__(self):
        for name, ds in sorted(self.datasets.items(), key=lambda x: x[1].priority):
            yield name, ds

    def __repr__(self):
        return "<Dataset with names [%s]>"%(', '.join(self.datasets.keys()))

    def __dir__(self):
        return self.__dict__.keys() + self.datasets.keys()

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
        self.attrs = kwargs
        
        self._check_size(mask)
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
        from . import utils
        mapper = utils.get_mapper(self.subject, self.xfmname, projection)
        return mapper(self)

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

    @property
    def volume(self):
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
        if 'priority' in self.attrs:
            return self.attrs['priority']
        return 1000

    @priority.setter
    def priority(self, val):
        self.attrs['priority'] = val

    def save(self, filename, name="data"):
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

class VertexData(BrainData):
    def __init__(self, data, subject, **kwargs):
        """Vertex Data possibilities

        raw linear movie: (t, v, c)
        reg linear movie: (t, v)
        raw linear image: (v, c)
        reg linear image: (v,)
        """
        self.data = data
        self.subject = subject
        self.attrs = kwargs

        self.raw = data.dtype == np.uint8
        if data.ndim == 3:
            self.movie = True
            if not self.raw:
                raise ValueError('Invalid data shape')
        elif data.ndim == 2:
            self.movie = not self.raw

        pts, polys = surfs.getSurf(self.subject, "fiducial", merge=True)
        self.nverts = self.data.shape[-2 if self.raw else -1]
        if len(pts) != self.nverts:
            raise ValueError('Invalid number of vertices for subject')

    def _check_size(self):
        raise NotImplementedError

    def volume(self, xfmname, projection='nearest', **kwargs):
        from . import utils
        mapper = utils.get_mapper(self.subject, xfmname, projection)
        return mapper.backwards(self, **kwargs)

    def __repr__(self):
        maskstr = ''
        if 'projection' in self.attrs:
            maskstr = '%s mapped'%self.attrs['projection']

        if self.raw:
            maskstr += " raw"
        if self.movie:
            maskstr += " movie"
        return "<%s vertex data for %s>"%(maskstr, self.subject)

    def _write_hdf(self, h5, name="data"):
        node = _hdf_write(h5, self.data, name=name)
        node.attrs.subject = self.subject
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
                return BrainData(data, **attrs)
            return VertexData(data, **attrs)
    elif isinstance(filename, tables.File):
        node = filename.getNode("/datasets", name)
        data, attrs = _hdf_read(node)
        if 'xfmname' in attrs:
            return BrainData(data, **attrs)
        return VertexData(data, **attrs)

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