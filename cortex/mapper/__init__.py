import os
import warnings

import nibabel
import numpy as np
from scipy import sparse
warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)

from .. import dataset

def get_mapper(subject, xfmname, type='nearest', recache=False, **kwargs):
    from ..database import db
    from . import point, patch, volume, line

    mapcls = dict(
        nearest=point.PointNN,
        trilinear=point.PointTrilin,
        gaussian=point.PointGauss,
        lanczos=point.PointLanczos,
        const_patch_nn=patch.ConstPatchNN,
        const_patch_trilin=patch.ConstPatchTrilin,
        const_patch_lanczos=patch.ConstPatchLanczos,
        line_nearest=line.LineNN,
        line_trilinear=line.LineTrilin)
    Map = mapcls[type]
    ptype = Map.__name__.lower()
    kwds ='_'.join(['%s%s'%(k,str(v)) for k, v in list(kwargs.items())])
    if len(kwds) > 0:
        ptype += '_'+kwds

    fname = "{xfmname}_{projection}.npz".format(xfmname=xfmname, projection=ptype)

    xfmfile = db.get_paths(subject)['xfmdir'].format(xfmname=xfmname)
    cachefile = os.path.join(db.get_cache(subject), fname)

    try:
        if not recache and (xfmname == "identity" or os.stat(cachefile).st_mtime > os.stat(xfmfile).st_mtime):
           return mapcls[type].from_cache(cachefile)
        raise Exception
    except Exception as e:
        return mapcls[type]._cache(cachefile, subject, xfmname, **kwargs)

def _savecache(filename, left, right, shape):
    np.savez(filename,
            left_data=left.data,
            left_indices=left.indices,
            left_indptr=left.indptr,
            left_shape=left.shape,
            right_data=right.data,
            right_indices=right.indices,
            right_indptr=right.indptr,
            right_shape=right.shape,
            shape=shape)

class Mapper(object):
    '''Maps data from epi volume onto surface using various projections'''
    def __init__(self, left, right, shape):
        self.idxmap = None
        self.masks = [left, right]
        self.nverts = left.shape[0] + right.shape[0]
        self.shape = shape

    @classmethod
    def from_cache(cls, cachefile):
        npz = np.load(cachefile)
        left = (npz['left_data'], npz['left_indices'], npz['left_indptr'])
        right = (npz['right_data'], npz['right_indices'], npz['right_indptr'])
        lsparse = sparse.csr_matrix(left, shape=npz['left_shape'])
        rsparse = sparse.csr_matrix(right, shape=npz['right_shape'])
        return cls(lsparse, rsparse, npz['shape'])

    @property
    def mask(self):
        mask = np.array(self.masks[0].sum(0) + self.masks[1].sum(0))
        return (mask.squeeze() != 0).reshape(self.shape)

    @property
    def hemimasks(self):
        func = lambda m: (np.array(m.sum(0)).squeeze() != 0).reshape(self.shape)
        return list(map(func, self.masks))

    def __repr__(self):
        ptype = self.__class__.__name__
        return '<%s mapper with %d vertices>'%(ptype, self.nverts)

    def __call__(self, data):
        if isinstance(data, tuple):
            data = dataset.Volume(*data)

        if isinstance(data, dataset.Vertex):
            llen = self.masks[0].shape[0]
            if data.raw:
                left, right = data.data[..., :llen,:], data.data[..., llen:,:]
                if self.idxmap is not None:
                    left = left[..., self.idxmap[0], :]
                    right = right[..., self.idxmap[1], :]
            else:
                left, right = data[..., :llen], data[..., llen:]
                if self.idxmap is not None:
                    left = left[..., self.idxmap[0]]
                    right = right[..., self.idxmap[1]]
            return left, right

        volume = np.ascontiguousarray(data.volume)
        volume.shape = len(volume), -1
        volume = volume.T

        mapped = []
        for mask in self.masks:
            mapped.append(np.array(mask * volume).T)

        if self.idxmap is not None:
            mapped[0] = mapped[0][:, self.idxmap[0]]
            mapped[1] = mapped[1][:, self.idxmap[1]]

        return dataset.Vertex(np.hstack(mapped).squeeze(), data.subject)

    def backwards(self, verts, fast=True):
        '''Projects vertex data back into volume space

        Parameters
        ----------
        verts : array_like
            If uint array and max <= nverts, assume binary mask of vertices
            If float array and len == nverts, project float values into volume
        '''
        left = np.zeros((self.masks[0].shape[0],), dtype=bool)
        right = np.zeros((self.masks[1].shape[0],), dtype=bool)

        if len(verts) > 0:
            if isinstance(verts, (list, tuple)) and len(verts) == 2:
                if len(verts[0]) == len(left):
                    left = verts[0]
                    right = verts[1]
                elif verts[0].max() < len(left):
                    left[verts[0]] = True
                    right[verts[1]] = True
                else:
                    raise ValueError
            else:
                if len(verts) == self.nverts:
                    left = verts[:len(left)]
                    right = verts[len(left):]
                elif verts.max() < self.nverts:
                    left[verts[verts < len(left)]] = True
                    right[verts[verts >= len(left)] - len(left)] = True
                else:
                    raise ValueError

        output = []
        for mask, data in zip(self.masks, [left, right]):
            proj = data * mask
            output.append(np.array(proj).reshape(self.shape))

        return output

    @classmethod
    def _cache(cls, filename, subject, xfmname, **kwargs):
        print('Caching mapper...')
        from ..database import db
        masks = []
        xfm = db.get_xfm(subject, xfmname, xfmtype='coord')
        fid = db.get_surf(subject, 'fiducial', merge=False, nudge=False)
        try:
            flat = db.get_surf(subject, 'flat', merge=False, nudge=False)
        except IOError:
            flat = fid

        for (pts, _), (_, polys) in zip(fid, flat):
            masks.append(cls._getmask(xfm(pts), polys, xfm.shape, **kwargs))

        _savecache(filename, masks[0], masks[1], xfm.shape)
        return cls(masks[0], masks[1], xfm.shape)
