import numpy as np
from scipy import sparse

from .. import dataset

import warnings
warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)

class Mapper(object):
    '''Maps data from epi volume onto surface using various projections'''
    def __init__(self, left, right, shape, subject, xfmname):
        self.idxmap = None
        self.masks = [left, right]
        self.nverts = left.shape[0] + right.shape[0]
        self.shape = shape
        self.subject = subject
        self.xfmname = xfmname

    @classmethod
    def from_cache(cls, cachefile, subject, xfmname):
        npz = np.load(cachefile)
        left = (npz['left_data'], npz['left_indices'], npz['left_indptr'])
        right = (npz['right_data'], npz['right_indices'], npz['right_indptr'])
        lsparse = sparse.csr_matrix(left, shape=npz['left_shape'])
        rsparse = sparse.csr_matrix(right, shape=npz['right_shape'])
        return cls(lsparse, rsparse, npz['shape'], subject, xfmname)

    @property
    def mask(self):
        mask = np.array(self.masks[0].sum(0) + self.masks[1].sum(0))
        return (mask.squeeze() != 0).reshape(self.shape)

    @property
    def hemimasks(self):
        func = lambda m: (np.array(m.sum(0)).squeeze() != 0).reshape(self.shape)
        return [func(x) for x in self.masks]

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

    def backwards(self, vertexdata):
        '''Projects vertex data back into volume space.

        Parameters
        ----------
        vertexdata : Vertex object or array
            The data that will be projected back into voxel space.
            If Vertex object is provided, a Volume object is returned
            If an array is provided, an array is returned
        '''
        Vert2Vol = isinstance(vertexdata, dataset.Vertex)
        if Vert2Vol:
            to_map = vertexdata.data
        else:
            to_map = vertexdata
        # stack the two mappers together
        bothmappers = sparse.vstack(self.masks)
        # dot the vertex data with the stacked mappers
        partial_vertex = bothmappers.T.dot(to_map)
        # solve the inverse mapping problem
        voxeldata = self._get_backmapper().solve(partial_vertex).reshape(self.shape)
        if Vert2Vol:
            # construct a volume object with the new data
            return dataset.Volume(voxeldata, self.subject, self.xfmname)
        else:
            return voxeldata

    def _get_backmapper(self):
        if not hasattr(self, '_backmapper'):
            # stack the two mappers together to get one voxel -> vertex mapper
            bothmappers = sparse.vstack(self.masks)
            # take inner product to get symmetric matrix
            symmappers = bothmappers.T.dot(bothmappers)
            # add (very) small diagonal to make sure it's full rank
            symmappers_reg = symmappers + 1e-9 * sparse.eye(symmappers.shape[0])
            # factorize it using splu so that inversion is fast
            self._backmapper = sparse.linalg.splu(symmappers_reg)

        return self._backmapper

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
        return cls(masks[0], masks[1], xfm.shape, subject, xfmname)

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
