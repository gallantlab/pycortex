import numpy as np
from scipy import sparse

from . import Mapper
from . import samplers

class WarpMapper(Mapper):
    sampler = staticmethod(samplers.nearest)
    @classmethod
    def _getmask(cls, coords, polys, shape, **kwargs):
        valid = np.unique(polys)
        mcoords = np.nan * np.ones_like(coords)
        mcoords[valid] = coords[valid]
        i, j, data = cls.sampler(mcoords, shape, **kwargs)
        csrshape = len(coords), np.prod(shape)
        return sparse.csr_matrix((data, np.array([i, j])), shape=csrshape)

    @classmethod
    def _cache(cls, filename, subject, xfmname, **kwargs):
        print('Caching mapper...')
        from ..database import db
        masks = []
        # the next line is where the big changes will have to start...
        xfm = db.get_xfm(subject, xfmname, xfmtype='coord')
        fid = db.get_surf(subject, 'fiducial', merge=False, nudge=False)
        flat = db.get_surf(subject, 'flat', merge=False, nudge=False)

        for (pts, _), (_, polys) in zip(fid, flat):
            masks.append(cls._getmask(xfm(pts), polys, xfm.shape, **kwargs))

        _savecache(filename, masks[0], masks[1], xfm.shape)
        return cls(masks[0], masks[1], xfm.shape)