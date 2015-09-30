import numpy as np
from scipy import sparse

from . import Mapper, _savecache
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
        from ..database import db
        from ..xfm import Transform

        print('Caching warpmapper...')
        print filename, subject, xfmname

        masks = []
        # the next line is where the big changes will have to start...
        # xfm = db.get_xfm(subject, xfmname, xfmtype='coord')
        re_anat = db.get_anat(subject,'reorient_anat')
        xfm = Transform(np.linalg.inv(re_anat.get_affine()),re_anat)
        warp = db.get_anat(subject,'mni2anat_field')
        warpfield = warp.get_data()
        fid = db.get_surf(subject, 'fiducial', merge=False, nudge=False)
        flat = db.get_surf(subject, 'flat', merge=False, nudge=False)

        for (pts, _), (_, polys) in zip(fid, flat):
            masks.append(cls._getmask(xfm(pts), polys, xfm.shape, **kwargs))

        _savecache(filename, masks[0], masks[1], xfm.shape)
        return cls(masks[0], masks[1], xfm.shape)