import numpy as np
from scipy import sparse

from . import Mapper
from . import samplers

class PointMapper(Mapper):
    @classmethod
    def _getmask(cls, coords, polys, shape, **kwargs):
        valid = np.unique(polys)
        mcoords = np.nan * np.ones_like(coords)
        mcoords[valid] = coords[valid]
        i, j, data = cls.sampler(mcoords, shape, **kwargs)
        csrshape = len(coords), np.prod(shape)
        return sparse.csr_matrix((data, np.array([i, j])), shape=csrshape)

class PointNN(PointMapper):
    sampler = staticmethod(samplers.nearest)

class PointTrilin(PointMapper):
    sampler = staticmethod(samplers.trilinear)

class PointGauss(PointMapper):
    sampler = staticmethod(samplers.gaussian)

class PointLanczos(PointMapper):
    sampler = staticmethod(samplers.lanczos)
