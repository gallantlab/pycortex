import numpy as np
from scipy import sparse

from . import Mapper
from . import samplers

class PointMapper(Mapper):
    @classmethod
    def _getmask(cls, coords, polys, shape, **kwargs):
        valid = np.nan * np.ones_like(coords)
        valid[np.unique(polys)] = coords[np.unique(polys)]
        i, j, data = cls.sampler(valid, shape, **kwargs)
        csrshape = len(coords), np.prod(shape)
        return sparse.csr_matrix((data, np.array([i, j])), shape=csrshape)

class PointNN(Mapper):
    sampler = samplers.nearest

class PointTrilin(Mapper):
    sampler = samplers.trilinear

class PointGauss(Mapper):
    sampler = samplers.gaussian

class PointLanczos(Mapper):
    sampler = samplers.lanczos
