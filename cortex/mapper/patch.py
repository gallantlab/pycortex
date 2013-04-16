import numpy as np
from scipy import sparse

from . import Mapper
from . import samplers
from .. import polyutils

class PatchMapper(Mapper):
    @classmethod
    def _getmask(cls, pts, polys, shape, npts=1024, mp=True, **kwargs):
        rand = np.random.rand(npts, 3)
        csrshape = len(wm), np.prod(shape)

        def func(pts):
            if len(pts) > 0:
                #generate points within the bounding box
                samples = rand * (pts.max(0) - pts.min(0)) + pts.min(0)
                #check which points are inside the polyhedron
                inside = polyutils.inside_convex_poly(pts)(samples)
                return cls._sample(samples[inside], shape, np.sum(inside))

        surf = polyutils.Surface(pts, polys)
        patches = surf.patches(n=cls.patchsize)
        if mp:
            from . import mp
            samples = mp.map(func, patches)
        else:
            samples = map(func, patches)
            
        ij, data = [], []
        for i, sample in enumerate(samples):
            if sample is not None:
                idx = np.zeros((2, len(sample[0])))
                idx[0], idx[1] = i, sample[0]
                ij.append(idx)
                data.append(sample[1])

        return sparse.csr_matrix((np.hstack(data), np.hstack(ij)), shape=csrshape)

class ConstPatch(PatchMapper):
    patchsize = 1

class ConstPatchNN(ConstPatch):
    sampler = staticmethod(samplers.nearest)

class ConstPatchTrilin(ConstPatch):
    sampler = staticmethod(samplers.trilinear)

class ConstPatchLanczos(ConstPatch):
    sampler = staticmethod(samplers.lanczos)