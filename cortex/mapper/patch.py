import numpy as np
from scipy import sparse

from . import Mapper
from . import samplers
from .. import polyutils

class PatchMapper(Mapper):
    @classmethod
    def _getmask(cls, pts, polys, shape, npts=64, mp=True, **kwargs):
        rand = np.random.rand(2, npts)

        def func(ipts):
            idx, pts = ipts
            if pts is not None:
                A = np.outer(1-np.sqrt(rand[0]), pts[:,0].ravel())
                B = np.outer(np.sqrt(rand[0]) * (1-rand[1]), pts[:,1].ravel())
                C = np.outer(rand[1] * np.sqrt(rand[0]), pts[:,2].ravel())
                randpts = (A+B+C).reshape(-1, pts.shape[0], pts.shape[2])
                areas = polyutils.face_area(pts)
                areas /= areas.sum()

                allj, alldata = [], []
                for tri, area in zip(randpts.swapaxes(0,1), areas):
                    i, j, data = cls.sampler(tri, shape, renorm=False, mp=False, **kwargs)
                    alldata.append(data / data.sum() * area)
                    allj.append(j)

                #print idx
                return samplers.collapse(np.hstack(allj), np.hstack(alldata))
            return None, None

        surf = polyutils.Surface(pts, polys)
        patches = surf.patches(n=cls.patchsize)
        if mp:
            from .. import mp
            samples = mp.map(func, enumerate(patches))
        else:
            samples = map(func, enumerate(patches))

        ij, alldata = [], []
        for i, (j, data) in enumerate(samples):
            if data is not None:
                ij.append(np.vstack(np.broadcast_arrays(i, j)).T)
                alldata.append(data)

        data, ij = np.hstack(alldata), np.vstack(ij).T
        csrshape = len(pts), np.prod(shape)
        return sparse.csr_matrix((data, ij), shape=csrshape)

class ConstPatch(PatchMapper):
    patchsize = 1

class ConstPatchNN(ConstPatch):
    sampler = staticmethod(samplers.nearest)

class ConstPatchTrilin(ConstPatch):
    sampler = staticmethod(samplers.trilinear)

class ConstPatchLanczos(ConstPatch):
    sampler = staticmethod(samplers.lanczos)