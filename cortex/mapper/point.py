from . import Mapper
from .utils import trilinear, lanczos

class Nearest(Mapper):
    '''Maps epi volume data to surface using nearest neighbor interpolation'''
    @staticmethod
    def _getmask(coords, polys, shape):
        valid = np.ones((len(coords),), dtype=bool)

        coords = np.where(np.mod(coords, 2) == 0.5, np.ceil(coords), np.around(coords)).astype(int)
        d1 = np.logical_and(0 <= coords[:,0], coords[:,0] < shape[2])
        d2 = np.logical_and(0 <= coords[:,1], coords[:,1] < shape[1])
        d3 = np.logical_and(0 <= coords[:,2], coords[:,2] < shape[0])
        valid = np.logical_and(np.logical_and(valid, d1), np.logical_and(d2, d3))

        ravelidx = np.ravel_multi_index(coords.T[::-1], shape, mode='clip')

        ij = np.array([np.nonzero(valid)[0], ravelidx[valid]])
        data = np.ones((len(ij.T),), dtype=bool)
        csrshape = len(coords), np.prod(shape)
        return sparse.csr_matrix((data, ij), dtype=bool, shape=csrshape)

class Trilinear(Mapper):
    @staticmethod
    def _getmask(coords, polys, shape):
        from .utils import trilinear
        csrshape = len(coords), np.prod(shape)
        valid = np.unique(polys)
        coords = coords[valid]
        idx, value = trilinear(coords)

        return sparse.csr_matrix((data, ij), shape=csrshape)

class Gaussian(Mapper):
    def _recache(self, subject, xfmname, std=2):
        raise NotImplementedError

class Lanczos(Mapper):
    @staticmethod
    def _getmask(coords, polys, shape, window=3, renorm=True):
        

        return mask.tocsr()
