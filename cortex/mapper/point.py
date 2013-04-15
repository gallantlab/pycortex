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
        nZ, nY, nX = shape
        dx = coords[:,0] - np.atleast_2d(np.arange(nX)).T
        dy = coords[:,1] - np.atleast_2d(np.arange(nY)).T
        dz = coords[:,2] - np.atleast_2d(np.arange(nZ)).T

        def lanczos(x):
            out = np.zeros_like(x)
            sel = np.abs(x)<window
            selx = x[sel]
            out[sel] = np.sin(np.pi * selx) * np.sin(np.pi * selx / window) * (window / (np.pi**2 * selx**2))
            return out

        Lx = lanczos(dx)
        Ly = lanczos(dy)
        Lz = lanczos(dz)
        
        mask = sparse.lil_matrix((len(coords), np.prod(shape)))
        for v in range(len(coords)):
            ix = np.nonzero(Lx[:,v])[0]
            iy = np.nonzero(Ly[:,v])[0]
            iz = np.nonzero(Lz[:,v])[0]

            vx = Lx[ix,v]
            vy = Ly[iy,v]
            vz = Lz[iz,v]
            try:
                inds = np.ravel_multi_index(np.array(list(product(iz, iy, ix))).T, shape)
                vals = np.prod(np.array(list(product(vz, vy, vx))), 1)
                if renorm:
                    vals /= vals.sum()
                mask[v,inds] = vals
            except ValueError:
                pass

            if not v % 1000:
                print(v)

        return mask.tocsr()
