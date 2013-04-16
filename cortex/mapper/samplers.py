import numpy as np

def nearest(coords, shape):
    valid = ~(np.isnan(coords).all(1))
    rcoords = coords[valid].round().astype(int)
    j = np.ravel_multi_index(rcoords.T[::-1], shape, mode='clip')
    return np.nonzero(valid)[0], j, np.ones((valid.sum(),))
    
def trilinear(coords, shape):
    #trilinear interpolation equation from http://paulbourke.net/miscellaneous/interpolation/
    valid = ~(np.isnan(coords).all(1))
    (x, y, z), floor = np.modf(coords[valid].T)
    floor = floor.astype(int)
    ceil = floor + 1
    x[x < 0] = 0
    y[y < 0] = 0
    z[z < 0] = 0

    i000 = np.array([floor[2], floor[1], floor[0]])
    i100 = np.array([floor[2], floor[1],  ceil[0]])
    i010 = np.array([floor[2],  ceil[1], floor[0]])
    i001 = np.array([ ceil[2], floor[1], floor[0]])
    i101 = np.array([ ceil[2], floor[1],  ceil[0]])
    i011 = np.array([ ceil[2],  ceil[1], floor[0]])
    i110 = np.array([floor[2],  ceil[1],  ceil[0]])
    i111 = np.array([ ceil[2],  ceil[1],  ceil[0]])

    v000 = (1-x)*(1-y)*(1-z)
    v100 = x*(1-y)*(1-z)
    v010 = (1-x)*y*(1-z)
    v110 = x*y*(1-z)
    v001 = (1-x)*(1-y)*z
    v101 = x*(1-y)*z
    v011 = (1-x)*y*z
    v111 = x*y*z
    
    i    = np.tile(np.nonzero(valid)[0], [8, 1]).T.ravel()
    j    = np.hstack([i000, i100, i010, i001, i101, i011, i110, i111]).T
    data = np.vstack([v000, v100, v010, v001, v101, v011, v110, v111]).T.ravel()
    return i, np.ravel_multi_index(j, shape, mode='clip'), data

def gaussian(coords, window=3):
    raise NotImplementedError

def lanczos(coords, window=3, renorm=True):
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

    import ipdb
    ipdb.set_trace()
    
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
    return i, j, data