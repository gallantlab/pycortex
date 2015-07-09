import numpy as np

def collapse(j, data):
    """Collapses samples into a single row"""
    uniques = np.unique(j)
    return uniques, np.array([data[j == u].sum() for u in uniques])

def nearest(coords, shape, **kwargs):
    valid = ~(np.isnan(coords).all(1))
    valid = np.logical_and(valid, np.logical_and(coords[:,0] > -.5, coords[:,0] < shape[2]+.5))
    valid = np.logical_and(valid, np.logical_and(coords[:,1] > -.5, coords[:,1] < shape[1]+.5))
    valid = np.logical_and(valid, np.logical_and(coords[:,2] > -.5, coords[:,2] < shape[0]+.5))

    rcoords = coords[valid].round().astype(int)
    j = np.ravel_multi_index(rcoords.T[::-1], shape, mode='clip')
    #return np.nonzero(valid)[0], j, (rcoords > 0).all(1) #np.ones((valid.sum(),))
    return np.nonzero(valid)[0], j, np.ones((valid.sum(),))
    
def trilinear(coords, shape, **kwargs):
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
    
    i    = np.tile(np.nonzero(valid)[0], [1, 8]).ravel()
    j    = np.hstack([i000, i100, i010, i001, i101, i011, i110, i111])
    data = np.vstack([v000, v100, v010, v001, v101, v011, v110, v111]).ravel()
    return i, np.ravel_multi_index(j, shape, mode='clip'), data

def distance_func(func, coords, shape, renorm=True, mp=True):
    """Generates masks for seperable distance functions"""
    nZ, nY, nX = shape
    dx = coords[:,0] - np.atleast_2d(np.arange(nX)).T
    dy = coords[:,1] - np.atleast_2d(np.arange(nY)).T
    dz = coords[:,2] - np.atleast_2d(np.arange(nZ)).T

    Lx, Ly, Lz = func(dx), func(dy), func(dz)
    ix, jx = np.nonzero(Lx)
    iy, jy = np.nonzero(Ly)
    iz, jz = np.nonzero(Lz)
    ba = np.broadcast_arrays
    def func(v):
        mx, my, mz = ix[jx == v], iy[jy == v], iz[jz == v]
        idx, idy, idz = [i.ravel() for i in ba(*np.ix_(mx, my, mz))]
        vx, vy, vz = [i.ravel() for i in ba(*np.ix_(Lx[mx, v], Ly[my, v], Lz[mz, v]))]

        i = v * np.ones((len(idx,)))
        j = np.ravel_multi_index((idz, idy, idx), shape, mode='clip')
        data = vx*vy*vz
        if renorm:
            data /= data.sum()
        return i, j, data

    if mp:
        from .. import mp
        ijdata = mp.map(func, range(len(coords)))
    else:
        ijdata = map(func, range(len(coords)))

    return np.hstack(ijdata)

def gaussian(coords, shape, sigma=1, window=3, **kwargs):
    raise NotImplementedError
    def gaussian(x):
        pass
    return distance_func(gaussian, coords, shape, **kwargs)

def lanczos(coords, shape, window=3, **kwargs):
    def lanczos(x):
        out = np.zeros_like(x)
        sel = np.abs(x)<window
        selx = x[sel]
        out[sel] = np.sin(np.pi * selx) * np.sin(np.pi * selx / window) * (window / (np.pi**2 * selx**2))
        return out

    return distance_func(lanczos, coords, shape, **kwargs)
