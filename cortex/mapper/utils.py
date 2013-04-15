import numpy as np

def trilinear(coords):
    #trilinear interpolation equation from http://paulbourke.net/miscellaneous/interpolation/
    (x, y, z), floor = np.modf(coords.T)
    floor = floor.astype(int)
    ceil = floor + 1
    x[x < 0] = 0
    y[y < 0] = 0
    z[z < 0] = 0

    i000 = (floor[2], floor[1], floor[0])
    i100 = (floor[2], floor[1],  ceil[0])
    i010 = (floor[2],  ceil[1], floor[0])
    i001 = ( ceil[2], floor[1], floor[0])
    i101 = ( ceil[2], floor[1],  ceil[0])
    i011 = ( ceil[2],  ceil[1], floor[0])
    i110 = (floor[2],  ceil[1],  ceil[0])
    i111 = ( ceil[2],  ceil[1],  ceil[0])

    v000 = (1-x)*(1-y)*(1-z)
    v100 = x*(1-y)*(1-z)
    v010 = (1-x)*y*(1-z)
    v110 = x*y*(1-z)
    v001 = (1-x)*(1-y)*z
    v101 = x*(1-y)*z
    v011 = (1-x)*y*z
    v111 = x*y*z
    
    idx   = np.vstack([i000, i100, i010, i001, i101, i011, i110, i111]).T.ravel()
    value = np.vstack([v000, v100, v010, v001, v101, v011, v110, v111]).T.ravel()
    return idx, value

def lancz