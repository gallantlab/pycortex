import os
import sys
import tempfile

import tables
import numpy as np

def mosaic(data, xy=(6, 5), trim=10, skip=1, show=True, **kwargs):
    '''mosaic(data, xy=(6, 5), trim=10, skip=1)

    Turns volume data into a mosaic, useful for quickly viewing volumetric data

    Parameters
    ----------
    data : array_like
        3D volumetric data to mosaic
    xy : tuple, optional
        tuple(x, y) for the grid of images. Default (6, 5)
    trim : int, optional
        How many pixels to trim from the edges of each image. Default 10
    skip : int, optional
        How many slices to skip in the beginning. Default 1
    '''
    assert len(data.shape) == 3, "Are you sure this is volumetric?"
    dat = data.copy()
    dat = dat[:, trim:-trim, trim:-trim]
    d = dat.shape[1:]
    output = np.zeros(d*np.array(xy))
    
    c = skip
    for i in range(xy[0]):
        for j in range(xy[1]):
            if c < len(dat):
                output[d[0]*i:d[0]*(i+1), d[1]*j:d[1]*(j+1)] = dat[c]
            c+= 1
    
    if show:
        from matplotlib import pyplot as plt
        plt.imshow(output, **kwargs)

    return output

def flatmap(data, subject='JG', show=True):
    '''A completely silly wrapper around matlab's brain2flatmap function...'''
    global runner
    if runner is None:
        runner = matlab.runner()

    subjects = dict(JG=18)
    import tables

    assert len(data.shape) == 3, "Not the right shape, did you mean mri.flatmap(mri.unmask(mask, data))?"
    tf = tempfile.NamedTemporaryFile()
    h5 = tables.openFile(tf.name, "w")
    h5.createArray("/", "data", data)
    h5.close()

    cmd = """
        data = h5read('{tf}', '/data'); 
        map = brain2flatmap(data, {subj}); 
        h5create('{tf}', '/map', size(map));
        h5write('{tf}', '/map', map);
    """.format(tf=tf.name, subj=subjects[subject])
    runner(cmd)

    h5 = tables.openFile(tf.name, "r")
    fm = h5.root.map[:]
    h5.close()
    tf.close()
    if show:
        import matplotlib.pyplot as plt
        from matplotlib import cm

        fmpos = fm.T.copy()
        fmpos[fmpos < 0] = 0
        plt.imshow(fmpos, aspect='equal', cmap=cm.hot)
        plt.colorbar()

    return fm.T

def flatmap_hist(corrs, experiment, subject='JG', bins=100):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fm = flatmap(unmask(experiment, corrs), subject='JG', show=False)
    fm[fm < 0] = 0
    fig = plt.figure()
    mappable = fig.add_subplot(2,1,1).imshow(fm, aspect='equal', cmap=cm.hot)
    fig.colorbar(mappable)
    fig.add_subplot(2,1,2).hist(corrs, bins)

def detrend_volume_poly(data, polyorder = 10, mask=None):
    from scipy.special import legendre
    polys = [legendre(i) for i in range(polyorder)]
    s = data.shape
    b = data.ravel()[:,np.newaxis]
    lins = np.mgrid[-1:1:s[0]*1j, -1:1:s[1]*1j, -1:1:s[2]*1j].reshape(3,-1)

    if mask is not None:
        lins = lins[:,mask.ravel() > 0]
        b = b[mask.ravel() > 0]
    
    A = np.vstack([[p(i) for i in lins] for p in polys]).T
    x, res, rank, sing = np.linalg.lstsq(A, b)

    detrended = b.ravel() - np.dot(A, x).ravel()
    if mask is not None:
        filled = np.zeros_like(mask)
        filled[mask > 0] = detrended
        return filled
    else:
        return detrended.reshape(*s)

def flatten(data, subject="JG", reference="20110909JG_dust"):
    xfm = dbflats["xfms"][subject][reference]
    if "raw" in dbflats[subject]:
        pts, polys = dbflats[subject]['raw']['fiducial']
    else:
        pts, polys = vtkread([dbflats[subject]['L']['fiducial'], dbflats[subject]['R']['fiducial']])
        dbflats[subject]['raw'] = dict(fiducial=(pts, polys))

    coords = np.dot(xfm, np.append(pts, np.ones((len(pts),1)), axis=-1).T)[:-1]
    return np.array([data[tuple(c)] for c in coords.T.round().astype(int)])