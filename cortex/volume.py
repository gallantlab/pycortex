import nibabel
import numpy as np

from .db import surfs

def unmask(mask, data):
    """unmask(mask, data)

    `Unmasks` the data, assuming it's been masked.

    Parameters
    ----------
    mask : array_like
        The data mask
    data : array_like
        Actual MRI data to unmask
    """
    if data.ndim > 1:
        output = np.zeros((len(data),)+mask.shape, dtype=data.dtype)
        output[:, mask > 0] = data
    else:
        output = np.zeros(mask.shape, dtype=data.dtype)
        output[mask > 0] = data
    return output

def detrend_median(data, kernel=15):
    from scipy.signal import medfilt
    lowfreq = medfilt(data, [1, kernel, kernel])
    return data - lowfreq

def detrend_gradient(data, diff=3):
    return (np.array(np.gradient(data, 1, diff, diff))**2).sum(0)

def detrend_poly(data, polyorder = 10, mask=None):
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

def mosaic(data, xy=(6, 5), trim=10, skip=1, show=True, **kwargs):
    """mosaic(data, xy=(6, 5), trim=10, skip=1)

    Turns volume data into a mosaic, useful for quickly viewing volumetric data
    IN RADIOLOGICAL COORDINATES (LEFT SIDE OF FIGURE IS RIGHT SIDE OF SUBJECT)

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
    """
    assert len(data.shape) == 3, "Are you sure this is volumetric?"
    dat = data.copy()
    if trim>0:
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
        plt.xticks([])
        plt.yticks([])

    return output

def show_slice(data, subject, xfmname, vmin=None, vmax=None, **kwargs):
    from matplotlib import cm
    import matplotlib.pyplot as plt

    if vmax is None:
        from scipy import stats
        vmax = stats.scoreatpercentile(data.ravel(), 99)

    anat = nibabel.load(surfs.getAnat(subject, 'raw')).get_data().T
    data, _ = epi2anatspace(data, subject, xfmname)
    data[data < vmin] = np.nan

    state = dict(slice=data.shape[0]*.66, dim=0, pad=())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    anatomical = ax.imshow(anat[state['pad'] + (state['slice'],)], cmap=cm.gray, aspect='equal')
    functional = ax.imshow(data[state['pad'] + (state['slice'],)], vmin=vmin, vmax=vmax, aspect='equal', **kwargs)

    def update():
        print "showing dim %d, slice %d"%(state['dim'] % 3, state['slice'])
        functional.set_data(data[state['pad'] + (state['slice'],)])
        anatomical.set_data(anat[state['pad'] + (state['slice'],)])
        fig.canvas.draw()

    def switchslice(event):
        state['dim'] += 1
        state['pad'] = (slice(None),)*(state['dim'] % 3)
        update()

    def scrollslice(event):
        if event.button == 'up':
            state['slice'] += 1
        elif event.button == 'down':
            state['slice'] -= 1
        update()

    fig.canvas.mpl_connect('scroll_event', scrollslice)
    fig.canvas.mpl_connect('button_press_event', switchslice)
    return fig

def show_mip(data, **kwargs):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.add_subplot(221).imshow(data.max(0), **kwargs)
    fig.add_subplot(222).imshow(data.max(1), **kwargs)
    fig.add_subplot(223).imshow(data.max(2), **kwargs)
    return fig

def epi2anatspace(data, subject, xfmname):
    """Resamples epi-space [data] into the anatomical space for the given [subject]
    using the given transformation [xfm].

    Returns the data and a temporary filename.
    """
    import tempfile
    import subprocess

    ## Get transform, save out into ascii file
    xfm = surfs.getXfm(subject, xfmname)
    fslxfm = xfm.to_fsl(surfs.getAnat(subject, 'raw'))

    xfmfilename = tempfile.mktemp(".mat")
    with open(xfmfilename, "w") as xfmh:
        for ll in fslxfm.tolist():
            xfmh.write(" ".join(["%0.5f"%f for f in ll])+"\n")

    ## Save out data into nifti file
    datafile = nibabel.Nifti1Image(data.T, xfm.epi.get_affine(), xfm.epi.get_header())
    datafilename = tempfile.mktemp(".nii")
    nibabel.save(datafile, datafilename)

    ## Reslice epi-space image
    raw = surfs.getAnat(subject, type='raw')
    outfilename = tempfile.mktemp(".nii")
    subprocess.call(["fsl5.0-flirt",
                     "-ref", raw,
                     "-in", datafilename,
                     "-applyxfm",
                     "-init", xfmfilename,
                     #"-interp", "sinc",
                     "-out", outfilename])

    ## Load resliced image
    outdata = nibabel.load(outfilename+".gz").get_data().T

    return outdata, outfilename

def fslview(*imfilenames):
    import subprocess
    subprocess.call(["fslview"] + list(imfilenames))
