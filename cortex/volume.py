"""Contains functions for working with volume data
"""
import os
from typing import Literal, Optional, TypeVar, Union
import numpy as np
import numpy.typing as npt

from . import dataset
from .database import db
from .xfm import Transform

DType = TypeVar('DType', bound=np.generic)
# TODO: MaskedArray typing might require newer numpy versions. If so, drop the generic typing.
def unmask(mask: np.ndarray[tuple[int, int, int], np.dtype[np.bool_]], data: npt.NDArray[DType]) -> Union[np.ma.MaskedArray[tuple[int, ...], np.dtype[DType]], npt.NDArray[DType], npt.NDArray[np.uint8]]:
    """unmask(mask, data)

    Unmask the data, assuming it's been masked. Creates a volume
    the same size as `mask` containing `data` at the locations
    where `mask` is True.

    If `data` is RGB valued (dtype uint8 and last dim is 3 or 4),
    the area outside the mask will be filled with zeros.

    Otherwise, a numpy MaskedArray will be returned.

    Parameters
    ----------
    mask : array_like
        The data mask
    data : array_like
        Actual MRI data to unmask

    Returns
    -------
    unmasked : array_like
        Volume same size as `mask` but same dtype as `data`.
    """
    nvox = mask.sum()
    if data.shape[0] == nvox:
        data = data[np.newaxis]
    elif nvox not in data.shape:
        raise ValueError('Invalid mask for the data')

    if data.shape[-1] in (3, 4):
        if data.dtype != np.uint8:
            raise TypeError('Invalid dtype for raw dataset')
        #raw dataset, unmask with an alpha channel
        output = np.zeros((len(data),)+mask.shape+(4,), dtype=np.uint8)
        output[:, mask > 0, :data.shape[-1]] = data
        if data.shape[-1] == 3:
            output[:, mask > 0, 3] = 255
    else:
        #output = (np.nan*np.ones((len(data),)+mask.shape)).astype(data.dtype)
        outdata = np.zeros((len(data),)+mask.shape).astype(data.dtype)
        outdata[:, mask>0] = data
        outmask = np.tile(~mask[None,:,:,:], (len(data), 1, 1, 1))
        output = np.ma.MaskedArray(outdata, mask=outmask)

    return output.squeeze()

def detrend_median(data: npt.NDArray, kernel: int=15) -> npt.NDArray:
    from scipy.signal import medfilt
    lowfreq = medfilt(data, [1, kernel, kernel])
    return data - lowfreq

def detrend_gradient(data: npt.ArrayLike, diff: int=3) -> npt.NDArray:
    return (np.array(np.gradient(data, 1, diff, diff))**2).sum(0)

def detrend_poly(data: npt.NDArray, polyorder: int = 10, mask: Optional[npt.NDArray] = None) -> npt.NDArray:
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

def mosaic(data: npt.NDArray[DType], dim: int=0, show: bool=True, **kwargs) -> tuple[Union[npt.NDArray[DType], npt.NDArray[np.uint8]], tuple[int, int]]:
    """
    Turns volume data into a mosaic, useful for quickly viewing volumetric data
    with radiological convention (left side of figure is right side of subject).

    Parameters
    ----------
    data : array_like
        3D volumetric data to mosaic
    dim : int
        Dimension across which to mosaic. Default 0.
    show : bool
        Display mosaic with matplotlib? Default True.
    **kwargs
        Additional keyword arguments are passed to `matplotlib.pyplot.imshow`
        (e.g. `cmap`, `vmin`, `vmax`, `interpolation`, `alpha`) when `show`
        is True.

    Returns
    -------
    mosaic : ndarray
        2D mosaic image formed by tiling slices of `data` along `dim`.
    grid_shape : tuple of int
        (nwide, ntall) grid dimensions used to tile the mosaic.
    """
    if data.ndim not in (3, 4):
        raise ValueError("Invalid data shape")
    plane = list(data.shape)
    slices = plane.pop(dim)
    height, width = plane[:2]
    aspect = width / float(height)
    square = np.sqrt(slices / aspect)
    nwide = int(np.ceil(square))
    ntall = int(np.ceil(float(slices) / nwide))

    shape = (ntall * (height+1) + 1, nwide * (width+1) + 1)
    if data.dtype == np.uint8:
        output = np.zeros(shape+(4,), dtype=np.uint8)
    else:
        output = (np.nan*np.ones(shape)).astype(data.dtype)

    sl: list[Union[int, slice]] = [slice(None), slice(None), slice(None)]
    for h in range(ntall):
        for w in range(nwide):
            sl[dim] = h*nwide+w
            if sl[dim] < slices:
                hsl = slice(h*(height+1)+1, (h+1)*(height+1))
                wsl = slice(w*(width+1)+1, (w+1)*(width+1))
                if data.dtype == np.uint8:
                    output[hsl, wsl, :data.shape[3]] = data[tuple(sl)]
                    if data.shape[3] == 3:
                        output[hsl, wsl, 3] = 255
                else:    
                    output[hsl, wsl] = data[tuple(sl)]
    
    if show:
        from matplotlib import pyplot as plt
        plt.imshow(output, **kwargs)
        plt.axis('off')

    return output, (nwide, ntall)

def show_slice(dataview, **kwargs):
    """Interactively display a single anatomical slice with a functional
    volume overlaid, letting the viewed slice and axis be changed with the
    mouse.

    Scroll to move through slices along the current axis; click to switch
    to the next axis (cycling through the three spatial dimensions).

    Parameters
    ----------
    dataview : Volume
        Volumetric data to overlay on the subject's anatomical image.
    **kwargs
        Additional keyword arguments are passed to `matplotlib.pyplot.imshow`
        for the functional overlay (e.g. `cmap`, `vmin`, `vmax`), overriding
        the defaults taken from `dataview`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The interactive figure.
    """
    import nibabel
    from matplotlib import cm
    import matplotlib.pyplot as plt

    dataview = dataset.normalize(dataview)
    if not isinstance(dataview, dataset.Volume):
        raise TypeError('Only volumetric data may be visualized in show_slice')

    subject = dataview.subject
    xfmname = dataview.xfmname
    imshow_kw = dict(vmin=dataview.vmin, vmax=dataview.vmax, cmap=dataview.cmap)
    imshow_kw.update(kwargs)

    anat = db.get_anat(subject, 'raw').get_fdata().T
    data = epi2anatspace(dataview)

    data[data < dataview.vmin] = np.nan

    state = dict(slice=data.shape[0]*.66, dim=0, pad=())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    anatomical = ax.imshow(anat[state['pad'] + (state['slice'],)], cmap=cm.gray, aspect='equal')
    functional = ax.imshow(data[state['pad'] + (state['slice'],)], aspect='equal', **imshow_kw)

    def update():
        print("showing dim %d, slice %d"%(state['dim'] % 3, state['slice']))
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
    '''Display a maximum intensity projection for the data, using three subplots

    Parameters
    ----------
    data : ndarray
        3D volumetric data to project.
    **kwargs
        Additional keyword arguments are passed to `matplotlib.pyplot.imshow`
        for each subplot (e.g. `cmap`, `vmin`, `vmax`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the three maximum-intensity-projection subplots
        (one per axis).
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.add_subplot(221).imshow(data.max(0), **kwargs)
    fig.add_subplot(222).imshow(data.max(1), **kwargs)
    fig.add_subplot(223).imshow(data.max(2), **kwargs)
    return fig

def show_glass(dataview, pad=10):
    '''Create a classic "glass brain" view of the data, with the outline

    Parameters
    ----------
    dataview : Dataview
        Volumetric data to display.
    pad : int, optional
        Padding (in voxels) around the brain outline. Default 10.
    '''
    import nibabel
    nib = db.get_anat(subject, 'fiducial')
    mask = nib.get_fdata()

    left, right = np.nonzero(np.diff(mask.max(0).max(0)))[0][[0,-1]]
    front, back = np.nonzero(np.diff(mask.max(0).max(1)))[0][[0,-1]]
    top, bottom = np.nonzero(np.diff(mask.max(1).max(1)))[0][[0,-1]]

    glass = np.zeros((mask.shape[1], mask.shape[2]*2), dtype=bool)
    glass[:, :mask.shape[2]] = mask.max(0)
    #this requires a lot of logic to put the image into a canonical orientation
    #too much work for something we'll never use
    raise NotImplementedError

def epi2anatspace(volumedata, order=1):
    """Resample an epi volume data into anatomical space using scipy.

    Parameters
    ----------
    volumedata : VolumeData
        The input epi volumedata object.
    order : int
        The order of the resampler, in terms of splines. 0 is nearest, 1 is linear.

    Returns
    -------
    anatspace : ndarray
        The ND array of the anatomy space data
    """
    from scipy.ndimage import affine_transform
    ds = dataset.normalize(volumedata)
    volumedata = ds#.data

    anat = db.get_anat(volumedata.subject, "raw")
    xfm = db.get_xfm(volumedata.subject, volumedata.xfmname, "coord")

    #allxfm =  Transform(anat.affine, anat.shape).inv * xfm.inv
    allxfm = xfm * Transform(anat.affine, anat.shape)

    rotpart = allxfm.xfm[:3, :3]
    transpart = allxfm.xfm[:3,-1]
    return affine_transform(volumedata.volume.T.squeeze(), rotpart,
                            offset=transpart, output_shape=anat.shape[::-1],
                            cval=np.nan, order=order).T

def anat2epispace(anatdata: npt.NDArray, subject: str, xfmname: str, order: Literal[0, 1, 2, 3, 4, 5]=1) -> npt.NDArray:
    """Resamples data from anatomical space into epi space
    
    Parameters
    ----------
    anatdata : ndarray
        data in anatomical space
    subject : str
        Name of subject
    xfmname : str
        Name of transform
    order : int
        Order of spline interpolation
        
    Returns
    -------
    epidata : ndarray
        data in EPI space
    """
    from scipy.ndimage import affine_transform
    anatref = db.get_anat(subject)
    target = db.get_xfm(subject, xfmname, "coord")

    allxfm =  Transform(anatref.affine, anatref.shape).inv * target.inv
    #allxfm = xfm * Transform(anat.affine, anat.shape)

    rotpart = allxfm.xfm[:3, :3]
    transpart = allxfm.xfm[:3,-1]
    
    return affine_transform(anatdata.T, rotpart, offset=transpart, output_shape=target.shape[::-1], cval=np.nan, order=order).T


def epi2anatspace_fsl(volumedata):
    """Resamples epi-space [data] into the anatomical space for the given [subject]
    using the given transformation [xfm].

    Currently broken -- always raises NotImplementedError. Do not use.

    Parameters
    ----------
    volumedata : VolumeData
        Data in EPI (functional) space to resample into anatomical space.

    Returns
    -------
    anatspace : ndarray
        `volumedata` resampled into anatomical space.
    """
    #This function is currently broken! do not use it!
    raise NotImplementedError

    import tempfile
    import subprocess
    import nibabel

    volumedata = dataset.normalize(volumedata).data
    subject = volumedata.subject
    xfmname = volumedata.xfmname
    data = volumedata.volume

    ## Get transform (pycortex estimates anat-to-epi)
    xfm = db.get_xfm(subject, xfmname)
    fslxfm = xfm.to_fsl(db.get_anat(subject, 'raw').get_filename())
    ## Invert transform to epi-to-anat
    fslxfm = np.linalg.inv(fslxfm)
    ## Save out into ascii file
    xfmfilename = tempfile.mktemp(".mat")
    with open(xfmfilename, "w") as xfmh:
        for ll in fslxfm.tolist():      
            xfmh.write(" ".join(["%0.5f"%f for f in ll])+"\n")

    ## Save out data into nifti file
    datafile = nibabel.Nifti1Image(data.T, xfm.reference.affine, xfm.reference.header)
    datafilename = tempfile.mktemp(".nii")
    nibabel.save(datafile, datafilename)

    ## Reslice epi-space image
    raw = db.get_anat(subject, type='raw').get_filename()
    outfilename = tempfile.mktemp(".nii")
    subprocess.call(["fsl5.0-flirt",
                     "-ref", raw,
                     "-in", datafilename,
                     "-applyxfm",
                     "-init", xfmfilename,
                     #"-interp", "sinc",
                     "-out", outfilename])

    ## Load resliced image
    outdata = nibabel.load(outfilename+".gz").get_fdata().T
    ## Clean up
    os.remove(outfilename+".gz")
    os.remove(datafilename)
    ## Done!
    return outdata

def anat2epispace_fsl(data,subject,xfmname):
    """Resamples anat-space data into the epi space for the given [subject]
    and transformation [xfm]

    Parameters
    ----------
    data : ndarray
        Data in anatomical space to resample into EPI (functional) space.
    subject : str
        Name of subject.
    xfmname : str
        Name of transform.

    Returns
    -------
    outdata : ndarray
        `data` resampled into EPI space, at the resolution/orientation of
        the transform's reference functional image.
    """
    import tempfile
    import subprocess
    import nibabel

    ## Get transform (pycortex estimates anat-to-epi)
    xfm = db.get_xfm(subject, xfmname)
    anatNII = db.get_anat(subject, type='raw')
    fslxfm = xfm.to_fsl(anatNII.get_filename())
    ## Save out into ascii file
    xfmfilename = tempfile.mktemp(".mat")
    print('xfm file: %s'%xfmfilename)
    with open(xfmfilename, "w") as xfmh:
        for ll in fslxfm.tolist():
            xfmh.write(" ".join(["%0.5f"%f for f in ll])+"\n")

    ## Save out data into nifti file
    datafile = nibabel.Nifti1Image(data.T, anatNII.affine, anatNII.header)
    datafilename = tempfile.mktemp(".nii")
    nibabel.save(datafile, datafilename)

    ## Reslice epi-space image
    epiNIIf = xfm.reference.get_filename()
    outfilename = tempfile.mktemp(".nii")
    subprocess.call(["fsl5.0-flirt",
                     "-in", datafilename,
                     "-ref", epiNIIf,
                     "-out", outfilename,
                     "-init", xfmfilename,
                     #"-interp", "sinc",
                     "-applyxfm"])

    ## Load resliced image
    outdata = nibabel.load(outfilename+".gz").get_fdata().T
    ## Clean up
    os.remove(outfilename+".gz")
    os.remove(datafilename)
    ## Done!
    return outdata

def fslview(*ims):
    import tempfile
    import subprocess

    fnames = []
    tempfiles = []
    for im in ims:
        if not (isinstance(im, str) and os.path.exists(im)):
            import nibabel
            tf = tempfile.NamedTemporaryFile(suffix=".nii.gz")
            tempfiles.append(tf)
            nib = nibabel.Nifti1Image(im.T, np.eye(4))
            nib.to_filename(tf.name)
            fnames.append(tf.name)
        else:
            fnames.append(im)

    subprocess.call(["fslview"] + fnames)
