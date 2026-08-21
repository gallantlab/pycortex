import os

import numpy as np

from .. import dataset
from .mapper import Mapper, _savecache
from .utils import nanproject, vol2surf


def get_mapper(subject, xfmname, type='nearest', recache=False, **kwargs):
    """
    Get a Mapper object that projects data between volume and vertex
    (surface) space for the given subject and transform, loading a cached
    version from disk if one already exists.

    Parameters
    ----------
    subject : str
        Subject identifier. Must exist in the pycortex database.
    xfmname : str
        Transform name. Must exist in the pycortex database.
    type : str, optional
        Mapping method to use for projecting between volume and surface
        space. One of 'nearest', 'trilinear', 'gaussian', 'lanczos',
        'const_patch_nn', 'const_patch_trilin', 'const_patch_lanczos',
        'line_nearest', 'line_trilinear', 'line_lanczos'. Default is
        'nearest'.
    recache : bool, optional
        If True, ignore any cached mapper file on disk and recompute it.
        Default is False.
    **kwargs
        Extra keyword arguments are passed to the chosen mapper's sampling
        function, and which ones matter depends on `type`:
            sigma : float (default 1)
                Standard deviation of the gaussian kernel. Only used when
                type='gaussian'.
            window : int (default 3)
                Kernel window size, in voxels. Used when type='gaussian' or
                type='lanczos'.
            npts : int (default 64)
                Number of random sample points per patch/line segment. Used
                by the 'const_patch_*' and 'line_*' types.
            mp : bool (default True)
                Whether to parallelize sampling across processes. Used by
                the 'const_patch_*' types.

    Returns
    -------
    Mapper
        A Mapper subclass instance (e.g. PointNN, LineTrilin, ...) that can
        project data between volume and vertex space for this
        subject/transform.
    """
    from ..database import db
    from . import point, patch, line

    mapcls = dict(
        nearest=point.PointNN,
        trilinear=point.PointTrilin,
        gaussian=point.PointGauss,
        lanczos=point.PointLanczos,
        const_patch_nn=patch.ConstPatchNN,
        const_patch_trilin=patch.ConstPatchTrilin,
        const_patch_lanczos=patch.ConstPatchLanczos,
        line_nearest=line.LineNN,
        line_trilinear=line.LineTrilin,
        line_lanczos=line.LineLanczos)
    Map = mapcls[type]
    ptype = Map.__name__.lower()
    kwds ='_'.join(['%s%s'%(k,str(v)) for k, v in list(kwargs.items())])
    if len(kwds) > 0:
        ptype += '_'+kwds

    fname = "{xfmname}_{projection}.npz".format(xfmname=xfmname, projection=ptype)

    xfmfile = db.get_paths(subject)['xfmdir'].format(xfmname=xfmname)
    cachefile = os.path.join(db.get_cache(subject), fname)

    try:
        if not recache and (xfmname == "identity" or os.stat(cachefile).st_mtime > os.stat(xfmfile).st_mtime):
            return Map.from_cache(cachefile, subject, xfmname)
        raise Exception
    except Exception:
        return Map._cache(cachefile, subject, xfmname, **kwargs)
