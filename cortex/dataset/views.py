import os
import glob
import json
import h5py
import numpy as np

from .. import options
from .braindata import BrainData, VolumeData, VertexData

default_cmap = options.config.get("basic", "default_cmap")


def normalize(data):
    if isinstance(data, tuple):
        if len(data) == 3:
            if data[0].dtype == np.uint8:
                return VolumeRGB(data[0][...,0], data[0][...,1], data[0][...,2], *data[1:])
            return Volume(*data)
        elif len(data) == 2:
            return Vertex(*data)
        else:
            raise TypeError("Invalid input for Dataview")
    elif isinstance(data, Dataview):
        return data
    else:
        raise TypeError("Invalid input for Dataview")

def _from_hdf_data(h5, name, xfmname=None, subject=None, **kwargs):
    """Decodes a __hash named node from an HDF file into the 
    constituent Vertex or Volume object"""
    dnode = h5.get("/data/%s"%name)
    if dnode is None:
        dnode = h5.get(name)

    attrs = {k: u(v) for (k, v) in dnode.attrs.items()}
    if subject is None:
        subject = attrs['subject']
    #support old style xfmname saving as attribute
    if xfmname is None and 'xfmname' in attrs:
        xfmname = attrs['xfmname']
    mask = None
    if 'mask' in attrs:
        if attrs['mask'].startswith("__"):
            mask = h5['/subjects/%s/transforms/%s/masks/%s' %
                      (attrs['subject'], xfmname, attrs['mask'])].value
        else:
            mask = attrs['mask']

    #support old style RGB volumes
    if dnode.dtype == np.uint8 and dnode.shape[-1] in (3, 4):
        alpha = None
        if dnode.shape[-1] == 4:
            alpha = dnode[..., 3]

        if xfmname is None:
            return VertexRGB(dnode[...,0], dnode[...,1], dnode[...,2], subject, 
                             alpha=alpha, **kwargs)

        return VolumeRGB(dnode[...,0], dnode[...,1], dnode[...,2], subject, xfmname, 
                         alpha=alpha, mask=mask, **kwargs)

    if xfmname is None:
        return Vertex(dnode, subject, **kwargs)
    
    return Volume(dnode, subject, xfmname, mask=mask, **kwargs)
        

def _from_hdf_view(h5, data, xfmname=None, vmin=None, vmax=None,  subject=None, **kwargs):

    if isinstance(data, str):
        return _from_hdf_data(h5, data, xfmname=xfmname, vmin=vmin, vmax=vmax, subject=subject, **kwargs)
        
    if len(data) == 2:
        dim1 = _from_hdf_data(h5, data[0], xfmname=xfmname[0], subject=subject)
        dim2 = _from_hdf_data(h5, data[1], xfmname=xfmname[1], subject=subject)
        cls = Vertex2D if isinstance(dim1, Vertex) else Volume2D
        return cls(dim1, dim2, vmin=vmin[0], vmin2=vmin[1], 
                   vmax=vmax[0], vmax2=vmax[1], subject=subject, **kwargs)
    elif len(data) == 4:
        red, green, blue = [_from_hdf_data(h5, d, xfmname=xfmname, subject=subject) for d in data[:3]]
        alpha = None 
        if data[3] is not None:
            alpha = _from_hdf_data(h5, data[3], xfmname=xfmname, subject=subject)

        cls = VertexRGB if isinstance(red, Vertex) else VolumeRGB
        return cls(red, green, blue, alpha=alpha, subject=subject, **kwargs)
    else:
        raise ValueError("Invalid Dataview specification")

class Dataview(object):
    def __init__(self, cmap=None, vmin=None, vmax=None, description="", state=None, **kwargs):
        if self.__class__ == Dataview:
            raise TypeError('Cannot directly instantiate Dataview objects')

        self.cmap = cmap if cmap is not None else default_cmap
        self.vmin = vmin
        self.vmax = vmax
        self.state = state
        self.attrs = kwargs
        if 'priority' not in self.attrs:
            self.attrs['priority'] = 1
        self.description = description

    def copy(self, *args, **kwargs):
        kwargs.update(self.attrs)
        return self.__class__(*args, 
                              cmap=self.cmap, 
                              vmin=self.vmin, 
                              vmax=self.vmax, 
                              description=self.description, 
                              state=self.state, 
                              **kwargs)

    @property
    def priority(self):
        return self.attrs['priority']

    @priority.setter
    def priority(self, value):
        self.attrs['priority'] = value

    def to_json(self, simple=False):
        if simple:
            return dict()

        desc = self.description
        if hasattr(desc, 'decode'):
            desc = desc.decode()
        sdict = dict(
            state=self.state,
            attrs=self.attrs.copy(),
            desc=desc)
        try:
            sdict.update(dict(
                cmap=[self.cmap],
                vmin=[self.vmin if self.vmin is not None else np.percentile(np.nan_to_num(self.data), 1)],
                vmax=[self.vmax if self.vmax is not None else np.percentile(np.nan_to_num(self.data), 99)]
                ))
        except AttributeError:
            pass
        return sdict

    @staticmethod
    def from_hdf(node, subject=None):
        data = json.loads(u(node[0]))
        desc = node[1]
        try:
            cmap = json.loads(u(node[2]))
        except:
            cmap = u(node[2])
        vmin = json.loads(u(node[3]))
        vmax = json.loads(u(node[4]))
        state = json.loads(u(node[5]))
        attrs = json.loads(u(node[6]))
        try:
            xfmname = json.loads(u(node[7]))
        except ValueError:
            xfmname = None

        if not isinstance(vmin, list):
            vmin = [vmin]
        if not isinstance(vmax, list):
            vmax = [vmax]
        if not isinstance(cmap, list):
            cmap = [cmap]

        if len(data) == 1:
            xfm = None if xfmname is None else xfmname[0]
            return _from_hdf_view(node.file, data[0], xfmname=xfm, cmap=cmap[0], description=desc, 
                                  vmin=vmin[0], vmax=vmax[0], state=state, subject=subject, **attrs)
        else:
            views = [_from_hdf_view(node.file, d, xfmname=x, subject=subject) for d, x in zip(data, xfmname)]
            raise NotImplementedError

    def _write_hdf(self, h5, name="data", data=None, xfmname=None):
        views = h5.require_group("/views")
        view = views.require_dataset(name, (8,), h5py.special_dtype(vlen=str))
        view[0] = json.dumps(data)
        view[1] = self.description
        try:
            view[2] = json.dumps([self.cmap])
            view[3] = json.dumps([self.vmin])
            view[4] = json.dumps([self.vmax])
        except AttributeError:
            #For VolumeRGB/Vertex, there is no cmap/vmin/vmax
            view[2] = "null"
            view[3:5] = "null"
        view[5] = json.dumps(self.state)
        view[6] = json.dumps(self.attrs)
        view[7] = json.dumps(xfmname)
        return view

    def get_cmapdict(self):
        """Returns a dictionary with cmap information."""

        from matplotlib import colors, pyplot as plt

        try:
            # plt.get_cmap accepts:
            # - matplotlib colormap names
            # - pycortex colormap names previously registered in matplotlib
            # - matplotlib.colors.Colormap instances
            cmap = plt.get_cmap(self.cmap)
        except ValueError:
            # unknown colormap, test whether it's in pycortex colormaps
            cmapdir = options.config.get('webgl', 'colormaps')
            colormaps = glob.glob(os.path.join(cmapdir, "*.png"))
            colormaps = dict(((os.path.split(c)[1][:-4], c) for c in colormaps))
            if not self.cmap in colormaps:
                raise ValueError('Unkown color map %s' % self.cmap)
            I = plt.imread(colormaps[self.cmap])
            cmap = colors.ListedColormap(np.squeeze(I))
            # Register colormap to matplotlib to avoid loading it again
            plt.register_cmap(self.cmap, cmap)

        return dict(cmap=cmap, vmin=self.vmin, vmax=self.vmax)

    @property
    def raw(self):
        from matplotlib import colors, cm

        cmap = self.get_cmapdict()['cmap']
        # Normalize colors according to vmin, vmax
        norm = colors.Normalize(self.vmin, self.vmax) 
        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        color_data = cmapper.to_rgba(self.data.flatten()).reshape(self.data.shape+(4,))
        # rollaxis puts the last color dimension first, to allow output of separate channels: r,g,b,a = dataset.raw
        color_data = (np.clip(color_data, 0, 1) * 255).astype(np.uint8)
        return np.rollaxis(color_data, -1)


class Multiview(Dataview):
    def __init__(self, views, description=""):
        for view in views:
            if not isinstance(view, Dataview):
                raise TypeError("Must be a View object!")
        raise NotImplementedError
        self.views = views

    def uniques(self, collapse=False):
        for view in self.views:
            for sv in view.uniques(collapse=collapse):
                yield sv

class Volume(VolumeData, Dataview):
    """
    Encapsulates a 3D volume or 4D volumetric movie. Includes information on how
    the volume should be colormapped for display purposes.

    Parameters
    ----------
    data : ndarray
        The data. Can be 3D with shape (z,y,x), 1D with shape (v,) for masked data,
        4D with shape (t,z,y,x), or 2D with shape (t,v). For masked data, if the
        size of the given array matches any of the existing masks in the database,
        that mask will automatically be loaded. If it does not, an error will be 
        raised.
    subject : str
        Subject identifier. Must exist in the pycortex database.
    xfmname : str
        Transform name. Must exist in the pycortex database.
    mask : ndarray, optional
        Binary 3D array with shape (z,y,x) showing which voxels are selected.
        If masked data is given, the mask will automatically be loaded if it 
        exists in the pycortex database.
    cmap : str or matplotlib colormap, optional
        Colormap (or colormap name) to use. If not given defaults to matplotlib
        default colormap.
    vmin : float, optional
        Minimum value in colormap. If not given, defaults to the 1st percentile
        of the data.
    vmax : float, optional
        Maximum value in colormap. If not given defaults to the 99th percentile
        of the data.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    **kwargs
        All additional arguments in kwargs are passed to the VolumeData and Dataview

    """
    def __init__(self, data, subject, xfmname, mask=None, 
                 cmap=None, vmin=None, vmax=None, description="", **kwargs):
        super(Volume, self).__init__(data, subject, xfmname, mask=mask, 
                                     cmap=cmap, vmin=vmin, vmax=vmax,
                                     description=description, **kwargs)
        # set vmin and vmax
        self.vmin = self.vmin if self.vmin is not None else \
            np.percentile(np.nan_to_num(self.data), 1)
        self.vmax = self.vmax if self.vmax is not None else \
            np.percentile(np.nan_to_num(self.data), 99)

    def _write_hdf(self, h5, name="data"):
        datanode = VolumeData._write_hdf(self, h5)
        viewnode = Dataview._write_hdf(self, h5, name=name,
                                       data=[self.name],
                                       xfmname=[self.xfmname])
        return viewnode

    @property
    def raw(self):
        r, g, b, a = super(Volume, self).raw
        return VolumeRGB(r, g, b, self.subject, self.xfmname, a, 
                         description=self.description, state=self.state,
                         **self.attrs)

class Vertex(VertexData, Dataview):
    """
    Encapsulates a 1D vertex map or 2D vertex movie. Includes information on how
    the data should be colormapped for display purposes.

    Parameters
    ----------
    data : ndarray
        The data. Can be 1D with shape (v,), or 2D with shape (t,v). Here, v can
        be the number of vertices in both hemispheres, or the number of vertices
        in either one of the hemispheres. In that case, the data for the other 
        hemisphere will be filled with zeros.
    subject : str
        Subject identifier. Must exist in the pycortex database.
    cmap : str or matplotlib colormap, optional
        Colormap (or colormap name) to use. If not given defaults to matplotlib
        default colormap.
    vmin : float, optional
        Minimum value in colormap. If not given, defaults to the 1st percentile
        of the data.
    vmax : float, optional
        Maximum value in colormap. If not given defaults to the 99th percentile
        of the data.
    description : str, optional
        String describing this dataset. Displayed in webgl viewer.
    **kwargs
        All additional arguments in kwargs are passed to the VolumeData and Dataview

    """
    def __init__(self, data, subject, cmap=None, vmin=None, vmax=None, description="", **kwargs):
        super(Vertex, self).__init__(data, subject, cmap=cmap, vmin=vmin, vmax=vmax, 
                                     description=description, **kwargs)
        # set vmin and vmax
        self.vmin = self.vmin if self.vmin is not None else \
            np.percentile(np.nan_to_num(self.data), 1)
        self.vmax = self.vmax if self.vmax is not None else \
            np.percentile(np.nan_to_num(self.data), 99)

    def _write_hdf(self, h5, name="data"):
        datanode = VertexData._write_hdf(self, h5)
        viewnode = Dataview._write_hdf(self, h5, name=name, data=[self.name])
        return viewnode

    @property
    def raw(self):
        r, g, b, a = super(Vertex, self).raw
        return VertexRGB(r, g, b, self.subject, a, 
                         description=self.description, state=self.state,
                         **self.attrs)

    def map(self, target_subj, surface_type='fiducial', 
            hemi='both', fs_subj=None, **kwargs):
        """Map this data from this surface to another surface
        
        Calls `cortex.freesurfer.vertex_to_vertex()`  with this 
        vertex object as the first argument.

        NOTE: Requires either previous computation of mapping matrices
        (with `cortex.db.get_mri_surf2surf_matrix`) or active 
        freesurfer environment.

        Parameters
        ----------
        target_subj : str
            freesurfer subject to which to map
        
        Other Parameters
        ----------------
        kwargs map to `cortex.freesurfer.vertex_to_vertex()`
        """
        # Input check
        if hemi not in ['lh', 'rh', 'both']:
            raise ValueError("`hemi` kwarg must be 'lh', 'rh', or 'both'")
        # lazy load
        from ..database import db
        mats = db.get_mri_surf2surf_matrix(self.subject, surface_type, 
                hemi='both', target_subj=target_subj, fs_subj=fs_subj, 
                **kwargs)
        new_data = [mats[0].dot(self.left), mats[1].dot(self.right)]
        if hemi == 'both':
            new_data = np.hstack(new_data)
        elif hemi == 'lh':
            new_data = np.hstack([new_data[0], np.nan * np.zeros(new_data[1].shape)])
        elif hemi == 'rh':
            new_data = np.hstack([np.nan * np.zeros(new_data[0].shape), new_data[1]])
        vx = Vertex(new_data, target_subj, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        return vx
        
def u(s, encoding='utf8'):
    try:
        return s.decode(encoding)
    except AttributeError:
        return s


from .viewRGB import VolumeRGB, VertexRGB, Colors
from .view2D import Volume2D, Vertex2D
