import json
import warnings
import h5py
import numpy as np
from .. import options
from ..database import db
from .braindata import BrainData, VolumeData, VertexData, _hash

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

def _from_hdf_data(h5, name, xfmname=None, **kwargs):
    """Decodes a __hash named node from an HDF file into the 
    constituent Vertex or Volume object"""
    dnode = h5.get("/data/%s"%name)
    if dnode is None:
        dnode = h5.get(name)

    subj = dnode.attrs['subject']
    #support old style xfmname saving as attribute
    if xfmname is None and 'xfmname' in dnode.attrs:
        xfmname = dnode.attrs['xfmname']

    mask = None
    if "mask" in dnode.attrs:
        if dnode.attrs['mask'].startswith("__"):
            mask = h5['/subjects/%s/transforms/%s/masks/%s'%(dnode.attrs['subject'], xfmname, dnode.attrs['mask'])].value
        else:
            mask = dnode.attrs['mask']

    #support old style RGB volumes
    if dnode.dtype == np.uint8 and dnode.shape[-1] in (3, 4):
        alpha = None
        if dnode.shape[-1] == 4:
            alpha = dnode[..., 3]

        if xfmname is None:
            return VertexRGB(dnode[...,0], dnode[...,1], dnode[...,2], subj, 
                alpha=alpha, **kwargs)

        return VolumeRGB(dnode[...,0], dnode[...,1], dnode[...,2], subj, xfmname, 
            alpha=alpha, mask=mask, **kwargs)

    if xfmname is None:
        return Vertex(dnode, subj, **kwargs)
    
    return Volume(dnode, subj, xfmname, mask=mask, **kwargs)
        

def _from_hdf_view(h5, data, xfmname=None, vmin=None, vmax=None,  **kwargs):
    try:
        basestring
        strcls = (unicode, str)
    except NameError:
        strcls = str

    if isinstance(data, strcls):
        return _from_hdf_data(h5, data, xfmname=xfmname, vmin=vmin, vmax=vmax, **kwargs)
        
    if len(data) == 2:
        dim1 = _from_hdf_data(h5, data[0], xfmname=xfmname[0])
        dim2 = _from_hdf_data(h5, data[1], xfmname=xfmname[1])
        cls = Vertex2D if isinstance(dim1, Vertex) else Volume2D
        return cls(dim1, dim2, vmin=vmin[0], vmin2=vmin[1], 
            vmax=vmax[0], vmax2=vmax[1], **kwargs)
    elif len(data) == 4:
        red, green, blue = [_from_hdf_data(h5, d, xfmname=xfmname) for d in data[:3]]
        alpha = None 
        if data[3] is not None:
            alpha = _from_hdf_data(h5, data[3], xfmname=xfmname)

        cls = VertexRGB if isinstance(red, Vertex) else VolumeRGB
        return cls(red, green, blue, alpha=alpha, **kwargs)
    else:
        raise ValueError("Invalid Dataview specification")

class Dataview(object):
    def __init__(self, cmap=None, vmin=None, vmax=None, description="", state=None, 
        cvmin=None,cvmax=None,cvthr=False,**kwargs):
        """
        MOAR HELP PLEASE. or maybe not. Is this even visible in inherited classes?

        cvmin : float,optional
            Minimum value for curvature colormap. Defaults to config file value.
        cvmax : float, optional
            Maximum value for background curvature colormap. Defaults to config file value.
        cvthr : bool,optional
            Apply threshold to background curvature
        """
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
            
        sdict = dict(
            state=self.state, 
            attrs=self.attrs.copy(), 
            desc=self.description)
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
    def from_hdf(node):
        data = json.loads(node[0])
        desc = node[1]
        try:
            cmap = json.loads(node[2])
        except:
            cmap = node[2]
        vmin = json.loads(node[3])
        vmax = json.loads(node[4])
        state = json.loads(node[5])
        attrs = json.loads(node[6])
        try:
            xfmname = json.loads(node[7])
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
                vmin=vmin[0], vmax=vmax[0], state=state, **attrs)
        else:
            views = [_from_hdf_view(node.file, d, xfmname=x) for d, x in zip(data, xfname)]
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
            view[2] = None
            view[3:5] = "null"
        view[5] = json.dumps(self.state)
        view[6] = json.dumps(self.attrs)
        view[7] = json.dumps(xfmname)
        return view

    @property
    def raw(self):
        from matplotlib import colors, cm, pyplot as plt
        import glob, os
        # Get colormap from matplotlib or pycortex colormaps
        ## -- redundant code, here and in cortex/quicklflat.py -- ##
        if isinstance(self.cmap,(str,unicode)):
            if not self.cmap in cm.__dict__:
                # unknown colormap, test whether it's in pycortex colormaps
                cmapdir = options.config.get('webgl', 'colormaps')
                colormaps = glob.glob(os.path.join(cmapdir, "*.png"))
                colormaps = dict(((os.path.split(c)[1][:-4],c) for c in colormaps))
                if not self.cmap in colormaps:
                    raise Exception('Unkown color map!')
                I = plt.imread(colormaps[self.cmap])
                cmap = colors.ListedColormap(np.squeeze(I))
                # Register colormap while we're at it
                cm.register_cmap(self.cmap,cmap)
            else:
                cmap = cm.get_cmap(self.cmap)
        elif isinstance(self.cmap,colors.Colormap):
            cmap = self.cmap
        # Normalize colors according to vmin, vmax
        norm = colors.Normalize(self.vmin, self.vmax) 
        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        color_data = cmapper.to_rgba(self.data.flatten()).reshape(self.data.shape+(4,))
        # rollaxis puts the last color dimension first, to allow output of separate channels: r,g,b,a = dataset.raw
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
    def __init__(self, data, subject, xfmname, mask=None, 
        cmap=None, vmin=None, vmax=None, description="", **kwargs):
        super(Volume, self).__init__(data, subject, xfmname, mask=mask, 
            cmap=cmap, vmin=vmin, vmax=vmax, description=description, **kwargs)

    def _write_hdf(self, h5, name="data"):
        datanode = VolumeData._write_hdf(self, h5)
        viewnode = Dataview._write_hdf(self, h5, name=name,
            data=[self.name], xfmname=[self.xfmname])
        return viewnode

    @property
    def raw(self):
        r, g, b, a = super(Volume, self).raw
        return VolumeRGB(r, g, b, self.subject, self.xfmname, a, 
            description=self.description, state=self.state, **self.attrs)

class Vertex(VertexData, Dataview):
    def __init__(self, data, subject, cmap=None, vmin=None, vmax=None, description="", **kwargs):
        super(Vertex, self).__init__(data, subject, cmap=cmap, vmin=vmin, vmax=vmax, 
            description=description, **kwargs)

    def _write_hdf(self, h5, name="data"):
        datanode = VertexData._write_hdf(self, h5)
        viewnode = Dataview._write_hdf(self, h5, name=name, data=[self.name])
        return viewnode

    @property
    def raw(self):
        r, g, b, a = super(Vertex, self).raw
        return VertexRGB(r, g, b, self.subject, a, 
            description=self.description, state=self.state, **self.attrs)

from .viewRGB import VolumeRGB, VertexRGB
from .view2D import Volume2D, Vertex2D
