import os
import json
import numpy as np

from .. import options
from .views import Dataview, Volume, Vertex, VolumeRGB, VertexRGB
from .braindata import VolumeData, VertexData

default_cmap2D = options.config.get("basic", "default_cmap2D")

class Dataview2D(Dataview):
    def __init__(self, description="", cmap=None, vmin=None, vmax=None, vmin2=None, vmax2=None, state=None, **kwargs):
        self.cmap = cmap or default_cmap2D
        self.vmin = vmin
        self.vmax = vmax
        self.vmin2 = vmin if vmin2 is None else vmin2
        self.vmax2 = vmax if vmax2 is None else vmax2
        self.state = state
        self.attrs = kwargs
        if 'priority' not in self.attrs:
            self.attrs['priority'] = 1
        self.description = description

    def uniques(self, collapse=False):
        yield self.dim1
        yield self.dim2

    def _write_hdf(self, h5, name="data"):
        self._cls._write_hdf(self.dim1, h5)
        self._cls._write_hdf(self.dim2, h5)

        viewnode = Dataview._write_hdf(self, h5, name=name)
        viewnode[0] = json.dumps([[self.dim1.name, self.dim2.name]])
        viewnode[3] = json.dumps([[self.vmin, self.vmin2]])
        viewnode[4] = json.dumps([[self.vmax, self.vmax2]])
        return viewnode

    def to_json(self, simple=False):
        sdict = dict(data=[[self.dim1.name, self.dim2.name]],
            state=self.state, 
            attrs=self.attrs, 
            desc=self.description,
            cmap=[self.cmap] )

        d1js = self.dim1.to_json()
        d2js = self.dim2.to_json()
        sdict.update(dict(
            vmin = [[self.vmin or d1js['vmin'][0], self.vmin2 or d2js['vmin'][0]]],
            vmax = [[self.vmax or d1js['vmax'][0], self.vmax2 or d2js['vmax'][0]]],
            ))

        if "xfm" in d1js:
            sdict['xfm'] = [[d1js['xfm'][0], d2js['xfm'][0]]]

        return sdict

    def _to_raw(self, data1, data2):
        from matplotlib import pyplot as plt
        from matplotlib.colors import Normalize
        cmapdir = options.config.get("webgl", "colormaps")
        cmap = plt.imread(os.path.join(cmapdir, "%s.png"%self.cmap))

        norm1 = Normalize(self.vmin, self.vmax)
        norm2 = Normalize(self.vmin2, self.vmax2)
        
        d1 = np.clip(norm1(data1), 0, 1)
        d2 = np.clip(1 - norm2(data2), 0, 1)
        dim1 = np.round(d1 * (cmap.shape[1]-1))
        # Nans in data seemed to cause weird interaction with conversion to uint32
        dim1 = np.nan_to_num(dim1).astype(np.uint32) 
        dim2 = np.round(d2 * (cmap.shape[0]-1))
        dim2 = np.nan_to_num(dim2).astype(np.uint32)

        colored = cmap[dim2.ravel(), dim1.ravel()]
        r, g, b, a = colored.T
        r.shape = dim1.shape
        g.shape = dim1.shape
        b.shape = dim1.shape
        a.shape = dim1.shape
        # Preserve nan values as alpha = 0
        aidx = np.logical_or(np.isnan(data1),np.isnan(data2))
        a[aidx] = 0
        return r, g, b, a

    @property
    def subject(self):
        return self.dim1.subject

class Volume2D(Dataview2D):
    _cls = VolumeData
    def __init__(self, dim1, dim2, subject=None, xfmname=None, description="", cmap=None,
        vmin=None, vmax=None, vmin2=None, vmax2=None, **kwargs):
        if isinstance(dim1, self._cls):
            if subject is not None or xfmname is not None:
                raise TypeError("Subject and xfmname cannot be specified with Volumes")
            if not isinstance(dim2, self._cls) or dim2.subject != dim1.subject:
                raise TypeError("Invalid data for second dimension")
            self.dim1 = dim1
            self.dim2 = dim2
            vmin = dim1.vmin if vmin is None else vmin
            vmin2 = dim2.vmin if vmin2 is None else vmin2
            vmax = dim1.vmax if vmax is None else vmax
            vmax2 = dim2.vmax if vmax2 is None else vmax2
        else:
            self.dim1 = Volume(dim1, subject, xfmname)
            self.dim2 = Volume(dim2, subject, xfmname)

        super(Volume2D, self).__init__(description=description, cmap=cmap, vmin=vmin,
                                       vmax=vmax, vmin2=vmin2, vmax2=vmax2, **kwargs)

    def __repr__(self):
        return "<2D volumetric data for (%s, %s)>"%(self.dim1.subject, self.dim1.xfmname)

    def _write_hdf(self, h5, name="data"):
        viewnode = super(Volume2D, self)._write_hdf(h5, name)
        viewnode[7] = json.dumps([[self.dim1.xfmname, self.dim2.xfmname]])
        return viewnode

    @property
    def raw(self):
        """Implement 2D colormapping for quickflat"""
        if self.dim1.xfmname != self.dim2.xfmname:
            raise ValueError("Both Volumes must have same xfmname to generate single raw volume")

        if ((self.dim1.linear and self.dim2.linear) and
            (self.dim1.mask.shape == self.dim2.mask.shape) and
            np.all(self.dim1.mask == self.dim2.mask)):
            r, g, b, a = self._to_raw(self.dim1.data, self.dim2.data)
        else:
            r, g, b, a = self._to_raw(self.dim1.volume, self.dim2.volume)
        if 'alpha'  in self.attrs:
            # Over-write alpha from colormap with alpha arg if provided.
            # Question: Might it be important tokeep alpha as an attr?
            a = self.attrs.pop('alpha')
        return VolumeRGB(r, g, b, alpha=a, subject=self.dim1.subject, xfmname=self.dim1.xfmname, 
            state=self.state, description=self.description, **self.attrs)

    @property
    def xfmname(self):
        return self.dim1.xfmname

class Vertex2D(Dataview2D):
    _cls = VertexData
    def __init__(self, dim1, dim2, subject=None, description="", cmap=None,
        vmin=None, vmax=None, vmin2=None, vmax2=None, **kwargs):
        if isinstance(dim1, VertexData):
            if subject is not None:
                raise TypeError("Subject cannot be specified with Volumes")
            if not isinstance(dim2, VertexData) or dim2.subject != dim1.subject:
                raise TypeError("Invalid data for second dimension")
            self.dim1 = dim1
            self.dim2 = dim2
            vmin = dim1.vmin if vmin is None else vmin
            vmin2 = dim2.vmin if vmin2 is None else vmin2
            vmax = dim1.vmax if vmax is None else vmax
            vmax2 = dim2.vmax if vmax2 is None else vmax2
        else:
            self.dim1 = Vertex(dim1, subject)
            self.dim2 = Vertex(dim2, subject)

        super(Vertex2D, self).__init__(description=description, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    def __repr__(self):
        return "<2D vertex data for (%s)>"%self.dim1.subject

    @property
    def raw(self):
        r, g, b, a = self._to_raw(self.dim1.data, self.dim2.data)
        return VertexRGB(r, g, b, alpha=a, subject=self.dim1.subject)

    @property
    def vertices(self):
        return self.raw.vertices
