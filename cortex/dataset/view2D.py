import json
import numpy as np

from .. import options
from .views import Dataview, Volume, Vertex
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

class Vertex2D(Dataview):
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
            vmin, vmin2 = dim1.vmin, dim2.vmin
            vmax, vmax2 = dim1.vmax, dim2.vmax
        else:
            self.dim1 = Vertex(dim1, subject)
            self.dim2 = Vertex(dim2, subject)

        super(Vertex2D, self).__init__(description=description, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    def __repr__(self):
        return "<2D vertex data for (%s)>"%self.dim1.subject
