import json
import numpy as np

from .views import Dataview, Volume, Vertex
from .braindata import VolumeData, VertexData, _hash

class DataviewRGB(Dataview):
    def __init__(self, subject=None, alpha=None, description="", state=None, **kwargs):
        self.alpha = alpha
        self.subject = self.red.subject
        self.movie = self.red.movie
        self.description = description
        self.state = state
        self.attrs = kwargs
        if 'priority' not in self.attrs:
            self.attrs['priority'] = 1

        # If movie, make sure each channel has the same number of time points
        if self.red.movie:
            if not self.red.data.shape[0] == self.green.data.shape[0] == self.blue.data.shape[0]:
                raise ValueError("For movie data, all three channels have to be the same length")

    def uniques(self, collapse=False):
        if collapse:
            yield self
        else:
            yield self.red
            yield self.green
            yield self.blue
            if self.alpha is not None:
                yield self.alpha

    def _write_hdf(self, h5, name="data", xfmname=None):
        self._cls._write_hdf(self.red, h5)
        self._cls._write_hdf(self.green, h5)
        self._cls._write_hdf(self.blue, h5)

        alpha = None
        if self.alpha is not None:
            self._cls._write_hdf(self.alpha, h5)
            alpha = self.alpha.name

        data = [self.red.name, self.green.name, self.blue.name, alpha]
        viewnode = Dataview._write_hdf(self, h5, name=name, 
            data=[data], xfmname=xfmname)

        return viewnode

    def to_json(self, simple=False):
        sdict = super(DataviewRGB, self).to_json(simple=simple)

        if simple:
            sdict['name'] = self.name
        else:
            sdict['data'] = [self.name]

        return sdict

class VolumeRGB(DataviewRGB):
    _cls = VolumeData
    def __init__(self, red, green, blue, subject=None, xfmname=None, alpha=None, description="", 
        state=None, **kwargs):
        if isinstance(red, VolumeData):
            if not isinstance(green, VolumeData) or red.subject != green.subject:
                raise TypeError("Invalid data for green channel")
            if not isinstance(blue, VolumeData) or red.subject != blue.subject:
                raise TypeError("Invalid data for blue channel")
            self.red = red
            self.green = green
            self.blue = blue
        else:
            if subject is None or xfmname is None:
                raise TypeError("Subject and xfmname are required")
            self.red = Volume(red, subject, xfmname)
            self.green = Volume(green, subject, xfmname)
            self.blue = Volume(blue, subject, xfmname)

        self.xfmname = self.red.xfmname
        super(VolumeRGB, self).__init__(subject, alpha, description=description, state=state, **kwargs)

    @property
    def volume(self):
        alpha = self.alpha
        if self.alpha is None:
            alpha = np.ones_like(self.red.volume)

        if not isinstance(alpha, Volume):
            alpha = Volume(alpha, self.subject, self.xfmname)

        volume = []
        for dv in (self.red, self.green, self.blue, alpha):
            vol = dv.volume
            if vol.dtype != np.uint8:
                if dv.vmin is None:
                    if vol.min() < 0:
                        vol -= vol.min()
                else:
                    vol -= dv.vmin()

                if dv.vmax is None:
                    if vol.max() > 1:
                        vol /= vol.max()
                else:
                    vol /= dv.vmax
                vol = (vol * 255).astype(np.uint8)
            volume.append(vol)

        return np.array(volume).transpose([1, 2, 3, 4, 0])

    def __repr__(self):
        return "<RGB volumetric data for (%s, %s)>"%(self.red.subject, self.red.xfmname)

    def __hash__(self):
        return hash(_hash(self.volume))

    @property
    def name(self):
        return "__%s"%_hash(self.volume)[:16]

    def _write_hdf(self, h5, name="data"):
        return super(VolumeRGB, self)._write_hdf(h5, name=name, xfmname=[self.xfmname])

class VertexRGB(DataviewRGB):
    _cls = VertexData
    def __init__(self, red, green, blue, subject=None, alpha=None, description="", 
        state=None, **kwargs):

        if isinstance(red, VertexData):
            if not isinstance(green, VertexData) or red.subject != green.subject:
                raise TypeError("Invalid data for green channel")
            if not isinstance(blue, VertexData) or red.subject != blue.subject:
                raise TypeError("Invalid data for blue channel")
            self.red = red
            self.green = green
            self.blue = blue
        else:
            if subject is None:
                raise TypeError("Subject name is required")
            self.red = Vertex(red, subject)
            self.green = Vertex(green, subject)
            self.blue = Vertex(blue, subject)

        super(VertexRGB, self).__init__(subject, alpha, description=description, 
            state=state, **kwargs)

    @property
    def vertices(self):
        alpha = self.alpha
        if alpha is None:
            alpha = np.ones_like(self.red.data)
        if not isinstance(alpha, Vertex):
            alpha = Vertex(alpha, self.subject)

        verts = []
        for dv in (self.red, self.green, self.blue, alpha):
            vert = dv.vertices
            if vert.dtype != np.uint8:
                if vert.min() < 0:
                    vert -= vert.min()
                if vert.max() > 1:
                    vert /= vert.max()
                vert = (vert * 255).astype(np.uint8)
            verts.append(vert)

        return np.array(verts).transpose([1, 2, 0])

    @property
    def left(self):
        return self.vertices[:,:self.red.llen]

    @property
    def right(self):
        return self.vertices[:,self.red.llen:]

    def __repr__(self):
        return "<RGB vertex data for (%s)>"%(self.subject)

    def __hash__(self):
        return hash(_hash(self.vertices))

    @property
    def name(self):
        return "__%s"%_hash(self.vertices)[:16]
