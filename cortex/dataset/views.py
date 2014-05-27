import json

import h5py
import numpy as np

from .. import options
from . import BrainData, VolumeData, VertexData

default_cmap = options.config.get("basic", "default_cmap")

def normalize(data):
    if isinstance(data, tuple):
        if len(data) == 3:
            return Volume(*data)
        elif len(data) == 2:
            return Vertex(*data)
        else:
            raise TypeError("Invalid input for DataView")
    elif isinstance(data, DataView):
        return data
    else:
        raise TypeError("Invalid input for DataView")

class View(object):
    def __init__(self, cmap=None, vmin=None, vmax=None, state=None, **kwargs):
        self.cmap = cmap if cmap is not None else default_cmap
        self.vmin = vmin
        self.vmax = vmax
        self.state = state
        self.attrs = kwargs
        if 'priority' not in self.attrs:
            self.attrs['priority'] = 1

    @property
    def priority(self):
        return self.attrs['priority']

    @priority.setter
    def priority(self, value):
        self.attrs['priority'] = value

    def to_json(self):
        return dict()

class DataView(View):
    def __init__(self, description="", **kwargs):
        super(DataView, self).__init__(**kwargs)
        self.description = description

    @classmethod
    def from_hdf(cls, ds, node):
        data = []
        for name in json.loads(node[0]):
            if isinstance(name, list):
                d = []
                for n in name:
                    bd = BrainData.from_hdf(ds, node.file.get(n))
                    d.append(bd)
                data.append(d)
            else:
                bd = BrainData.from_hdf(ds, node.file.get(name))
                data.append(bd)
        if len(data) < 2 and isinstance(data[0], BrainData):
            data = data[0]
        desc = node[1]
        cmap = node[2]
        vmin = json.loads(node[3])
        vmax = json.loads(node[4])
        state = json.loads(node[5])
        attrs = json.loads(node[6])
        return cls(data, cmap=cmap, vmin=vmin, vmax=vmax, description=desc, **attrs)

    def to_json(self):
        sdict = super(DataView, self).to_json(self)

        sdict.update(dict(
            data=dnames, 
            cmap=self.cmap, 
            vmin=self.vmin, 
            vmax=self.vmax, 
            desc=self.description, 
            state=self.state, 
            attrs=self.attrs))
        return sdict

    def __iter__(self):
        if isinstance(self.data, BrainData):
            yield self.data
        else:
            for data in self.data:
                if isinstance(data, BrainData):
                    yield data
                else:
                    for d in data:
                        yield d

    def view(self, cmap=None, vmin=None, vmax=None, state=None, **kwargs):
        """Generate a new view on the contained data. Any variable that is not 
        None will be updated"""
        cmap = self.cmap if cmap is None else cmap
        vmin = self.vmin if vmin is None else vmin
        vmax = self.vmax if vmax is None else vmax
        state = self.state if state is None else state

        for key, value in self.attrs.items():
            if key not in kwargs:
                kwargs[key] = value

        return DataView(self.data, cmap=cmap, vmin=vmin, vmax=vmax, state=state, **kwargs)

    def _write_hdf(self, h5, name="data"):
        #Must support 3 optional layers of stacking
        if isinstance(self.data, BrainData):
            dnode = self.data._write_hdf(h5)
            nname = [dnode.name]
        else:
            nname = []
            for data in self.data:
                if isinstance(data, BrainData):
                    dnode = data._write_hdf(h5)
                    nname.append(dnode.name)
                else:
                    dnames = []
                    for d in data:
                        dnode = d._write_hdf(h5)
                        dnames.append(dnode.name)
                    nname.append(dnames)

        views = h5.require_group("/views")
        view = views.require_dataset(name, (8,), h5py.special_dtype(vlen=str))
        view[0] = json.dumps(nname)
        view[1] = self.description
        view[2] = self.cmap
        view[3] = json.dumps(self.vmin)
        view[4] = json.dumps(self.vmax)
        view[5] = json.dumps(self.state)
        view[6] = json.dumps(self.attrs)
        return view

class MultiView(View):
    def __init__(self, views, description=""):
        for view in views:
            if not isinstance(view, View):
                raise TypeError("Must be a View object!")
        raise NotImplementedError

class Volume(DataView, VolumeData):
    def __init__(self, data, subject, xfmname, cmap=None, vmin=None, vmax=None, description="", **kwargs):
        super(Volume, self).__init__(data, subject, xfmname, cmap=cmap, vmin=vmin, vmax=vmax, 
            description=description, **kwargs)

class Vertex(DataView, VertexData):
    def __init__(self, data, subject, cmap=None, vmin=None, vmax=None, description="", **kwargs):
        super(Vertex, self).__init__(data, subject, cmap=cmap, vmin=vmin, vmax=vmax, 
            description=description, **kwargs)

class RGBVolume(DataView):
    def __init__(self, red, green, blue, subject=None, xfmname=None, alpha=None, description="", **kwargs):
        if "cmap" in kwargs or "vmin" in kwargs or "vmax" in kwargs:
            raise TypeError("RGBViews does not have colormap options")

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

        self.alpha = alpha

        super(RGBVolume, self).__init__(description=description, **kwargs)

    @property
    def volume(self):
        alpha = self.alpha
        if self.alpha is None:
            alpha = np.ones_like(red)

        volume = []
        for dv in (self.red, self.green, self.blue, alpha)
            vol = dv.volume
            if vol.dtype != np.uint8:
                if vol.min() < 0:
                    vol -= vol.min()
                if vol.max() > 1:
                    vol /= vol.max()
                vol = (vol * 255).astype(np.uint8)
            volume.append(vol)

        return np.array(volume).transpose([1, 2, 3, 4, 0])

class RGBVertex(DataView):
    def __init__(self, red, green, blue, subject=None, alpha=None, description="", **kwargs):
        if "cmap" in kwargs or "vmin" in kwargs or "vmax" in kwargs:
            raise TypeError("RGBViews does not have colormap options")
                if "cmap" in kwargs or "vmin" in kwargs or "vmax" in kwargs:
            raise TypeError("RGBViews does not have colormap options")

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
            self.red = VertexData(red, xfmname)
            self.green = VertexData(green, xfmname)
            self.blue = VertexData(blue, xfmname)

        if alpha is None:
            self.alpha = VertexData()

        #Normalize the RGB channels to be between 0 and 255
        for dv in (self.red, self.green, self.blue, self.alpha):
            if self.red.data.dtype != np.uint8:
                if dv.min() < 0:
                    dv -= dv.min()
                if dv.max() > 1:
                    dv /= dv.max()

        super(RGBVertex, self).__init__(description=description, **kwargs)

    @property
    def volume(self):
        alpha = self.alpha
        if self.alpha is None:
            alpha = np.ones_like(red)

        volume = []
        for dv in (self.red, self.green, self.blue, alpha)
            vol = dv.volume
            if vol.dtype != np.uint8:
                if vol.min() < 0:
                    vol -= vol.min()
                if vol.max() > 1:
                    vol /= vol.max()
                vol = (vol * 255).astype(np.uint8)
            volume.append(vol)

        return np.array(volume).transpose([1, 2, 0])

class TwoDVolume(View):
    def __init__(self, dim1, dim2, subject=None, xfmname=None, description="", vmin2=None, vmax2=None, **kwargs):
        self.vmin2 = vmin2
        self.vmax2 = vmax2

        super(TwoDVolume, self).__init__(description=description, **kwargs)

class TwoDVertex(View):
    def __init__(self, dim1, dim2, subject=None, description="", vmin2=None, vmax2=None, **kwargs):
        self.dim1 = normalize(dim1)
        self.dim2 = normalize(dim2)
        self.vmin2 = vmin2
        self.vmax2 = vmax2
        super(TwoDVertex, self).__init__(description=description, **kwargs)