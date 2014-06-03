import json

import h5py
import numpy as np

from .. import options
from . import BrainData, VolumeData, VertexData

default_cmap = options.config.get("basic", "default_cmap")

def normalize(data):
    if isinstance(data, tuple):
        if len(data) == 3:
            if data[0].dtype == np.uint8:
                return RGBVolume(data[0][...,0], data[0][...,1], data[0][...,2], *data[1:])
            return Volume(*data)
        elif len(data) == 2:
            return Vertex(*data)
        else:
            raise TypeError("Invalid input for DataView")
    elif isinstance(data, DataView):
        return data
    else:
        raise TypeError("Invalid input for DataView")

class DataView(object):
    def __init__(self, cmap=None, vmin=None, vmax=None, description="", state=None, **kwargs):
        if self.__class__ == DataView:
            raise TypeError('Cannot directly instantiate DataView objects')

        self.cmap = cmap if cmap is not None else default_cmap
        self.vmin = vmin
        self.vmax = vmax
        self.state = state
        self.attrs = kwargs
        if 'priority' not in self.attrs:
            self.attrs['priority'] = 1
        self.description = description

    def copy(self, *args):
        return self.__class__(*args, 
            cmap=self.cmap, 
            vmin=self.vmin, 
            vmax=self.vmax, 
            description=self.description, 
            state=self.state, 
            **self.attrs)

    @property
    def priority(self):
        return self.attrs['priority']

    @priority.setter
    def priority(self, value):
        self.attrs['priority'] = value

    def to_json(self):
        return dict(
            cmap=self.cmap, 
            vmin=self.vmin, 
            vmax=self.vmax, 
            state=self.state, 
            attrs=self.attrs, 
            desc=self.description)

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
    @staticmethod
    def from_hdf(ds, node):
        data, desc, cmap = node[:3]
        vmin = json.loads(node[3])
        vmax = json.loads(node[4])
        state = json.loads(node[5])
        attrs = json.loads(node[6])

        subj = node.attrs['subject']
        if "xfmname" in node.attrs:
            xfmname = node.attrs['xfmname']
            mask = None
            if "mask" in node.attrs:
                try:
                    db.get_mask(subj, xfmname, node.attrs['mask'])
                    mask = node.attrs['mask']
                except IOError:
                    mask = dataset.get_mask(subj, xfmname, node.attrs['mask'])
            return Volume(node, subj, xfmname, mask=mask)
        else:
            return Vertex(node, subj)

    def _write_hdf(self, h5, name="data", writeview=True):
        datanode = super(DataView, self)._write_hdf(h5)
        if writeview:
            views = h5.require_group("/views")
            view = views.require_dataset(name, (8,), h5py.special_dtype(vlen=str))
            view[1] = self.description
            view[2] = self.cmap
            view[3] = json.dumps(self.vmin)
            view[4] = json.dumps(self.vmax)
            view[5] = json.dumps(self.state)
            view[6] = json.dumps(self.attrs)
            return view
        return datanode

class MultiView(DataView):
    def __init__(self, views, description=""):
        for view in views:
            if not isinstance(view, DataView):
                raise TypeError("Must be a View object!")
        raise NotImplementedError

class Volume(VolumeData, DataView):
    def __init__(self, data, subject, xfmname, mask=None, 
        cmap=None, vmin=None, vmax=None, description="", **kwargs):
        super(Volume, self).__init__(data, subject, xfmname, mask=mask, 
            cmap=cmap, vmin=vmin, vmax=vmax, description=description, **kwargs)

class Vertex(VertexData, DataView):
    def __init__(self, data, subject, cmap=None, vmin=None, vmax=None, description="", **kwargs):
        super(Vertex, self).__init__(data, subject, cmap=cmap, vmin=vmin, vmax=vmax, 
            description=description, **kwargs)

class RGBVolume(DataView):
    def __init__(self, red, green, blue, subject=None, xfmname=None, alpha=None, description=""):
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
        self.description = self.description

    @property
    def volume(self):
        if self.alpha is None:
            alpha = np.ones_like(red)
        if not isinstance(self.alpha, Volume):
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

class RGBVertex(DataView):
    def __init__(self, red, green, blue, subject=None, alpha=None, description=""):
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

        self.alpha = alpha
        self.description = description

    @property
    def vertices(self):
        if self.alpha is None:
            alpha = np.ones_like(red)
        if not isinstance(self.alpha, Vertex):
            alpha = Vertex(self.alpha, self.subject)

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

class TwoDVolume(DataView):
    def __init__(self, dim1, dim2, subject=None, xfmname=None, description="", cmap=None,
        vmin=None, vmax=None, vmin2=None, vmax2=None, **kwargs):
        if isinstance(dim1, VolumeData):
            if subject is not None or xfmname is not None:
                raise TypeError("Subject and xfmname cannot be specified with Volumes")
            if not isinstance(dim2, VolumeData) or dim2.subject != dim1.subject:
                raise TypeError("Invalid data for second dimension")
            self.dim1 = dim1
            self.dim2 = dim2
        else:
            self.dim1 = Volume(dim1, subject, xfmname)
            self.dim2 = Volume(dim2, subject, xfmname)

        self.vmin2 = vmin2 if vmin2 is not None else vmin
        self.vmax2 = vmax2 if vmax2 is not None else vmax

        super(TwoDVolume, self).__init__(description=description, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

class TwoDVertex(DataView):
    def __init__(self, dim1, dim2, subject=None, description="", cmap=None,
        vmin=None, vmax=None, vmin2=None, vmax2=None, **kwargs):
        if isinstance(dim1, VertexData):
            if subject is not None:
                raise TypeError("Subject cannot be specified with Volumes")
            if not isinstance(dim2, VertexData) or dim2.subject != dim1.subject:
                raise TypeError("Invalid data for second dimension")
            self.dim1 = dim1
            self.dim2 = dim2
            
        else:
            self.dim1 = Vertex(dim1, subject)
            self.dim2 = Vertex(dim2, subject)

        self.vmin2 = vmin2
        self.vmax2 = vmax2
        super(TwoDVertex, self).__init__(description=description, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)