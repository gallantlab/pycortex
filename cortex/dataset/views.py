import json

import h5py
import numpy as np

from .. import options
from . import BrainData, VolumeData, VertexData

default_cmap = options.config.get("basic", "default_cmap")

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

    def __call__(self, data, description=""):
        return DataView(data, description, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, state=self.state)

class DataView(View):
    def __init__(self, data, description="", **kwargs):
        super(DataView, self).__init__(**kwargs)
        if isinstance(data, list):
            #validate if the input is of a recognizable form
            if len(data) != 1 or len(data[0]) != 2:
                raise TypeError("Sorry, multi views are currently not supported...")
            xdim, ydim = map(DataView.normalize, data[0])
            if xdim.subject != ydim.subject:
                raise TypeError("2D data views require the same subject")
            if xdim.raw or ydim.raw:
                raise TypeError("2D data views cannot be raw")
            #this awful bit of logic checks validity of movies
            if xdim.movie ^ ydim.movie or (
                xdim.movie and 
                len(xdim.data) != len(ydim.data)):
                raise TypeError('2D movies must be same length')

            no_nan_x = xdim.data[~np.isnan(xdim.data)]
            no_nan_y = ydim.data[~np.isnan(ydim.data)]
            if self.vmin is None:
                self.vmin = [(
                    float(np.percentile(no_nan_x, 1)), 
                    float(np.percentile(no_nan_y, 1)))]
            if self.vmax is None:
                self.vmax = [(
                    float(np.percentile(no_nan_x, 99)), 
                    float(np.percentile(no_nan_y, 99)))]
            self.data = [(xdim, ydim)]
        else:
            self.data = DataView.normalize(data)
            no_nan = self.data.data[~np.isnan(self.data.data)]

            if self.vmin is None:
                self.vmin = float(np.percentile(no_nan, 1))
            if self.vmax is None:
                self.vmax = float(np.percentile(no_nan, 99))

        self.description = description

    @staticmethod
    def normalize(data):
        if isinstance(data, tuple):
            if len(data) == 3:
                return VolumeData(*data)
            elif len(data) == 2:
                return VertexData(*data)
            else:
                raise TypeError("Invalid input for DataView")
        elif isinstance(data, BrainData):
            return data
        elif isinstance(data, DataView) and isinstance(data.data, BrainData):
            return data.data
        else:
            raise TypeError("Invalid input for DataView")

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
        dnames = []
        if isinstance(self.data, BrainData):
            dnames.append(self.data.name)
        elif isinstance(self.data, list):
            for data in self.data:
                if isinstance(data, BrainData):
                    dnames.append(data.name)
                else:
                    dnames.append([d.name for d in data])

        return dict(
            data=dnames, 
            cmap=self.cmap, 
            vmin=self.vmin, 
            vmax=self.vmax, 
            desc=self.description, 
            state=self.state, 
            attrs=self.attrs)

    def copy(self, data=None):
        if data is None:
            data = self.data
        return DataView(data, 
            self.description, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, state=self.state, **self.attrs)

    @property
    def raw(self):
        if not isinstance(self.data, BrainData):
            raise ValueError('Can only colormap single data views')
        if self.data.raw:
            raise ValueError('Data is already colormapped')
        from matplotlib import cm, colors
        cmap = cm.get_cmap(self.cmap)
        norm = colors.Normalize(vmin=self.vmin, vmax=self.vmax, clip=True)
        raw = (cmap(norm(self.data.data)) * 255).astype(np.uint8)
        return self.copy(self.data.copy(raw))

    def map(self, sampler="nearest"):
        if not isinstance(self.data, VolumeData):
            raise ValueError("Can only map volumedata views")

        return self.copy(self.data.map(sampler))

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
