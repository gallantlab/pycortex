"""This module defines a class Package which is used by webgl to encode pycortex datasets into json objects.
The general structure of the object that's transmitted looks like this:

dict(
    views = [ dict(name="proper name", cmap=cmap, vmin=vmin, vmax=vmax, data=["__braindata_name"]) ],
    data  = dict(__braindata_name=dict(subject=subject, min=min, max=max)),
    images=(__braindata_name=["img1.png", "img2.png"]),
)
"""
import os
import json
from io import BytesIO
import numpy as np

from .. import dataset
from .. import volume


#TODO: How to package multiviews?
class Package(object):
    """Package the data into a form usable by javascript"""
    def __init__(self, data):
        self.dataset = dataset.normalize(data)
        self.uniques = data.uniques(collapse=True)
        
        self.brains = dict()
        self.images = dict()
        for brain in self.uniques:
            name = brain.name
            self.brains[name] = brain.to_json(simple=True)
            if isinstance(brain, (dataset.Vertex, dataset.VertexRGB)):
                encdata = brain.vertices
            else:
                encdata = brain.volume
            if isinstance(brain, (dataset.VolumeRGB, dataset.VertexRGB)):
                encdata = encdata.astype(np.uint8)
                self.brains[name]['raw'] = True
            else:
                encdata = encdata.astype(np.float32)
                self.brains[name]['raw'] = False

            #VertexData requires reordering, only save normalized version for now
            if isinstance(brain, (dataset.Vertex, dataset.VertexRGB)):
                self.images[name] = [encdata]
            else:
                self.images[name] = [volume.mosaic(vol, show=False) for vol in encdata]
                if len(set([shape for m, shape in self.images[name]])) != 1:
                    raise ValueError('Internal error in mosaic')
                self.brains[name]['mosaic'] = self.images[name][0][1]
                self.images[name] = [_pack_png(m) for m, shape in self.images[name]]

    @property
    def views(self):
        metadata = []
        for name, view in self.dataset:
            meta = view.to_json(simple=False)
            meta['name'] = name
            if 'stim' in meta['attrs']:
                meta['attrs']['stim'] = os.path.split(meta['attrs']['stim'])[1] 
            metadata.append(meta)
        return metadata

    @property
    def subjects(self):
        return set(braindata.subject for braindata in self.uniques)

    def reorder(self, subjects):
        indices = dict((k, np.load(os.path.splitext(v)[0]+".npz")) for k, v in subjects.items())
        for brain in self.uniques:
            if isinstance(brain, (dataset.Vertex, dataset.VertexRGB)):
                data = np.array(self.images[brain.name])[0]
                npyform = BytesIO()
                if self.brains[brain.name]['raw']:
                    data = data[..., indices[brain.subject]['index'], :]
                else:
                    data = data[..., indices[brain.subject]['index']]
                np.save(npyform, np.ascontiguousarray(data))
                npyform.seek(0)
                self.images[brain.name] = [npyform.read()]
        for npz in indices.values():
            npz.close()

    def metadata(self, submap=None, **kwargs):
        if submap is not None:
            for data in self.brains.values():
                data['subject'] = submap[data['subject']]
        return dict(views=self.views, data=self.brains, images=self.image_names(**kwargs))

    def image_names(self, fmt="/data/{name}/{frame}/"):
        names = dict()
        for name, imgs in self.images.items():
            names[name] = [fmt.format(name=name, frame=i) for i in range(len(imgs))]
        return names

def _pack_png(mosaic):
    from PIL import Image
    buf = BytesIO()
    if mosaic.dtype not in (np.float32, np.uint8):
        raise TypeError

    y, x = mosaic.shape[:2]
    im = Image.frombuffer('RGBA', (x,y), mosaic.data, 'raw', 'RGBA', 0, 1)
    im.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()
