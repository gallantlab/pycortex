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
import numpy as np

from .. import dataset
from .. import volume

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
            voldata = brain.volume
            if isinstance(brain, (dataset.VolumeRGB, dataset.VertexRGB)):
                voldata = voldata.astype(np.uint8)
                self.brains[name]['raw'] = True
            else:
                voldata = voldata.astype(np.float32)
                self.brains[name]['raw'] = False
            self.images[name] = [volume.mosaic(vol, show=False) for vol in voldata]
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

    def metadata(self, **kwargs):
        return dict(views=self.views, data=self.brains, images=self.image_names(**kwargs))

    def image_names(self, fmt="/data/{name}/{frame}/"):
        names = dict()
        for name, imgs in self.images.items():
            names[name] = [fmt.format(name=name, frame=i) for i in range(len(imgs))]
        return names

def _pack_png(mosaic):
    from PIL import Image
    import cStringIO
    buf = cStringIO.StringIO()
    if mosaic.dtype not in (np.float32, np.uint8):
        raise TypeError

    y, x = mosaic.shape[:2]
    im = Image.frombuffer('RGBA', (x,y), mosaic.data, 'raw', 'RGBA', 0, 1)
    im.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()