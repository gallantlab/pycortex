"""This module defines a class Package which is used by webgl to encode pycortex datasets into json objects.
The general structure of the object that's transmitted looks like this:

dict(
    views = [ dict(name="proper name", cmap=cmap, vmin=vmin, vmax=vmax, data=["__braindata_name"]) ],
    data  = dict(__braindata_name=dict(subject=subject, min=min, max=max)),
    images=(__braindata_name=["img1.png", "img2.png"]),
)
"""
import json
import numpy as np

from .. import dataset
from .. import volume

class Package(object):
    """Package the data into a form usable by javascript"""
    def __init__(self, data):
        self.dataset = dataset.normalize(data)
        self.uniques = data.uniques()
        
        self.brains = dict()
        self.images = dict()
        for brain in self.uniques:
            name = brain.name
            self.brains[name] = brain.to_json()
            voldata = brain.volume
            if not brain.movie:
                voldata = voldata[np.newaxis]
            if brain.raw:
                voldata = voldata.astype(np.uint8)
            else:
                voldata = voldata.astype(np.float32)
            self.images[name] = [volume.mosaic(vol, show=False) for vol in voldata]
            if len(set([shape for m, shape in self.images[name]])) != 1:
                raise ValueError('Internal error in mosaic')
            self.brains[name]['mosaic'] = self.images[name][0][1]
            self.images[name] = [_pack_png(m) for m, shape in self.images[name]]

    @property
    def views(self):
        metadata = []
        for name, view in self.dataset:
            meta = view.to_json()
            meta['name'] = name
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
    import Image
    import cStringIO
    buf = cStringIO.StringIO()
    if mosaic.dtype not in (np.float32, np.uint8):
        raise TypeError

    y, x = mosaic.shape[:2]
    im = Image.frombuffer('RGBA', (x,y), mosaic.data, 'raw', 'RGBA', 0, 1)
    im.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()