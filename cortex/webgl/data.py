import json
import numpy as np

from .. import dataset
from .. import volume

class Package(object):
    """Package the data into a form usable by javascript"""
    def __init__(self, data):
        self.dataset = dataset.normalize(data)
        self.uniques = data.uniques
        
        self.brains = dict()
        self.images = dict()
        for ds in self.uniques:
            name = ds.name
            self.brains[name] = ds.to_json()
            voldata = ds.data
            if not ds.movie:
                voldata = voldata[np.newaxis]
            if ds.raw:
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

def _pack_png(mosaic):
    import Image
    import cStringIO
    buf = cStringIO.StringIO()
    if mosaic.dtype not in (np.float32, np.uint8):
        raise TypeError

    im = Image.frombuffer('RGBA', mosaic.shape[:2], mosaic.data, 'raw', 'RGBA', 0, 1)
    im.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()