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


# TODO: How to package multiviews?
class Package(object):
    """Package the data into a form usable by javascript"""

    def __init__(self, data, lazy=False):
        # lazy=True defers volume mosaic/PNG packing to get_image(). Only the first frame is packed up front.
        self.dataset = dataset.normalize(data)
        self.uniques = list(data.uniques(collapse=True))
        self.subjects = set()

        self.brains = dict()
        self.images = dict()
        self._pending = dict()
        for brain in self.uniques:
            name = brain.name
            self.subjects.add(brain.subject)
            self.brains[name] = brain.to_json(simple=True)
            if isinstance(brain, (dataset.Vertex, dataset.VertexRGB)):
                encdata = brain.vertices
            else:
                encdata = brain.volume
            if isinstance(brain, (dataset.VolumeRGB, dataset.VertexRGB)):
                encdata = encdata.astype(np.uint8)
                # The WebGL fragment shader (shaderlib.js) composites with a
                # premultiplied-alpha "over" formula
                # (gl_FragColor = vColor + (1-α)·bg). We only need to pre-
                # multiply on the Python side for VertexRGB, where the bytes
                # are uploaded as raw vertex attributes and Three.js does NOT
                # premultiply (see dataset.js VertexData path). VolumeRGB ships
                # through the PNG texture path (dataset.js:335-338, raw=true),
                # where Three.js sets `tex.premultiplyAlpha = true` and the
                # WebGL UNPACK_PREMULTIPLY_ALPHA_WEBGL hook premultiplies the
                # texture once on upload -- premultiplying here would double-
                # attenuate it. The .vertices/.volume properties stay
                # non-premultiplied so the matplotlib (quickshow) path keeps
                # using matplotlib's straight-alpha imshow compositor.
                if isinstance(brain, dataset.VertexRGB):
                    # Note: encdata is already a fresh uint8 copy from the
                    # .astype(np.uint8) call above, so we can write into it
                    # in place. The assignment to a uint8 slice handles the
                    # float→uint8 cast for us.
                    a = encdata[..., 3:4].astype(np.float32) / 255.0
                    encdata[..., :3] = np.round(
                        encdata[..., :3].astype(np.float32) * a
                    )
                self.brains[name]["raw"] = True
            else:
                self.brains[name]["raw"] = False
                if isinstance(brain, dataset.Vertex) or not lazy:
                    encdata = encdata.astype(np.float32, copy=False)

            # VertexData requires reordering, only save normalized version for now
            if isinstance(brain, (dataset.Vertex, dataset.VertexRGB)):
                self.images[name] = [encdata]
            elif lazy:
                # The mosaic grid depends only on the volume shape, 
                # so the first frame determines the metadata for every frame.
                mosaic0, shape = volume.mosaic(self._pack_dtype(name, encdata[0]),
                                               show=False)
                self.brains[name]["mosaic"] = shape
                self._pending[name] = encdata
                self.images[name] = [_pack_png(mosaic0)] + [None] * (len(encdata) - 1)
            else:
                self.images[name] = [volume.mosaic(vol, show=False) for vol in encdata]
                if len(set([shape for m, shape in self.images[name]])) != 1:
                    raise ValueError("Internal error in mosaic")
                self.brains[name]["mosaic"] = self.images[name][0][1]
                self.images[name] = [_pack_png(m) for m, shape in self.images[name]]

    @property
    def views(self):
        metadata = []
        for name, view in self.dataset:
            meta = view.to_json(simple=False)
            meta["name"] = name
            if "stim" in meta["attrs"]:
                meta["attrs"]["stim"] = os.path.split(meta["attrs"]["stim"])[1]
            metadata.append(meta)
        return metadata

    def reorder(self, subjects):
        indices = dict(
            (k, np.load(os.path.splitext(v)[0] + ".npz")) for k, v in subjects.items()
        )
        for brain in self.uniques:
            if isinstance(brain, (dataset.Vertex, dataset.VertexRGB)):
                data = np.array(self.images[brain.name])[0]
                npyform = BytesIO()
                if self.brains[brain.name]["raw"]:
                    data = data[..., indices[brain.subject]["index"], :]
                else:
                    data = data[..., indices[brain.subject]["index"]]
                np.save(npyform, np.ascontiguousarray(data))
                npyform.seek(0)
                self.images[brain.name] = [npyform.read()]
        for npz in indices.values():
            npz.close()

    def metadata(self, submap=None, **kwargs):
        if submap is not None:
            for data in self.brains.values():
                data["subject"] = submap[data["subject"]]
        return dict(
            views=self.views, data=self.brains, images=self.image_names(**kwargs)
        )

    def _pack_dtype(self, name, framedata):
        """One frame in the dtype _pack_png accepts: raw data stays uint8,
        everything else becomes float32 (a per-frame no-op-sized copy)."""
        if self.brains[name]["raw"]:
            return framedata
        return np.asarray(framedata, dtype=np.float32)

    def get_image(self, name, frame):
        """Return the packed PNG (or npy blob) for one frame, packing it on
        first request when the package was built with lazy=True."""
        img = self.images[name][frame]
        if img is None:
            mosaic, _ = volume.mosaic(
                self._pack_dtype(name, self._pending[name][frame]), show=False)
            img = self.images[name][frame] = _pack_png(mosaic)
        return img

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
    im = Image.frombuffer("RGBA", (x, y), mosaic.data, "raw", "RGBA", 0, 1)
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
