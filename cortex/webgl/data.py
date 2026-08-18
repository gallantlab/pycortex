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
import numpy.typing as npt

from .. import dataset
from .. import volume
from typing import Any, Optional, TypedDict, Union

class PackageMetadata(TypedDict):
    views: list[dataset.DataviewJSON]
    data: dict[str, dataset.DataviewJSON]
    images: dict[str, list[str]]

# TODO: How to package multiviews?
class Package(object):
    """Package the data into a form usable by javascript"""

    def __init__(self, data):
        self.dataset = dataset.normalize(data)
        # `uniques` yields Packables -- the units of transport, each with the
        # content-addressed `name` used as the browser's key for it. It was
        # `Iterator[Dataview]`, which has no `name`; this list being untyped is
        # the only reason that ever type-checked.
        self.uniques: list[dataset.Packable] = list(data.uniques(collapse=True))
        self.subjects: set[str] = set()

        self.brains: dict[str, dataset.DataviewJSON] = dict()
        # Two-phase, which is why this is not `list[bytes]`: the volumetric path
        # finishes as PNG bytes here, but the surface path leaves the raw float
        # array in place and only `reorder` turns it into `.npy` bytes, because it
        # cannot serialise until it knows the CTM's vertex order. Anything reading
        # `images` between the two is holding arrays, not bytes.
        self.images: dict[str, list[Any]] = dict()
        for brain in self.uniques:
            name = brain.name
            self.subjects.add(brain.subject)
            self.brains[name] = brain.to_json(simple=True)
            # The array to ship is `spatial_data`, which each spatial interface
            # points at its own storage -- `volume` for volumetric kinds,
            # `vertices` for surface ones. This was a fork on
            # `isinstance(brain, (Vertex, VertexRGB))` whose `else` branch read
            # `.volume`, so a spatial kind this module had not heard of died on
            # AttributeError (or, worse, was silently mosaicked as a volume if it
            # happened to have one).
            encdata = brain.spatial_data
            if isinstance(brain, dataset.DataviewRGB):
                encdata = encdata.astype(np.uint8)
                # The WebGL fragment shader (shaderlib.js) composites with a
                # premultiplied-alpha "over" formula
                # (gl_FragColor = vColor + (1-α)·bg). We only need to pre-
                # multiply on the Python side for a surface kind, whose bytes
                # are uploaded as raw vertex attributes and which Three.js does
                # NOT premultiply (see dataset.js VertexData path). A volumetric
                # kind ships through the PNG texture path (dataset.js:335-338, raw=true),
                # where Three.js sets `tex.premultiplyAlpha = true` and the
                # WebGL UNPACK_PREMULTIPLY_ALPHA_WEBGL hook premultiplies the
                # texture once on upload -- premultiplying here would double-
                # attenuate it. The .vertices/.volume properties stay
                # non-premultiplied so the matplotlib (quickshow) path keeps
                # using matplotlib's straight-alpha imshow compositor.
                if isinstance(brain, dataset.SurfaceView):
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
                encdata = encdata.astype(np.float32)
                self.brains[name]["raw"] = False

            # How the array reaches the browser is a per-spatial-kind fact, and the two
            # encodings are not interchangeable: surface data ships as raw
            # per-vertex attributes (and must be permuted into the CTM's vertex
            # order by `reorder`), volumetric data as a mosaicked PNG texture the
            # shader samples through the transform. `webgl/resources/js/dataset.js`
            # picks the path by `mosaic === undefined`, so a spatial kind needing a third
            # encoding has to change the JS too. See INHERITANCE.md.
            if isinstance(brain, dataset.SurfaceView):
                # TODO: how does this work? check if tests run this part
                self.images[name] = [encdata]
            elif not isinstance(brain, dataset.VolumetricView):
                # Neither encoding applies. Say so here: this used to fall into
                # the branch below and surface as "Invalid data shape" from
                # `volume.mosaic`, several frames deep and naming neither the
                # view nor the real problem.
                raise TypeError(
                    "%s has no webgl wire encoding: it is neither a "
                    "VolumetricView (mosaicked texture) nor a SurfaceView "
                    "(per-vertex attributes). Inherit one of those spatial interfaces, "
                    "a third encoding to webgl/resources/js/dataset.js. It can "
                    "still be drawn by quickflat." % type(brain).__name__
                )
            else:
                # TODO: make temporary typing work
                self.images[name] = [volume.mosaic(vol, show=False) for vol in encdata]
                if len(set([shape for m, shape in self.images[name]])) != 1:
                    raise ValueError("Internal error in mosaic")
                self.brains[name]["mosaic"] = self.images[name][0][1]
                self.images[name] = [_pack_png(m) for m, shape in self.images[name]]

    @property
    def views(self) -> list[dataset.DataviewJSON]:
        metadata = []
        for name, view in self.dataset:
            meta = view.to_json(simple=False)
            meta["name"] = name
            if "stim" in meta["attrs"]:
                meta["attrs"]["stim"] = os.path.split(meta["attrs"]["stim"])[1]
            metadata.append(meta)
        return metadata

    def reorder(self, subjects: dict[str, str]) -> None:
        indices = dict(
            (k, np.load(os.path.splitext(v)[0] + ".npz")) for k, v in subjects.items()
        )
        for brain in self.uniques:
            # Only the per-vertex-attribute encoding needs permuting; see __init__.
            if isinstance(brain, dataset.SurfaceView):
                # `name` comes from Packable and `subject` from Dataview, so both
                # survive narrowing to the spatial interface: mypy reads this as a subclass of
                # both Packable and SurfaceView.
                name = brain.name
                data = np.array(self.images[name])[0]
                npyform = BytesIO()
                if self.brains[name]["raw"]:
                    data = data[..., indices[brain.subject]["index"], :]
                else:
                    data = data[..., indices[brain.subject]["index"]]
                np.save(npyform, np.ascontiguousarray(data))
                npyform.seek(0)
                self.images[name] = [npyform.read()]
        for npz in indices.values():
            npz.close()

    # TODO: submap?
    def metadata(self, submap: Optional[dict[str, str]]=None, **kwargs) -> PackageMetadata:
        if submap is not None:
            for data in self.brains.values():
                data["subject"] = submap[data["subject"]]
        return PackageMetadata(
            views=self.views, data=self.brains, images=self.image_names(**kwargs)
        )

    def image_names(self, fmt: str="/data/{name}/{frame}/") -> dict[str, list[str]]:
        names: dict[str, list[str]] = dict()
        for name, imgs in self.images.items():
            names[name] = [fmt.format(name=name, frame=i) for i in range(len(imgs))]
        return names


def _pack_png(mosaic: Union[npt.NDArray[np.float32], npt.NDArray[np.uint8]]) -> bytes:
    from PIL import Image

    buf = BytesIO()
    if mosaic.dtype not in (np.float32, np.uint8):
        raise TypeError

    y, x = mosaic.shape[:2]
    im = Image.frombuffer("RGBA", (x, y), mosaic.data, "raw", "RGBA", 0, 1)
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
