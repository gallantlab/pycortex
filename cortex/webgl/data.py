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
from typing import Any, Optional, TypedDict, cast

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
        # Two-phase, which is why this is not `list[bytes]`: the mosaicked-texture
        # encoding finishes as PNG bytes here, but the per-vertex one leaves an
        # array in place and only `reorder` turns it into `.npy` bytes, because it
        # cannot serialise until it knows the CTM's vertex order. Anything reading
        # `images` between the two is holding arrays, not bytes.
        self.images: dict[str, list[Any]] = dict()
        # Kept so `reorder` can ask the same encoding that produced the frames what
        # to do with them, instead of re-deriving it from the view's class.
        self._payloads: dict[str, dataset.WebGLPayload] = dict()
        for brain in self.uniques:
            name = brain.name
            self.subjects.add(brain.subject)
            self.brains[name] = brain.to_json(simple=True)
            # Two questions, both answered by the view rather than by its class:
            # `spatial_data` is the array to ship (each spatial interface points it
            # at its own storage), and `space.pack_for_webgl` is how that array
            # reaches the browser. This module used to answer the second itself, in
            # three `isinstance(brain, SurfaceView)` forks -- the premultiplied-alpha
            # asymmetry, the mosaic-versus-attributes choice and the vertex
            # reordering -- plus a guard for "neither", so one fact was restated
            # once per consequence and a space this module had not heard of hit
            # whichever `else` came first. See INHERITANCE.md.
            payload = brain.space.pack_for_webgl(
                brain.spatial_data, raw=isinstance(brain, dataset.DataviewRGB)
            )
            self._payloads[name] = payload
            self.images[name] = payload.frames
            # `describe()` returns only keys of DataviewJSON, but says so as a
            # plain dict: it lives in `cortex.dataset._webgl`, which `views` (where
            # DataviewJSON is declared) cannot be imported from without a cycle.
            self.brains[name].update(cast(dataset.DataviewJSON, payload.describe()))

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
            # Whether permuting applies is the encoding's business, and only the
            # per-vertex one says yes -- the default `reorder` returns the frames
            # untouched without reading the index, so a mosaicked view does not
            # decompress an index array it has no use for.
            name = brain.name
            self.images[name] = self._payloads[name].reorder(
                self.images[name], indices[brain.subject]
            )
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

