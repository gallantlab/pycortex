"""This module defines a class Package which is used by webgl to encode pycortex datasets into json objects.
The general structure of the object that's transmitted looks like this:

dict(
    views = [ dict(name="proper name", cmap=cmap, vmin=vmin, vmax=vmax, data=["__braindata_name"]) ],
    data  = dict(__braindata_name=dict(subject=subject, min=min, max=max)),
    images=(__braindata_name=["img1.png", "img2.png"]),
    tracts = dict(name=dict(subject=subject, n_points=N, n_streamlines=M, ...,
                            urls=dict(points=url, offsets=url, colors=url))),
)

Tractograms (`cortex.Tractogram`) are handled separately from the BrainData
based dataviews: they are not colormapped in the browser and carry no
volume/vertex arrays, so they never appear in ``views``/``data``/``images``.
Instead they contribute four little-endian binary buffers each -- ``points``
(float32, N x 3), ``offsets`` (uint32, M + 1), ``colors`` (uint8, N x 3) and
``groups`` (uint32, concatenation of every group's streamline indices, in
the order they appear in ``tract_meta[name]["groups"]``, whose values are
``[start, stop]`` slice bounds into this buffer rather than counts) -- which
the viewer fetches and turns into a THREE.js line geometry
(``resources/js/tractogram.js``).
"""

import os
import json
from io import BytesIO
import numpy as np

from .. import dataset
from .. import volume


# TODO: How to package multiviews?
class Package(object):
    """Package the data into a form usable by javascript

    Parameters
    ----------
    data : Dataset or Dataview
        The data to package.
    require_brains : bool
        If True (the default), packaging a dataset that holds only
        tractograms raises `ValueError`: the viewer cannot boot without a
        Volume/Vertex dataview to build its surfaces from. Set to False when
        pushing data into an already-running viewer.
    """

    def __init__(self, data, require_brains=True):
        self.dataset = dataset.normalize(data)
        self._require_brains = require_brains

        # `normalize` returns a bare Dataview untouched (it does not wrap it
        # into a Dataset), so handle both shapes here.
        if isinstance(self.dataset, dataset.Dataset):
            items = list(self.dataset)
        else:
            items = [(None, self.dataset)]

        # Tractograms are not BrainData: they are pulled out first, both
        # because Dataset.uniques() would choke on them (only BrainData
        # implements .uniques()) and because every existing javascript path
        # expects views/data/images to hold colormapped brain data only.
        self.tracts = dict()
        self.tract_meta = dict()
        brain_views = []
        for name, view in items:
            if isinstance(view, dataset.Tractogram):
                tname = name or view.description or view.name
                if view.n_points > np.iinfo(np.uint32).max:
                    raise ValueError(
                        "Tractogram %r has %d points, more than the uint32 "
                        "offsets sent to the viewer can address; use "
                        "Tractogram.subsample() first." % (tname, view.n_points)
                    )
                group_indices, _ = view.groups_wire()
                self.tracts[tname] = dict(
                    points=view.points.astype("<f4").tobytes(),
                    offsets=view.offsets.astype("<u4").tobytes(),
                    colors=view.vertex_colors().astype(np.uint8).tobytes(),
                    groups=group_indices.astype("<u4").tobytes(),
                )
                self.tract_meta[tname] = view.to_json()
            else:
                brain_views.append(view)

        # Deduplicate while keeping the dataset order (BrainData hashes by
        # identity, so a plain set would shuffle the brains between runs).
        self.uniques = []
        for view in brain_views:
            for sv in view.uniques(collapse=True):
                if sv not in self.uniques:
                    self.uniques.append(sv)
        # Tract subjects count too: `show` builds CTM packs per subject,
        # `addData` rejects unknown subjects and `make_static(anonymize=True)`
        # renames every subject it knows about.
        self.subjects = set(meta["subject"] for meta in self.tract_meta.values())

        self.brains = dict()
        self.images = dict()
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
                encdata = encdata.astype(np.float32)
                self.brains[name]["raw"] = False

            # VertexData requires reordering, only save normalized version for now
            if isinstance(brain, (dataset.Vertex, dataset.VertexRGB)):
                self.images[name] = [encdata]
            else:
                self.images[name] = [volume.mosaic(vol, show=False) for vol in encdata]
                if len(set([shape for m, shape in self.images[name]])) != 1:
                    raise ValueError("Internal error in mosaic")
                self.brains[name]["mosaic"] = self.images[name][0][1]
                self.images[name] = [_pack_png(m) for m, shape in self.images[name]]

        if require_brains and self.tracts and not self.brains:
            raise ValueError(
                "A Tractogram cannot be displayed on its own: the webgl "
                "viewer needs at least one Volume or Vertex dataview to "
                "build the surfaces and boot. Pass them together, e.g. "
                "cortex.webshow(cortex.Dataset(curvature=vertex, af=tract))."
            )

    @property
    def views(self):
        metadata = []
        for name, view in self._brain_items():
            meta = view.to_json(simple=False)
            meta["name"] = name
            if "stim" in meta["attrs"]:
                meta["attrs"]["stim"] = os.path.split(meta["attrs"]["stim"])[1]
            metadata.append(meta)
        return metadata

    def _brain_items(self):
        """(name, dataview) pairs, excluding tractograms."""
        if isinstance(self.dataset, dataset.Dataset):
            items = list(self.dataset)
        else:
            items = [(self.dataset.description or "data", self.dataset)]
        return [
            (name, view)
            for name, view in items
            if not isinstance(view, dataset.Tractogram)
        ]

    def reorder(self, subjects):
        # Tractograms carry no vertex-indexed data, so there is nothing to
        # reorder for them; `self.uniques` holds BrainData only.
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

    def metadata(self, submap=None, tract_fmt="/tract/{name}/{buf}/", **kwargs):
        """Assemble the metadata dict handed to the javascript viewer.

        Parameters
        ----------
        submap : dict or None
            Optional subject renaming map (used by `make_static` when
            anonymizing).
        tract_fmt : str
            Format string for the tractogram buffer urls, with ``{name}``
            (tractogram name) and ``{buf}`` (one of points/offsets/colors)
            fields. Defaults to the live-server route served by
            `cortex.webgl.view.TractHandler`.
        kwargs
            Passed to `image_names` (i.e. its ``fmt``).
        """
        if submap is not None:
            for data in self.brains.values():
                data["subject"] = submap[data["subject"]]
            for meta in self.tract_meta.values():
                if meta["subject"] in submap:
                    meta["subject"] = submap[meta["subject"]]
        meta = dict(
            views=self.views, data=self.brains, images=self.image_names(**kwargs)
        )
        urls = self.tract_names(fmt=tract_fmt)
        meta["tracts"] = dict(
            (name, dict(self.tract_meta[name], urls=urls[name]))
            for name in self.tracts
        )
        return meta

    def image_names(self, fmt="/data/{name}/{frame}/"):
        names = dict()
        for name, imgs in self.images.items():
            names[name] = [fmt.format(name=name, frame=i) for i in range(len(imgs))]
        return names

    def tract_names(self, fmt="/tract/{name}/{buf}/"):
        """Urls of the binary buffers of every tractogram, keyed by name."""
        names = dict()
        for name, bufs in self.tracts.items():
            names[name] = dict(
                (buf, fmt.format(name=name, buf=buf)) for buf in sorted(bufs)
            )
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
