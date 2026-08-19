"""The two webgl wire encodings, one class each.

This module is a **compatibility surface**, not a place to tidy up. Everything
here is read by ``webgl/resources/js/dataset.js``, which dispatches on the
*shape* of what it receives -- ``mosaic === undefined`` means per-vertex
attributes, anything else means a mosaicked texture -- so a change in these bytes
breaks the viewer silently rather than noisily. See INHERITANCE.md, "The wire
format is a hard interface".

A space says which encoding its arrays use by returning one of these from
:meth:`~cortex.dataset._space.BrainSpace.pack_for_webgl`. That is the whole
extension point: ``webgl/data.py`` used to answer the same question with three
``isinstance(brain, SurfaceView)`` branches plus a guard for "neither", so the
premultiplied-alpha asymmetry, the mosaic-versus-attributes choice and the vertex
reordering were three separate forks on one fact -- and a space inheriting neither
built-in spatial interface could not be packaged at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt


class WebGLPayload(ABC):
    """One view's array, encoded for the browser.

    Subclasses do the encoding in ``__init__``: by the time a payload exists,
    :attr:`frames` is what will be served. The two that exist are the two
    ``dataset.js`` can read; a third needs a matching branch there.
    """

    #: What gets served, one entry per frame. PNG bytes for a mosaicked texture;
    #: for per-vertex attributes a single array that only becomes ``.npy`` bytes
    #: in :meth:`reorder`, since it cannot be serialised before the CTM's vertex
    #: order is known.
    frames: list[Any]

    #: Whether the array is 4-channel uint8 (an RGB view) rather than scalar
    #: floats. Shipped as the ``raw`` JSON key and, for per-vertex attributes,
    #: decides whether reordering indexes a trailing channel axis.
    raw: bool

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """The JSON keys this encoding contributes to the view's data record.

        Merged into what :meth:`~cortex.dataset._space.BrainSpace.describe_layout`
        already supplies, so between them the wire contract is stated by the
        space and never assembled by the consumer. Keys must be *absent* rather
        than null when they do not apply: ``dataset.js`` selects the texture path
        by testing ``mosaic === undefined``.
        """

    def reorder(self, frames: list[Any], vertex_index: Any) -> list[Any]:
        """Frames permuted into the CTM's vertex order, if that applies.

        Concrete and a no-op by default: only the per-vertex encoding cares, and
        ``vertex_index`` -- the opened ``.npz`` of index arrays -- is deliberately
        passed unread so an encoding that does not need it never decompresses it.
        """
        return frames


class MosaicTexture(WebGLPayload):
    """Volumetric encoding: each frame tiled into one PNG, sampled as a texture.

    Alpha is *not* premultiplied here. Three.js sets ``tex.premultiplyAlpha`` on
    upload and ``UNPACK_PREMULTIPLY_ALPHA_WEBGL`` does it once on the GPU, so
    doing it in Python as well would double-attenuate. The asymmetry with
    :class:`VertexAttributes` is a fact about the browser, not a bug.
    """

    def __init__(self, data: npt.NDArray, *, raw: bool) -> None:
        # Deferred: `cortex.volume` imports `cortex.dataset` at module level.
        from ..volume import mosaic

        self.raw = raw
        data = data.astype(np.uint8 if raw else np.float32)
        tiles = [mosaic(frame, show=False) for frame in data]
        shapes = {shape for _, shape in tiles}
        if len(shapes) != 1:
            raise ValueError(
                "Frames of one view tiled to different mosaic shapes: %r" % (shapes,)
            )
        self.mosaic: tuple[int, int] = tiles[0][1]
        self.frames = [pack_png(tile) for tile, _ in tiles]

    def describe(self) -> dict[str, Any]:
        return {"raw": self.raw, "mosaic": self.mosaic}


class VertexAttributes(WebGLPayload):
    """Surface encoding: raw per-vertex attributes, served as ``.npy`` bytes.

    Alpha *is* premultiplied here, because these bytes reach the shader as vertex
    attributes and nothing else premultiplies them: the fragment shader
    composites with ``gl_FragColor = vColor + (1-a)*bg``, which only gives the
    right answer for premultiplied colour (issue #631). The ``vertices``/``volume``
    properties stay straight-alpha so the matplotlib path keeps working.
    """

    def __init__(self, data: npt.NDArray, *, raw: bool) -> None:
        self.raw = raw
        # `astype` copies even when the dtype already matches, so premultiplying
        # below writes into an array nothing else holds.
        data = data.astype(np.uint8 if raw else np.float32)
        if raw:
            alpha = data[..., 3:4].astype(np.float32) / 255.0
            data[..., :3] = np.round(data[..., :3].astype(np.float32) * alpha)
        self.frames = [data]

    def describe(self) -> dict[str, Any]:
        return {"raw": self.raw}

    def reorder(self, frames: list[Any], vertex_index: Any) -> list[Any]:
        index = vertex_index["index"]
        data = np.array(frames)[0]
        # An RGB view carries a trailing channel axis; a scalar one does not.
        data = data[..., index, :] if self.raw else data[..., index]
        buf = BytesIO()
        np.save(buf, np.ascontiguousarray(data))
        buf.seek(0)
        return [buf.read()]


def pack_png(tile: npt.NDArray) -> bytes:
    """One mosaic tile as PNG bytes."""
    from PIL import Image

    if tile.dtype not in (np.float32, np.uint8):
        raise TypeError("Cannot pack %s as an RGBA PNG" % tile.dtype)

    y, x = tile.shape[:2]
    # `tobytes()` rather than the buffer protocol on `.data`: same bytes for the
    # contiguous array `mosaic` returns, and typed as the `bytes` PIL declares.
    im = Image.frombuffer(
        "RGBA", (x, y), np.ascontiguousarray(tile).tobytes(), "raw", "RGBA", 0, 1
    )
    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def no_encoding(space: Any) -> NoReturn:
    """Raise the "this space cannot be shipped to a browser" error.

    Its own function so the message lives next to the two encodings it names.
    """
    raise TypeError(
        "%s has no webgl wire encoding, so its data cannot be sent to a browser. "
        "Implement pack_for_webgl to return one of the two encodings dataset.js "
        "can read -- MosaicTexture (a mosaicked texture sampled through a "
        "transform) or VertexAttributes (raw per-vertex attributes) -- or add a "
        "third to webgl/resources/js/dataset.js. Data in this space can still be "
        "drawn by quickflat." % type(space).__name__
    )
