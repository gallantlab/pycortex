"""Tests for the WebGL serialization layer in cortex.webgl.data."""

import numpy as np
import pytest

import cortex
from cortex import dataset
from cortex.webgl.data import Package


subj, xfmname, nverts, volshape = "S1", "fullhead", 304380, (31, 100, 100)


def _packaged_rgba(brain):
    """Run a dataview through Package and recover the pre-mosaic uint8 RGBA bytes."""
    pkg = Package(dataset.Dataset(view=brain))
    images = pkg.images[brain.name]
    # Vertex* paths store the raw uint8 array directly; Volume* paths mosaic+PNG.
    if isinstance(brain, dataset.VertexRGB):
        return images[0]
    raise NotImplementedError("Use _packaged_rgba_volume for VolumeRGB")


def _expected_premultiplied(raw_uint8):
    """Compute alpha-premultiplied RGB bytes the same way Package should."""
    a = raw_uint8[..., 3:4].astype(np.float32) / 255.0
    out = raw_uint8.copy()
    out[..., :3] = np.round(raw_uint8[..., :3].astype(np.float32) * a).astype(np.uint8)
    return out


def test_vertexrgb_alpha_is_premultiplied_in_package():
    """WebGL Package should ship alpha-premultiplied RGB bytes (issue #631).

    The shader formula is gl_FragColor = vColor + (1-α)·bg, which only yields
    correct "over" compositing when vColor is premultiplied. The fix lives in
    Package.__init__; this test pins the contract.
    """
    rng = np.random.default_rng(0)
    r = rng.uniform(0, 1, nverts).astype(np.float32)
    g = rng.uniform(0, 1, nverts).astype(np.float32)
    b = rng.uniform(0, 1, nverts).astype(np.float32)
    alpha = rng.uniform(0, 1, nverts).astype(np.float32)

    vrgb = cortex.VertexRGB(
        r,
        g,
        b,
        subj,
        alpha=cortex.Vertex(alpha, subj, vmin=0, vmax=1),
    )

    # Sanity: the .vertices property itself stays NON-premultiplied so the
    # quickshow (matplotlib) path keeps working.
    raw = vrgb.vertices
    assert raw.dtype == np.uint8
    assert raw.shape == (1, nverts, 4)
    # Raw alpha must round-trip the input alpha to within 1 LSB.
    expected_a = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    assert np.allclose(raw[0, :, 3].astype(int), expected_a.astype(int), atol=1)
    # Raw RGB bytes must NOT already be premultiplied -- if they were, the
    # fix is in the wrong layer and quickshow would double-attenuate.
    nontrivial = expected_a < 200  # avoid α≈1 where premult ≈ raw
    naive_premult = np.round(
        raw[0, :, 0].astype(np.float32) * raw[0, :, 3].astype(np.float32) / 255.0
    ).astype(np.uint8)
    assert (
        np.mean(
            np.abs(
                raw[0, nontrivial, 0].astype(int)
                - naive_premult[nontrivial].astype(int)
            )
        )
        > 5
    ), "vertices property looks already-premultiplied; quickshow would break"

    # Now check what Package serializes for the WebGL viewer.
    packaged = _packaged_rgba(vrgb)
    expected = _expected_premultiplied(raw)
    assert packaged.shape == raw.shape
    assert packaged.dtype == np.uint8
    # Alpha channel must be passed through unchanged (shader needs it for 1-α).
    assert np.array_equal(packaged[..., 3], expected[..., 3])
    # RGB channels must be alpha-premultiplied.
    assert np.array_equal(packaged[..., :3], expected[..., :3])

    # And the un-packaged property must NOT have been mutated by Package
    # (Package should defensive-copy).
    raw_after = vrgb.vertices
    assert np.array_equal(raw_after, raw)


def test_vertexrgb_alpha_one_is_passthrough():
    """When α=1 everywhere, premultiplication is a no-op (bug was invisible at α=1)."""
    rng = np.random.default_rng(1)
    r = rng.uniform(0, 1, nverts).astype(np.float32)
    g = rng.uniform(0, 1, nverts).astype(np.float32)
    b = rng.uniform(0, 1, nverts).astype(np.float32)

    vrgb = cortex.VertexRGB(r, g, b, subj)  # default alpha = 1
    raw = vrgb.vertices
    packaged = _packaged_rgba(vrgb)
    assert np.array_equal(packaged[..., 3], raw[..., 3])  # alpha all 255
    # RGB unchanged because α=255 → premultiply by 1.
    assert np.array_equal(packaged[..., :3], raw[..., :3])


def test_vertexrgb_alpha_zero_zeros_rgb():
    """α=0 must drive packaged RGB to 0 -- the shader then renders pure curvature."""
    rng = np.random.default_rng(2)
    r = rng.uniform(0, 1, nverts).astype(np.float32)
    g = rng.uniform(0, 1, nverts).astype(np.float32)
    b = rng.uniform(0, 1, nverts).astype(np.float32)
    alpha = np.zeros(nverts, dtype=np.float32)

    vrgb = cortex.VertexRGB(
        r,
        g,
        b,
        subj,
        alpha=cortex.Vertex(alpha, subj, vmin=0, vmax=1),
    )
    packaged = _packaged_rgba(vrgb)
    assert np.array_equal(packaged[..., 3], np.zeros_like(packaged[..., 3]))
    assert np.array_equal(packaged[..., :3], np.zeros_like(packaged[..., :3]))


def test_volumergb_alpha_is_NOT_premultiplied_in_package():
    """VolumeRGB must ship straight-alpha bytes -- Three.js premultiplies on upload.

    Three.js sets ``tex.premultiplyAlpha = true`` for raw VolumeRGB textures
    (cortex/webgl/resources/js/dataset.js:335-338), which makes WebGL apply
    UNPACK_PREMULTIPLY_ALPHA_WEBGL on texture upload. So the shader sees
    premultiplied RGB by the time vColor is sampled, but ONLY because the
    texture-upload pipeline does it for us. If Package also premultiplied
    here, partial-alpha VolumeRGB would render double-attenuated (too dark).
    """
    rng = np.random.default_rng(3)
    shape = volshape
    r = rng.uniform(0, 1, shape).astype(np.float32)
    g = rng.uniform(0, 1, shape).astype(np.float32)
    b = rng.uniform(0, 1, shape).astype(np.float32)
    alpha = rng.uniform(0, 1, shape).astype(np.float32)

    vrgb = cortex.VolumeRGB(
        r,
        g,
        b,
        subj,
        xfmname,
        alpha=cortex.Volume(alpha, subj, xfmname, vmin=0, vmax=1),
    )

    raw = vrgb.volume
    assert raw.dtype == np.uint8

    # Spy on the Package internals: monkey-patch volume.mosaic to capture the
    # array Package actually ships before it gets PNG-encoded.
    from cortex.webgl import data as webgl_data

    captured = {}
    real_mosaic = webgl_data.volume.mosaic

    def spy_mosaic(arr, show=False):
        captured.setdefault("frames", []).append(arr.copy())
        return real_mosaic(arr, show=show)

    webgl_data.volume.mosaic = spy_mosaic
    try:
        Package(dataset.Dataset(view=vrgb))
    finally:
        webgl_data.volume.mosaic = real_mosaic

    assert len(captured["frames"]) == 1
    packaged_frame = captured["frames"][0]

    # Bytes shipped to the browser must equal the raw .volume bytes verbatim
    # (NOT premultiplied) -- Three.js will premultiply once on texture upload.
    assert packaged_frame.shape == raw[0].shape
    assert np.array_equal(packaged_frame, raw[0])

    # And specifically, RGB should NOT have been premultiplied by alpha.
    naive_premult = _expected_premultiplied(raw[0])
    nontrivial = (raw[0, ..., 3] < 200) & (raw[0, ..., 0] > 50)
    assert (
        np.mean(
            np.abs(
                packaged_frame[nontrivial][..., 0].astype(int)
                - naive_premult[nontrivial][..., 0].astype(int)
            )
        )
        > 5
    ), "VolumeRGB Package output looks premultiplied; Three.js will then double-attenuate"
