"""Tests that the WebGL shader samples cortical depth like quickflat (gh-749).

Both renderers average ``n`` samples between the white matter and pial
surfaces (``thick=n`` in quickflat, ``layers=n`` in the viewer). They must put
those samples at the same depths, or data that varies on the scale of the
sample spacing renders differently in the two.

These tests need no rendering. They evaluate the viewer's shader-generating
JavaScript under node and compare the depths it emits to the depths at which
quickflat's own pixel mapper (``quickflat.utils._make_pixel_cache``) samples
the volume, measured by running that function rather than restating its grid.
"""

import json
import os
import re
import shutil
import subprocess
from typing import Any, Callable, List

import numpy as np
import pytest

import cortex.webgl
from cortex.mapper import samplers
from cortex.quickflat import utils as qf_utils

SHADERLIB = os.path.join(
    os.path.dirname(cortex.webgl.__file__), "resources", "js", "shaderlib.js"
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required"
)

# Runs shaderlib.js with a stubbed THREE (only ShaderChunk is touched while
# building shader source) and prints, as JSON, the sample depths of the
# generated fragment shader for each requested number of layers.
NODE_SCRIPT = """
const fs = require('fs');
const vm = require('vm');
const ctx = {THREE: {ShaderChunk: new Proxy({}, {get: () => ''})}};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8') + '\\n;this.Shaders = Shaders; this.Shaderlib = Shaderlib;', ctx);
const opts = JSON.parse(process.argv[2]);
const out = {};
for (const layers of opts.layers) {
    const frag = ctx.Shaders.surface_pixel({morphs: 3, volume: 1, layers: layers,
        sampler: 'nearest', rgb: opts.rgb, twod: opts.twod, dither: false}).fragment;
    out[layers] = {
        fragment: frag,
        helper: ctx.Shaderlib.sampleDepths ? ctx.Shaderlib.sampleDepths(layers) : null,
    };
}
console.log(JSON.stringify(out));
"""


def _run_shaderlib(layers: List[int], rgb: bool = False, twod: bool = False) -> dict:
    opts = json.dumps(dict(layers=layers, rgb=rgb, twod=twod))
    result = subprocess.run(
        ["node", "-e", NODE_SCRIPT, SHADERLIB, opts],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return {int(k): v for k, v in json.loads(result.stdout).items()}


_NEAREST = samplers.nearest


def _sample_coords(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> List[np.ndarray]:
    """Volume coordinates quickflat's pixel mapper hands its sampler, per call."""
    calls: List[np.ndarray] = []

    def recording_sampler(coords: np.ndarray, shape: Any) -> Any:
        calls.append(np.array(coords))
        return _NEAREST(coords, shape)

    monkeypatch.setattr(samplers, "nearest", recording_sampler)
    qf_utils._make_pixel_cache("S1", "fullhead", height=64, sampler="nearest", **kwargs)
    return calls


@pytest.fixture(scope="module")
def quickflat_depths() -> Callable[[int], np.ndarray]:
    """Fractions toward pial at which quickflat samples with ``thick`` samples.

    The pixel mapper only exposes sample positions in voxel coordinates, so
    they are converted to depths using the two ends of the ribbon, which the
    same function yields as the single sample of ``thick=1`` at depth 0 (white
    matter) and depth 1 (pial). Results are cached: each call runs the mapper.
    """
    with pytest.MonkeyPatch.context() as mp:
        (wm,) = _sample_coords(mp, thick=1, depth=0.0)
        (pial,) = _sample_coords(mp, thick=1, depth=1.0)
        ribbon = pial - wm
        assert np.abs(ribbon).max() > 0, "S1 has no cortical thickness to sample"
        depths = {}
        for thick in LAYERS:
            coords = _sample_coords(mp, thick=thick)
            assert len(coords) == thick
            depths[thick] = np.sort([
                np.sum((c - wm) * ribbon) / np.sum(ribbon * ribbon) for c in coords
            ])
    return depths.__getitem__


def _shader_depths(fragment: str) -> np.ndarray:
    """Fractions toward the white matter surface the shader samples at."""
    pattern = r"coord_x = mix\(vPos_x\[0\], vPos_x\[1\], ([0-9.]+)\);"
    return np.array([float(d) for d in re.findall(pattern, fragment)])


LAYERS = [2, 4, 8, 16, 32]


@pytest.fixture(scope="module")
def shaderlib() -> dict:
    return _run_shaderlib(LAYERS)


@pytest.mark.parametrize("layers", LAYERS)
def test_sample_depths_match_quickflat(
    layers: int, shaderlib: dict, quickflat_depths: Callable[[int], np.ndarray]
) -> None:
    """The shader's depth grid equals quickflat's, as a set of positions.

    The shader mixes pial (0) to white matter (1) while quickflat weights pial
    by t, so the shader depth d corresponds to quickflat's t = 1 - d.
    """
    depths = _shader_depths(shaderlib[layers]["fragment"])
    assert len(depths) == layers
    np.testing.assert_allclose(
        np.sort(1 - depths), quickflat_depths(layers), atol=1e-5
    )


@pytest.mark.parametrize("layers", LAYERS)
def test_sample_depths_avoid_surfaces(layers: int, shaderlib: dict) -> None:
    """No sample sits exactly on the white matter or pial surface."""
    depths = _shader_depths(shaderlib[layers]["fragment"])
    assert depths.min() > 0
    assert depths.max() < 1


@pytest.mark.parametrize("layers", LAYERS)
def test_sample_depths_evenly_spaced_and_symmetric(
    layers: int, shaderlib: dict
) -> None:
    depths = np.sort(_shader_depths(shaderlib[layers]["fragment"]))
    np.testing.assert_allclose(np.diff(depths), 1.0 / (layers + 1), atol=1e-5)
    # Symmetric about the middle of the sheet, so the mean depth is 0.5.
    np.testing.assert_allclose(depths, 1 - depths[::-1], atol=1e-5)
    assert depths.mean() == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize("layers", LAYERS)
def test_sample_depths_helper_matches_shader(layers: int, shaderlib: dict) -> None:
    """The exported helper is what the generated shader actually uses."""
    np.testing.assert_allclose(
        _shader_depths(shaderlib[layers]["fragment"]),
        shaderlib[layers]["helper"],
        atol=1e-6,
    )


@pytest.mark.parametrize("rgb,twod", [(True, False), (False, True)])
def test_sample_depths_same_for_all_data_types(
    rgb: bool, twod: bool, quickflat_depths: Callable[[int], np.ndarray]
) -> None:
    """RGB and 2D shaders sample the same depths as the plain one."""
    out = _run_shaderlib([32], rgb=rgb, twod=twod)
    np.testing.assert_allclose(
        np.sort(1 - _shader_depths(out[32]["fragment"])),
        quickflat_depths(32),
        atol=1e-5,
    )


def test_sample_weights_sum_to_one(shaderlib: dict) -> None:
    """Each sample is weighted 1/layers, matching quickflat's data/thick."""
    for layers in LAYERS:
        weights = re.findall(
            r"values\.x \+= ([0-9.]+)\*", shaderlib[layers]["fragment"]
        )
        assert len(weights) == layers
        assert sum(float(w) for w in weights) == pytest.approx(1.0, abs=1e-4)
