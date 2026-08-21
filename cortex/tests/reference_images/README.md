# Reference images

Stored renders that `cortex/tests/test_webgl_headless.py` asserts against, so a
change in rendering output fails a test instead of needing to be spotted by eye.

## `alpha_dataviews/`

Twelve images: each of the six public dataview classes (`Volume`, `Vertex`,
`Volume2D`, `Vertex2D`, `VolumeRGB`, `VertexRGB`) rendered through both paths --
`qs_*` via `cortex.quickshow` (matplotlib) and `wg_*` via
`cortex.export.plot_panels` (headless WebGL).

Between them they exercise every way pycortex encodes alpha: `Volume`/`Vertex` are
a no-alpha baseline, `Volume2D`/`Vertex2D` use the 2D alpha colormap
`RdBu_r_alpha`, and `VolumeRGB`/`VertexRGB` use the native `alpha=` keyword. All
six also composite the curvature underlay.

## Provenance

Generated on the `main` branch at commit `5af26a86`, deliberately: they are a
pre-change baseline, so a restructure of `cortex.dataset` has to reproduce main's
output pixel for pixel.

| | |
| --- | --- |
| chromium | 151.0.7922.34 (headless shell, SwiftShader software rendering) |
| matplotlib | 3.11.1 |
| pillow | 12.3.0 |

## Format

Lossless WebP (`method=6`, `quality=100`, `exact=True`): bit-exact after decode,
and 59% the size of optimized PNG (669 KiB versus 1129 KiB for the set).

Two alternatives were measured and rejected:

- **Higher PNG compression** does nothing. With `optimize=True`, `compress_level=6`
  and `compress_level=9` produce byte-identical output.
- **AVIF** is smaller again, but Pillow cannot write it losslessly. Even at
  `qmin=0, qmax=0` it decoded with max\|difference\| 27-29 -- larger than the
  test's own 16-unit per-pixel threshold, so it would corrupt the comparison it
  exists to feed.

Bit-exactness matters because the test compares all four channels; a format that
perturbs values would eat into the tolerance budget meant for genuine
environmental differences.

## Distribution

These are test fixtures with no runtime use, so they are **kept out of the wheel**
(`exclude_package_data` in `setup.py`) and **kept in the source tarball**
(`MANIFEST.in`'s `recursive-include cortex *`). A build from source can therefore
run the test; a `pip install` does not carry ~700 KiB of PNGs into site-packages
for data no user will read.

The test skips, rather than fails, when the images are absent, so a test run
against an installed wheel degrades gracefully.

## Regenerating

The renders are deterministic: repeated runs on one machine produce bit-identical
output, including the WebGL ones under software rendering. They are, however,
coupled to the Chromium and matplotlib builds above, so a browser or matplotlib
upgrade can shift anti-aliasing and rasterization slightly. The test's tolerances
absorb that; if an upgrade moves output beyond them, inspect the `diff_*.png`
files the failure writes, confirm the change is cosmetic, then:

```
REGENERATE_REFERENCE_IMAGES=1 pytest cortex/tests/test_webgl_headless.py -k visual_comparison
```

Review the resulting diff before committing -- regenerating is how a real
regression gets silently blessed.
