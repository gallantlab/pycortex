# Reference images

Stored renders that `cortex/tests/test_visual_regression.py` asserts against.

## Contents

| directory | images | contents |
| --- | --- | --- |
| `alpha_dataviews/` | 10 | five of the six public dataview classes (`Volume`, `Vertex`, `Volume2D`, `VolumeRGB`, `VertexRGB`), both renderers |
| `nan_dataviews/` | 10 | the same five, with NaNs over roughly half the primary data channel |
| `nan_alpha_dataviews/` | 4 | `VolumeRGB`/`VertexRGB` only, with the NaNs in the `alpha=` map |
| `nonflat_views/` | 4 | `Volume`/`Vertex` on the inflated and fiducial surfaces at `lateral_pivot`, webgl only |

Filenames are `quickflat_<Class>` and `webgl_<Class>`, except `nonflat_views/`,
which uses `webgl_<surface>_<angle>_<Class>`.

`Vertex2D` is the sixth class and has no images: its webgl flatmap renders
blank (gh-714) and `save_3d_views` raises, so it cannot be tested through the
webgl path at all. The two `Vertex2D` tests are **xfailed** on that
`RuntimeError`, strictly — if the render ever succeeds the XPASS says so rather
than passing silently.

## Render settings

The three flatmap directories render `quickflat_*` with
`cortex.quickflat.make_png` and `webgl_*` with `save_3d_views`, both with
curvature **un-thresholded** (`curvature_threshold=False` and
`surface.{subject}.curvature.smoothness=1.0`). (This is to avoid failures from
differences in the renderers' anti-aliasing implementations.)
Everything else is at its default.

`nonflat_views/` keeps pycortex's default thresholded curvature, unlike the
flatmap groups.

The exact keyword arguments are in `_render_and_check_dataview` and
`_render_and_check_webgl_only`; change either and the references must be
regenerated.

## Checks

The three flatmap tests check each render twice: against its own stored
reference at a tight tolerance (`MAX_MEAN_ABS_DIFF`, `MAX_FRACTION_DIFFERING`,
`MAX_FRACTION_GROSSLY_DIFFERING`, `MAX_SSIM_LOSS`, all four of which must pass),
and against the other renderer's render of the same dataview at a loose one
(`CROSS_MAX_MEAN_ABS_DIFF`, `CROSS_MAX_FRACTION_DIFFERING`), with no stored
fixture. `test_visual_comparison_nonflat_views` runs the reference check only.

Both renderers write their flatmap content-tight and transparent outside it, so
the cross-renderer check only resizes webgl to quickflat's size before diffing.
RGB under fully transparent pixels is normalized first: it is undefined there,
and matplotlib leaves white where the browser leaves black.

## Provenance

Generated on `main` (`3779f7ca`), from the demo subject `S1` in the filestore
bundled with pycortex, which is pinned by `cortex/tests/conftest.py`.

| | |
| --- | --- |
| chromium | 151.0.7922.34 (headless shell, SwiftShader software rendering) |
| playwright | 1.62.0 (fixes the chromium build above) |
| matplotlib | 3.10.9 |

Both are pinned in the `test` dependency group, and re-pinning is part of
regenerating. playwright fixes the chromium build, which determines the 16 webgl
references; matplotlib rasterizes the 12 quickflat ones.

Update matplotlib beyond 3.10.9 once Python 3.10 is dropped.

## Format

Lossless WebP (`method=6`, `quality=100`, `exact=True`): bit-exact after decode,
and 59% the size of optimized PNG (1229 KiB versus 2061 KiB for the set of 28).

## Storage

Tracked with **git LFS**. If yours are 130-byte text files rather than images,
the clone has not fetched them:

```
git lfs install && git lfs pull
```

The tests skip on that, and on the images being absent altogether, rather than
failing.

## Distribution

Kept out of the wheel (`exclude_package_data` in `setup.py`) and kept in the
source tarball (`MANIFEST.in`'s `recursive-include cortex *`), so a run against
an installed wheel degrades gracefully.

## Regenerating

The renders are deterministic: repeated runs on one machine produce
bit-identical output, including the WebGL ones under software rendering. They
are coupled to the Chromium and matplotlib builds above, so an upgrade can shift
anti-aliasing and rasterization; the tolerances absorb small shifts. If a
failure exceeds them, inspect the `diff_*.png` files it writes, confirm the
change is cosmetic, then:

```
REGENERATE_REFERENCE_IMAGES=1 pytest cortex/tests/test_visual_regression.py
```

That rewrites all four directories in one run. Review the resulting diff before
committing.
