# Pycortex testing survey and recommendation plan

This document surveys the current state of the pycortex test suite and lays out a
prioritized plan to increase test coverage of the most critical parts of the
library. The goal is **not** 100% coverage: it is protecting the pieces where a
silent bug corrupts scientific results or user data, and the pieces that break
most often in practice.

Coverage numbers below come from running the suite on Linux/Python 3.11 without
Playwright or Inkscape available (74 passed, 63 skipped, overall **37% line
coverage**). In CI both tools are installed, so the WebGL headless and
Inkscape-dependent numbers are higher there; the numbers for the core numerical
modules are unaffected by those skips and are the ones this plan focuses on.

## 1. Current state of the suite

**Infrastructure** (in good shape):

- pytest + pytest-cov + pytest-timeout, Codecov upload, CI matrix over Python
  3.10–3.14 (`.github/workflows/run_tests.yml`).
- A bundled 41 MB `S1` subject in `filestore/` with real surfaces (flat, pia,
  wm, inflated), two transforms (`fullhead`, `retinotopy`), an anatomical
  volume, and `overlays.svg` — this makes true integration tests possible.
- Playwright-driven headless browser tests for the WebGL viewer
  (`test_webgl_headless.py`, 1056 lines) that do pixel-level verification of
  rendering. These were added in response to real regressions and are the
  strongest part of the suite.
- An `isolated_filestore` fixture pattern (in `test_utils.py`) that redirects
  `cortex.db` to a temp directory.

**What is well covered today:**

| Area | Coverage | Tests |
|---|---|---|
| `dataset/` (Volume/Vertex/RGB/2D views) | 72–88% | `test_dataset.py` (34 tests: construction, save/load, RGB, NaN/alpha semantics) |
| WebGL data packaging (alpha premultiplication) | — | `test_webgl_data.py` |
| WebGL rendering (CI only) | — | `test_webgl_headless.py`, `test_headless.py` |
| `hcp.py` (fs_LR resampling) | 59% | `test_hcp.py` (barycentric properties, CIFTI handling) |
| `freesurfer` surf2surf resampling | — | `test_freesurfer.py` (row sums, constants preserved) |
| `segment` config handling | — | `test_segment.py` (mocked subprocess) |
| `download_subject` | — | `test_utils.py` |

**Where the recent bugs actually were** (last ~60 fix commits): WebGL viewer
rendering and async races (majority — now regression-tested), NaN/alpha
handling in dataviews (now regression-tested), vmin/vmax handling, Inkscape
version detection, a left-hemisphere-only flatmap smoothing bug, Windows path
handling. The pattern: bugs cluster in the visualization pipeline and in
platform/tool integration, and the team already writes regression tests for
them. What has *no* safety net is the numerical core described below.

## 2. Critical gaps, prioritized

### Tier 1 — scientific correctness core

These modules turn fMRI volumes into the surface values users publish. A bug
here silently produces wrong science. They are also pure numerics — cheap to
test, no browser or external tool needed.

**1. `cortex/mapper/` — volume↔surface projection.**
`mapper/volume.py` 0%, `samplers.py` 19%, `mapper/utils.py` 11%, `line.py` 53%,
`mapper.py` 52%. This is the single most important untested subsystem.
Recommended tests:

- Sampler unit tests (`nearest`, `trilinear`, `gaussian`, `lanczos`) on tiny
  synthetic coordinate sets against hand-computed weights; trilinear weights
  sum to 1; nearest picks the correct voxel at ties/edges.
- Mapper properties on the bundled `S1`/`fullhead`: for each mapper type
  (`nearest`, `trilinear`, `line_nearest`, `line_trilinear`, ...), a constant
  volume maps to a constant vertex array (mass preservation), output length
  equals `lh + rh` vertex counts, `mask`/`hemimasks` shapes match the
  transform shape.
- `backwards()` round-trip sanity: forward-then-backward of a smooth volume
  correlates highly with the input inside the mask.
- Cache correctness: `get_mapper(..., recache=True)` equals the cached result
  loaded via `Mapper.from_cache` (guards the npz serialization).

**2. `cortex/xfm.py` — coordinate transforms (24%).**
Every volume that enters pycortex passes through `Transform`. Recommended
tests:

- Algebra: `inv`, `__mul__`, `__call__` against plain numpy on random affines;
  `T * T.inv ≈ identity`.
- `from_fsl`/`to_fsl` round-trip on small NIfTI fixtures with non-trivial
  affines, **including a negative-determinant (radiological) affine** — the
  `_x_flipper` branch is exactly the kind of code that regresses silently.
- `from_freesurfer`/`to_freesurfer` round-trip; `_vox2ras_tkr` against known
  values (can mock the FreeSurfer subject dir; the math is local).

**3. `cortex/volume.py` — masking (28%).**
`unmask`/mask round-trip property tests (`data == unmask(mask, data)[mask]`)
for bool, uint8-RGB and masked-array paths; `mosaic` output shape;
`anat2epispace`/`epi2anatspace` scipy-path round-trip on a smooth volume
(FSL-wrapping variants can be mocked or skipped).

**4. `cortex/polyutils/` — surface geometry.**
`surface.py` 45%, `subsurface.py` 8%, `distortion.py` 16%. `geodesic_distance`,
`smooth`, and the distortion metrics feed flatmap QC and `get_roi_masks`.
Recommended tests on *analytically known meshes* (a flat triangulated grid, a
unit icosphere — a small fixture-mesh factory is a few lines with
`scipy.spatial` or hand-written):

- Geodesic distance ≈ Euclidean on a planar mesh; symmetric; zero at source.
- `face_normals`/`vertex_normals`/`face_areas` on a unit cube.
- `smooth(constant) == constant`; smoothing reduces variance.
- Areal/metric distortion of an identity "flattening" is ~0.
- `SubsurfaceMixin`: subsurface of a patch contains exactly the seeded
  vertices' neighborhood; index maps are consistent.

### Tier 2 — data integrity and IO

**5. `cortex/database.py` — the filestore (51%).**
This is the central mutable state; a path bug destroys user data (see the
recent "sulcus install instructions destroyed existing sulci" doc fix and the
Windows path fix). Using the existing `isolated_filestore` pattern promoted
into a shared `conftest.py`:

- `save_xfm`/`get_xfm` round-trip for each `xfmtype`, including the
  "refuse to silently overwrite with a different reference" branch.
- `save_mask`/`get_mask` round-trip; wrong-shape mask rejected.
- `get_surf` for each hemisphere/`merge`/`nudge` combination on `S1`
  (shapes, left-right vertex offset when merged).
- `get_coords`, `get_surfinfo` smoke on `S1`; `make_subj` creates the
  expected directory skeleton.

**6. `cortex/formats.pyx` — mesh IO (only GIfTI tested).**
Add write→read round-trip tests for VTK, OFF, STL, and OBJ on a tiny mesh
(exact equality of `pts`/`polys`). This is compiled Cython, so line coverage
won't show it — the round-trip tests are the coverage. Guards against numpy
API drift (a `fromstring`→`frombuffer` breakage already happened once).

**7. Dataset HDF5 round-trips.** `test_dataset.py` covers save/load for the
basics; extend to `VolumeRGB`, `Vertex2D`, and masked data with
`load → repack → compare` equality.

### Tier 3 — visualization pipeline (regression-prone)

**8. `cortex/quickflat/` — flatmap plotting.**
`composite.py` 14%, `view.py` 22%, `utils.py` 65%. `quickflat` is the most
used public API after the dataset classes. Much of it *can* be tested without
Inkscape by keeping ROIs/labels off:

- `make_flatmap_image` for every dataview type (Volume, Vertex, RGB, 2D,
  masked, NaN-containing): output extents, alpha handling, `nanmean`
  behavior (partially exists — extend the matrix).
- Pure helpers: `_convert_svg_kwargs`, `_color2hex`, `_get_extents`,
  `_check_colorbar_location` (cheap, table-driven).
- `make_figure` smoke with `with_rois=False, with_labels=False` +
  `make_png` writes a nonempty file.
- **Fix the local-vs-CI split:** `test_warn_non_perceptually_uniform_2D_cmap`
  currently fails on machines without Inkscape because `quickshow` defaults
  to rendering ROIs. Add an `inkscape` pytest marker with an auto-skip (the
  version-probing helper already exists in `cortex/testing_utils.py`) so a
  bare `pytest` run is green everywhere.

**9. `cortex/svgoverlay.py` (56%).** Parses and mutates `overlays.svg` — it
feeds both quickflat *and* `get_roi_verts`/`get_roi_masks` (scientific
output, not just pictures). Test against the bundled `S1/overlays.svg`:
layer enumeration, `get_mask` for a known ROI returns a plausible vertex set,
`add_layer` is idempotent, `toxml` round-trips through a re-parse. Keep
`get_texture` tests behind the `inkscape` marker.

**10. `cortex/utils.py` (36%).** Big grab-bag; target only the scientific
entry points: `get_roi_verts`/`get_roi_masks` on `S1` (masks are disjoint
under `split_lr`, contained in the cortical mask, known ROI is non-empty),
`get_cortical_mask` for each type, `get_cmap`/`add_cmap`. Skip the
Inkscape/ROI-drawing and movie helpers.

**11. WebGL non-browser layer.** `webgl/serve.py`, `view.py`, `htmlembed.py`
are 0% locally but exercised by the CI headless suite — that is the right
tool for them; don't duplicate with mocks. Worth adding: unit tests for
`webgl/data.py` (58%) packaging edge cases (int dtypes, masked volumes) that
don't need a browser.

### Tier 4 — explicitly deprioritized

Wrap external tools or GUIs; chasing coverage here is low-value. Test only
argument/matrix assembly with mocked `subprocess` (the pattern in
`test_segment.py`), or nothing at all:

- `align.py` (9%) — FSL + Mayavi GUI. Pure candidates: FLIRT matrix handling
  via `xfm.from_fsl` tests above.
- `freesurfer.py` (23%) — the resampling math is already tested; add tiny
  binary fixtures for `parse_surf`/`parse_curv`/`parse_patch` +
  `write_patch` round-trip, mock the `mri_*` command assembly. Skip the
  import/export orchestration.
- `mni.py` (0%) — mock FSL; test only transform composition.
- `mayavi_aligner.py`, `blender/`, `segment.py` GUI paths, `rois.py`
  (legacy), `formats_old.py`, `fmriprep.py`, `brainctm.py` (exercised
  indirectly by CI WebGL tests) — leave alone, and exclude the pure-GUI
  modules from the coverage denominator (see below).

## 3. Infrastructure recommendations

1. **Create `cortex/tests/conftest.py`** and move/promote shared fixtures:
   `isolated_filestore`, a `tiny_mesh` factory (planar grid + icosphere), a
   `tiny_nifti` factory (parameterized affine, incl. negative determinant),
   and a `tiny_subject` builder that writes a minimal subject into the
   isolated filestore via `formats.write_gii`. Tier 1–2 tests then run in
   milliseconds without the 41 MB `S1`, which stays for integration tests.
2. **Add pytest markers** `headless` (Playwright) and `inkscape`, with
   auto-skip logic in `conftest.py` based on tool availability. Document
   `pytest -m "not headless"` as the fast local loop. This also fixes the one
   currently-failing local test.
3. **Trim the coverage denominator**: add a `[tool.coverage.run] omit` for
   `mayavi_aligner.py`, `blender/*`, `formats_old.py`, and `appdirs.py`
   (vendored) so the Codecov number tracks code that can realistically be
   tested. ~1100 statements of GUI/vendored code currently dilute the signal.
4. **Optional**: `hypothesis` property-based tests are a natural fit for
   `unmask`/`Transform` algebra, but plain parameterized tests cover the plan
   above; adopt only if the team wants the dependency.

## 4. Suggested order of work

| Phase | Content | Effort | Why first |
|---|---|---|---|
| 1 | conftest fixtures + markers + coverage `omit` (infra) | ~1 day | Everything else builds on it; makes local runs green |
| 2 | `xfm.py` algebra + FSL/FreeSurfer round-trips; `formats` round-trips; `volume.py` mask/unmask; `database.py` save/get round-trips | 1–2 days | Highest value per effort; pure Python/numpy |
| 3 | `mapper/` correctness suite (samplers + mapper properties + cache) | 2–3 days | The scientific core; needs the Phase 1 fixtures |
| 4 | `polyutils` geometry on analytic meshes | 1–2 days | Protects geodesic/ROI machinery |
| 5 | `quickflat` image-matrix + helpers; `svgoverlay` parsing; `utils.get_roi_masks` | 1–2 days | Most-used public API; regression-prone |
| 6 | `freesurfer` parsers + mocked command assembly | 1 day | Rounds out import paths |

Phases 2–3 alone should lift the meaningful (post-`omit`) coverage well above
50% and, more importantly, put invariant checks around every number pycortex
produces for downstream analysis.

## Appendix: coverage snapshot (local, Playwright/Inkscape skipped)

```
Module                          Cover   Module                          Cover
cortex/dataset/view2D.py          88%   cortex/quickflat/utils.py         65%
cortex/dataset/viewRGB.py         86%   cortex/hcp.py                     59%
cortex/dataset/views.py           80%   cortex/webgl/data.py              58%
cortex/dataset/dataset.py         74%   cortex/svgoverlay.py              56%
cortex/dataset/braindata.py       72%   cortex/mapper/line.py             53%
cortex/mapper/point.py           100%   cortex/mapper/mapper.py           52%
                                        cortex/database.py                51%
cortex/polyutils/surface.py       45%   cortex/volume.py                  28%
cortex/polyutils/misc.py          43%   cortex/xfm.py                     24%
cortex/utils.py                   36%   cortex/freesurfer.py              23%
cortex/mapper/patch.py            34%   cortex/quickflat/view.py          22%
cortex/exact_geodesic.py          34%   cortex/surfinfo.py                20%
cortex/mapper/samplers.py         19%   cortex/polyutils/distortion.py    16%
cortex/quickflat/composite.py     14%   cortex/mapper/utils.py            11%
cortex/align.py                    9%   cortex/polyutils/subsurface.py     8%
cortex/mapper/volume.py            0%   cortex/mni.py                      0%
cortex/rois.py                     0%   cortex/brainctm.py                 0%
cortex/webgl/{serve,view}.py       0%*  cortex/fmriprep.py                 0%

* covered in CI by the Playwright headless suite
TOTAL (local): 37%
```
