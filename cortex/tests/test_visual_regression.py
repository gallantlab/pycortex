"""Visual regression tests: quickflat and webgl renders vs stored references.

Four suites. Three of them render flatmaps of the six public dataview classes
(``Volume``, ``Vertex``, ``Volume2D``, ``Vertex2D``, ``VolumeRGB``,
``VertexRGB``) through both matplotlib (``cortex.quickshow``) and the headless
WebGL viewer (``cortex.export.plot_panels``), varying what the data carries:
alpha-bearing values, NaNs in the data channels, and NaNs in the alpha map. The
last covers only the two RGB classes, the only ones taking an explicit
``alpha=``. Every one of those renders is checked twice -- against its own
stored reference at a tight tolerance, and directly against the other renderer's
render of the same dataview at a loose one.

The fourth suite renders non-flatmap views, ``Volume`` and ``Vertex`` on the
inflated and fiducial surfaces, through ``save_3d_views``. Those are
webgl-only and get the reference check alone: ``cortex.quickshow`` produces
flatmaps and nothing else, so there is nothing to diff them against.

See ``reference_images/README.md`` for how the references were produced and how
to regenerate them.

All tests are skipped if playwright is not installed.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pytest

import cortex
import cortex.export
import cortex.polyutils
from cortex.dataset import Dataview
from cortex.tests.testing_utils import has_playwright

pytestmark = pytest.mark.skipif(
    not has_playwright, reason="playwright and chromium are required"
)

# Vertex2D cannot be tested through the webgl path: its flatmap renders blank
# (gh-714) and save_3d_views raises, so no reference can be generated. #679's
# lighting refactor ported the HASFLAT bump-displacement block into the vertex
# shader, which under headless/SwiftShader leaves that flatmap unrendered. The
# mark is strict and on RuntimeError specifically, so a render that starts
# succeeding reports an XPASS rather than passing silently.
DATAVIEW_NAMES = [
    "Volume",
    "Vertex",
    "Volume2D",
    pytest.param(
        "Vertex2D",
        marks=pytest.mark.xfail(
            raises=RuntimeError,
            strict=True,
            reason="gh-714: the Vertex2D flatmap shader fails to link",
        ),
    ),
    "VolumeRGB",
    "VertexRGB",
]

subj = "S1"
xfmname = "fullhead"

#: Stored renders this test asserts against. See that directory's README for how
#: they were produced and how to regenerate them.
REFERENCE_ROOT = Path(__file__).parent / "reference_images"

REFERENCE_DIR = REFERENCE_ROOT / "alpha_dataviews"

#: As REFERENCE_DIR, but for dataviews whose source data contains NaNs.
NAN_REFERENCE_DIR = REFERENCE_ROOT / "nan_dataviews"

#: As NAN_REFERENCE_DIR, but with the NaNs in the *alpha map* rather than in the
#: data. Only the RGB dataviews take an explicit ``alpha=``, so only those two.
NAN_ALPHA_REFERENCE_DIR = REFERENCE_ROOT / "nan_alpha_dataviews"

#: Dataviews that accept an explicit alpha map, and so can carry NaNs in it.
NAN_ALPHA_DATAVIEW_NAMES = ["VolumeRGB", "VertexRGB"]

#: Non-flatmap views, checked against a webgl reference only. quickflat renders
#: nothing but flatmaps, so these have no counterpart to diff against and no
#: cross-renderer leg -- see test_visual_comparison_nonflat_views.
NONFLAT_REFERENCE_DIR = REFERENCE_ROOT / "nonflat_views"

#: (surface, angle, dataview). Volume and Vertex cover both shader paths, which
#: matters because the flatmap suite exercises them under conditions that turn
#: out to be a different regime: the two known webgl lighting bugs reproduce on
#: flatmaps only.
NONFLAT_VIEWS = [
    ("inflated", "lateral_pivot", "Volume"),
    ("inflated", "lateral_pivot", "Vertex"),
    ("fiducial", "lateral_pivot", "Volume"),
    ("fiducial", "lateral_pivot", "Vertex"),
]

#: Lossless WebP: bit-exact after decode and 59% the size of optimized PNG. AVIF
#: is smaller still but Pillow cannot write it losslessly -- it was measured at
#: max|difference| 27-29, larger than DIFF_THRESHOLD below, so it would corrupt
#: the comparison it is meant to feed. Higher PNG compression levels are pointless
#: here: level 6 and level 9 produce byte-identical output.
REFERENCE_SUFFIX = ".webp"

#: First bytes of a git-lfs pointer. The reference images are LFS-tracked, so
# if LFS hasn't been properly initialized, these 130-byte text stubs exist in
# place of the images.
LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"

#: Rewrite the references from this run instead of comparing against them.
REGENERATE_REFERENCES = bool(os.environ.get("REGENERATE_REFERENCE_IMAGES"))

# Tolerances. The renders are deterministic -- repeated runs on one machine are
# bit-identical -- so these are not absorbing noise. They exist because the
# references are coupled to the Chromium and matplotlib builds that produced them,
# and an upgrade can shift anti-aliasing and rasterization slightly. They are far
# tighter than any real regression: a wrong colormap, a dropped alpha channel or
# swapped color channels all move large areas of the image by much more.
MAX_MEAN_ABS_DIFF = 2.0        # mean |difference| over all pixels/channels, of 255
DIFF_THRESHOLD = 16            # a pixel "differs" if any channel moves by more
MAX_FRACTION_DIFFERING = 0.02  # at most this fraction of pixels may differ

# The two limits above are both weak against a change that moves a *small* number
# of pixels by a *large* amount, which is what a geometry or contour shift looks
# like: the mean is diluted by the ~97% of pixels that did not move, and a shifted
# contour only just clears the fraction limit. The #679 vertex-shader change was
# caught with 1.4x margin on the fraction and would have passed the mean outright
# (1.574 against a limit of 2.0), despite moving 2.8% of pixels by a median of
# 67/255.
#
# So two further criteria, each covering what the others miss. Measured against
# simulated cosmetic drift (a 596<->594 resample, +-1 LSB quantisation, gamma
# 1.02) versus the real #679 change:
#
#            metric                cosmetic     #679    separation
#            mean                    0.596      1.574       2.6x
#            fraction > 16           0.48%      2.79%       5.8x
#            fraction > 32          0.0052%     2.16%       large
#            SSIM loss              0.0022     0.0393      18.3x
#
# They are complementary rather than ranked, so all four are checked:
#   - mean and fraction>16 catch broad, low-amplitude shifts (a gamma change
#     scores 6.68 mean but 0.00% on fraction>32).
#   - fraction>32 catches sparse, high-amplitude ones (the #679 case).
#   - SSIM catches structural change, but is computed on luminance and is
#     therefore blind to a channel permutation -- an R/B swap scores 0.0000 SSIM
#     loss while scoring 21.75 on the mean. It only adds sensitivity alongside
#     the others; it cannot replace them.
#
# Caveat on the limits below: the cosmetic figures are *simulated*, not measured
# across a real Chromium or matplotlib upgrade. A genuine toolchain bump changes
# anti-aliasing, which can move a few pixels a long way and so does show up in
# fraction>32. Both limits are set well above the simulated drift rather than at
# it, to leave room for that.
#
# The gross threshold is 32 rather than 64 because of a second worked example.
# gh-695 ("unify NaN and alpha handling", cb976270 on fix/nan-alpha-parity as of
# writing) changes four quickflat volumetric renders through premultiplied-alpha
# thickness averaging, and at 64 every one of
# them scores 0.000% -- the suite missed it entirely. Its mean cannot be tightened
# into a catch either: at 0.19 it sits *below* the simulated cosmetic floor of
# 0.60, so a tighter mean yields false positives before it yields a catch. At 32
# the signal is 0.256% against this 0.1% limit while the worst simulated cosmetic
# drift is 0.0052%, i.e. 19x below it. Dropping to 32 also strengthens the #679
# catch (1.46% -> 2.16%). The cost is headroom: at 64 no cosmetic perturbation
# registered at all, so a real toolchain bump now has 19x of room rather than
# effectively unlimited.
GROSS_DIFF_THRESHOLD = 32             # a pixel differs "grossly" if any channel moves by more
MAX_FRACTION_GROSSLY_DIFFERING = 0.001
MAX_SSIM_LOSS = 0.01                  # 1 - mean SSIM over the luminance channel

# quickflat and webgl's flatmap renders disagree by a fixed anisotropic
# scale + translation, not a genuine content difference: fitting a homography
# (cv2.findTransformECC, MOTION_HOMOGRAPHY) between the two on curvature-only
# content (data alpha=0, so no dataview-specific signal) landed on essentially
# zero rotation and zero projective terms -- i.e. no real perspective
# distortion -- but scale_x=0.9690, scale_y=0.9365, and a ~7px translation.
# Applying just that (affine, webgl -> quickflat's frame) dropped curvature-only
# mean|diff| from 18.9 to 4.4. Translation is stored as a fraction of
# quickflat's own (width, height) so it scales if that size varies slightly
# between dataviews; the scale factors are already dimensionless.
CROSS_RENDERER_SCALE_X = 0.9690
CROSS_RENDERER_SCALE_Y = 0.9365
CROSS_RENDERER_TRANSLATE_X_FRAC = 6.9008 / 392
CROSS_RENDERER_TRANSLATE_Y_FRAC = 6.9650 / 204

# Cross-renderer tolerances: quickflat vs webgl for the *same* dataview, rather
# than each against its own reference. Looser than the within-renderer ones
# above because matplotlib and Three.js still differ in anti-aliasing and
# colormap sampling even after the affine correction above removes the
# dominant coordinate-frame mismatch. What must not happen is the kind of
# broad color/shape disagreement (wrong colormap, dropped alpha, swapped
# channels) that a real regression causes, which is far larger than this.
#
# Calibrated across all twelve flatmap pairs, which with the affine correction
# span mean|diff| 1.06-2.01 and 0.10-1.64% of pixels differing. The limits sit
# 3-4x above those worst cases: tight enough that losing the correction, or a
# real colormap/alpha/channel regression, fails immediately, loose enough to
# absorb a Chromium or matplotlib upgrade.
CROSS_MAX_MEAN_ABS_DIFF = 8.0
CROSS_DIFF_THRESHOLD = 64
CROSS_MAX_FRACTION_DIFFERING = 0.05

# Render settings chosen to minimize cross-renderer disagreement, from a factorial
# sweep of both renderers' settings (curvature brightness/contrast/threshold,
# depth, sampler, thick/layers, and webgl's three lighting controls).
#
# Only one setting was worth changing from the defaults: curvature thresholding.
# Thresholded curvature puts a hard binary edge at curvature=0 whose sub-pixel
# placement each rasterizer resolves differently, so it is maximally sensitive to
# the residual misalignment; smooth curvature is low-frequency and resamples
# cleanly. Averaged over the brightness/contrast grid the thresholded pairing
# scored 8.50 against 3.66 for the smooth one, and 4.98 vs 2.66 at the default
# brightness/contrast. quickflat's ``curvature_threshold`` and webgl's
# ``curvature.smoothness`` are the same knob from opposite ends -- smoothness 0.0
# *is* thresholded -- so both have to move together or they disagree by more.
#
# Everything else stayed at its default, on the evidence:
#   - lighting: the default (all off) already ties the best combination at 2.304;
#     `uniform_illumination=1` renders identically on a flatmap (0.003). Only
#     `topleft_lighting=1` hurts (15.30), and it is off by default.
#   - sampler: trilinear beat nearest by ~1% (2.336 vs 2.365), i.e. noise.
#   - depth / thick / layers: total spread 0.38 over the whole grid, below the
#     noise floor -- the polarity difference between quickflat's `depth` and
#     webgl's `thickmix` is not resolvable and does not matter here.
#
# NB this deliberately renders with curvature *un*-thresholded, which is not
# pycortex's default appearance, so these references do not cover the default
# curvature path. That is the trade for a ~2x tighter cross-renderer floor.
QUICKFLAT_CURVATURE_THRESHOLD = False
WEBGL_CURVATURE_SMOOTHNESS = 1.0


def _ssim(a: npt.NDArray, b: npt.NDArray) -> float:
    """Mean structural similarity between two RGBA images, over luminance.

    Standard SSIM with an 11x11 Gaussian window (sigma 1.5) and the usual
    stabilising constants, implemented on scipy because scikit-image, which
    would otherwise supply it, is not a pycortex dependency. Returns 1.0 for
    identical input.

    Computed on the channel mean, so it is invariant to a channel permutation --
    see the note on MAX_SSIM_LOSS. It is a structural check, not a color one.
    """
    from scipy.ndimage import gaussian_filter

    x = a[..., :3].mean(axis=-1).astype(np.float64)
    y = b[..., :3].mean(axis=-1).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    blur = lambda img: gaussian_filter(img, sigma=1.5, truncate=(11 - 1) / 2 / 1.5)
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov = blur(x * y) - mu_x * mu_y

    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float((num / den).mean())


def _unusable_reference(ref_path: Path) -> Optional[str]:
    """Why ``ref_path`` cannot be compared against, or None if it can.

    Absent and unfetched-from-LFS are separate cases with separate remedies, and
    neither should fail the run: an installed wheel legitimately has no
    references, and a pointer means the clone simply has not fetched them.
    """
    if not ref_path.exists():
        return (
            f"No reference {ref_path.name} in {ref_path.parent}. "
            "See that directory's README for regeneration."
        )
    with open(ref_path, "rb") as handle:
        if handle.read(len(LFS_POINTER_MAGIC)) == LFS_POINTER_MAGIC:
            return (
                f"{ref_path.name} is an unfetched git-lfs pointer, not an "
                "image. Run `git lfs pull`."
            )
    return None


def _reference_store_unusable() -> Optional[str]:
    """Why the whole reference store is unusable, or None if it is fine.

    Absent-entirely and pointers-everywhere are properties of the checkout, not
    of one render, so they are settled once at import. Leaving them to the
    per-file check would render all eighteen views before skipping -- about 90
    seconds to produce eighteen skips. The per-file check still runs, for the
    case this cannot see: a populated directory missing one render.
    """
    stored = sorted(REFERENCE_ROOT.glob(f"*/*{REFERENCE_SUFFIX}"))
    if not stored:
        return (
            f"No reference images under {REFERENCE_ROOT}. "
            "See that directory's README for regeneration."
        )
    return _unusable_reference(stored[0])


if not REGENERATE_REFERENCES:
    _store_problem = _reference_store_unusable()
    if _store_problem is not None:
        pytest.skip(_store_problem, allow_module_level=True)


def _check_against_reference(
    name: str,
    actual_path: Path,
    debug_dir: Path,
    reference_dir: Path,
) -> Optional[str]:
    """Compare one render to its reference.

    Checks four criteria, all of which must pass -- see ``MAX_MEAN_ABS_DIFF``
    and the tolerances below it for why they are complementary rather than
    redundant. Returns a description of every breach, or None if the render
    matches.

    A shape mismatch is not an immediate failure: the render is resized to the
    reference's size before diffing, since the two are still expected to show
    the same content at a slightly different trim/crop. That is a plain resize,
    without the affine that ``_check_cross_renderer`` also applies -- one
    renderer against itself has no coordinate-frame mismatch to undo.

    On any breach, writes the (possibly resized) render and an amplified
    difference image into ``debug_dir``, so the change can be inspected rather
    than guessed at. With ``REGENERATE_REFERENCES`` set it overwrites the
    reference instead and reports no mismatch.
    """
    from PIL import Image

    ref_path = reference_dir / f"{name}{REFERENCE_SUFFIX}"

    if REGENERATE_REFERENCES:
        reference_dir.mkdir(parents=True, exist_ok=True)
        Image.open(actual_path).convert("RGBA").save(
            ref_path, format="WEBP", lossless=True, method=6, quality=100, exact=True
        )
        return None

    actual_im = Image.open(actual_path).convert("RGBA")
    ref_im = Image.open(ref_path).convert("RGBA")
    shape_note = ""
    if actual_im.size != ref_im.size:
        shape_note = f" (aligned: render was {actual_im.size}, reference is {ref_im.size})"
        actual_im = actual_im.resize(ref_im.size, Image.BILINEAR)

    actual = np.asarray(actual_im).astype(np.int16)
    ref = np.asarray(ref_im).astype(np.int16)

    diff = np.abs(actual - ref)
    per_pixel = diff.max(axis=-1)
    mean_abs = float(diff.mean())
    fraction = float((per_pixel > DIFF_THRESHOLD).mean())
    gross = float((per_pixel > GROSS_DIFF_THRESHOLD).mean())
    ssim_loss = 1.0 - _ssim(actual, ref)

    breaches = []
    if mean_abs > MAX_MEAN_ABS_DIFF:
        breaches.append(f"mean|diff|={mean_abs:.3f} (limit {MAX_MEAN_ABS_DIFF})")
    if fraction > MAX_FRACTION_DIFFERING:
        breaches.append(
            f"{fraction:.2%} of pixels differ by more than {DIFF_THRESHOLD} "
            f"(limit {MAX_FRACTION_DIFFERING:.0%})"
        )
    if gross > MAX_FRACTION_GROSSLY_DIFFERING:
        breaches.append(
            f"{gross:.3%} of pixels differ by more than {GROSS_DIFF_THRESHOLD} "
            f"(limit {MAX_FRACTION_GROSSLY_DIFFERING:.1%})"
        )
    if ssim_loss > MAX_SSIM_LOSS:
        breaches.append(f"SSIM loss={ssim_loss:.4f} (limit {MAX_SSIM_LOSS})")
    if not breaches:
        return None

    actual_im.save(debug_dir / f"actual_{name}.png")
    amplified = np.clip(diff[..., :3] * 8, 0, 255).astype("uint8")
    Image.fromarray(amplified).save(debug_dir / f"diff_{name}.png")
    return f"{name}: " + "; ".join(breaches) + shape_note


def _check_cross_renderer(
    name: str,
    quickflat_path: Path,
    webgl_path: Path,
    debug_dir: Path,
) -> Optional[str]:
    """Compare a quickflat render directly against its webgl counterpart.

    Unlike ``_check_against_reference`` this has no stored fixture: it diffs
    the two renders produced by *this* test run against each other. webgl's
    render is resized to quickflat's own size, then the fixed affine
    correction above (scale + translation) is applied to align it onto
    quickflat's coordinate frame before diffing -- see
    ``CROSS_RENDERER_SCALE_X`` for how that was derived. Returns a mismatch
    description, or None if they agree within the (loose) cross-renderer
    tolerance.
    """
    from PIL import Image

    qf_im = Image.open(quickflat_path).convert("RGBA")
    wg_im = Image.open(webgl_path).convert("RGBA")
    shape_note = f" (aligned: quickflat was {qf_im.size}, webgl was {wg_im.size})"

    size = qf_im.size
    w, h = size
    tx = CROSS_RENDERER_TRANSLATE_X_FRAC * w
    ty = CROSS_RENDERER_TRANSLATE_Y_FRAC * h
    wg_aligned = wg_im.resize(size, Image.BILINEAR).transform(
        size, Image.AFFINE,
        (CROSS_RENDERER_SCALE_X, 0.0, tx, 0.0, CROSS_RENDERER_SCALE_Y, ty),
        resample=Image.BILINEAR,
    )

    qf = np.asarray(qf_im).astype(np.int16)
    wg = np.asarray(wg_aligned).astype(np.int16)

    diff = np.abs(qf - wg)
    mean_abs = float(diff.mean())
    fraction = float((diff.max(axis=-1) > CROSS_DIFF_THRESHOLD).mean())
    if mean_abs <= CROSS_MAX_MEAN_ABS_DIFF and fraction <= CROSS_MAX_FRACTION_DIFFERING:
        return None

    amplified = np.clip(diff[..., :3] * 4, 0, 255).astype("uint8")
    Image.fromarray(amplified).save(debug_dir / f"cross_diff_{name}.png")
    return (
        f"cross_{name}: quickflat vs webgl mean|diff|={mean_abs:.3f} "
        f"(limit {CROSS_MAX_MEAN_ABS_DIFF}), {fraction:.2%} of pixels differ by "
        f"more than {CROSS_DIFF_THRESHOLD} (limit {CROSS_MAX_FRACTION_DIFFERING:.0%})"
        f"{shape_note}"
    )


# Gaussian falloff from `seed`, used as the accuracy/alpha channel.
def _bump(
    surf: cortex.polyutils.Surface, seed: int, sigma: float
) -> npt.NDArray[np.floating]:
    d = np.linalg.norm(surf.pts - surf.pts[seed], axis=1)
    return np.exp(-(d**2) / (2 * sigma**2))


def _synth_arrays() -> dict:
    """The clean volume and surface data all three builders start from.

    Shared so they cannot drift apart: what distinguishes the suites is which
    elements they then NaN out, and that is the thing under test.
    """
    zz, yy, xx = np.mgrid[0:31, 0:100, 0:100]
    center = np.array([15, 50, 50])
    sigma_v = 25.0
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2

    # Vertex data is encoded by spatial coordinate, not by vertex index.
    surfs = [
        cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subj, "fiducial")
    ]
    num_verts = [s.pts.shape[0] for s in surfs]
    pts = np.vstack([surfs[0].pts, surfs[1].pts])
    y_centered = pts[:, 1] - pts[:, 1].mean()

    return dict(
        xx=xx, yy=yy, zz=zz,
        num_verts=num_verts,
        data_vol=(xx - 50) / 50.0,                        # ~ [-1, 1]
        accuracy_vol=np.exp(-dist2 / (2 * sigma_v**2)),   # [0, 1] bump
        red_vol=np.clip(xx / 99.0, 0, 1),
        green_vol=np.clip(yy / 99.0, 0, 1),
        blue_vol=np.clip(zz / 30.0, 0, 1),
        data_vtx=y_centered / np.abs(y_centered).max(),   # [-1, 1]
        xyz_norm=(pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0)),
        accuracy_vtx=np.hstack([
            _bump(surfs[0], num_verts[0] // 2, sigma=40.0),
            _bump(surfs[1], num_verts[1] // 2, sigma=40.0),
        ]),
    )


def _dataview(
    name: str,
    *,
    data_vol: npt.NDArray,
    dim2_vol: npt.NDArray,
    rgb_vol: tuple,
    alpha_vol: npt.NDArray,
    data_vtx: npt.NDArray,
    dim2_vtx: npt.NDArray,
    rgb_vtx: tuple,
    alpha_vtx: npt.NDArray,
) -> Dataview:
    """Construct one of the six dataview classes from prepared channels.

    One dispatch shared by all three suites, so adding a dataview class is a
    single edit rather than three kept in lockstep.
    """
    cmap_plain, cmap_2d = "viridis", "RdBu_r_alpha"

    if name == "Volume":
        return cortex.Volume(data_vol, subj, xfmname, cmap=cmap_plain, vmin=-1, vmax=1)
    elif name == "Vertex":
        return cortex.Vertex(data_vtx, subj, cmap=cmap_plain, vmin=-1, vmax=1)
    elif name == "Volume2D":
        return cortex.Volume2D(
            data_vol, dim2_vol, subj, xfmname, cmap=cmap_2d,
            vmin=-1, vmax=1, vmin2=0, vmax2=1,
        )
    elif name == "Vertex2D":
        return cortex.Vertex2D(
            data_vtx, dim2_vtx, subj, cmap=cmap_2d,
            vmin=-1, vmax=1, vmin2=0, vmax2=1,
        )
    elif name == "VolumeRGB":
        red, green, blue = rgb_vol
        return cortex.VolumeRGB(
            cortex.Volume(red, subj, xfmname, vmin=0, vmax=1),
            cortex.Volume(green, subj, xfmname, vmin=0, vmax=1),
            cortex.Volume(blue, subj, xfmname, vmin=0, vmax=1),
            subj, xfmname,
            alpha=cortex.Volume(alpha_vol, subj, xfmname, vmin=0, vmax=1),
        )
    elif name == "VertexRGB":
        red, green, blue = rgb_vtx
        return cortex.VertexRGB(
            cortex.Vertex(red, subj, vmin=0, vmax=1),
            cortex.Vertex(green, subj, vmin=0, vmax=1),
            cortex.Vertex(blue, subj, vmin=0, vmax=1),
            subj,
            alpha=cortex.Vertex(alpha_vtx, subj, vmin=0, vmax=1),
        )
    else:
        raise ValueError(f"Unknown dataview: {name}")


def _build_alpha_dataview(name: str) -> Dataview:
    """Build a single alpha-bearing dataview by name, with no NaNs anywhere."""
    a = _synth_arrays()
    return _dataview(
        name,
        data_vol=a["data_vol"],
        dim2_vol=a["accuracy_vol"],
        rgb_vol=(a["red_vol"], a["green_vol"], a["blue_vol"]),
        alpha_vol=a["accuracy_vol"],
        data_vtx=a["data_vtx"],
        dim2_vtx=a["accuracy_vtx"],
        rgb_vtx=tuple(a["xyz_norm"][:, i] for i in range(3)),
        alpha_vtx=a["accuracy_vtx"],
    )


def _build_nan_dataview(name: str) -> Dataview:
    """Build a dataview with NaNs over roughly half of the primary data channel."""
    a = _synth_arrays()
    xx, yy, zz = a["xx"], a["yy"], a["zz"]

    # The rule, as gh-695 states it, is that a NaN *anywhere* at a voxel -- the
    # data, either 2D dimension, any RGB channel, or the alpha map -- renders
    # fully transparent. These references pin the behavior as it is on main.
    # Each of those gets NaN'd over its own region, so a single render exercises
    # several branches of the rule at once and a failure still says which one
    # moved. The vertex regions are disjoint; the volume ones (x>=50, y>=50,
    # z>=15) overlap, which is harmless and additionally covers voxels carrying
    # more than one NaN at once. Blue is deliberately left clean, as a control
    # that not everything has simply gone transparent.
    #
    # The alpha map is NaN'd here too, on a third axis. That is not a duplicate
    # of the nan_alpha suite: this covers alpha NaNs superposed on color NaNs,
    # where nan_alpha isolates them with every color channel clean.
    #
    # Expect the alpha map's surviving NaN fraction to look smaller than its mask.
    # Where a color channel is already NaN the pipeline writes alpha's vmin over
    # it, so of the z>=15 region only the quarter with red and green both clean
    # stays NaN; the rest is resolved to a hard 0. Both halves of that are worth
    # rendering, which is why the regions are allowed to overlap.
    def vol_nan(arr, mask):
        out = arr.copy()
        out[mask] = np.nan
        return out

    primary = xx >= 50      # data, and red for RGB
    secondary = yy >= 50    # 2D dimension 2, and green for RGB
    tertiary = zz >= 15     # the alpha map, on a third independent axis

    # As above, in disjoint index ranges rather than spatial ones.
    total = sum(a["num_verts"])
    idx = np.arange(total)
    vtx_primary = idx >= total // 2
    vtx_secondary = idx < total // 4
    vtx_tertiary = (idx >= total // 4) & (idx < total // 2)
    xyz = a["xyz_norm"]

    return _dataview(
        name,
        data_vol=vol_nan(a["data_vol"], primary),
        dim2_vol=vol_nan(a["accuracy_vol"], secondary),
        rgb_vol=(
            vol_nan(a["red_vol"], primary),
            vol_nan(a["green_vol"], secondary),
            a["blue_vol"],
        ),
        alpha_vol=vol_nan(a["accuracy_vol"], tertiary),
        data_vtx=vol_nan(a["data_vtx"], vtx_primary),
        dim2_vtx=vol_nan(a["accuracy_vtx"], vtx_secondary),
        rgb_vtx=(
            vol_nan(xyz[:, 0], vtx_primary),
            vol_nan(xyz[:, 1], vtx_secondary),
            xyz[:, 2],
        ),
        alpha_vtx=vol_nan(a["accuracy_vtx"], vtx_tertiary),
    )


def _build_nan_alpha_dataview(name: str) -> Dataview:
    """Build an RGB dataview whose *alpha map* carries NaNs, color channels clean.

    The other NaN suite puts NaNs in the data; this puts them in the alpha map,
    which is a separate code path -- alpha is not color-mapped, it is used
    directly as a blend weight, so a NaN reaches the compositing arithmetic
    rather than a colormap lookup.

    Current behavior is that those elements render fully transparent, i.e. the
    curvature underlay shows through, which is what the other NaN cases do too.
    """
    a = _synth_arrays()
    alpha_vol = a["accuracy_vol"].copy()
    alpha_vol[a["xx"] >= 50] = np.nan

    total = sum(a["num_verts"])
    alpha_vtx = a["accuracy_vtx"].copy()
    alpha_vtx[np.arange(total) >= total // 2] = np.nan

    return _dataview(
        name,
        data_vol=a["data_vol"],
        dim2_vol=a["accuracy_vol"],
        rgb_vol=(a["red_vol"], a["green_vol"], a["blue_vol"]),
        alpha_vol=alpha_vol,
        data_vtx=a["data_vtx"],
        dim2_vtx=a["accuracy_vtx"],
        rgb_vtx=tuple(a["xyz_norm"][:, i] for i in range(3)),
        alpha_vtx=alpha_vtx,
    )


def _assert_no_failures(failures: list[str], tmp_path: Path) -> None:
    """Fail with every mismatch at once, and say where to look at them."""
    assert not failures, (
        "Renders differ from expectations:\n  "
        + "\n  ".join(failures)
        + f"\n\nFor details, see {tmp_path} and {REFERENCE_ROOT}/README.md"
    )


def _render_and_check_dataview(
    name: str,
    view: Dataview,
    reference_dir: Path,
    tmp_path: Path,
) -> list[str]:
    """Render a single dataview via quickshow + webgl and check against reference.

    Each render is checked three ways: quickflat vs its own reference, webgl vs
    its own reference (both tight tolerances, see ``_check_against_reference``),
    and quickflat vs webgl directly (loose tolerance, see
    ``_check_cross_renderer``).

    Returns a list of failure messages (empty if no failures). Skips the test
    if reference images are missing, and regenerates them if ``REGENERATE_REFERENCES``
    is set.
    """
    import matplotlib.pyplot as plt

    # quickshow → low-res PNG
    qf_path = tmp_path / f"quickflat_{name}.png"
    qf_fig = cortex.quickshow(
        view,
        with_curvature=True,
        with_rois=False,
        with_labels=False,
        with_colorbar=False,
        with_sulci=False,
        with_borders=False,
        height=256,
        curvature_threshold=QUICKFLAT_CURVATURE_THRESHOLD,
    )
    qf_fig.savefig(qf_path, bbox_inches="tight", pad_inches=0, dpi=80)
    plt.close(qf_fig)

    # webgl → trimmed flatmap PNG via plot_panels (single flatmap panel).
    # Not save_3d_views, whose output is transparent where quickshow's figure is
    # opaque; going through matplotlib keeps both sides comparable for the
    # cross-renderer check.
    flatmap_panel = [
        {
            "extent": [0.0, 0.0, 1.0, 1.0],
            "view": {"angle": "flatmap", "surface": "flatmap"},
        }
    ]
    wg_path = tmp_path / f"webgl_{name}.png"
    wg_fig = cortex.export.plot_panels(
        view,
        panels=flatmap_panel,
        figsize=(6, 3),
        windowsize=(512, 384),
        save_name=str(wg_path),
        sleep=10,
        # ``curvature_smoothness`` is a named parameter of ``show()``, which
        # ``plot_panels`` forwards to. Dotted viewer-state names are *not* --
        # ``show()`` drops those into the HTML template unused -- so anything
        # without a declared keyword has to go through ``_set_view`` instead.
        viewer_params=dict(
            labels_visible=[], overlays_visible=[],
            curvature_smoothness=WEBGL_CURVATURE_SMOOTHNESS,
        ),
        headless=True,
    )
    plt.close(wg_fig)

    # Checked after rendering, not before, so that a dataview which cannot be
    # rendered at all fails here rather than skipping on the missing reference
    # its failure is the reason for -- see the xfail on Vertex2D.
    if not REGENERATE_REFERENCES:
        for prefix in ("quickflat", "webgl"):
            reason = _unusable_reference(
                reference_dir / f"{prefix}_{name}{REFERENCE_SUFFIX}"
            )
            if reason is not None:
                pytest.skip(reason)

    failures = []

    # _check_against_reference rewrites the reference and returns None when
    # regenerating, so this collects nothing on that path.
    for prefix, path in [("quickflat", qf_path), ("webgl", wg_path)]:
        msg = _check_against_reference(f"{prefix}_{name}", path, tmp_path, reference_dir)
        if msg is not None:
            failures.append(msg)

    if REGENERATE_REFERENCES:
        pytest.skip(f"Regenerated {name} references in {reference_dir}")

    # Cross-renderer check (never regenerates, always compares)
    msg = _check_cross_renderer(name, qf_path, wg_path, tmp_path)
    if msg is not None:
        failures.append(msg)

    return failures


def _render_and_check_webgl_only(
    tag: str,
    view: Dataview,
    surface: str,
    angle: str,
    reference_dir: Path,
    tmp_path: Path,
) -> list[str]:
    """Render one non-flatmap view through webgl and check it against a reference.

    A cut-down ``_render_and_check_dataview``: there is no quickflat render to
    compare against, because ``cortex.quickshow`` produces flatmaps and nothing
    else, so both the quickflat reference check and the cross-renderer check are
    absent by necessity rather than by choice.

    It therefore calls ``save_3d_views`` directly rather than ``plot_panels``,
    which would compose the screenshot into a matplotlib figure and store an
    upsampled interpolation of the render instead of the render. The flatmap
    path pays that cost to keep both renderers comparable; without a second
    renderer there is nothing to buy with it.

    Curvature is left at pycortex's default (thresholded) here, unlike the
    flatmap suites. Those un-threshold it to reduce cross-renderer
    disagreement -- a reason that does not apply when there is no second
    renderer -- so using the default recovers coverage of the default curvature
    path, which the flatmap references explicitly do not provide.
    """
    from cortex.export.save_views import save_3d_views

    wg_path = save_3d_views(
        view,
        base_name=str(tmp_path / f"webgl_{tag}"),
        list_angles=[angle],
        list_surfaces=[surface],
        trim=True,
        size=(512, 384),
        sleep=10,
        viewer_params=dict(labels_visible=[], overlays_visible=[]),
        headless=True,
    )[0]

    # After the render, as in _render_and_check_dataview.
    if not REGENERATE_REFERENCES:
        reason = _unusable_reference(
            reference_dir / f"webgl_{tag}{REFERENCE_SUFFIX}"
        )
        if reason is not None:
            pytest.skip(reason)

    msg = _check_against_reference(
        f"webgl_{tag}", Path(wg_path), tmp_path, reference_dir
    )
    if REGENERATE_REFERENCES:
        pytest.skip(f"Regenerated webgl_{tag} in {reference_dir}")
    return [msg] if msg is not None else []


@pytest.mark.parametrize("name", DATAVIEW_NAMES)
def test_visual_comparison_alpha_dataviews(tmp_path, name):
    """Render an alpha-bearing dataview via quickshow + webgl, and assert it matches.

    Plain Volume / Vertex have no native per-element alpha (pycortex's
    bundled ``*_alpha`` colormaps are all 2D and only apply to the 2D
    dataview types), so those two act as a no-alpha baseline. The other four
    exercise alpha: Volume2D / Vertex2D via the 2D-alpha cmap ``RdBu_r_alpha``,
    VolumeRGB / VertexRGB via the ``alpha=`` kwarg.

    Compared both within-renderer (against a stored reference) and
    cross-renderer (quickflat vs webgl); see ``_render_and_check_dataview``. A
    mismatch leaves ``actual_*.png`` and an amplified ``diff_*.png`` in the
    test's ``tmp_path``.
    """
    view = _build_alpha_dataview(name)
    failures = _render_and_check_dataview(name, view, REFERENCE_DIR, tmp_path)
    _assert_no_failures(failures, tmp_path)


@pytest.mark.parametrize("name", DATAVIEW_NAMES)
def test_visual_comparison_nan_dataviews(tmp_path, name):
    """Render a NaN-bearing dataview via quickshow + webgl, and assert it matches.

    NaN is pycortex's convention for "no data at this voxel/vertex" -- both
    renderers are expected to draw those elements as fully transparent (falling
    through to the curvature underlay) rather than mapping NaN through the
    colormap as if it were a real value. This test renders the six dataview
    classes with the *primary* data channel (not alpha) containing NaNs over
    roughly half of each volume/surface.

    Compared both within-renderer (against a stored reference) and
    cross-renderer (quickflat vs webgl); see ``_render_and_check_dataview``. A
    mismatch leaves ``actual_*.png`` and an amplified ``diff_*.png`` in the
    test's ``tmp_path``.
    """
    view = _build_nan_dataview(name)
    failures = _render_and_check_dataview(name, view, NAN_REFERENCE_DIR, tmp_path)
    _assert_no_failures(failures, tmp_path)


@pytest.mark.parametrize("name", NAN_ALPHA_DATAVIEW_NAMES)
def test_visual_comparison_nan_alpha_dataviews(tmp_path, name):
    """Render an RGB dataview whose alpha map carries NaNs, and assert it matches.

    The other NaN suite puts NaNs in the data channels; this one puts them in the
    alpha map. That is a distinct path -- alpha is not color-mapped, it is used
    directly as a blend weight, so the NaN lands in the compositing arithmetic
    rather than in a colormap lookup. Only ``VolumeRGB``/``VertexRGB`` take an
    explicit ``alpha=``, so only those two are covered.

    Current behavior, which these references encode, is that NaN-alpha elements
    render fully transparent and the curvature underlay shows through -- the same
    outcome as a NaN in the data.

    Be aware that this behavior is not settled. gh-695, which unifies NaN and
    alpha handling across quickflat, WebGL and the RGB dataviews, changes how the
    surviving RGB is blended without changing the transparency itself. If that lands, expect these four references to need
    regenerating; the transparency assertion should survive, the exact blend will
    not.
    """
    view = _build_nan_alpha_dataview(name)
    failures = _render_and_check_dataview(name, view, NAN_ALPHA_REFERENCE_DIR, tmp_path)
    _assert_no_failures(failures, tmp_path)


@pytest.mark.parametrize("surface,angle,name", NONFLAT_VIEWS)
def test_visual_comparison_nonflat_views(tmp_path, surface, angle, name):
    """Render a non-flatmap view through webgl and assert it matches its reference.

    Everything else in this file renders the flatmap. That leaves the 3D views
    covered only by test_webgl_headless.py's smoke tests, which assert the file
    is larger than 1000 bytes -- enough to catch a render that never happened,
    not one that came out wrong.

    webgl only, necessarily: quickshow renders flatmaps and nothing else, so
    these views have no quickflat counterpart and therefore no cross-renderer
    check. Volume and Vertex both appear because they take different shader
    paths, and the flatmap is demonstrably not representative of how those
    behave -- both known webgl lighting bugs reproduce on flatmaps only.
    """
    view = _build_alpha_dataview(name)
    tag = f"{surface}_{angle}_{name}"
    failures = _render_and_check_webgl_only(
        tag, view, surface, angle, NONFLAT_REFERENCE_DIR, tmp_path
    )
    _assert_no_failures(failures, tmp_path)
