"""Visual regression tests: quickflat and webgl renders vs stored references.

Five suites. Three of them render flatmaps of the six public dataview classes
(``Volume``, ``Vertex``, ``Volume2D``, ``Vertex2D``, ``VolumeRGB``,
``VertexRGB``) through both matplotlib (``cortex.quickflat.make_png``) and the
headless WebGL viewer (``save_3d_views``), varying what the data carries:
alpha-bearing values, NaNs in the data channels, and NaNs in the alpha map. The
last covers the four classes taking an explicit ``alpha=``. Every one of those
renders is checked twice -- against its own stored reference at a tight
tolerance, and directly against the other renderer's render of the same
dataview at a loose one.

The fourth renders the three volumetric classes with NaNs arranged to fall
between depth samples, with both renderers averaging over the same number of
them, at both values of the ``nanmean`` setting. The other suites leave depth
sampling alone and NaN whole columns at a time, so nothing in them depends on
how a column of samples is combined.

The fifth renders non-flatmap views, ``Volume`` and ``Vertex`` on the
inflated and fiducial surfaces, through ``save_3d_views``. Those are
webgl-only and get the reference check alone: quickflat produces flatmaps and
nothing else, so there is nothing to diff them against.

Every render is transparent outside the flatmap, so the two renderers are
directly comparable without compositing or a coordinate correction.

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

#: The render tests need a browser. The builder checks at the bottom of this
#: file do not, and they are the ones that catch a suite masking away the very
#: thing it meant to render, so this is applied per test rather than as a
#: module-level ``pytestmark`` -- a checkout without playwright still runs them.
requires_playwright = pytest.mark.skipif(
    not has_playwright, reason="playwright and chromium are required"
)

DATAVIEW_NAMES = [
    "Volume",
    "Vertex",
    "Volume2D",
    "Vertex2D",
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
#: This was the two RGB classes until gh-695. Volume2D/Vertex2D took an
#: ``alpha=`` as well, but it was stashed as a raw ndarray in ``attrs`` instead
#: of becoming an attribute, so quickflat-Volume2D painted its NaNs opaque,
#: quickflat-Vertex2D ignored the map outright, and the webgl viewer failed to
#: load the dataview at all. It is now a real attribute, multiplied into the
#: colormap's own alpha on both renderers, so all four classes belong here.
NAN_ALPHA_DATAVIEW_NAMES = ["Volume2D", "Vertex2D", "VolumeRGB", "VertexRGB"]

#: As NAN_REFERENCE_DIR, but with the NaNs arranged to fall between depth
#: samples rather than covering a whole column, so what is rendered depends on
#: how the two renderers average across the cortical thickness.
MULTILAYER_REFERENCE_DIR = REFERENCE_ROOT / "multilayer_nan_dataviews"

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

#: Lossless WebP: bit-exact after decode and appreciably smaller than optimized
#: PNG.
REFERENCE_SUFFIX = ".webp"

#: First bytes of a git-lfs pointer. The reference images are LFS-tracked, so
#: if LFS hasn't been properly initialized, these 130-byte text stubs exist in
#: place of the images.
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

# The two limits above are both weak against a change that moves a *small*
# number of pixels by a *large* amount, which is what a geometry or contour
# shift looks like: the mean is diluted by the pixels that did not move. So two
# further criteria, each covering what the others miss:
#   - mean and fraction>16 catch broad, low-amplitude shifts, which the gross
#     fraction misses entirely.
#   - fraction>32 catches sparse, high-amplitude ones. 32 rather than 64 because
#     gh-695's premultiplied-alpha change scores 0% at 64 -- the suite would miss
#     it -- and its mean sits below the cosmetic floor, so no tighter mean helps.
#   - SSIM catches structural change, but is computed on luminance and so is
#     blind to a channel permutation. It adds sensitivity alongside the others;
#     it cannot replace them.
#
# The limits sit well above measured cosmetic drift rather than at it, since a
# real toolchain bump changes anti-aliasing and does show up in fraction>32.
GROSS_DIFF_THRESHOLD = 32             # a pixel differs "grossly" if any channel moves by more
MAX_FRACTION_GROSSLY_DIFFERING = 0.001
MAX_SSIM_LOSS = 0.01                  # 1 - mean SSIM over the luminance channel

# Cross-renderer tolerances: quickflat vs webgl for the *same* dataview, rather
# than each against its own reference. Looser than the within-renderer ones
# above, since the two renderers genuinely differ in anti-aliasing and colormap
# sampling. Re-measure after any change to either renderer's output size.
CROSS_MAX_MEAN_ABS_DIFF = 3.5
CROSS_DIFF_THRESHOLD = 32  # same as GROSS_DIFF_THRESHOLD.
CROSS_MAX_FRACTION_DIFFERING = 0.032

# Render settings chosen to minimize cross-renderer disagreement, from a
# factorial sweep of both renderers' settings. Only curvature thresholding was
# worth changing from the defaults: a thresholded curvature puts a hard binary
# edge at curvature=0 that each rasterizer resolves differently, where smooth
# curvature is low-frequency and resamples cleanly. On curvature-only content
# it costs 1.8x on the mean -- passing, now that the renders are the same size,
# but eating most of the headroom for no benefit. quickflat's
# ``curvature_threshold`` and webgl's ``curvature.smoothness`` are the same knob
# from opposite ends -- smoothness 0.0 *is* thresholded -- so both have to move
# together.
#
# Lighting, sampler, depth and thick/layers all stayed at their defaults; none
# moved the disagreement measurably on a flatmap, whose normals face the camera.
#
# NB this is not pycortex's default appearance, so these references do not cover
# the default curvature path. That is the trade for a tighter floor;
# nonflat_views/ keeps the default and recovers the coverage.
#: Height of the quickflat render. make_png scales the width to the subject's
#: flatmap aspect, giving roughly 490x256.
QUICKFLAT_HEIGHT = 256

#: Browser canvas for every webgl render. After trimming, lands at roughly
# quickflat's 490x256 (from QUICKFLAT_HEIGHT).
WEBGL_CANVAS = (925, 695)

#: Depth samples each renderer averages over, at their own defaults. They do
#: not match: quickflat has always averaged 32 samples across the cortical
#: thickness, the webgl viewer one. That asymmetry is pre-existing and the
#: cross-renderer tolerance absorbs it on smooth data; the multilayer suite
#: below is the one that sets them equal, because it is about exactly this.
QUICKFLAT_THICK = 32
WEBGL_LAYERS = 1

#: Depth samples for the multilayer suite, applied to both renderers.
MULTILAYER_DEPTHS = 32

#: Why the multilayer suite's cross-renderer leg is allowed to breach. Setting
#: both renderers to the same number of depth samples does not put those samples
#: at the same depths: quickflat takes ``linspace(0, 1, thick + 2)[1:-1]``, the
#: shader ``i / (layers - 1)``, so one grid is interior and the other reaches
#: both the white-matter and pial surfaces. On smooth data that is invisible --
#: every other cross check here passes at mean|diff| 1.3-1.9 -- but this suite's
#: NaN slabs are two voxels thick, at the sampling limit, so a sub-sample offset
#: in depth flips which samples are NaN and the two renders disagree pixel by
#: pixel. The disagreement is pixel-scale phase noise, not a difference in what
#: is drawn: the signed bias is within +-2 of 255, transparency agrees to the
#: same 1.19% outline as the passing suites, and a sigma=4 blur brings mean|diff|
#: back to 1.3 against a 1.18 floor. The reference legs stay strict; only this
#: one is conceded. See gh-749.
MULTILAYER_CROSS_XFAIL_REASON = (
    "quickflat and webgl sample cortical depth on different grids "
    "(gh-749), which this suite's two-voxel NaN slabs resolve differently "
    "per pixel"
)

#: Classes the multilayer suite covers. Volumetric only: vertex dataviews hold
#: one value per vertex, so every depth sample at a vertex is the same number
#: and there is nothing for the across-depth averaging to do. All three are
#: here because each takes a different branch of the sampling shader -- scalar,
#: 2D, and RGB, the last combining premultiplied RGBA rather than values.
MULTILAYER_DATAVIEW_NAMES = ["Volume", "Volume2D", "VolumeRGB"]

#: The volumetric classes, which are the ones whose renders have any depth to
#: sample. Mirrors the test in ``save_3d_views``, which only forwards the
#: sampler and layer settings for these.
VOLUMETRIC = (cortex.Volume, cortex.Volume2D, cortex.VolumeRGB)

# Don't threshold curvature to reduce cross-renderer disagreement due to
# anti-aliasing implementations.
QUICKFLAT_CURVATURE_THRESHOLD = False
WEBGL_CURVATURE_SMOOTHNESS = 1.0


def _normalize_transparent(rgba: npt.NDArray) -> npt.NDArray:
    """Zero the RGB of fully transparent pixels, which is undefined there."""
    out = rgba.astype(np.int16).copy()
    out[out[..., 3] == 0, :3] = 0
    return out


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
    the same content at a slightly different trim/crop.

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
    the two renders produced by *this* test run against each other. Both are
    content-tight and transparent outside the flatmap, so webgl is resized to
    quickflat's size and diffed as RGBA -- no coordinate correction is needed.

    RGB under fully transparent pixels is normalized first. It is undefined
    there, and the two writers disagree: matplotlib leaves white, the browser
    leaves black. Alpha itself stays in the comparison, so a render that
    lost its transparency is still caught.
    """
    from PIL import Image

    qf_im = Image.open(quickflat_path).convert("RGBA")
    wg_im = Image.open(webgl_path).convert("RGBA")
    shape_note = f" (resized: quickflat was {qf_im.size}, webgl was {wg_im.size})"

    qf = _normalize_transparent(np.asarray(qf_im))
    wg = _normalize_transparent(np.asarray(wg_im.resize(qf_im.size, Image.BILINEAR)))

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
    twod_alpha: bool = False,
) -> Dataview:
    """Construct one of the six dataview classes from prepared channels.

    One dispatch shared by every suite, so adding a dataview class is a single
    edit rather than several kept in lockstep.

    ``twod_alpha`` decides whether Volume2D/Vertex2D are also handed the
    ``alpha_*`` map as an explicit ``alpha=``, on top of the alpha their 2D
    colormap already carries. Off by default, so that a suite covers one alpha
    source at a time: with it off the 2D classes exercise the colormap's alpha
    alone, with it on they exercise the product of the two, which is what
    gh-695 made both renderers compute.
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
            **(dict(alpha=alpha_vol) if twod_alpha else {}),
        )
    elif name == "Vertex2D":
        return cortex.Vertex2D(
            data_vtx, dim2_vtx, subj, cmap=cmap_2d,
            vmin=-1, vmax=1, vmin2=0, vmax2=1,
            **(dict(alpha=alpha_vtx) if twod_alpha else {}),
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
    # moved. Both sets of regions are three overlapping halves on independent
    # axes, which covers elements carrying more than one NaN at once and, more
    # importantly, leaves a clean remainder: the union is about 7/8, so an
    # eighth of the data survives every mask. Blue is left clean as well, as a
    # control that not everything has simply gone transparent -- but the
    # remainder is what makes that control meaningful, since a channel being
    # clean does nothing for an element the other masks have already hit.
    #
    # The alpha map is NaN'd here too, on a third axis. That is not a duplicate
    # of the nan_alpha suite: this covers alpha NaNs superposed on color NaNs,
    # where nan_alpha isolates them with every color channel clean.
    #
    # Expect the alpha map's surviving NaN fraction to look smaller than its
    # mask: where a color channel is already NaN the pipeline writes alpha's
    # vmin over it. Both halves of that are worth rendering, which is why the
    # regions are allowed to overlap.
    def vol_nan(arr, mask):
        out = arr.copy()
        out[mask] = np.nan
        return out

    primary = xx >= 50      # data, and red for RGB
    secondary = yy >= 50    # 2D dimension 2, and green for RGB
    tertiary = zz >= 15     # the alpha map, on a third independent axis

    # As above, on the vertex coordinates. These were three disjoint index
    # ranges -- idx >= total/2, idx < total/4, and the quarter between -- which
    # tile the surface exactly and so left no clean remainder at all: every
    # vertex carried a NaN in some channel, and the two classes using all three
    # (Vertex2D, and VertexRGB via red/green/alpha) rendered as bare curvature.
    # Their references pinned an empty flatmap and covered nothing. Splitting on
    # coordinates instead mirrors the volume regions above and leaves 12% of the
    # surface clean; the median rather than 0.5 keeps each mask at half exactly.
    #
    # The z comparison runs the other way on purpose. Leaving a remainder is not
    # enough on its own -- it also has to be somewhere the data is visible, and
    # for Vertex2D visibility is governed by the alpha the 2D colormap derives
    # from dim2, which is the accuracy bump. Of the eight orientations, >= on
    # every axis is the worst: its survivors carry mean accuracy 0.23, only 3.5%
    # of them above 0.5, and Vertex2D renders nearly blank (1.2% of its opaque
    # pixels colored, against 9.9% for Volume2D). Flipping z puts the remainder
    # on the bump -- mean accuracy 0.62, 70% above 0.5 -- for the same 12%.
    xyz = a["xyz_norm"]
    vtx_mid = np.median(xyz, axis=0)
    vtx_primary = xyz[:, 0] >= vtx_mid[0]     # data, and red for RGB
    vtx_secondary = xyz[:, 1] >= vtx_mid[1]   # 2D dimension 2, and green for RGB
    vtx_tertiary = xyz[:, 2] <= vtx_mid[2]    # the alpha map

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
        # The rule above names the alpha map as one of the places a NaN makes
        # an element transparent, and it is only reachable for the 2D classes
        # since gh-695 turned their ``alpha=`` into a real attribute.
        twod_alpha=True,
    )


def _build_nan_alpha_dataview(name: str) -> Dataview:
    """Build a dataview whose *alpha map* carries NaNs, every data channel clean.

    The other NaN suite puts NaNs in the data; this puts them in the alpha map,
    which is a separate code path -- alpha is not color-mapped, it is used
    directly as a blend weight, so a NaN reaches the compositing arithmetic
    rather than a colormap lookup.

    For Volume2D/Vertex2D the map is passed as an explicit ``alpha=`` and has
    to survive being multiplied into the alpha their 2D colormap already
    supplies, which is a second path again.

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
        twod_alpha=True,
    )


def _build_multilayer_nan_dataview(name: str) -> Dataview:
    """Build a volumetric dataview whose NaNs alternate across cortical depth.

    The other NaN suites NaN out broad regions, so a surface point is either
    NaN at every depth or at none, and how the renderers combine depth samples
    never comes into it. This one NaNs diagonal slabs two voxels thick, which
    is fine enough that nearly every surface point has both NaN and valid
    samples under it, whatever the local orientation of the ribbon. Diagonal so
    that the ribbon is nowhere parallel to the slabs for long.

    That is the case gh-695 changed. The webgl viewer summed its layer samples,
    so a single NaN anywhere in the column hid the whole fragment; it now
    averages the valid layers and is transparent only where none is valid,
    which is what quickflat's ``nanmean`` already did. Rendered at
    ``nanmean=True`` the data should be broadly visible, at ``nanmean=False``
    broadly transparent, and the two renderers should agree either way.
    """
    a = _synth_arrays()
    slab = ((((a["xx"] + a["yy"] + a["zz"]) // 2) % 2) == 1)

    def vol_nan(arr):
        out = arr.copy()
        out[slab] = np.nan
        return out

    return _dataview(
        name,
        data_vol=vol_nan(a["data_vol"]),
        dim2_vol=vol_nan(a["accuracy_vol"]),
        rgb_vol=(
            vol_nan(a["red_vol"]),
            a["green_vol"],
            a["blue_vol"],
        ),
        alpha_vol=a["accuracy_vol"],
        # Unused: every class in MULTILAYER_DATAVIEW_NAMES is volumetric.
        data_vtx=a["data_vtx"],
        dim2_vtx=a["accuracy_vtx"],
        rgb_vtx=tuple(a["xyz_norm"][:, i] for i in range(3)),
        alpha_vtx=a["accuracy_vtx"],
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
    *,
    thick: int = QUICKFLAT_THICK,
    layers: int = WEBGL_LAYERS,
    nanmean: Optional[bool] = None,
    cross_xfail_reason: Optional[str] = None,
) -> list[str]:
    """Render a single dataview through both renderers and check it.

    Each render is checked three ways: quickflat vs its own reference, webgl vs
    its own reference (both tight tolerances, see ``_check_against_reference``),
    and quickflat vs webgl directly (loose tolerance, see
    ``_check_cross_renderer``).

    ``thick`` and ``layers`` are the two renderers' depth-sampling counts, and
    ``nanmean`` whether either skips NaN samples when averaging over them.
    ``nanmean=None`` leaves both renderers at their own default rather than
    setting it, so the suites that predate gh-695 render exactly as before.

    ``cross_xfail_reason``, if given, xfails the test when the cross-renderer
    leg breaches rather than failing it, for a caller whose content the two
    renderers are not expected to agree on pixel for pixel. The two reference
    legs stay strict either way: a reference mismatch is asserted before the
    xfail is conceded, so a real regression is not swallowed by it. A caller
    that passes it and then agrees anyway simply passes -- this is the
    imperative ``pytest.xfail``, not a mark, so there is no xpass to configure.

    Returns a list of failure messages (empty if no failures). Skips the test
    if reference images are missing, and regenerates them if ``REGENERATE_REFERENCES``
    is set.
    """
    from cortex.export.save_views import angle_view_params, save_3d_views

    flatmap_angle = (
        "flatmap",
        {
            **angle_view_params["flatmap"],
            "surface.{subject}.curvature.smoothness": WEBGL_CURVATURE_SMOOTHNESS,
            # save_3d_views drops these for vertex dataviews, which have no
            # depth to sample; setting them there would hang its wait loop.
            **(
                {"surface.{subject}.nanmean": nanmean}
                if nanmean is not None and isinstance(view, VOLUMETRIC)
                else {}
            ),
        },
    )

    # quickflat -> transparent PNG via make_png.
    qf_path = tmp_path / f"quickflat_{name}.png"
    cortex.quickflat.make_png(
        str(qf_path),
        view,
        height=QUICKFLAT_HEIGHT,
        with_curvature=True,
        with_rois=False,
        with_labels=False,
        with_colorbar=False,
        with_sulci=False,
        with_borders=False,
        curvature_threshold=QUICKFLAT_CURVATURE_THRESHOLD,
        thick=thick,
        **({} if nanmean is None else dict(nanmean=nanmean)),
    )

    # webgl -> trimmed flatmap screenshot.
    wg_path = Path(
        save_3d_views(
            view,
            base_name=str(tmp_path / f"webgl_{name}"),
            list_angles=[flatmap_angle],
            list_surfaces=["flatmap"],
            layers=layers,
            trim=True,
            size=WEBGL_CANVAS,
            sleep=10,
            viewer_params=dict(labels_visible=[], overlays_visible=[]),
            headless=True,
        )[0]
    )

    # Fails rather than skips: wholesale absence -- an installed wheel, or a
    # clone that has not fetched LFS -- is caught at import, so a single gap in
    # a populated store means an incomplete regeneration.
    if not REGENERATE_REFERENCES:
        for prefix in ("quickflat", "webgl"):
            reason = _unusable_reference(
                reference_dir / f"{prefix}_{name}{REFERENCE_SUFFIX}"
            )
            if reason is not None:
                pytest.fail(reason)

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
        if cross_xfail_reason is not None:
            # Only the cross-renderer leg is expected to breach. A reference
            # mismatch is a real regression whatever this leg does, so surface
            # those first rather than letting the xfail swallow them.
            _assert_no_failures(failures, tmp_path)
            pytest.xfail(f"{cross_xfail_reason}\n  {msg}")
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

    A cut-down ``_render_and_check_dataview``: no cross-renderer check because
    we don't use quickflat.

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
        size=WEBGL_CANVAS,
        sleep=10,
        viewer_params=dict(labels_visible=[], overlays_visible=[]),
        headless=True,
    )[0]

    # Fails rather than skips, as in _render_and_check_dataview.
    if not REGENERATE_REFERENCES:
        reason = _unusable_reference(
            reference_dir / f"webgl_{tag}{REFERENCE_SUFFIX}"
        )
        if reason is not None:
            pytest.fail(reason)

    msg = _check_against_reference(
        f"webgl_{tag}", Path(wg_path), tmp_path, reference_dir
    )
    if REGENERATE_REFERENCES:
        pytest.skip(f"Regenerated webgl_{tag} in {reference_dir}")
    return [msg] if msg is not None else []


@requires_playwright
@pytest.mark.parametrize("name", DATAVIEW_NAMES)
def test_visual_comparison_alpha_dataviews(tmp_path, name):
    """Render an alpha-bearing dataview through both renderers, and assert it matches.

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


@requires_playwright
@pytest.mark.parametrize("name", DATAVIEW_NAMES)
def test_visual_comparison_nan_dataviews(tmp_path, name):
    """Render a NaN-bearing dataview through both renderers, and assert it matches.

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


@requires_playwright
@pytest.mark.parametrize("name", NAN_ALPHA_DATAVIEW_NAMES)
def test_visual_comparison_nan_alpha_dataviews(tmp_path, name):
    """Render an RGB dataview whose alpha map carries NaNs, and assert it matches.

    The other NaN suite puts NaNs in the data channels; this one puts them in the
    alpha map. That is a distinct path -- alpha is not color-mapped, it is used
    directly as a blend weight, so the NaN lands in the compositing arithmetic
    rather than in a colormap lookup. All four classes that take an explicit
    ``alpha=`` are covered. That was ``VolumeRGB``/``VertexRGB`` alone until
    gh-695: ``Volume2D``/``Vertex2D`` accepted the argument but dropped it into
    ``attrs`` as a bare ndarray, which quickflat then mishandled and the webgl
    viewer choked on. For those two the map also has to survive being
    multiplied into the alpha their 2D colormap already carries.

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


@requires_playwright
@pytest.mark.parametrize("name", MULTILAYER_DATAVIEW_NAMES)
@pytest.mark.parametrize("nanmean", [True, False], ids=["nanmean", "no_nanmean"])
@pytest.mark.timeout(600)
def test_visual_comparison_multilayer_nan_dataviews(tmp_path, name, nanmean):
    """Render NaNs that fall between depth samples, with both renderers averaging.

    The other suites leave depth sampling alone, and at their defaults the two
    renderers do not even agree on how many samples to take (quickflat 32, the
    viewer 1), so nothing there covers how a column of samples containing NaNs
    is combined. This sets both to ``MULTILAYER_DEPTHS`` and NaNs the data
    finely enough that most surface points have both NaN and valid samples --
    see ``_build_multilayer_nan_dataview``.

    Parametrized over the ``nanmean`` setting itself, since the point is that
    the two renderers agree on *both* of its values: with it on the valid
    samples are averaged and the data shows, with it off any NaN in the column
    hides the fragment. Before gh-695 the viewer had no such setting and always
    behaved as if it were off, while quickflat defaulted to on.

    Given a longer timeout than the suite default: 32 layers under software
    rendering is appreciably slower than the single-layer renders elsewhere.

    The cross-renderer leg is xfailed here, and only here -- see
    ``MULTILAYER_CROSS_XFAIL_REASON`` for the measurements and gh-749 for the
    underlying difference. In short, this suite sets both renderers to the same
    number of depth samples but cannot put those samples at the same depths, and
    its two-voxel NaN slabs are fine enough to resolve that offset. Five of the
    six parameter sets breach; the sixth stays inside the tolerance and simply
    passes. What the suite is actually for is unaffected: both renderers still
    have to match their own references exactly, and both respond to ``nanmean``
    the same way (toggling it moves quickflat by mean 10.9 and webgl by 11.2 on
    Volume).

    The Volume2D case is flaky, through no fault of this suite. Setting
    ``layers`` above 1 intermittently leaves the viewer's RPC proxy answering
    ``{}`` to every subsequent query, so ``save_3d_views`` dies on the next
    parameter it sets. It never recovers, waiting it out does not help, and the
    browser reports no error at all. It reproduces on main at about the same
    rate (roughly one run in four) with the dataview built inline rather than
    here, so it is a pre-existing bug in the viewer or in JSProxy, not
    something gh-695 introduced -- this is simply the first test to drive
    ``layers`` above 1 on a 2D dataview. Left in rather than skipped, because
    the coverage is wanted and the flake is worth fixing on its own.
    """
    view = _build_multilayer_nan_dataview(name)
    tag = f"{name}_{'nanmean' if nanmean else 'no_nanmean'}"
    failures = _render_and_check_dataview(
        tag,
        view,
        MULTILAYER_REFERENCE_DIR,
        tmp_path,
        thick=MULTILAYER_DEPTHS,
        layers=MULTILAYER_DEPTHS,
        nanmean=nanmean,
        cross_xfail_reason=MULTILAYER_CROSS_XFAIL_REASON,
    )
    _assert_no_failures(failures, tmp_path)


@requires_playwright
@pytest.mark.parametrize("surface,angle,name", NONFLAT_VIEWS)
def test_visual_comparison_nonflat_views(tmp_path, surface, angle, name):
    """Render a non-flatmap view through webgl and assert it matches its reference.

    The other tests render flatmaps, which in webgl can have different behavior
    from 3D views (for example, lighting).

    This is webgl only (quickflat only renders flatmaps), so no cross-renderer
    check. Test both Volume and Vertex because they take different shader
    paths.
    """
    view = _build_alpha_dataview(name)
    tag = f"{surface}_{angle}_{name}"
    failures = _render_and_check_webgl_only(
        tag, view, surface, angle, NONFLAT_REFERENCE_DIR, tmp_path
    )
    _assert_no_failures(failures, tmp_path)


#: Least fraction of elements a NaN suite may leave untouched by every one of
#: its masks. The suites sit far above it -- the thinnest is 12% -- so this is
#: not a tuned threshold, it is a floor under the one way these builders fail
#: silently. The vertex masks did cover the whole surface once: every element
#: carried a NaN somewhere, Vertex2D and VertexRGB rendered as bare curvature,
#: and their references pinned an empty flatmap. Nothing caught it, because an
#: empty render matches an empty reference exactly, and agrees with the other
#: renderer's empty render exactly as well.
MIN_CLEAN_FRACTION = 0.05

#: (suite, class) pairs, one per dataview each NaN suite actually renders.
NAN_BUILDER_CASES = (
    [("nan", name) for name in DATAVIEW_NAMES]
    + [("nan_alpha", name) for name in NAN_ALPHA_DATAVIEW_NAMES]
    + [("multilayer", name) for name in MULTILAYER_DATAVIEW_NAMES]
)


def _clean_fraction(view: Dataview) -> float:
    """Fraction of a dataview's elements carrying no NaN in any channel.

    Reads whichever of the channel attributes the class actually has, so it
    covers all six without knowing which is which: a scalar class has ``data``,
    the 2D ones ``dim1``/``dim2``, the RGB ones ``red``/``green``/``blue``, and
    any of them may carry ``alpha``. Squeezed because VolumeRGB keeps its alpha
    with a leading axis the color channels do not have.

    This is the quantity that decides whether anything is drawn at all: the rule
    under test is that a NaN anywhere at an element renders it transparent, so
    an element is visible only if every channel is finite there.
    """
    channels = []
    for attr in ("data", "dim1", "dim2", "red", "green", "blue", "alpha"):
        channel = getattr(view, attr, None)
        if channel is None:
            continue
        values = np.asarray(getattr(channel, "data", channel), dtype=float)
        channels.append(np.isfinite(np.squeeze(values)).ravel())
    return float(np.logical_and.reduce(channels).mean())


@pytest.mark.parametrize("suite,name", NAN_BUILDER_CASES)
def test_nan_builders_leave_clean_elements(suite, name):
    """Assert each NaN suite leaves elements no mask touched.

    A suite whose masks between them cover everything renders nothing, and
    nothing is exactly what the rest of this file cannot see -- the reference
    check passes against a blank reference and the cross-renderer check passes
    comparing one blank flatmap to another. So it is asserted on the dataviews
    directly, before any of it is rendered.

    Needs no browser, unlike every other test here, which is the point: this is
    the check that says the suites are testing something, and it should not be
    contingent on a working playwright install.
    """
    builders = {
        "nan": _build_nan_dataview,
        "nan_alpha": _build_nan_alpha_dataview,
        "multilayer": _build_multilayer_nan_dataview,
    }
    clean = _clean_fraction(builders[suite](name))
    assert clean >= MIN_CLEAN_FRACTION, (
        f"{suite}/{name}: only {clean:.2%} of elements are free of NaNs, under "
        f"the {MIN_CLEAN_FRACTION:.0%} floor. Its masks cover nearly everything "
        "between them, so this renders as bare curvature and the reference "
        "pinned from it would assert nothing."
    )
