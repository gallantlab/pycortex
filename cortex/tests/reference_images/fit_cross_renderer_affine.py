"""Re-derive the cross-renderer affine transform used by test_visual_regression.py.

quickflat and webgl do not agree on a common pixel grid: each computes its own
trim/extent, so webgl's flatmap lands in quickflat's frame off by a fixed
anisotropic scale plus a translation. ``_check_cross_renderer`` corrects for
that before diffing, using the ``CROSS_RENDERER_*`` constants in
``cortex/tests/test_visual_regression.py``. This script recomputes them.

Run it if the cross-renderer check starts failing broadly with a mean|diff|
around 5-9 rather than 1-2, which is what losing the correction looks like --
e.g. after a change to ``plot_panels``' figure composition, to quickflat's
``height=``/``dpi=``, or to either renderer's trim logic. (The ~19 this script
reports for its own uncorrected baseline is a different number: curvature-only
content, with no opaque data covering the sulcal texture.)

    uv run --with opencv-python-headless \
        python cortex/tests/reference_images/fit_cross_renderer_affine.py

OpenCV is only needed for the fit, so it is deliberately not a project
dependency -- hence ``--with``.

Method: render curvature-only content (a VolumeRGB with alpha=0 everywhere, so
the data layer is fully transparent and only the curvature underlay is drawn)
through both renderers. Curvature is the one layer both paths composite
identically, so the fit measures the coordinate-frame mismatch rather than any
dataview-specific colormap difference. Then fit webgl -> quickflat with
``cv2.findTransformECC``, first as a full homography to confirm there is no real
projective component, then as an affine to read off the constants.

Note on conventions: ``cv2.warpAffine(..., WARP_INVERSE_MAP)`` and PIL's
``Image.AFFINE`` use the same inverse mapping (output pixel -> source coords),
so the fitted matrix rows drop straight into PIL's coefficient tuple in
``_check_cross_renderer`` with no inversion or transpose.
"""

import numpy as np

import cortex
import cortex.export

SUBJ = "S1"
XFMNAME = "fullhead"

# Must match what test_visual_regression.py's _render_and_check_dataview does,
# or the fitted transform will not apply to the renders the test produces.
#
# The curvature settings are the exception: the test renders un-thresholded
# (curvature_threshold=False, curvature.smoothness=1.0), this renders pycortex's
# default. That does shift the fit -- matching them gives 0.9668/0.9335 and a
# ~7.27px translation rather than 0.9690/0.9365 and ~6.90px -- but not in a way
# that matters: across the twelve stored pairs the two constant sets score
# mean|diff| 1.65 and 1.70 on average. Left as-is so the committed constants
# stay reproducible from this script unmodified.
QUICKFLAT_HEIGHT = 256
QUICKFLAT_DPI = 80
WEBGL_FIGSIZE = (6, 3)
WEBGL_WINDOWSIZE = (512, 384)
WEBGL_SLEEP = 10

FLATMAP_PANEL = [
    {"extent": [0.0, 0.0, 1.0, 1.0], "view": {"angle": "flatmap", "surface": "flatmap"}}
]


def _curvature_only_dataview():
    """A dataview whose data layer is fully transparent, leaving only curvature."""
    zeros = np.zeros((31, 100, 100))
    chan = lambda: cortex.Volume(zeros, SUBJ, XFMNAME, vmin=0, vmax=1)
    return cortex.VolumeRGB(
        chan(), chan(), chan(), SUBJ, XFMNAME, alpha=chan(),
    )


def main():
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image

    view = _curvature_only_dataview()

    qf_path = "fit_affine_quickflat.png"
    fig = cortex.quickshow(
        view, with_curvature=True, with_rois=False, with_labels=False,
        with_colorbar=False, with_sulci=False, with_borders=False,
        height=QUICKFLAT_HEIGHT,
    )
    fig.savefig(qf_path, bbox_inches="tight", pad_inches=0, dpi=QUICKFLAT_DPI)
    plt.close(fig)

    wg_path = "fit_affine_webgl.png"
    fig = cortex.export.plot_panels(
        view, panels=FLATMAP_PANEL, figsize=WEBGL_FIGSIZE,
        windowsize=WEBGL_WINDOWSIZE, save_name=wg_path, sleep=WEBGL_SLEEP,
        viewer_params=dict(labels_visible=[], overlays_visible=[]),
        headless=True,
    )
    plt.close(fig)

    qf_im = Image.open(qf_path).convert("L")
    wg_im = Image.open(wg_path).convert("L")
    print(f"quickflat {qf_im.size}, webgl {wg_im.size}")

    # Fit in quickflat's frame, which is what _check_cross_renderer diffs in.
    qf = np.asarray(qf_im).astype(np.float32) / 255.0
    wg = np.asarray(wg_im.resize(qf_im.size, Image.BILINEAR)).astype(np.float32) / 255.0
    h, w = qf.shape
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

    print(f"\nmean|diff| with a plain resize and no correction: "
          f"{np.abs(qf - wg).mean() * 255:.3f}")

    # Homography first, purely as a check that there is nothing projective to
    # undo. webgl uses a THREE.PerspectiveCamera, but the flatmap view looks
    # straight down at a planar surface, where a pinhole projection degenerates
    # to a uniform scale -- so the bottom row should come out as [0, 0, 1].
    hom = np.eye(3, dtype=np.float32)
    _, hom = cv2.findTransformECC(qf, wg, hom, cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
    print(f"\nhomography projective row: [{hom[2, 0]:.3e}, {hom[2, 1]:.3e}, {hom[2, 2]:.3f}]")
    print("  near [0, 0, 1] => no real perspective distortion; affine is sufficient")

    aff = np.eye(2, 3, dtype=np.float32)
    cc, aff = cv2.findTransformECC(qf, wg, aff, cv2.MOTION_AFFINE, criteria, None, 5)
    aligned = cv2.warpAffine(
        wg, aff, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    mask = aligned != 0
    print(f"\naffine fit: correlation={cc:.4f}")
    print(f"mean|diff| after correction: {np.abs(qf[mask] - aligned[mask]).mean() * 255:.3f}")

    (a, b, tx), (c, d, ty) = aff[0], aff[1]
    rotation = np.degrees(np.arctan2(c, a))
    print(f"rotation={rotation:.4f} deg (expect ~0), shear terms b={b:.2e} c={c:.2e}")

    print("\nPaste into cortex/tests/test_visual_regression.py:\n")
    print(f"CROSS_RENDERER_SCALE_X = {np.hypot(a, c):.4f}")
    print(f"CROSS_RENDERER_SCALE_Y = {np.hypot(b, d):.4f}")
    print(f"CROSS_RENDERER_TRANSLATE_X_FRAC = {tx:.4f} / {w}")
    print(f"CROSS_RENDERER_TRANSLATE_Y_FRAC = {ty:.4f} / {h}")

    amp = np.clip(np.abs(qf - aligned) * 255 * 4, 0, 255).astype("uint8")
    Image.fromarray(amp).save("fit_affine_residual.png")
    print("\nresidual (4x amplified) written to fit_affine_residual.png -- expect a")
    print("faint sulcal outline only; broad structure means the fit did not take")


if __name__ == "__main__":
    main()
