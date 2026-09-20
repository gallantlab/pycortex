"""Tests for the surface opacity slider (gallantlab/pycortex#353).

Most of this module needs playwright + Chromium (see ``has_playwright`` in
``testing_utils``) since it drives the real webgl viewer headlessly. The
config-default check does not.
"""
import numpy as np
import pytest

import cortex

from .testing_utils import has_playwright

subj = "S1"


def test_surface_opacity_default_is_one():
    """The shipped default must be fully opaque so existing renders are
    unaffected by this feature (see cortex/defaults.cfg)."""
    assert (
        cortex.options.config.getfloat("webgl_viewopts", "surface_opacity") == 1.0
    )


@pytest.mark.skipif(
    not has_playwright, reason="playwright + Chromium not available"
)
def test_surface_opacity_renders_translucent(tmp_path):
    """A translucent surface_opacity should visibly differ from the opaque
    default, with brain-silhouette pixels shifting toward the background
    color as the surface goes translucent."""
    from PIL import Image

    from cortex.export.save_views import save_3d_views, unfold_view_params

    surfs = [
        cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subj, "fiducial")
    ]
    nverts = sum(s.pts.shape[0] for s in surfs)
    view = cortex.Vertex(np.zeros(nverts), subj)

    def _render(opacity, name):
        surface_params = {
            **unfold_view_params["inflated"],
            "surface.{subject}.surface_opacity": opacity,
        }
        path = save_3d_views(
            view,
            base_name=str(tmp_path / name),
            list_angles=["lateral_pivot"],
            list_surfaces=[surface_params],
            size=(640, 480),
            trim=False,
            viewer_params=dict(labels_visible=[], overlays_visible=[]),
            headless=True,
        )[0]
        return np.asarray(Image.open(path).convert("RGBA")).astype(np.int16)

    opaque = _render(1.0, "opaque")
    translucent = _render(0.25, "translucent")

    assert opaque.shape == translucent.shape

    # The two renders should differ substantially -- not just anti-aliasing
    # noise -- once the surface goes translucent.
    diff = np.abs(opaque.astype(np.int32) - translucent.astype(np.int32))
    assert diff.mean() > 1.0, (
        "translucent (surface_opacity=0.25) render is nearly identical to "
        "the opaque one; the slider may not be wired up"
    )

    # Corners are outside the inflated brain's silhouette in this view, so
    # sample one as the background color, and take the brain silhouette to
    # be every pixel of the opaque render that differs from it.
    bg_color = opaque[0, 0, :3].astype(np.int32)
    brain = np.any(opaque[..., :3].astype(np.int32) != bg_color, axis=-1)
    assert brain.mean() > 0.05, "opaque render shows (almost) no brain"

    dist_opaque_to_bg = np.abs(opaque[brain, :3].astype(np.int32) - bg_color).sum(-1).mean()
    dist_translucent_to_bg = (
        np.abs(translucent[brain, :3].astype(np.int32) - bg_color).sum(-1).mean()
    )

    assert dist_translucent_to_bg < dist_opaque_to_bg, (
        "translucent brain pixels did not move toward the background color "
        f"(opaque->bg={dist_opaque_to_bg:.1f}, "
        f"translucent->bg={dist_translucent_to_bg:.1f}, bg={bg_color.tolist()})"
    )
