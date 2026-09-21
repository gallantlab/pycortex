"""Tests for the surface opacity slider (gallantlab/pycortex#353).

Most of this module needs playwright + Chromium (see ``has_playwright`` in
``testing_utils``) since it drives the real webgl viewer headlessly. The
config-default check does not.
"""
import configparser
import os.path

import numpy as np
import pytest

import cortex

from .testing_utils import has_playwright

subj = "S1"


def test_surface_opacity_default_is_one():
    """The shipped default must be fully opaque so existing renders are
    unaffected by this feature.

    Read ``cortex/defaults.cfg`` through its own parser rather than
    ``cortex.options.config``, which has already overlaid the user's
    ``options.cfg`` (``cortex/options.py``) -- a contributor who set their
    own ``surface_opacity`` would otherwise fail this.
    """
    path = os.path.join(os.path.dirname(cortex.__file__), "defaults.cfg")
    defaults = configparser.ConfigParser()
    assert defaults.read(path), f"could not read {path}"
    assert defaults.getfloat("webgl_viewopts", "surface_opacity") == 1.0


@pytest.mark.skipif(
    not has_playwright, reason="playwright + Chromium not available"
)
def test_surface_opacity_renders_translucent(tmp_path):
    """A translucent surface_opacity must reach the rendered fragment alpha
    without darkening the surface color along with it.

    The saved PNG carries straight (non-premultiplied) alpha, so a
    ``surface_opacity`` of 0.25 has to show up as alpha 0.25 over the brain
    while the RGB stays the color the opaque render gives. Folding the
    opacity into RGB as well would make whatever composites the image fade
    it a second time.
    """
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
        return np.asarray(Image.open(path).convert("RGBA")).astype(np.int32)

    opacity = 0.25
    opaque = _render(1.0, "opaque")
    translucent = _render(opacity, "translucent")

    assert opaque.shape == translucent.shape

    # The opaque render is the historical one: the render target has no
    # multisampling, so every pixel is either fully drawn or fully empty.
    assert set(np.unique(opaque[..., 3]).tolist()) <= {0, 255}
    brain = opaque[..., 3] == 255
    assert brain.mean() > 0.05, "opaque render shows (almost) no brain"

    # Most of the silhouette is a single layer of surface, and there the
    # fragment alpha must be exactly the slider value. (Where the surface
    # folds over itself the layers composite to a higher alpha, which is
    # what makes the far side show through.)
    single = np.abs(translucent[..., 3] - opacity * 255) <= 1
    assert not (single & ~brain).any(), "translucent render leaked outside the brain"
    assert single.sum() > 0.5 * brain.sum(), (
        "surface_opacity did not reach the rendered alpha; the slider may not "
        f"be wired up (alphas seen: {np.unique(translucent[..., 3]).tolist()})"
    )

    # Straight, not premultiplied: those pixels keep the opaque color.
    rgb_error = np.abs(translucent[single][:, :3] - opaque[single][:, :3]).max()
    assert rgb_error <= 8, (
        "translucent surface color drifted from the opaque one by "
        f"{rgb_error}/255; the exported image looks premultiplied, so it "
        "will be faded twice once composited"
    )


@pytest.mark.skipif(
    not has_playwright, reason="playwright + Chromium not available"
)
def test_translucent_surface_keeps_labels_on_top():
    """ROI labels must keep drawing after the surface once it turns translucent.

    Labels are ``depthTest: false`` -- they do their own occlusion against the
    depth texture ``SVGOverlay.prerender`` bakes -- so they are only correct
    while they are drawn last. An opaque surface gave that for free, since
    three.js r69 renders the whole opaque list before the transparent one. Any
    ``surface_opacity`` below 1 moves the surface into the transparent list,
    where it sorted against the labels by projected centre depth; both sit at
    the origin, so the tie fell to object id and the surface painted over
    them. Nudging the slider from 1 to 0.99 -- far too small a change to see
    on the surface itself -- wiped the labels off the brain.

    The bundled S1 overlay carries no label text, so there is nothing to count
    in a screenshot; assert the ordering invariant on the live objects
    instead. r69 sorts the transparent list ascending by ``renderDepth``
    (falling back to the projected z, which clipping keeps inside [-1, 1]) and
    then walks it backwards, so a label needs a ``renderDepth`` well below -1
    to stay on top of the surface.
    """
    surfs = [
        cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subj, "fiducial")
    ]
    nverts = sum(s.pts.shape[0] for s in surfs)
    view = cortex.Vertex(np.zeros(nverts), subj)

    with cortex.export.headless_viewer(view, viewer_params={}) as handle:
        handle._set_view(**{"surface.{subject}.surface_opacity": 0.5})

        svg = handle.surfs[0].surf.svg
        checked = 0
        for layer in ("rois", "sulci"):
            meshes = getattr(svg, layer).labels.meshes
            for hemi in ("left", "right"):
                depth = getattr(meshes, hemi).renderDepth
                # An unpinned renderDepth is null, which JSProxy hands back as
                # a proxy object rather than a number -- hence the type check
                # instead of a bare comparison.
                assert isinstance(depth, (int, float)), (
                    f"{layer}/{hemi} labels have no pinned renderDepth, so a "
                    "translucent surface can sort in front of them"
                )
                assert depth < -1, (
                    f"{layer}/{hemi} labels sort at {depth}, which a surface "
                    "can beat once surface_opacity drops below 1"
                )
                checked += 1
        assert checked == 4
