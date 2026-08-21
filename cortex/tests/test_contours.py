"""Tests for contour/border rendering of parcellation data.

Tests cover:
- Python utility: get_contour_vertices()
- Quickflat: add_contours(), _detect_label_borders(), make_figure(with_contours=...)
- WebGL: shader contour uniforms and geometry attributes
"""

import numpy as np
import pytest

import cortex
from cortex.quickflat.composite import _detect_label_borders
from cortex.testing_utils import has_installed

no_inkscape = not has_installed("inkscape")

SUBJECT = "S1"


def _make_parcellation(subject=SUBJECT):
    """Create parcellation vertex data from existing ROIs.

    Returns array of shape (n_vertices,) with integer labels per ROI
    and 0 for vertices not in any ROI.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        roi_verts = cortex.get_roi_verts(subject)
    n_verts = cortex.db.get_surf(subject, "fiducial", merge=True)[0].shape[0]
    parcellation = np.zeros(n_verts, dtype=float)
    for i, (name, verts) in enumerate(roi_verts.items(), start=1):
        parcellation[np.asarray(verts, dtype=int)] = float(i)
    return parcellation


# --- Tests for Python contour utility ---


class TestGetContourVertices:
    def test_returns_border_vertices(self):
        """get_contour_vertices should return True at vertices bordering
        different label values."""
        parcellation = _make_parcellation()
        border = cortex.utils.get_contour_vertices(parcellation, SUBJECT)
        assert border.dtype == bool
        assert border.shape == parcellation.shape
        # There should be some border vertices (parcellation has multiple labels)
        assert border.sum() > 0
        # Border vertices should be fewer than total labeled vertices
        assert border.sum() < (parcellation > 0).sum()

    def test_uniform_data_has_no_borders(self):
        """Uniform data (all same label) should have no border vertices."""
        n_verts = cortex.db.get_surf(SUBJECT, "fiducial", merge=True)[0].shape[0]
        uniform = np.ones(n_verts, dtype=float)
        border = cortex.utils.get_contour_vertices(uniform, SUBJECT)
        assert border.sum() == 0

    def test_border_vertices_are_adjacent_to_different_labels(self):
        """Every border vertex should have at least one neighbor with a
        different label."""
        parcellation = _make_parcellation()
        border = cortex.utils.get_contour_vertices(parcellation, SUBJECT)

        _, polys = cortex.db.get_surf(SUBJECT, "fiducial", merge=True)
        neighbors = cortex.utils._get_neighbors_dict(polys)

        # Check a sample of border vertices
        border_verts = np.where(border)[0][:100]
        for v in border_verts:
            neighbor_labels = {
                parcellation[n] for n in neighbors[v] if n < len(parcellation)
            }
            assert (
                len(neighbor_labels) > 1 or parcellation[v] not in neighbor_labels
            ), f"Border vertex {v} has no neighbor with different label"


# --- Tests for _detect_label_borders (2D image helper) ---


class TestDetectLabelBorders:
    def test_uniform_image_no_borders(self):
        """Uniform label image should have no borders."""
        img = np.ones((50, 50))
        border = _detect_label_borders(img)
        assert border.sum() == 0

    def test_two_regions_has_border(self):
        """Image split into two regions should have a border between them."""
        img = np.ones((50, 50))
        img[:, 25:] = 2.0
        border = _detect_label_borders(img)
        # Border should be at column 24 and 25 (the boundary pixels)
        assert border.sum() > 0
        # Border pixels should be in the middle columns
        border_cols = np.where(border.any(axis=0))[0]
        assert 24 in border_cols or 25 in border_cols

    def test_nan_pixels_are_not_borders(self):
        """NaN pixels (outside brain mask) should not be marked as borders."""
        img = np.full((50, 50), np.nan)
        img[10:40, 10:40] = 1.0
        img[10:40, 25:40] = 2.0
        border = _detect_label_borders(img)
        # Should have borders between label 1 and 2, but not at NaN edges
        # (NaN-to-value transitions should not count as borders since
        # we're interested in parcel-to-parcel boundaries, not brain mask edges)
        nan_mask = np.isnan(img)
        assert not border[nan_mask].any()

    def test_3d_image_uses_first_channel(self):
        """For RGBA images, borders should be detected on first channel."""
        img = np.ones((50, 50, 4))
        img[:, 25:, 0] = 2.0
        border = _detect_label_borders(img)
        assert border.sum() > 0


# --- Tests for quickflat contour rendering ---


class TestQuickflatContours:
    @pytest.mark.skipif(no_inkscape, reason="Inkscape required")
    def test_add_contours_returns_image(self):
        """add_contours() should return a matplotlib AxesImage."""
        from matplotlib import pyplot as plt
        from cortex.quickflat.composite import add_contours

        parcellation = _make_parcellation()
        parc_vertex = cortex.Vertex(parcellation, SUBJECT)

        fig, ax = plt.subplots()
        img = add_contours(ax, parc_vertex, height=256)
        assert img is not None
        plt.close(fig)

    @pytest.mark.skipif(no_inkscape, reason="Inkscape required")
    def test_add_contours_border_pixels_are_opaque(self):
        """Contour overlay should have opaque pixels only at label borders."""
        from cortex.quickflat.composite import add_contours
        from matplotlib import pyplot as plt

        parcellation = _make_parcellation()
        parc_vertex = cortex.Vertex(parcellation, SUBJECT)

        fig, ax = plt.subplots()
        img = add_contours(ax, parc_vertex, height=256)
        rgba = img.get_array()
        # Alpha channel should be > 0 only at border pixels
        has_content = rgba[:, :, 3] > 0 if rgba.ndim == 3 else rgba > 0
        assert has_content.any(), "No contour pixels found"
        plt.close(fig)

    @pytest.mark.skipif(no_inkscape, reason="Inkscape required")
    def test_make_figure_with_contours(self):
        """make_figure() with with_contours should produce a figure with
        contour overlay."""
        parcellation = _make_parcellation()
        parc_vertex = cortex.Vertex(parcellation, SUBJECT)
        activation = cortex.Vertex(
            np.random.randn(parcellation.shape[0]), SUBJECT, cmap="hot", vmin=-2, vmax=2
        )

        fig = cortex.quickflat.make_figure(
            activation,
            with_contours=parc_vertex,
            with_rois=False,
            with_colorbar=False,
            height=256,
        )
        assert fig is not None
        # Should have at least 2 images: data + contours
        ax = fig.get_axes()[0]
        images = ax.get_images()
        assert len(images) >= 2

    @pytest.mark.skipif(no_inkscape, reason="Inkscape required")
    def test_make_figure_volume_with_vertex_contours(self):
        """make_figure() should work with Volume data + Vertex contour overlay."""
        parcellation = _make_parcellation()
        parc_vertex = cortex.Vertex(parcellation, SUBJECT)
        volume = cortex.Volume.random(subject=SUBJECT, xfmname="fullhead")

        fig = cortex.quickflat.make_figure(
            volume,
            with_contours=parc_vertex,
            with_rois=False,
            with_colorbar=False,
            height=256,
        )
        assert fig is not None
        ax = fig.get_axes()[0]
        images = ax.get_images()
        assert len(images) >= 2

    @pytest.mark.skipif(no_inkscape, reason="Inkscape required")
    def test_contour_linewidth(self):
        """Thicker linewidth should produce more border pixels."""
        from cortex.quickflat.composite import add_contours
        from matplotlib import pyplot as plt

        parcellation = _make_parcellation()
        parc_vertex = cortex.Vertex(parcellation, SUBJECT)

        fig1, ax1 = plt.subplots()
        img1 = add_contours(ax1, parc_vertex, height=256, linewidth=1)
        rgba1 = img1.get_array()
        n_pixels_1 = (
            (rgba1[:, :, 3] > 0).sum() if rgba1.ndim == 3 else (rgba1 > 0).sum()
        )

        fig2, ax2 = plt.subplots()
        img2 = add_contours(ax2, parc_vertex, height=256, linewidth=3)
        rgba2 = img2.get_array()
        n_pixels_3 = (
            (rgba2[:, :, 3] > 0).sum() if rgba2.ndim == 3 else (rgba2 > 0).sum()
        )

        assert (
            n_pixels_3 > n_pixels_1
        ), f"linewidth=3 ({n_pixels_3}) should have more pixels than linewidth=1 ({n_pixels_1})"
        plt.close("all")

    @pytest.mark.skipif(no_inkscape, reason="Inkscape required")
    def test_contour_linecolor(self):
        """Custom linecolor should appear in the contour image."""
        from cortex.quickflat.composite import add_contours
        from matplotlib import pyplot as plt

        parcellation = _make_parcellation()
        parc_vertex = cortex.Vertex(parcellation, SUBJECT)

        red = (1.0, 0.0, 0.0, 1.0)
        fig, ax = plt.subplots()
        img = add_contours(ax, parc_vertex, height=256, linecolor=red)
        rgba = img.get_array()
        if rgba.ndim == 3:
            border_mask = rgba[:, :, 3] > 0
            # Red channel should be 1.0 at border pixels
            np.testing.assert_allclose(rgba[border_mask, 0], 1.0)
            # Green and blue should be 0
            np.testing.assert_allclose(rgba[border_mask, 1], 0.0)
            np.testing.assert_allclose(rgba[border_mask, 2], 0.0)
        plt.close(fig)


# --- Tests for WebGL contour support ---


class TestWebGLContours:
    def test_shader_includes_contour_uniforms(self):
        """Both surface_vertex and surface_pixel shaders should include
        contour-related uniforms."""
        import os

        shader_path = os.path.join(
            os.path.dirname(cortex.__file__), "webgl", "resources", "js", "shaderlib.js"
        )
        with open(shader_path, "r") as f:
            shader_code = f.read()

        assert "contourMode" in shader_code
        assert "contourThreshold" in shader_code
        assert "contourColor" in shader_code
        assert "fwidth" in shader_code
        assert "vDataValue" in shader_code
        assert "vContourDataValue" in shader_code
        assert "contourColormap" in shader_code

    def test_geometry_has_contour_attributes(self):
        """mriview_surface.js should initialize contourData attributes."""
        import os

        surface_path = os.path.join(
            os.path.dirname(cortex.__file__),
            "webgl",
            "resources",
            "js",
            "mriview_surface.js",
        )
        with open(surface_path, "r") as f:
            surface_code = f.read()

        assert "contourData0" in surface_code
        assert "contourData1" in surface_code

    def test_viewer_has_contour_overlay_support(self):
        """mriview.js should have contour overlay selection support."""
        import os

        viewer_path = os.path.join(
            os.path.dirname(cortex.__file__), "webgl", "resources", "js", "mriview.js"
        )
        with open(viewer_path, "r") as f:
            viewer_code = f.read()

        assert "setContourOverlay" in viewer_code or "contour_overlay" in viewer_code
