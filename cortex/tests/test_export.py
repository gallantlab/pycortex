"""Tests for webviewer export helpers that require the headless viewer.

These tests require ``playwright`` and Chromium to be installed::

    pip install playwright
    playwright install chromium
"""
import os
import tempfile

import numpy as np
import pytest

import cortex

from .testing_utils import has_playwright

pytestmark = pytest.mark.skipif(
    not has_playwright,
    reason="playwright + Chromium not available",
)


subj, xfmname, volshape = "S1", "fullhead", (31, 100, 100)


def test_save_3d_views_headless():
    """save_3d_views with headless=True should produce an image file."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, "test_img")
        file_names = cortex.export.save_3d_views(
            vol,
            base_name=base,
            list_angles=["lateral_pivot"],
            list_surfaces=["inflated"],
            size=(1024, 768),
            trim=False,
            # The WebGL scene needs time to initialise surfaces before
            # _set_view can succeed; sleep=10 (the default) is safe.
            sleep=10,
            headless=True,
        )

        assert len(file_names) == 1
        assert os.path.isfile(file_names[0])
        assert os.path.getsize(file_names[0]) > 0

        # Check that the file is a valid image and has the expected dimensions.
        from PIL import Image
        with Image.open(file_names[0]) as img:
            assert img.size == (1024, 768)


def test_plot_panels_headless():
    """plot_panels with headless=True should produce an output image file."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    panels = [
        cortex.export.PanelParams({
            "extent": (0.0, 0.0, 1.0, 1.0),
            "view": cortex.export.PanelView(angle="lateral_pivot", surface="inflated"),
        })
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_name = os.path.join(tmpdir, "panels.png")
        fig = cortex.export.plot_panels(
            vol,
            panels=panels,
            figsize=(8, 6),
            windowsize=(1024, 768),
            save_name=save_name,
            sleep=10,
            viewer_params={},
            headless=True,
        )

        # The function returns a matplotlib Figure and, when save_name is
        # provided, should have written the file to disk.
        assert fig is not None
        assert os.path.isfile(save_name)
        assert os.path.getsize(save_name) > 0