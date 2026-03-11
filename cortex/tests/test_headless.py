"""Smoke tests for the headless viewer (cortex.export.headless).

These tests require ``playwright`` and Chromium to be installed::

    pip install playwright
    playwright install chromium

They also require an active pycortex filestore with at least the S1 subject.
"""
import os
import tempfile

import numpy as np
import pytest

import cortex
import cortex.export

subj, xfmname, volshape = "S1", "fullhead", (31, 100, 100)

# Skip the entire module if playwright or Chromium is not available.
try:
    from playwright.sync_api import sync_playwright

    _pw = sync_playwright().start()
    try:
        _b = _pw.chromium.launch(headless=True, args=["--no-sandbox"])
        _b.close()
    finally:
        _pw.stop()
    _has_playwright = True
except Exception:
    _has_playwright = False

pytestmark = pytest.mark.skipif(
    not _has_playwright,
    reason="playwright + Chromium not available",
)


def test_headless_viewer_opens_and_closes():
    """The headless viewer context manager should yield a working handle and
    tear down cleanly."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        # The handle should have a .server attribute (the WebApp)
        assert hasattr(handle, "server")
        # The server should be serving on some port
        assert handle.server.port > 0


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
