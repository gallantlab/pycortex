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
from cortex.export.headless import _PlaywrightThread

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


def test_browser_errors_collected():
    """_PlaywrightThread should capture console.error and pageerror from the
    browser and make them available via browser_errors."""
    pw = _PlaywrightThread()
    # Start a trivial HTTP server that serves a page triggering JS errors.
    import http.server
    import threading as _threading

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            # Emit a console.error and an uncaught exception.
            self.wfile.write(b"""<html><body><script>
                console.error("test-console-error-message");
                throw new Error("test-uncaught-exception");
            </script></body></html>""")

        def log_message(self, *args, **kwargs):
            pass  # suppress request logging

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = _threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        pw.start(f"http://127.0.0.1:{port}/", timeout=15)
        # Give Playwright listeners a moment to fire.
        import time
        time.sleep(1)
        errors = pw.browser_errors
        assert any("test-console-error-message" in e for e in errors), (
            f"Expected console.error to be captured, got: {errors}"
        )
        assert any("test-uncaught-exception" in e for e in errors), (
            f"Expected pageerror to be captured, got: {errors}"
        )
    finally:
        pw.shutdown()
        server.shutdown()


def test_browser_errors_on_handle():
    """The headless_viewer handle should expose browser errors via
    handle._pw_thread.browser_errors."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        assert hasattr(handle, "_pw_thread")
        # browser_errors should return a list (possibly empty if no errors).
        errors = handle._pw_thread.browser_errors
        assert isinstance(errors, list)
