"""Smoke tests for the headless viewer (cortex.export.headless).

These tests require ``playwright`` and Chromium to be installed::

    pip install playwright
    playwright install chromium
"""
import os
import tempfile

import numpy as np
import pytest

import cortex
from cortex.export.headless import _PlaywrightThread

from .testing_utils import has_playwright, wait_for_file

pytestmark = pytest.mark.skipif(
    not has_playwright,
    reason="playwright + Chromium not available",
)


subj, xfmname, volshape = "S1", "fullhead", (31, 100, 100)


def test_headless_viewer_opens_and_closes():
    """The headless viewer context manager should yield a working handle and
    tear down cleanly."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        # The handle should have a .server attribute (the WebApp)
        assert hasattr(handle, "server")
        # The server should be serving on some port
        assert handle.server.port > 0



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


def test_headless_viewer_with_running_asyncio_loop():
    """headless_viewer must work when an asyncio event loop is already running,
    as is the case inside Jupyter notebooks.

    Before the _PlaywrightThread fix, sync_playwright() would raise:
        Error: It looks like you are using Playwright Sync API inside the
        asyncio loop.  Please use the Async API instead.
    """
    import asyncio

    async def _inner():
        # Inside this coroutine the event loop is running — same as Jupyter.
        vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
            assert hasattr(handle, "server")
            assert handle.server.port > 0

    asyncio.run(_inner())


def test_headless_viewer_in_notebook():
    """Execute headless_viewer inside a real Jupyter kernel via nbclient.

    This is the most faithful reproduction of the original bug: an IPython
    kernel has a running asyncio event loop, and all the Jupyter-specific
    machinery (display hooks, IOPub, etc.) is active.

    Skipped when ``nbclient`` or ``nbformat`` are not installed.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell(
            "import numpy as np\n"
            "import cortex\n"
            "import cortex.export\n"
            f"subj, xfmname, volshape = {subj!r}, {xfmname!r}, {volshape!r}\n"
        ),
        nbformat.v4.new_code_cell(
            "vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)\n"
            "with cortex.export.headless_viewer(vol, viewer_params={}) as handle:\n"
            "    assert hasattr(handle, 'server')\n"
            "    assert handle.server.port > 0\n"
            "print('headless_viewer OK')\n"
        ),
    ]

    client = nbclient.NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
    )
    client.execute()

    # Verify the last cell ran without error and printed "OK".
    last_outputs = nb.cells[-1].outputs
    assert any(
        "headless_viewer OK" in out.get("text", "")
        for out in last_outputs
        if out["output_type"] == "stream"
    ), f"Notebook cell did not produce expected output. Outputs: {last_outputs}"


def test_filter_webgl_failures_keeps_only_real_failures():
    """Only genuine WebGL failures match; ordinary console noise does not.

    The filter has to stay narrow: a healthy viewer always logs a console.error
    for the Leap Motion websocket it cannot reach.
    """
    from cortex.export.headless import filter_webgl_failures

    noise = [
        "[console.error] WebSocket connection to 'ws://127.0.0.1:6437/v6.json' "
        "failed: Error in connection establishment: net::ERR_CONNECTION_REFUSED",
        "[console.warning] THREE.WebGLShader: gl.getShaderInfoLog() WARNING: 0:87",
        "[console.warning] [.WebGL-0x21b400157a00]GL Driver Message (OpenGL, Perf)",
        # Emitted alongside a link failure but meaningless alone: nothing calls
        # gl.validateProgram(), so VALIDATE_STATUS is false for want of a run.
        "[console.error] gl.VALIDATE_STATUS false",
        "[console.error] gl.getError() 0",
    ]
    assert filter_webgl_failures(noise) == []

    link_failure = "[console.error] THREE.WebGLProgram: Could not initialise shader."
    context_failure = "[pageerror] Error creating WebGL context."
    assert filter_webgl_failures(noise + [link_failure]) == [link_failure]
    assert filter_webgl_failures(noise + [context_failure]) == [context_failure]


def test_browser_errors_are_delivered_before_teardown():
    """Console messages must arrive while the viewer is alive, not at teardown.

    Playwright's sync API dispatches queued events only while something is
    calling into it, so without the polling loop in ``_PlaywrightThread._run``
    nothing reaches ``browser_errors`` until ``_cleanup()`` -- long after
    ``save_3d_views`` reads it. Rendering therefore has to grow the count.

    Note the growth comes from console noise the viewer emits anyway
    (leapmotion websocket errors), not from the render. This test fails once
    leapmotion is removed, and it would blame the polling loop. A fix would be
    to emit a known message from the page, making it self-contained.
    """
    import time

    from cortex.export.headless import EVENT_POLL_INTERVAL

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        after_load = len(handle._pw_thread.browser_errors)
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "poll.png")
            handle.getImage(outfile, (512, 384))
            wait_for_file(outfile)
        # Poll rather than sleep a fixed interval; a few cycles is enough.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            during_render = len(handle._pw_thread.browser_errors)
            if during_render > after_load:
                break
            time.sleep(EVENT_POLL_INTERVAL)
        else:
            during_render = len(handle._pw_thread.browser_errors)

    assert during_render > after_load, (
        "browser_errors did not grow while the viewer was alive (%d -> %d); "
        "queued console events are not being dispatched"
        % (after_load, during_render)
    )
