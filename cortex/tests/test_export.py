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

from .testing_utils import has_playwright, wait_for_file

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
    calling into it, so without the pump in ``_PlaywrightThread._run`` nothing
    reaches ``browser_errors`` until ``_cleanup()`` -- long after
    ``save_3d_views`` reads it. Rendering therefore has to grow the count.
    """
    import time

    from cortex.export.headless import EVENT_PUMP_INTERVAL

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        after_load = len(handle._pw_thread.browser_errors)
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, "pump.png")
            handle.getImage(outfile, (512, 384))
            wait_for_file(outfile)
        # Poll rather than sleep a fixed interval; a few pump cycles is enough.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            during_render = len(handle._pw_thread.browser_errors)
            if during_render > after_load:
                break
            time.sleep(EVENT_PUMP_INTERVAL)
        else:
            during_render = len(handle._pw_thread.browser_errors)

    assert during_render > after_load, (
        "browser_errors did not grow while the viewer was alive (%d -> %d); "
        "queued console events are not being dispatched"
        % (after_load, during_render)
    )


def test_save_3d_views_raises_on_webgl_failure(monkeypatch):
    """A reported WebGL failure aborts the render rather than writing a blank png.

    Injected rather than provoked, so it does not depend on a broken shader in
    the tree under test.
    """
    monkeypatch.setattr(
        "cortex.export.headless.filter_webgl_failures",
        lambda errors: ["[console.error] THREE.WebGLProgram: Could not initialise shader."],
    )
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="WebGL failed while rendering"):
            cortex.export.save_3d_views(
                vol,
                base_name=os.path.join(tmpdir, "boom"),
                list_angles=["lateral_pivot"],
                list_surfaces=["inflated"],
                size=(512, 384),
                trim=False,
                sleep=10,
                headless=True,
            )
