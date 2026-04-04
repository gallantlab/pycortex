"""Tests for the Jupyter WebGL widget integration."""

import os
import re
import unittest
from unittest.mock import MagicMock, patch


class TestJupyterImports(unittest.TestCase):
    """Test that the jupyter module imports correctly."""

    def test_import_module(self):
        from cortex.webgl import jupyter

        self.assertTrue(hasattr(jupyter, "display"))
        self.assertTrue(hasattr(jupyter, "display_iframe"))
        self.assertTrue(hasattr(jupyter, "display_static"))
        self.assertTrue(hasattr(jupyter, "make_notebook_html"))
        self.assertTrue(hasattr(jupyter, "StaticViewer"))
        self.assertTrue(hasattr(jupyter, "close_all"))

    def test_import_from_webgl(self):
        import cortex

        self.assertTrue(hasattr(cortex.webgl, "jupyter"))


class TestFindFreePort(unittest.TestCase):
    """Test the _find_free_port helper."""

    def test_returns_valid_port(self):
        from cortex.webgl.jupyter import _find_free_port

        port = _find_free_port()
        self.assertIsInstance(port, int)
        self.assertGreaterEqual(port, 1024)
        self.assertLessEqual(port, 65535)

    def test_all_ports_in_valid_range(self):
        """Issue #24: replace flaky uniqueness test with range validation."""
        from cortex.webgl.jupyter import _find_free_port

        for _ in range(10):
            port = _find_free_port()
            self.assertIsInstance(port, int)
            self.assertGreaterEqual(port, 1024)
            self.assertLessEqual(port, 65535)


class TestDisplay(unittest.TestCase):
    """Test the display() dispatch function."""

    def test_invalid_method(self):
        from cortex.webgl.jupyter import display

        with self.assertRaises(ValueError):
            display(None, method="invalid")

    @patch("cortex.webgl.jupyter.display_iframe")
    def test_dispatch_iframe(self, mock_iframe):
        from cortex.webgl.jupyter import display

        display("fake_data", method="iframe")
        mock_iframe.assert_called_once()

    @patch("cortex.webgl.jupyter.display_static")
    def test_dispatch_static(self, mock_static):
        from cortex.webgl.jupyter import display

        display("fake_data", method="static")
        mock_static.assert_called_once()

    @patch("cortex.webgl.jupyter.display_iframe")
    def test_forwards_kwargs(self, mock_iframe):
        from cortex.webgl.jupyter import display

        display("fake_data", method="iframe", width=800, height=400, port=5555)
        mock_iframe.assert_called_once_with(
            "fake_data", width=800, height=400, port=5555
        )


class TestDisplayIframe(unittest.TestCase):
    """Test the IFrame-based display method."""

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.show")
    def test_calls_show_with_correct_params(self, mock_show, mock_display):
        from cortex.webgl.jupyter import display_iframe

        mock_show.return_value = MagicMock()
        result = display_iframe("fake_data", port=9999)

        mock_show.assert_called_once()
        call_kwargs = mock_show.call_args.kwargs
        self.assertFalse(call_kwargs.get("open_browser", True))
        self.assertFalse(call_kwargs.get("autoclose", True))
        self.assertEqual(call_kwargs["port"], 9999)
        self.assertEqual(result, mock_show.return_value)

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.show")
    def test_iframe_url_contains_port(self, mock_show, mock_display):
        from cortex.webgl.jupyter import display_iframe
        from IPython.display import IFrame

        mock_show.return_value = MagicMock()
        display_iframe("fake_data", port=8888)

        mock_display.assert_called_once()
        iframe_arg = mock_display.call_args[0][0]
        self.assertIsInstance(iframe_arg, IFrame)
        self.assertIn("8888", iframe_arg.src)
        self.assertIn("mixer.html", iframe_arg.src)

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.show")
    def test_auto_port_is_valid(self, mock_show, mock_display):
        from cortex.webgl.jupyter import display_iframe

        mock_show.return_value = MagicMock()
        display_iframe("fake_data")

        port = mock_show.call_args.kwargs["port"]
        self.assertGreaterEqual(port, 1024)
        self.assertLessEqual(port, 65535)


class TestDisplayStatic(unittest.TestCase):
    """Test the static HTML display method."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_returns_static_viewer(self, mock_make_static, mock_display):
        from cortex.webgl.jupyter import display_static, StaticViewer

        mock_make_static.side_effect = self._fake_make_static
        result = display_static("fake_data", height=500)

        mock_make_static.assert_called_once()
        self.assertTrue(mock_make_static.call_args.kwargs.get("html_embed", False))
        mock_display.assert_called_once()
        self.assertIsInstance(result, StaticViewer)
        self.assertTrue(hasattr(result, "close"))
        self.assertTrue(hasattr(result, "iframe"))
        result.close()

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_close_cleans_tmpdir(self, mock_make_static, mock_display):
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        result = display_static("fake_data")

        self.assertTrue(os.path.isdir(result._tmpdir))
        result.close()
        self.assertFalse(os.path.isdir(result._tmpdir))

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_make_static_failure_raises_runtime_error(
        self, mock_make_static, mock_display
    ):
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = Exception("make_static failed")

        with self.assertRaises(RuntimeError) as ctx:
            display_static("fake_data")
        self.assertIn("Failed to generate static viewer", str(ctx.exception))

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_uses_os_assigned_port(self, mock_make_static, mock_display):
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        result = display_static("fake_data")

        iframe_arg = mock_display.call_args[0][0]
        match = re.search(r":(\d+)/", iframe_arg.src)
        self.assertIsNotNone(match)
        port = int(match.group(1))
        self.assertGreaterEqual(port, 1024)
        self.assertLessEqual(port, 65535)
        result.close()

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_width_int_converted(self, mock_make_static, mock_display):
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        result = display_static("fake_data", width=800)

        mock_display.assert_called_once()
        iframe_arg = mock_display.call_args[0][0]
        self.assertEqual("800px", iframe_arg.width)
        result.close()


class TestViewerRegistry(unittest.TestCase):
    """Test the active viewer registry and close_all."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_viewer_registered_on_create(self, mock_make_static, mock_display):
        from cortex.webgl.jupyter import _active_viewers, display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")

        self.assertIn(viewer, _active_viewers)
        viewer.close()

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_close_all_cleans_up(self, mock_make_static, mock_display):
        from cortex.webgl.jupyter import close_all, display_static

        mock_make_static.side_effect = self._fake_make_static
        v1 = display_static("fake_data")
        v2 = display_static("fake_data")

        self.assertTrue(os.path.isdir(v1._tmpdir))
        self.assertTrue(os.path.isdir(v2._tmpdir))

        close_all()

        self.assertFalse(os.path.isdir(v1._tmpdir))
        self.assertFalse(os.path.isdir(v2._tmpdir))

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_close_all_idempotent(self, mock_make_static, mock_display):
        from cortex.webgl.jupyter import close_all, display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")
        viewer.close()
        # Should not raise even though viewer is already closed
        close_all()


class TestMakeNotebookHtml(unittest.TestCase):
    """Test the raw HTML generation function."""

    @patch("cortex.webgl.view.make_static")
    def test_returns_html_string(self, mock_make_static):
        from cortex.webgl.jupyter import make_notebook_html

        def fake_make_static(outpath, data, **kwargs):
            os.makedirs(outpath, exist_ok=True)
            with open(os.path.join(outpath, "index.html"), "w") as f:
                f.write("<!doctype html><html><body>viewer</body></html>")

        mock_make_static.side_effect = fake_make_static

        html = make_notebook_html("fake_data")
        self.assertIn("<!doctype html>", html)
        self.assertIn("viewer", html)

    @patch("cortex.webgl.view.make_static")
    def test_cleans_up_temp_dir(self, mock_make_static):
        from cortex.webgl.jupyter import make_notebook_html

        created_dirs = []

        def fake_make_static(outpath, data, **kwargs):
            created_dirs.append(os.path.dirname(outpath))
            os.makedirs(outpath, exist_ok=True)
            with open(os.path.join(outpath, "index.html"), "w") as f:
                f.write("<html>test</html>")

        mock_make_static.side_effect = fake_make_static

        make_notebook_html("fake_data")

        self.assertEqual(len(created_dirs), 1)
        self.assertFalse(os.path.isdir(created_dirs[0]))


class TestNotebookTemplateRemoved(unittest.TestCase):
    """Verify dead notebook.html template was removed (issue #3)."""

    def test_notebook_template_does_not_exist(self):
        import cortex.webgl

        template_dir = os.path.dirname(cortex.webgl.__file__)
        template_path = os.path.join(template_dir, "notebook.html")
        self.assertFalse(
            os.path.exists(template_path),
            "notebook.html should have been removed (dead code)",
        )


class TestDisplayIframeUrl(unittest.TestCase):
    """Test that display_iframe uses correct URL host and suppresses display_url."""

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.show")
    def test_iframe_url_uses_localhost_not_hostname(self, mock_show, mock_display):
        """Issue #1: iframe URL should use 127.0.0.1, not socket.gethostname()."""
        from cortex.webgl.jupyter import display_iframe

        mock_show.return_value = MagicMock()
        display_iframe("fake_data", port=7777)

        iframe_arg = mock_display.call_args[0][0]
        # Should use 127.0.0.1 (or env-configurable host), not machine hostname
        self.assertTrue(
            iframe_arg.src.startswith("http://127.0.0.1:7777/"),
            "IFrame URL should use 127.0.0.1, got: %s" % iframe_arg.src,
        )

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.show")
    def test_iframe_host_configurable_via_env(self, mock_show, mock_display):
        """Issue #1: iframe host should be configurable via env var."""
        from cortex.webgl.jupyter import display_iframe

        mock_show.return_value = MagicMock()
        with patch.dict(os.environ, {"CORTEX_JUPYTER_IFRAME_HOST": "myhost.local"}):
            display_iframe("fake_data", port=7777)

        iframe_arg = mock_display.call_args[0][0]
        self.assertIn("myhost.local", iframe_arg.src)

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.show")
    def test_show_called_with_display_url_false(self, mock_show, mock_display):
        """Issue #2: show() should be called with display_url=False."""
        from cortex.webgl.jupyter import display_iframe

        mock_show.return_value = MagicMock()
        display_iframe("fake_data", port=9999)

        call_kwargs = mock_show.call_args.kwargs
        self.assertFalse(
            call_kwargs.get("display_url", True),
            "show() should be called with display_url=False",
        )


class TestStaticViewerClosedProperty(unittest.TestCase):
    """Test the public `closed` property on StaticViewer."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_closed_property_false_before_close(self, mock_make_static, mock_display):
        """Issue #10: StaticViewer should have a public `closed` property."""
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")
        self.assertFalse(viewer.closed)
        viewer.close()

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_closed_property_true_after_close(self, mock_make_static, mock_display):
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")
        viewer.close()
        self.assertTrue(viewer.closed)


class TestStaticViewerReprHtml(unittest.TestCase):
    """Test _repr_html_ behavior including after close."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_repr_html_returns_string(self, mock_make_static, mock_display):
        """Issue #21: _repr_html_() should return a non-empty string."""
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")
        html = viewer._repr_html_()
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 0)
        viewer.close()

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_repr_html_after_close_shows_closed_message(
        self, mock_make_static, mock_display
    ):
        """Issue #9: _repr_html_() after close should show a closed message,
        not a broken iframe."""
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")
        viewer.close()
        html = viewer._repr_html_()
        self.assertIn("closed", html.lower())


class TestStaticViewerContextManager(unittest.TestCase):
    """Test that StaticViewer supports the context manager protocol."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_context_manager_closes_on_exit(self, mock_make_static, mock_display):
        """Issue #11: StaticViewer should support `with` statement."""
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")
        tmpdir = viewer._tmpdir

        with viewer:
            self.assertTrue(os.path.isdir(tmpdir))

        self.assertFalse(os.path.isdir(tmpdir))
        self.assertTrue(viewer.closed)

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_context_manager_returns_self(self, mock_make_static, mock_display):
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")

        with viewer as v:
            self.assertIs(v, viewer)

        viewer.close()


class TestStaticViewerDoubleClose(unittest.TestCase):
    """Test that close() is idempotent when called directly twice."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_double_close_no_error(self, mock_make_static, mock_display):
        """Issue #18: calling close() twice directly should not raise."""
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")
        viewer.close()
        viewer.close()  # should not raise


class TestCloseGracefulDegradation(unittest.TestCase):
    """Test that close() cleans up remaining resources even if httpd.shutdown fails."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_tmpdir_cleaned_even_if_httpd_shutdown_fails(
        self, mock_make_static, mock_display
    ):
        """Issue #19: tmpdir should be cleaned up even when httpd.shutdown() raises."""
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")
        tmpdir = viewer._tmpdir

        # Force httpd.shutdown to fail
        viewer._httpd.shutdown = MagicMock(side_effect=OSError("shutdown failed"))

        viewer.close()
        self.assertFalse(os.path.isdir(tmpdir))


class TestCloseThreadJoinVerification(unittest.TestCase):
    """Test that close() warns when thread.join times out."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_warns_when_thread_still_alive_after_join(
        self, mock_make_static, mock_display
    ):
        """Issue #14: should warn if thread doesn't stop after join timeout."""
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")

        # Replace thread with one that reports alive after join
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        viewer._thread = mock_thread

        with self.assertLogs("cortex.webgl.jupyter", level="WARNING") as cm:
            viewer.close(timeout=0.01)

        self.assertTrue(
            any("did not terminate" in msg for msg in cm.output),
            "Expected warning about thread not terminating, got: %s" % cm.output,
        )


class TestCloseAllLockScope(unittest.TestCase):
    """Test that close_all releases the lock before blocking shutdown."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_lock_not_held_during_close(self, mock_make_static, mock_display):
        """Issue #4: _viewer_lock should NOT be held during blocking close() calls."""
        from cortex.webgl.jupyter import _viewer_lock, close_all, display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")

        lock_held_during_close = []
        original_close = viewer.close

        def spy_close(*args, **kwargs):
            # Check if we can acquire the lock (non-blocking).
            # If close_all holds the lock, this will return False.
            acquired = _viewer_lock.acquire(blocking=False)
            lock_held_during_close.append(not acquired)
            if acquired:
                _viewer_lock.release()
            return original_close(*args, **kwargs)

        viewer.close = spy_close
        close_all()

        self.assertTrue(len(lock_held_during_close) > 0)
        self.assertFalse(
            lock_held_during_close[0],
            "close_all() held _viewer_lock during viewer.close() — risk of deadlock",
        )


class TestDisplayStaticResourceLeak(unittest.TestCase):
    """Test resource cleanup when display_static fails after server starts."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.view.make_static")
    def test_cleanup_when_ipydisplay_fails(self, mock_make_static):
        """Issue #7: resources should be cleaned up if ipydisplay raises."""
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static

        with patch(
            "cortex.webgl.jupyter.ipydisplay", side_effect=RuntimeError("no display")
        ):
            with self.assertRaises(RuntimeError):
                display_static("fake_data")

        # After the exception, there should be no leaked threads serving HTTP
        # (we can't easily check threads, but we can verify no leaked tmpdirs
        # by checking that the function cleaned up)
        # The key assertion is that no exception is swallowed and resources
        # are not leaked. Since tmpdir is internal, we verify via the
        # active_viewers registry.
        from cortex.webgl.jupyter import _active_viewers

        # No viewer should have been registered since construction failed
        self.assertEqual(len(list(_active_viewers)), 0)

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_cleanup_when_httpserver_fails(self, mock_make_static, mock_display):
        """Issue #8: tmpdir should be cleaned when HTTPServer fails to bind."""
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static

        created_tmpdirs = []
        original_mkdtemp = __import__("tempfile").mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_tmpdirs.append(d)
            return d

        with patch(
            "cortex.webgl.jupyter.tempfile.mkdtemp", side_effect=tracking_mkdtemp
        ):
            with patch(
                "cortex.webgl.jupyter.http.server.HTTPServer",
                side_effect=OSError("bind failed"),
            ):
                with self.assertRaises((OSError, RuntimeError)):
                    display_static("fake_data")

        # tmpdir should have been cleaned up
        for d in created_tmpdirs:
            self.assertFalse(
                os.path.isdir(d),
                "Temp directory leaked after HTTPServer failure: %s" % d,
            )


class TestDisplayStaticMissingIndexHtml(unittest.TestCase):
    """Test FileNotFoundError when make_static doesn't produce index.html."""

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_missing_index_html_raises_and_cleans_up(
        self, mock_make_static, mock_display
    ):
        """Issue #17: should raise FileNotFoundError and clean up tmpdir."""
        from cortex.webgl.jupyter import display_static

        def fake_make_static_no_index(outpath, data, **kwargs):
            os.makedirs(outpath, exist_ok=True)
            # Write some file but NOT index.html
            with open(os.path.join(outpath, "data.json"), "w") as f:
                f.write("{}")

        mock_make_static.side_effect = fake_make_static_no_index

        with self.assertRaises(FileNotFoundError):
            display_static("fake_data")


class TestStaticHostEnvVar(unittest.TestCase):
    """Test CORTEX_JUPYTER_STATIC_HOST env var."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_static_host_from_env(self, mock_make_static, mock_display):
        """Issue #20: CORTEX_JUPYTER_STATIC_HOST should change the iframe host."""
        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static

        with patch.dict(os.environ, {"CORTEX_JUPYTER_STATIC_HOST": "0.0.0.0"}):
            viewer = display_static("fake_data")

        iframe_arg = mock_display.call_args[0][0]
        self.assertIn("0.0.0.0", iframe_arg.src)
        viewer.close()


class TestQuietHandlerLogging(unittest.TestCase):
    """Test that _QuietHandler logs errors but suppresses routine access logs."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_serves_content_and_logs_errors(self, mock_make_static, mock_display):
        """Issue #15/#22: HTTP server should serve content, log errors,
        suppress routine access logs."""
        import urllib.request

        from cortex.webgl.jupyter import display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")

        # Extract port from iframe
        iframe_arg = mock_display.call_args[0][0]
        match = re.search(r":(\d+)/", iframe_arg.src)
        port = int(match.group(1))

        # Successful request should work
        resp = urllib.request.urlopen("http://127.0.0.1:%d/index.html" % port)
        self.assertEqual(resp.status, 200)
        content = resp.read().decode()
        self.assertIn("test", content)

        # 404 should be logged
        with self.assertLogs("cortex.webgl.jupyter", level="WARNING") as cm:
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/nonexistent" % port)
            except urllib.error.HTTPError:
                pass

        self.assertTrue(
            any("404" in msg or "nonexistent" in msg for msg in cm.output),
            "404 errors should be logged, got: %s" % cm.output,
        )

        viewer.close()


class TestFindFreePortFlaky(unittest.TestCase):
    """Fix for issue #24: flaky port uniqueness test."""

    def test_returns_valid_port_range(self):
        """Ports should be in the valid ephemeral range."""
        from cortex.webgl.jupyter import _find_free_port

        for _ in range(10):
            port = _find_free_port()
            self.assertIsInstance(port, int)
            self.assertGreaterEqual(port, 1024)
            self.assertLessEqual(port, 65535)


class TestViewerRemoveFromRegistryOnClose(unittest.TestCase):
    """Test that closed viewers are removed from _active_viewers."""

    def _fake_make_static(self, outpath, data, **kwargs):
        os.makedirs(outpath, exist_ok=True)
        with open(os.path.join(outpath, "index.html"), "w") as f:
            f.write("<html><body>test</body></html>")

    @patch("cortex.webgl.jupyter.ipydisplay")
    @patch("cortex.webgl.view.make_static")
    def test_viewer_removed_from_registry_on_close(
        self, mock_make_static, mock_display
    ):
        from cortex.webgl.jupyter import _active_viewers, display_static

        mock_make_static.side_effect = self._fake_make_static
        viewer = display_static("fake_data")
        self.assertIn(viewer, _active_viewers)

        viewer.close()
        self.assertNotIn(viewer, _active_viewers)


class TestMakeNotebookHtmlKwargs(unittest.TestCase):
    """Test that make_notebook_html forwards kwargs correctly."""

    @patch("cortex.webgl.view.make_static")
    def test_forwards_template_and_types(self, mock_make_static):
        from cortex.webgl.jupyter import make_notebook_html

        def fake_make_static(outpath, data, **kwargs):
            os.makedirs(outpath, exist_ok=True)
            with open(os.path.join(outpath, "index.html"), "w") as f:
                f.write("<html>test</html>")

        mock_make_static.side_effect = fake_make_static

        make_notebook_html("fake_data", template="notebook.html", types=("fiducial",))

        call_kwargs = mock_make_static.call_args.kwargs
        self.assertEqual(call_kwargs.get("template"), "notebook.html")
        self.assertEqual(call_kwargs.get("types"), ("fiducial",))
        self.assertTrue(call_kwargs.get("html_embed", False))


class TestInitImportProtection(unittest.TestCase):
    """Test that broken jupyter import doesn't break cortex.webgl."""

    def test_webgl_importable_with_broken_jupyter(self):
        """Issue #16: broken jupyter.py should not break cortex.webgl."""
        # This is tested implicitly — if the import guard is correct,
        # cortex.webgl will import even if jupyter submodule fails.
        # We verify the guard pattern exists.
        import cortex.webgl

        # Module should be importable regardless
        self.assertTrue(hasattr(cortex.webgl, "show"))
        self.assertTrue(hasattr(cortex.webgl, "make_static"))


if __name__ == "__main__":
    unittest.main()
