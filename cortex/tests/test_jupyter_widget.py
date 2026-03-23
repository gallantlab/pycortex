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

    def test_returns_different_ports(self):
        from cortex.webgl.jupyter import _find_free_port

        ports = {_find_free_port() for _ in range(10)}
        self.assertGreater(len(ports), 1)


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
        result.close()


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


class TestNotebookTemplate(unittest.TestCase):
    """Test that the notebook HTML template exists and is valid."""

    def test_template_exists(self):
        import cortex.webgl

        template_dir = os.path.dirname(cortex.webgl.__file__)
        template_path = os.path.join(template_dir, "notebook.html")
        self.assertTrue(
            os.path.exists(template_path),
            "notebook.html template not found at %s" % template_path,
        )

    def test_template_extends_base(self):
        import cortex.webgl

        template_dir = os.path.dirname(cortex.webgl.__file__)
        template_path = os.path.join(template_dir, "notebook.html")
        with open(template_path) as f:
            content = f.read()
        self.assertIn("extends template.html", content)
        self.assertIn("block onload", content)
        self.assertIn("block jsinit", content)


if __name__ == "__main__":
    unittest.main()
