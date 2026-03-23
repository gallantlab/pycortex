"""Tests for the Jupyter WebGL widget integration."""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestJupyterImports(unittest.TestCase):
    """Test that the jupyter module imports correctly."""

    def test_import_module(self):
        from cortex.webgl import jupyter
        self.assertTrue(hasattr(jupyter, 'display'))
        self.assertTrue(hasattr(jupyter, 'display_iframe'))
        self.assertTrue(hasattr(jupyter, 'display_static'))
        self.assertTrue(hasattr(jupyter, 'make_notebook_html'))

    def test_import_from_webgl(self):
        import cortex
        self.assertTrue(hasattr(cortex.webgl, 'jupyter'))

    def test_display_invalid_method(self):
        from cortex.webgl.jupyter import display
        with self.assertRaises(ValueError):
            display(None, method="invalid")


class TestDisplayIframe(unittest.TestCase):
    """Test the IFrame-based display method."""

    @patch('cortex.webgl.jupyter.ipydisplay')
    @patch('cortex.webgl.view.show')
    def test_iframe_calls_show(self, mock_show, mock_display):
        """Test that display_iframe calls show() with correct params."""
        from cortex.webgl.jupyter import display_iframe

        mock_server = MagicMock()
        mock_show.return_value = mock_server

        result = display_iframe("fake_data", port=9999)

        mock_show.assert_called_once()
        call_kwargs = mock_show.call_args
        self.assertFalse(call_kwargs.kwargs.get('open_browser', True))
        self.assertFalse(call_kwargs.kwargs.get('autoclose', True))
        self.assertEqual(result, mock_server)

    @patch('cortex.webgl.jupyter.ipydisplay')
    @patch('cortex.webgl.view.show')
    def test_iframe_displays_iframe(self, mock_show, mock_display):
        """Test that an IFrame is displayed."""
        from cortex.webgl.jupyter import display_iframe
        from IPython.display import IFrame

        mock_show.return_value = MagicMock()
        display_iframe("fake_data", port=8888, height=400)

        mock_display.assert_called_once()
        iframe_arg = mock_display.call_args[0][0]
        self.assertIsInstance(iframe_arg, IFrame)


class TestDisplayStatic(unittest.TestCase):
    """Test the static HTML display method."""

    @patch('cortex.webgl.jupyter.ipydisplay')
    @patch('cortex.webgl.view.make_static')
    def test_static_creates_tempdir(self, mock_make_static, mock_display):
        """Test that display_static creates a temp directory and HTML."""
        from cortex.webgl.jupyter import display_static

        # Mock make_static to create a fake index.html
        def fake_make_static(outpath, data, **kwargs):
            os.makedirs(outpath, exist_ok=True)
            with open(os.path.join(outpath, "index.html"), "w") as f:
                f.write("<html><body>test</body></html>")

        mock_make_static.side_effect = fake_make_static

        result = display_static("fake_data", height=500)

        mock_make_static.assert_called_once()
        call_kwargs = mock_make_static.call_args
        self.assertTrue(call_kwargs.kwargs.get('html_embed', False))
        mock_display.assert_called_once()
        # Result should be an IFrame
        from IPython.display import IFrame
        self.assertIsInstance(result, IFrame)

    @patch('cortex.webgl.jupyter.ipydisplay')
    @patch('cortex.webgl.view.make_static')
    def test_static_width_int(self, mock_make_static, mock_display):
        """Test integer width is converted to px string."""
        from cortex.webgl.jupyter import display_static

        def fake_make_static(outpath, data, **kwargs):
            os.makedirs(outpath, exist_ok=True)
            with open(os.path.join(outpath, "index.html"), "w") as f:
                f.write("<html><body>test</body></html>")

        mock_make_static.side_effect = fake_make_static

        result = display_static("fake_data", width=800)
        mock_display.assert_called_once()
        from IPython.display import IFrame
        self.assertIsInstance(result, IFrame)


class TestMakeNotebookHtml(unittest.TestCase):
    """Test the raw HTML generation function."""

    @patch('cortex.webgl.view.make_static')
    def test_returns_html_string(self, mock_make_static):
        """Test that make_notebook_html returns an HTML string."""
        from cortex.webgl.jupyter import make_notebook_html

        def fake_make_static(outpath, data, **kwargs):
            os.makedirs(outpath, exist_ok=True)
            with open(os.path.join(outpath, "index.html"), "w") as f:
                f.write("<!doctype html><html><body>viewer</body></html>")

        mock_make_static.side_effect = fake_make_static

        html = make_notebook_html("fake_data")
        self.assertIn("<!doctype html>", html)
        self.assertIn("viewer", html)


class TestNotebookTemplate(unittest.TestCase):
    """Test that the notebook HTML template exists and is valid."""

    def test_template_exists(self):
        """Test that notebook.html template exists."""
        import cortex.webgl
        template_dir = os.path.dirname(cortex.webgl.__file__)
        template_path = os.path.join(template_dir, "notebook.html")
        self.assertTrue(os.path.exists(template_path),
                        "notebook.html template not found at %s" % template_path)

    def test_template_extends_base(self):
        """Test that notebook.html extends template.html."""
        import cortex.webgl
        template_dir = os.path.dirname(cortex.webgl.__file__)
        template_path = os.path.join(template_dir, "notebook.html")
        with open(template_path) as f:
            content = f.read()
        self.assertIn("extends template.html", content)
        self.assertIn("block onload", content)
        self.assertIn("block jsinit", content)


if __name__ == '__main__':
    unittest.main()
