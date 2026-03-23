"""Jupyter notebook integration for pycortex WebGL viewer.

Provides two approaches for displaying brain surfaces in Jupyter notebooks:

1. **IFrame-based** (``display_iframe``): Starts a Tornado server and embeds
   the viewer in an IFrame. Full interactivity with WebSocket support.

2. **Static HTML** (``display_static``): Generates a self-contained HTML viewer
   with all resources embedded. Works in static notebooks (nbviewer, GitHub).

Usage
-----
>>> import cortex
>>> vol = cortex.Volume.random("S1", "fullhead")
>>> cortex.webgl.jupyter.display(vol)  # auto-detects best approach
"""

import http.server
import logging
import os
import shutil
import socket
import tempfile
import threading

from IPython.display import HTML, IFrame
from IPython.display import display as ipydisplay

logger = logging.getLogger(__name__)


def _find_free_port():
    """Find a free TCP port by binding to port 0 and reading the OS-assigned port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class StaticViewer:
    """Handle for a static viewer served via a local HTTP server.

    Call ``close()`` to shut down the server and clean up temp files.

    Attributes
    ----------
    iframe : IPython.display.IFrame
        The IFrame used to display the viewer.
    """

    def __init__(self, iframe, httpd, thread, tmpdir):
        self.iframe = iframe
        self._httpd = httpd
        self._thread = thread
        self._tmpdir = tmpdir

    def close(self):
        """Shut down the HTTP server and remove temp files."""
        try:
            self._httpd.shutdown()
        except Exception:
            logger.warning("Failed to shut down static viewer server", exc_info=True)
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            logger.warning(
                "Failed to clean up temp dir %s", self._tmpdir, exc_info=True
            )

    def __del__(self):
        try:
            self._httpd.shutdown()
        except Exception:
            pass
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass

    def _repr_html_(self):
        """Allow Jupyter to display this object directly."""
        return self.iframe._repr_html_()


def display(data, method="iframe", width="100%", height=600, **kwargs):
    """Display brain data in a Jupyter notebook using the WebGL viewer.

    Parameters
    ----------
    data : Dataset, Volume, Vertex, or dict
        Brain data to display.
    method : str, optional
        Display method: "iframe" for server-based (interactive, default),
        "static" for self-contained HTML (works in nbviewer).
    width : str or int, optional
        Widget width. Default "100%".
    height : int, optional
        Widget height in pixels. Default 600.
    **kwargs
        Additional keyword arguments passed to ``show()`` or ``make_static()``.

    Returns
    -------
    For "iframe": the server object (WebApp)
    For "static": a StaticViewer handle (call ``.close()`` to clean up)
    """
    if method == "iframe":
        return display_iframe(data, width=width, height=height, **kwargs)
    elif method == "static":
        return display_static(data, width=width, height=height, **kwargs)
    else:
        raise ValueError("method must be 'iframe' or 'static', got %r" % method)


def display_iframe(data, width="100%", height=600, port=None, **kwargs):
    """Display brain data via an embedded IFrame connected to a Tornado server.

    Starts the pycortex Tornado server and embeds it in an IFrame within the
    notebook. Provides full interactivity including surface morphing, data
    switching, and WebSocket-based Python control.

    Parameters
    ----------
    data : Dataset, Volume, Vertex, or dict
        Brain data to display.
    width : str or int, optional
        IFrame width. Default "100%".
    height : int, optional
        IFrame height in pixels. Default 600.
    port : int or None, optional
        Port for the Tornado server. If None, a free port is chosen
        automatically.
    **kwargs
        Additional keyword arguments passed to ``cortex.webgl.show()``.

    Returns
    -------
    server : WebApp
        The Tornado server object. Can be used to get a JSMixer client for
        programmatic control.
    """
    from . import view, serve

    if port is None:
        port = _find_free_port()

    # Start the server without opening a browser
    kwargs["open_browser"] = False
    kwargs["autoclose"] = False
    server = view.show(data, port=port, **kwargs)

    url = "http://%s:%d/mixer.html" % (serve.hostname, port)

    # Format width for IFrame
    if isinstance(width, int):
        width = "%dpx" % width

    ipydisplay(IFrame(src=url, width=width, height=height))

    return server


def display_static(data, width="100%", height=600, **kwargs):
    """Display brain data as a self-contained HTML viewer inline.

    Generates a complete static viewer with all JS/CSS/data embedded,
    then displays it in the notebook via a lightweight local HTTP server.

    Note: The embedded HTML is large (~4-5MB) because all JavaScript
    libraries and CSS are inlined.

    Parameters
    ----------
    data : Dataset, Volume, Vertex, or dict
        Brain data to display.
    width : str or int, optional
        Viewer width. Default "100%".
    height : int, optional
        Viewer height in pixels. Default 600.
    **kwargs
        Additional keyword arguments passed to ``cortex.webgl.make_static()``.

    Returns
    -------
    viewer : StaticViewer
        Handle for the static viewer. Call ``viewer.close()`` to shut down the
        HTTP server and clean up temporary files.
    """
    from . import view

    tmpdir = tempfile.mkdtemp(prefix="pycortex_jupyter_")
    outpath = os.path.join(tmpdir, "viewer")

    try:
        view.make_static(outpath, data, html_embed=True, **kwargs)
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(
            "Failed to generate static viewer. "
            "Check that data is valid and cortex is properly configured."
        ) from e

    index_html = os.path.join(outpath, "index.html")
    if not os.path.isfile(index_html):
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise FileNotFoundError(
            "make_static() did not produce index.html. "
            "This may indicate a problem with the static template."
        )

    # Format width
    if isinstance(width, int):
        width_str = "%dpx" % width
    else:
        width_str = width

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **handler_kwargs):
            super().__init__(*args, directory=outpath, **handler_kwargs)

        def log_message(self, format, *args):
            # Log HTTP errors, suppress routine access logs
            if args and len(args) >= 2:
                try:
                    status = int(args[1])
                    if status >= 400:
                        logger.warning("Static viewer HTTP %s: %s", args[1], args[0])
                except (ValueError, IndexError):
                    pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _QuietHandler)
    port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    iframe = IFrame(
        src="http://127.0.0.1:%d/index.html" % port, width=width_str, height=height
    )
    ipydisplay(iframe)

    return StaticViewer(iframe, httpd, thread, tmpdir)


def make_notebook_html(data, template="static.html", types=("inflated",), **kwargs):
    """Generate a self-contained HTML string for the WebGL viewer.

    This is a lower-level function that returns the raw HTML string rather
    than displaying it. Useful for saving or embedding in custom contexts.

    Parameters
    ----------
    data : Dataset, Volume, Vertex, or dict
        Brain data to display.
    template : str, optional
        HTML template name. Default "static.html".
    types : tuple, optional
        Surface types to include. Default ("inflated",).
    **kwargs
        Additional keyword arguments passed to ``make_static()``.

    Returns
    -------
    html : str
        The self-contained HTML string.
    """
    from . import view

    with tempfile.TemporaryDirectory(prefix="pycortex_nb_") as tmpdir:
        outpath = os.path.join(tmpdir, "viewer")
        view.make_static(
            outpath, data, template=template, types=types, html_embed=True, **kwargs
        )

        index_html = os.path.join(outpath, "index.html")
        with open(index_html, "r") as f:
            return f.read()
