"""Jupyter notebook integration for pycortex WebGL viewer.

Provides two approaches for displaying brain surfaces in Jupyter notebooks:

1. **IFrame-based** (``display_iframe``): Starts a Tornado server and embeds
   the viewer in an IFrame. Full interactivity with WebSocket support.

2. **Static viewer** (``display_static``): Generates a static viewer directory
   served via a local HTTP server and embedded in an IFrame. Requires a live
   Jupyter environment (will not work in static notebook renderers).

Usage
-----
>>> import cortex
>>> vol = cortex.Volume.random("S1", "fullhead")
>>> cortex.webgl.jupyter.display(vol)  # defaults to iframe method
"""

import atexit
import http.server
import logging
import os
import shutil
import socket
import tempfile
import threading
import weakref

from IPython.display import IFrame
from IPython.display import display as ipydisplay

logger = logging.getLogger(__name__)

# Registry of active StaticViewer instances for cleanup.
# Uses weak references so viewers that are garbage-collected don't linger here.
_active_viewers = weakref.WeakSet()
_viewer_lock = threading.Lock()


def close_all():
    """Close all active static viewers, shutting down servers and removing temp files."""
    with _viewer_lock:
        viewers = list(_active_viewers)
    closed = 0
    for viewer in viewers:
        try:
            viewer.close()
            closed += 1
        except Exception:
            logger.warning("Failed to close viewer during close_all", exc_info=True)
    if closed:
        logger.info("Closed %d static viewer(s)", closed)


atexit.register(close_all)


def _find_free_port():
    """Find a free TCP port by binding to port 0 and reading the OS-assigned port.

    Note: There is an inherent TOCTOU race between releasing this socket and
    the caller binding the port. For ``display_static`` this is avoided by
    binding ``HTTPServer`` to port 0 directly. For ``display_iframe`` the
    underlying Tornado server does not support port 0, so this helper is used
    as a best-effort fallback.
    """
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
        self._closed = False
        self._lock = threading.Lock()
        with _viewer_lock:
            _active_viewers.add(self)

    def close(self, timeout=1.0):
        """Shut down the HTTP server, wait for the thread, and remove temp files.

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait for the server thread to finish.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                logger.warning(
                    "Failed to shut down static viewer server", exc_info=True
                )

        if self._thread is not None and self._thread.is_alive():
            try:
                self._thread.join(timeout=timeout)
            except Exception:
                logger.warning(
                    "Failed to join static viewer server thread", exc_info=True
                )

        if self._tmpdir is not None:
            try:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
            except Exception:
                logger.warning(
                    "Failed to clean up temp dir %s", self._tmpdir, exc_info=True
                )

    def __del__(self):
        try:
            self.close(timeout=0.1)
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
    """Display brain data using a temporary static WebGL viewer inline.

    Uses ``cortex.webgl.make_static`` to generate a *directory* containing
    ``index.html`` plus all required JS/CSS/data assets, then serves that
    directory via a lightweight local HTTP server and embeds it in the
    notebook inside an IFrame.

    Note
    ----
    The output is **not** a single self-contained HTML string; it is a static
    viewer directory that must be served for the page to function. This works
    in live Jupyter environments but most static notebook renderers will not
    display the interactive viewer.

    The bind host defaults to ``127.0.0.1``. For remote notebook setups
    (JupyterHub, SSH tunnels), set the ``CORTEX_JUPYTER_STATIC_HOST``
    environment variable to the appropriate hostname.

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

    host = os.environ.get("CORTEX_JUPYTER_STATIC_HOST", "127.0.0.1")

    httpd = http.server.HTTPServer((host, 0), _QuietHandler)
    port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    iframe = IFrame(
        src="http://%s:%d/index.html" % (host, port), width=width_str, height=height
    )
    ipydisplay(iframe)

    return StaticViewer(iframe, httpd, thread, tmpdir)


def make_notebook_html(data, template="static.html", types=("inflated",), **kwargs):
    """Generate the ``index.html`` for a static WebGL viewer.

    This is a lower-level function that returns the raw HTML string produced
    by ``make_static()``. Note that the HTML references external asset files
    (CTM meshes, JSON data, PNG colormaps) that ``make_static()`` writes
    alongside ``index.html``. The returned string alone is **not** a fully
    self-contained viewer -- it must be served from a directory containing
    those assets for the viewer to function.

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
        The generated HTML string (requires adjacent assets to function).
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
