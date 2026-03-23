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
import json
import os
import random
import tempfile
import warnings

from IPython.display import HTML, IFrame
from IPython.display import display as ipydisplay


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
    For "iframe": the server object (JSMixer or WebApp)
    For "static": the IPython HTML display object
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
        Port for the Tornado server. If None, a random port is chosen.
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
        port = random.randint(1024, 65536)

    # Start the server without opening a browser
    kwargs['open_browser'] = False
    kwargs['autoclose'] = False
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
    then displays it in the notebook. This works in static notebook
    renderers like nbviewer and GitHub.

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
    iframe : IPython.display.IFrame
        The IFrame display object.
    """
    from . import view

    # Create a temporary directory for the static viewer
    tmpdir = tempfile.mkdtemp(prefix="pycortex_jupyter_")
    outpath = os.path.join(tmpdir, "viewer")

    # Generate the static viewer
    view.make_static(outpath, data, html_embed=True, **kwargs)

    # Read the generated HTML
    index_html = os.path.join(outpath, "index.html")
    with open(index_html, "r") as f:
        html_content = f.read()

    # Format width
    if isinstance(width, int):
        width_str = "%dpx" % width
    else:
        width_str = width

    # Serve via a minimal local HTTP server to avoid srcdoc size limits
    # and cross-origin issues with data URIs in the embedded HTML
    import http.server
    import threading

    # Find a free port
    port = random.randint(10000, 65000)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **handler_kwargs):
            super().__init__(*args, directory=outpath, **handler_kwargs)

        def log_message(self, format, *args):
            pass  # Suppress log output in notebook

    httpd = http.server.HTTPServer(("127.0.0.1", port), QuietHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    iframe = IFrame(src="http://127.0.0.1:%d/index.html" % port,
                    width=width_str, height=height)
    ipydisplay(iframe)
    return iframe


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
    tmpdir = tempfile.mkdtemp(prefix="pycortex_nb_")
    outpath = os.path.join(tmpdir, "viewer")

    from . import view
    view.make_static(outpath, data, template=template, types=types,
                     html_embed=True, **kwargs)

    index_html = os.path.join(outpath, "index.html")
    with open(index_html, "r") as f:
        return f.read()
