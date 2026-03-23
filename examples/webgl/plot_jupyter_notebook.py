"""
=============================================
Display WebGL Viewer in a Jupyter Notebook
=============================================

Pycortex can embed the interactive WebGL brain viewer directly in a Jupyter
notebook cell. There are two methods:

1. **IFrame mode** (default): Starts a background Tornado server and embeds
   it in an IFrame. Provides full interactivity including surface morphing,
   data switching, and programmatic control from Python via WebSocket.

2. **Static mode**: Generates a self-contained HTML viewer served by a
   lightweight local HTTP server. Useful when you want to avoid the Tornado
   dependency or need a simpler setup.

Both methods are **non-blocking** — subsequent notebook cells execute
immediately.

.. note::

   This example cannot be executed during the documentation build because it
   requires a running Jupyter kernel and a browser. The code below is shown
   for reference only.
"""
# sphinx_gallery_thumbnail_path = ''
# sphinx_gallery_dummy_images = 1

import cortex
import numpy as np

np.random.seed(1234)

###############################################################################
# Create some example data
# ------------------------
volume = cortex.Volume.random(subject='S1', xfmname='fullhead')

###############################################################################
# Method 1: IFrame mode (recommended for interactive exploration)
# ---------------------------------------------------------------
# This starts a Tornado server in a background thread and embeds the viewer
# in an IFrame. The cell returns immediately — you can keep running code.

server = cortex.webgl.jupyter.display(volume)

###############################################################################
# The server runs in the background, so you can execute more cells right away.
# For example, create a second dataset and display it in another viewer:

volume2 = cortex.Volume.random(subject='S1', xfmname='fullhead')
server2 = cortex.webgl.jupyter.display(volume2)

###############################################################################
# Programmatic control via JSMixer
# --------------------------------
# After the viewer has loaded in the IFrame, you can get a control handle.
# NOTE: ``get_client()`` blocks until the browser connects via WebSocket,
# so call it in a separate cell from ``display()``.

client = server.get_client()

# Rotate the view
client._set_view(azimuth=45, altitude=30)

# Switch to inflated surface (mix=1.0 is fully inflated)
client._set_view(mix=1.0)

# Capture a screenshot
client.getImage("notebook_screenshot.png")

###############################################################################
# Method 2: Static mode
# ---------------------
# Generates a self-contained HTML viewer. The ``make_static()`` call takes
# a few seconds to generate CTM meshes and embed resources, then serves
# the result via a local HTTP server.

cortex.webgl.jupyter.display(volume, method="static")

###############################################################################
# Customizing the viewer
# ----------------------
# Both methods accept the same keyword arguments as ``cortex.webgl.show()``
# and ``cortex.webgl.make_static()``.

cortex.webgl.jupyter.display(
    volume,
    types=("inflated",),            # Surface types to include
    overlays_visible=("sulci",),    # Show sulci overlay by default
    height=400,                      # Shorter viewer
    title="My Experiment",
)

###############################################################################
# Lower-level: get raw HTML
# -------------------------
# ``make_notebook_html()`` returns the self-contained HTML as a string,
# useful for saving to disk or embedding in custom web pages.

html = cortex.webgl.jupyter.make_notebook_html(volume)
print("Generated HTML: %d bytes" % len(html))
