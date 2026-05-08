"""MRE for PR #627: Vertex objects without NaNs rendering as fully transparent.

Reproducer from the issue:
    import cortex
    vtx = cortex.Vertex.random("S1")
    cortex.webgl.show(vtx)

Before PR #627 (post #612): surface fully transparent.
After PR #627: surface renders normally.

Set PCX_MODE=nan to also exercise the regression check (Vertex with NaNs
should still render NaN positions as transparent).
"""

import os
import sys

_WORKTREE_ROOT = "/Users/mvdoc/Documents/04Archive/repos/pycortex/.claude/worktrees/stoic-snyder-f92c2a"
sys.path.insert(0, _WORKTREE_ROOT)
os.chdir("/tmp")

import numpy as np
import cortex

assert cortex.__file__.startswith(_WORKTREE_ROOT), (
    "cortex was loaded from %s, expected worktree" % cortex.__file__
)

PORT = 27583
MODE = os.environ.get("PCX_MODE", "plain")  # "plain" or "nan"

if MODE == "nan":
    # Regression check: NaN positions should still render transparent.
    vtx = cortex.Vertex.random("S1")
    data = vtx.data.copy()
    data[: len(data) // 2] = np.nan
    vtx = cortex.Vertex(data, "S1", cmap="RdBu_r", vmin=-3, vmax=3)
    title = "PR627 MRE: Vertex with NaNs (half transparent expected)"
else:
    # The exact MRE from PR #627 / issue #626.
    vtx = cortex.Vertex.random("S1")
    title = "PR627 MRE: cortex.Vertex.random('S1') (should render, not be transparent)"

server = cortex.webgl.show(
    vtx,
    port=PORT,
    open_browser=False,
    autoclose=False,
    title=title,
)
server.join()
