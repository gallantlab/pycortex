"""Launcher for previewing the pycortex WebGL viewer.

Spawns cortex.webgl.show() with a Vertex2D so we can debug NaN-mask
rendering visually via mcp__Claude_Preview__.
"""

import sys

import numpy as np

import cortex


def main(port: int = 0) -> None:
    np.random.seed(0)
    vtx1 = cortex.Vertex.random("S1")
    vtx2 = cortex.Vertex.random("S1")
    vtx12 = cortex.Vertex2D(vtx1, vtx2)

    server = cortex.webgl.show(vtx12, port=port, open_browser=False)
    print(f"viewer ready on port {server.port}", flush=True)
    # tornado server runs on a daemon thread; main must block
    server.join()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(port)
