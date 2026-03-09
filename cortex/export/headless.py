"""headless.py – context manager that runs the pycortex WebGL viewer inside a
headless Chromium browser via Playwright, allowing ``save_3d_views`` to
produce screenshots without any manual browser interaction.

The approach:
1. Start the Tornado WebGL server with ``open_browser=False`` so that it does
   not try to open a real browser window; this returns the bare ``WebApp``
   server object.
2. Launch a headless Chromium page via Playwright and navigate it to the
   server URL.  The page executes ``python_interface.js``, which opens a
   WebSocket to ``/wsconnect/`` and immediately sends the ``"connect"``
   message, setting the ``threading.Event`` inside ``server.get_client()``.
3. Call ``server.get_client()`` (which blocks on that event) to obtain the
   ``JSMixer`` handle – the same object that ``cortex.webshow()`` normally
   returns after the user navigates a real browser.
4. Yield the handle; the caller drives it exactly as in the normal flow.
5. On exit, tear down the viewer, server, and Playwright in reverse order.

Requirements
------------
    pip install playwright
    playwright install chromium
"""

import concurrent.futures
import contextlib
from typing import Any, Mapping, Union

import cortex


@contextlib.contextmanager
def headless_viewer(
    volume: Union[cortex.Volume, cortex.Vertex],
    viewer_params: Mapping[str, Any],
    *,
    timeout: float = 60.0,
):
    """Context manager that yields a connected ``JSMixer`` handle rendered in a
    headless Chromium browser.

    Parameters
    ----------
    volume : cortex.Volume or cortex.Vertex
        Data to display.
    viewer_params : dict
        Keyword arguments forwarded verbatim to ``cortex.webshow``.
    timeout : float
        Seconds to wait for the browser to establish the WebSocket connection
        and for ``server.get_client()`` to return (default: 60).

    Yields
    ------
    handle : JSMixer
        The viewer handle, ready for ``_set_view`` / ``getImage`` calls.
        ``handle.server`` is the underlying ``WebApp`` instance.
        ``handle._playwright_page`` is the Playwright ``Page`` object so
        callers can run additional ``page.evaluate(...)`` calls if needed.

    Raises
    ------
    ImportError
        If ``playwright`` is not installed.
    RuntimeError
        If the browser fails to connect within ``timeout`` seconds.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "playwright is required for headless rendering. Install it with:\n"
            "    pip install playwright && playwright install chromium"
        ) from exc

    # ------------------------------------------------------------------
    # 1. Start the Tornado server without opening a real browser window.
    #    open_browser=False suppresses webbrowser.open() and returns the
    #    raw WebApp server object instead of a JSMixer handle.
    # ------------------------------------------------------------------
    server = cortex.webshow(volume, open_browser=False, **viewer_params)
    url = f"http://localhost:{server.port}/mixer.html"

    # ------------------------------------------------------------------
    # 2. Launch headless Chromium with software WebGL (SwiftShader).
    #    --use-gl=swiftshader provides a full WebGL implementation that does
    #    not require a GPU or display server, making it usable in CI / Docker.
    #    On GPU-equipped machines swap to --use-gl=egl for hardware rendering.
    # ------------------------------------------------------------------
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--enable-webgl",
            "--use-gl=swiftshader",       # software WebGL; no GPU / display needed
            "--no-sandbox",               # required in most container environments
            "--disable-dev-shm-usage",    # avoids /dev/shm exhaustion in containers
        ],
    )
    page = browser.new_page()

    handle = None
    try:
        # ------------------------------------------------------------------
        # 3. Begin waiting for the WebSocket "connect" message in a thread
        #    *before* navigating, so we cannot miss it even if the browser
        #    connects before page.goto() returns.
        # ------------------------------------------------------------------
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(server.get_client)

            # Navigate the browser.  python_interface.js runs on load and
            # immediately sends "connect" over WebSocket, which unblocks
            # server.get_client() (it waits on a threading.Event).
            page.goto(url, timeout=timeout * 1000, wait_until="load")

            # Retrieve the handle; it should already be ready by this point,
            # but the timeout guard ensures we surface hung state clearly.
            try:
                handle = fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise RuntimeError(
                    f"Headless browser connected to {url} but the WebSocket "
                    f'"connect" message was not received within {timeout:.0f} s. '
                    "Check that WebGL initialised successfully in Chromium."
                )

        assert not isinstance(handle, list) # type narrowing to JSMixer, without having to import it
        handle.server = server
        # Expose the Playwright page for optional JS evaluation by callers.
        handle._playwright_page = page

        yield handle

    finally:
        # ------------------------------------------------------------------
        # 4. Tear-down in reverse connection order:
        #    a) Close the JS viewer (sends a WS RPC to the browser).
        #    b) Close the Playwright page — this disconnects the WebSocket
        #       from Chromium to the Tornado server cleanly, before the
        #       server is told to stop.  Closing the page *after* stopping
        #       the server is what causes browser.close() to hang: Chromium
        #       refuses to exit while it still has an open socket.
        #    c) Stop the Tornado server + IOLoop (no active connections now).
        #    d) Close the browser process, then stop Playwright.
        #    Each step is individually guarded so a failure in one does not
        #    prevent the others from running.
        # ------------------------------------------------------------------
        if handle is not None:
            try:
                assert not isinstance(handle, list) # type narrowing to JSMixer, without having to import it
                handle.close()
            except Exception:
                pass

        try:
            page.close()
        except Exception:
            pass

        try:
            server.stop()
        except Exception:
            pass

        try:
            browser.close()
        except Exception:
            pass

        try:
            playwright.stop()
        except Exception:
            pass
