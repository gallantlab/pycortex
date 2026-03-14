"""headless.py - context manager that runs the pycortex WebGL viewer inside a
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
   ``JSMixer`` handle - the same object that ``cortex.webshow()`` normally
   returns after the user navigates a real browser.
4. Yield the handle; the caller drives it exactly as in the normal flow.
5. On exit, tear down the viewer, server, and Playwright in reverse order.

Jupyter / asyncio compatibility
-------------------------------
Playwright's *sync* API internally starts an asyncio event loop via a
greenlet.  When called from inside an already-running loop (e.g. a Jupyter
notebook), ``sync_playwright()`` raises::

    Error: It looks like you are using Playwright Sync API inside the
    asyncio loop. Please use the Async API instead.

To avoid this, **all** Playwright sync API calls are executed in a dedicated
daemon thread (``_PlaywrightThread``) that has no running asyncio loop.  The
main thread never touches Playwright objects directly; it only communicates
with the worker thread through ``concurrent.futures.Future`` and
``threading.Event``.  This makes the context manager work identically in
plain Python scripts and in Jupyter notebooks.

Requirements
------------
    pip install playwright
    playwright install chromium
"""

import concurrent.futures
import contextlib
from logging import warning
import threading
from typing import Any, Mapping, Optional, Union

import cortex


# --------------------------------------------------------------------------- #
# Helper: run Playwright in a dedicated thread to avoid asyncio conflicts     #
# --------------------------------------------------------------------------- #

class _PlaywrightThread:
    """Manages the Playwright lifecycle on a private daemon thread.

    Playwright's sync API requires that no asyncio event loop is running on
    the calling thread.  Jupyter (and similar environments) always have a
    running loop, so we isolate all Playwright calls on a background thread
    that is guaranteed to be loop-free.

    Usage::

        pw_thread = _PlaywrightThread()
        pw_thread.start(url, timeout=60)   # blocks until page is loaded
        # ... use the pycortex handle (which talks via Tornado, not Playwright) ...
        pw_thread.shutdown()               # tears down browser + playwright
    """

    def __init__(self) -> None:
        self._ready_future: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._shutdown_event = threading.Event()
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        # Set during worker startup; only touched by the worker thread.
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        # Browser-side errors collected by Playwright event listeners.
        # Written by the worker thread, read by the main thread via
        # the ``browser_errors`` property.
        self._browser_errors: list[str] = []
        self._errors_lock = threading.Lock()

    # -- public API -------------------------------------------------------- #

    def start(self, url: str, *, timeout: float = 60.0) -> None:
        """Launch the worker thread, open Chromium, and navigate to *url*.

        Blocks until the page has finished loading (or raises on failure).

        Raises
        ------
        RuntimeError
            If the browser fails to start or navigate within *timeout* seconds.
        ImportError
            If ``playwright`` is not installed.
        """
        self._url = url
        self._nav_timeout = timeout
        self._thread = threading.Thread(
            target=self._worker, name="PlaywrightThread", daemon=True
        )
        self._thread.start()

        # Block until the worker signals that the page is loaded, or raises.
        try:
            self._ready_future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(
                f"Playwright worker did not finish navigating to {url} "
                f"within {timeout:.0f} s."
            ) from None
        # Re-raise any exception that happened inside the worker thread.
        if self._error is not None:
            raise self._error

    @property
    def browser_errors(self) -> list[str]:
        """Return a snapshot of browser-side errors collected so far.

        Includes uncaught JS exceptions (``pageerror``) and ``console.error``
        / ``console.warning`` messages.  Thread-safe.
        """
        with self._errors_lock:
            return list(self._browser_errors)

    def shutdown(self) -> None:
        """Signal the worker to tear down Playwright and wait for it to finish."""
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    # -- private worker ---------------------------------------------------- #

    def _worker(self) -> None:
        """Entry point for the daemon thread - runs the full Playwright lifecycle."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            self._error = ImportError(
                "playwright is required for headless rendering. Install it with:\n"
                "    pip install playwright && playwright install chromium"
            )
            self._error.__cause__ = exc
            self._ready_future.set_result(None)  # unblock caller
            return

        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--enable-webgl",
                    "--use-gl=swiftshader",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            self._page = self._browser.new_page()

            # Register listeners *before* navigation so we capture
            # errors that fire during page load (e.g. WebGL failures).
            self._page.on("pageerror", self._on_pageerror)
            self._page.on("console", self._on_console)

            self._page.goto(
                self._url,
                timeout=self._nav_timeout * 1000,
                wait_until="load",
            )
            # Navigation succeeded - signal the main thread.
            self._ready_future.set_result(None)
        except BaseException as exc:
            self._error = exc
            # Unblock the main thread so it can see the error.
            if not self._ready_future.done():
                self._ready_future.set_result(None)
            self._cleanup()
            return

        # Keep the thread (and therefore Playwright) alive until shutdown.
        self._shutdown_event.wait()
        self._cleanup()

    # -- Playwright event handlers (called on the worker thread) ---------- #

    def _on_pageerror(self, error: Any) -> None:
        """Listener for uncaught JS exceptions in the browser page."""
        with self._errors_lock:
            self._browser_errors.append(f"[pageerror] {error}")

    def _on_console(self, msg: Any) -> None:
        """Listener for console.error / console.warning messages."""
        if msg.type in ("error", "warning"):
            with self._errors_lock:
                self._browser_errors.append(f"[console.{msg.type}] {msg.text}")

    def _cleanup(self) -> None:
        """Tear down Playwright objects in reverse order.  Each step is
        individually guarded so a failure in one does not prevent the rest."""
        for obj, method in [
            (self._page, "close"),
            (self._browser, "close"),
            (self._playwright, "stop"),
        ]:
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# Public context manager                                                      #
# --------------------------------------------------------------------------- #

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

    Raises
    ------
    ImportError
        If ``playwright`` is not installed.
    RuntimeError
        If the browser fails to connect within ``timeout`` seconds.
    """

    # ------------------------------------------------------------------
    # 1. Start the Tornado server without opening a real browser window.
    #    open_browser=False suppresses webbrowser.open() and returns the
    #    raw WebApp server object instead of a JSMixer handle.
    # ------------------------------------------------------------------
    if 'port' in viewer_params and viewer_params['port'] is not None:
        # This is not recommended or necessary, since port tunneling is not needed!
        UserWarning(
            "Warning: headless_viewer ignores the 'port' argument in viewer_params. "
            "The Tornado server will listen on a random free port to avoid conflicts."
        )
    server = cortex.webshow(volume, open_browser=False, display_url=False, **{k: v for k, v in viewer_params.items() if k not in ["port", "display_url"]})
    url = f"http://localhost:{server.port}/mixer.html"

    # Prevent the server from auto-stopping when the last WebSocket client
    # disconnects (ClientSocket.on_close calls server.stop() when
    # disconnect_on_close is True).  The headless context manager owns the
    # full lifecycle and will call server.stop() explicitly during teardown.
    # Without this, server.stop() is called twice: once by the auto-close
    # mechanism when Playwright's page closes the WebSocket, and once by
    # our finally block — producing a duplicate "Stopping server" message.
    server.disconnect_on_close = False

    # ------------------------------------------------------------------
    # 2. Launch headless Chromium with software WebGL (SwiftShader) in a
    #    dedicated thread.  This avoids the "Playwright Sync API inside
    #    the asyncio loop" error that occurs in Jupyter notebooks.
    #    --use-gl=swiftshader provides a full WebGL implementation that
    #    does not require a GPU or display server, making it usable in
    #    CI / Docker / notebooks.
    # ------------------------------------------------------------------
    pw_thread = _PlaywrightThread()

    handle = None
    try:
        # ------------------------------------------------------------------
        # 3. Begin waiting for the WebSocket "connect" message in a thread
        #    *before* navigating, so we cannot miss it even if the browser
        #    connects before page.goto() returns.
        # ------------------------------------------------------------------
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(server.get_client)

            # Launch the browser and navigate.  python_interface.js runs on
            # load and sends "connect" over WebSocket, which unblocks
            # server.get_client().
            pw_thread.start(url, timeout=timeout)

            # Retrieve the handle; it should already be ready by this point,
            # but the timeout guard surfaces hung state clearly.
            try:
                handle = fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                browser_errors = pw_thread.browser_errors
                detail = (
                    "\nBrowser errors:\n" + "\n".join(browser_errors)
                    if browser_errors
                    else "\nNo browser errors were captured."
                )
                raise RuntimeError(
                    f"Headless browser connected to {url} but the WebSocket "
                    f'"connect" message was not received within {timeout:.0f} s. '
                    f"Check that WebGL initialised successfully in Chromium."
                    f"{detail}"
                )

        assert not isinstance(handle, list)  # type narrowing to JSMixer
        handle.server = server
        # Expose a live reference so callers can query browser errors at
        # any point during the session (each call returns a fresh snapshot).
        handle._pw_thread = pw_thread

        yield handle

    finally:
        # ------------------------------------------------------------------
        # 4. Tear-down in reverse connection order:
        #    a) Close the JS viewer (sends a WS RPC to the browser).
        #    b) Shut down the Playwright thread (closes page → browser →
        #       playwright, all on the worker thread).  This disconnects
        #       the WebSocket from Chromium cleanly *before* stopping the
        #       Tornado server, preventing browser.close() hangs.
        #    c) Stop the Tornado server + IOLoop.
        #    Each step is individually guarded so a failure in one does not
        #    prevent the others from running.
        # ------------------------------------------------------------------
        if handle is not None:
            try:
                assert not isinstance(handle, list)  # type narrowing
                handle.close()
            except Exception:
                pass

        try:
            pw_thread.shutdown()
        except Exception:
            pass

        try:
            server.stop()
        except Exception:
            pass
