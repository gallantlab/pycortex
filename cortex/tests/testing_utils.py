# Skip any test that relies on playwright if it's not available.
try:
    from playwright.sync_api import sync_playwright

    _pw = sync_playwright().start()
    try:
        _b = _pw.chromium.launch(headless=True, args=["--no-sandbox"])
        _b.close()
    finally:
        _pw.stop()
    has_playwright = True
except Exception:
    has_playwright = False


def wait_for_file(path, timeout=30):
    """Poll until `path` exists and has nonzero size; raise after `timeout`.

    TODO: cortex/export/save_views.py has a weaker inline copy of this loop --
    it checks existence only, so it accepts a file the browser has created but
    not finished writing. Consolidating means promoting this into cortex/export/
    (the library cannot import from cortex/tests/), not deleting either copy.

    If that happens, keep the ``time.sleep(1)`` that follows that loop. It reads
    as slack on the file wait but is not: it is the window in which event
    polling delivers console messages, which the WebGL failure check on the next
    line depends on. Re-label it rather than removing it.
    """
    import os
    import time

    for _ in range(int(timeout / 0.1)):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        time.sleep(0.1)
    raise RuntimeError(f"File {path!r} not written within {timeout}s")


# --------------------------------------------------------------------------- #
# Driving a headless viewer (cortex.export.headless_viewer) without sleeps.   #
# Software WebGL makes every redraw slow, so fixed sleeps sized for the worst #
# case dominated the runtime of the tests that switch data or view state.     #
# --------------------------------------------------------------------------- #


def js_eval(handle, expr):
    """Evaluate a javascript expression in the viewer page; one roundtrip."""
    result = handle.send(method="run", params=["window.eval", [expr]])
    return result[0] if isinstance(result, list) and result else result


def wait_js(handle, expr, what, timeout=60.0):
    """Poll until the javascript expression `expr` evaluates to ``true``."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if js_eval(handle, expr) is True:
            return
        time.sleep(0.05)
    raise RuntimeError("timed out after %.0fs waiting for %s" % (timeout, what))


def settle(handle):
    """Wait until the viewer has drawn at least one frame since this call.

    ``getImage`` renders synchronously, but some state only takes effect in
    the viewer's own draw loop (e.g. camera changes reach the camera through
    ``controls.update``), so a change is complete once a frame was drawn.
    """
    js_eval(handle, "window._settled = false; requestAnimationFrame(function() "
                    "{ requestAnimationFrame(function() { window._settled = true; }); })")
    wait_js(handle, "window._settled === true", "a redraw")


def wait_active(handle, name):
    """Wait until dataview `name` is shown with all of its data loaded.

    Call after ``setData``/``addData``: the dataview's ``loaded`` deferred
    resolves once its data arrived, and ``mriview.dataBuffersReady`` covers
    the tick in which its textures/vertex buffers are not populated yet.
    """
    wait_js(
        handle,
        "(function() { var v = window.viewer, d = v.dataviews[%r];"
        " return d !== undefined && v.active === d"
        " && d.loaded.state() === 'resolved'"
        " && mriview.dataBuffersReady(d.data); })()" % name,
        "dataview %s to load" % name,
    )
    settle(handle)


def set_view(handle, view, subject="S1"):
    """``handle._set_view(**view)`` in a single javascript task.

    Every ``ui.set`` redraws the viewer and under software WebGL a redraw
    costs about half a second; setting all keys in one task costs one. Only
    for keys ``_set_view`` accepts as they are (no legacy names). Unfolding
    goes first, as in ``_set_view``.
    """
    import json

    items = sorted(view.items(), key=lambda kv: not kv[0].endswith(".unfold"))
    js_eval(handle, "".join(
        "viewer.ui.set(%s, %s);"
        % (json.dumps(k.format(subject=subject)), json.dumps(v))
        for k, v in items
    ))
    settle(handle)


def render(handle, path, size=(512, 384), timeout=30):
    """``handle.getImage(path, size)`` and wait until the PNG is complete.

    The server writes the posted image in place, so a file that merely exists
    may still be partial: wait until it decodes instead.
    """
    import time

    from PIL import Image

    handle.getImage(path, size)
    deadline = time.monotonic() + timeout
    while True:
        try:
            with Image.open(path) as im:
                im.load()
            return
        except (OSError, SyntaxError):
            if time.monotonic() > deadline:
                raise RuntimeError("image not written: %s" % path)
            time.sleep(0.05)


def page_errors(handle):
    """Uncaught javascript exceptions the viewer page raised so far.

    Browser events reach Python only on the headless worker's next poll, so
    first wait out two poll intervals.
    """
    import time

    from cortex.export.headless import EVENT_POLL_INTERVAL

    time.sleep(2 * EVENT_POLL_INTERVAL)
    return [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]
