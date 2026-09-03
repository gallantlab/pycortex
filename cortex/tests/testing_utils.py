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
