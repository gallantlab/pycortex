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
