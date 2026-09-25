"""Browser-free tests of the Python end of the viewer's websocket RPC."""

import json
from typing import Any, Callable, Optional

import pytest

from cortex.webgl import serve


class _FakeIOLoop:
    """Stands in for the tornado IOLoop; ``answer`` plays the browser."""

    def __init__(self, answer: Callable[[dict], None]):
        self.answer = answer

    def add_callback(self, _send: Any, _sockets: Any, msg: str) -> None:
        self.answer(json.loads(msg))


@pytest.fixture
def app():
    app = serve.WebApp([], 0)
    # One connected client; send() only reads len(sockets).
    app.sockets = [object()]  # type: ignore[list-item]
    yield app
    for sock in app._sockets:
        sock.close()


def test_late_response_is_not_read_as_the_next_one(app):
    """An answer that arrives after send() gave up must not answer the next call.

    Before requests were tagged, it did -- and every call after it was then
    answered one behind for the rest of the session (the gh-695 Volume2D flake).
    """
    late: list[dict] = []

    def answer(request: dict) -> None:
        if request["params"] == ["slow"]:
            late.append(request)  # the browser is busy; answers later
            return
        for req in late:
            app.response.put(json.dumps({"id": req["id"], "result": "stale"}))
        late.clear()
        app.response.put(json.dumps({"id": request["id"], "result": "fresh"}))

    app.ioloop = _FakeIOLoop(answer)
    assert app.send(method="query", params=["slow"]) == [None]
    assert app.send(method="query", params=["fast"]) == ["fresh"]
    assert app.response.empty()


def test_getattr_retries_after_a_timed_out_query():
    """A query that timed out (None) is retried, not iterated over."""
    attrs = {"layers": ["number", 32]}
    answers: list[Optional[dict]] = []

    def send(*, method: str, params: list) -> list:
        return [answers.pop(0) if answers else attrs]

    proxy = serve.JSProxy(send, "window.viewer")
    answers.extend([None, None])  # attrs itself retries once
    assert proxy.layers == 32
