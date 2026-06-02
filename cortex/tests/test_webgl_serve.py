import cortex.webgl.serve as serve


def test_webapp_run_uses_current_ioloop(monkeypatch):
    class FakeIOLoop:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

    class FakeHTTPServer:
        def __init__(self, application, io_loop=None):
            self.application = application
            self.io_loop = io_loop
            self.listen_port = None

        def listen(self, port):
            self.listen_port = port

    fake_ioloop = FakeIOLoop()

    monkeypatch.setattr(serve.tornado.ioloop.IOLoop, "current", lambda: fake_ioloop)
    monkeypatch.setattr(serve.tornado.httpserver, "HTTPServer", FakeHTTPServer)

    app = serve.WebApp([], 0)
    app.run()

    assert app.ioloop is fake_ioloop
    assert fake_ioloop.started
    assert app.server.listen_port == 0
