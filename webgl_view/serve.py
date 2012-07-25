import os
import struct
import mimetypes
import multiprocessing as mp

import numpy as np
import tornado.web
import tornado.ioloop
from tornado import websocket

import db

msgid = ["raw", "__call__"]

class MainHandler(tornado.web.RequestHandler):
    def get(self, path):
        if path == '':
            self.write(open("index.html").read())
        elif os.path.isfile(path):
            mtype = mimetypes.guess_type(path)[0]
            if mtype is None:
                mtype = "application/octet-stream"
            self.set_header("Content-Type", mtype)
            self.write(open(path).read())
        else:
            self.write_error(404)

class ClientSocket(websocket.WebSocketHandler):
    def initialize(self, parent):
        self.parent = parent

    def open(self):
        self.parent.sockets.append(self)
        print "WebSocket opened"

    def on_close(self):
        print "WebSocket closed"
        self.parent.sockets.remove(self)

    def on_message(self, message):
        self.parent._pipe.send(message)

class WebApp(mp.Process):
    def __init__(self,  port):
        super(WebApp, self).__init__()
        self._pipe, self.pipe = os.pipe()
        self._clients, self.clients = mp.Pipe()
        self.port = port

    def run(self):
        self.sockets = []
        application = tornado.web.Application([
            (r"/wsconnect/", ClientSocket, dict(parent=self)),
            (r"/(.*)", MainHandler),
        ], gzip=True)
        application.listen(self.port)
        self.ioloop = tornado.ioloop.IOLoop.instance()
        self.ioloop.add_handler(self._pipe, self._send, self.ioloop.READ)
        self.ioloop.start()

    def _send(self, fd, event):
        nbytes, = struct.unpack('I', os.read(fd, 4))
        msg = os.read(fd, nbytes)
        if msg == "stop":
            self.ioloop.stop()
        else:
            for sock in self.sockets:
                sock.write_message(msg)

    def send(self, msg):
        if not isinstance(msg, (str, unicode)):
            msg = json.dumps(msg)

        os.write(self.pipe, struct.pack('I', len(msg))+msg)

    def get_client(self):
        msg = self.clients.recv()
        return JSProxy(self, json.loads(msg))

class JSProxy(object):
    def __init__(self, server, methods):
        self.server = server
        self.methods = dict([(name, FuncProxy(server.send, name)) for name in methods])
    
    def __getattr__(self, attr):
        return self.methods[attr]

    def run(self, jseval):
        msgtype = dict([(m, i) for i, m in enumerate(msgid)])
        msg = struct.pack('I', msgtype['raw'])+jseval
        self.server.send(msg)

class FuncProxy(object):
    def __init__(self, _send, _recv, name):
        self._send = _send
        self.name = name

    def __call__(self, *args):
        msgtype = dict([(m, i) for i, m in enumerate(msgid)])
        jsdat = json.dumps(dict(name=name, args=args))
        msg = struct.pack('I', msgtype['__call__'])+jsdat
        self._send(msg)

if __name__ == "__main__":
    proc = WebApp(8888)
    proc.start()
    proc.join()
