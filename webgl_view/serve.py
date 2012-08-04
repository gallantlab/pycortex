import os
import json
import struct
import socket
import binascii
import mimetypes
import multiprocessing as mp

import numpy as np
import tornado.web
import tornado.ioloop
from tornado import websocket

cwd = os.path.split(os.path.abspath(__file__))[0]
hostname = socket.gethostname()

dtypemap = {
    np.float: "float32",
    np.int: "int32",
    np.int32: "int32",
    np.float32: "float32",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.int16: "int16",
    np.int8:"int8",
}

class NPEncode(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float:
                obj = obj.astype(np.float32)
            elif obj.dtype == np.int:
                obj = obj.astype(np.int32)

            return dict(
                __class__="NParray",
                dtype=dtypemap[obj.dtype.type], 
                shape=obj.shape, 
                data=binascii.b2a_base64(obj.tostring()))
        else:
            return super(NPEncode, self).default(obj)

class MainHandler(tornado.web.RequestHandler):
    def get(self, path):
        fpath = os.path.join(cwd, path)
        if path == '':
            self.write(open(os.path.join(cwd, "index.html")).read())
        elif os.path.exists(fpath):
            mtype = mimetypes.guess_type(fpath)[0]
            if mtype is None:
                mtype = "application/octet-stream"
            self.set_header("Content-Type", mtype)
            self.write(open(fpath).read())
        else:
            print "could not find %s"%fpath
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
        self.parent.clients.value -= 1
        if self.parent.clients.value == 0:
            self.parent.stop()

    def on_message(self, message):
        if (message == "connect"):
            self.parent.clients.value += 1
            self.parent.c_evt.set()
        else:
            self.parent._response.send(message)

class WebApp(mp.Process):
    def __init__(self, handlers, port):
        super(WebApp, self).__init__()
        self.handlers = handlers + [
            (r"/wsconnect/", ClientSocket, dict(parent=self)),
            (r"/(.*)", MainHandler),
        ]
        self._pipe, self.pipe = os.pipe()
        self._response, self.response = mp.Pipe()
        self.port = port
        self.clients = mp.Value('i', 0)
        self.c_evt = mp.Event()

    def run(self):
        self.sockets = []
        application = tornado.web.Application(self.handlers, gzip=True)
        application.listen(self.port)
        self.ioloop = tornado.ioloop.IOLoop.instance()
        self.ioloop.add_handler(self._pipe, self._send, self.ioloop.READ)
        self.ioloop.start()

    def stop(self):
        print "Stopping server"
        self.ioloop.stop()

    def _send(self, fd, event):
        nbytes, = struct.unpack('I', os.read(fd, 4))
        msg = os.read(fd, nbytes)
        if msg == "stop":
            self.ioloop.stop()
        else:
            for sock in self.sockets:
                sock.write_message(msg)

    def send(self, **msg):
        if not isinstance(msg, (str, unicode)):
            msg = json.dumps(msg, cls=NPEncode)

        os.write(self.pipe, struct.pack('I', len(msg))+msg)        
        return [json.loads(self.response.recv()) for _ in range(self.clients.value)]

    def get_client(self):
        self.c_evt.wait()
        self.c_evt.clear()
        return JSProxy(self.send)

class JSProxy(object):
    def __init__(self, sendfunc, name="window"):
        self.send = sendfunc
        self.name = name
        self.attrs = set(self.send(method='query', params=[name])[0])
    
    def __getattr__(self, attr):
        assert attr in self.attrs
        return JSProxy(self.send, "%s.%s"%(self.name, attr))

    def __repr__(self):
        return "<JS: %s>"%self.name

    def __dir__(self):
        return list(self.attrs)

    def __call__(self, *args):
        resp = self.send(method='run', params=[self.name, args])
        if isinstance(resp[0], dict) and "error" in resp:
            raise Exception(resp[0]['error'])
        else:
            return resp

    def __getitem__(self, idx):
        assert not isinstance(idx, (slice, list, tuple, np.ndarray))
        resp = self.send(method='index', params=[self.name, args])
        if isinstance(resp[0], dict) and "error" in resp:
            raise Exception(resp[0]['error'])
        else:
            return resp 
