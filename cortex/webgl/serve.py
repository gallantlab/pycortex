import os
import re
import time
import json
import stat
import email
import Queue
import struct
import socket
import logging
import binascii
import datetime
import mimetypes
import threading

import numpy as np
import tornado.web
import tornado.ioloop
import tornado.httpserver
from tornado import websocket
from tornado.web import HTTPError

cwd = os.path.split(os.path.abspath(__file__))[0]
hostname = socket.gethostname()

def make_base64(imgfile):
    with open(imgfile) as img:
        mtype = mimetypes.guess_type(imgfile)[0]
        data = binascii.b2a_base64(img.read()).strip()
        return "data:{mtype};base64,{data}".format(mtype=mtype, data=data)

class NPEncode(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float:
                obj = obj.astype(np.float32)
            elif obj.dtype == np.int:
                obj = obj.astype(np.int32)

            return dict(
                __class__="NParray",
                dtype=obj.dtype.descr[0][1], 
                shape=obj.shape, 
                data=binascii.b2a_base64(obj.tostring()))
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return super(NPEncode, self).default(obj)

class ClientSocket(websocket.WebSocketHandler):
    def initialize(self, parent):
        self.parent = parent

    def open(self):
        self.parent.sockets.append(self)

    def on_close(self):
        self.parent.sockets.remove(self)
        if self.parent.n_clients == 0 and self.parent.disconnect_on_close:
            self.parent.stop()

    def on_message(self, message):
        if (message == "connect"):
            self.parent.connect.set()
        else:
            self.parent.response.put(message)

class WebApp(threading.Thread):
    daemon = True
    disconnect_on_close = True

    def __init__(self, handlers, port):
        super(WebApp, self).__init__()
        self.handlers = handlers + [
            (r"/wsconnect/", ClientSocket, dict(parent=self)),
            (r"/(.*)", tornado.web.StaticFileHandler, dict(path=cwd)),
        ]
        self.port = port
        self.response = Queue.Queue()
        self.connect = threading.Event()
        self.sockets = []

    @property
    def n_clients(self):
        num = len(self.sockets)
        return num

    def run(self):
        ioloop = tornado.ioloop.IOLoop()
        ioloop.clear_current()
        ioloop.make_current()
        application = tornado.web.Application(self.handlers, gzip=True)
        self.server = tornado.httpserver.HTTPServer(application, io_loop=ioloop)
        self.server.listen(self.port)
        ioloop.start()

    def stop(self):
        print("Stopping server")
        self.server.stop()
        tornado.ioloop.IOLoop.current().stop()

    def send(self, **msg):
        if not isinstance(msg, str):
            msg = json.dumps(msg, cls=NPEncode)

        for sock in self.sockets:
            sock.write_message(msg)
        return [json.loads(self.response.get(timeout=2)) for _ in range(self.n_clients)]

    def get_client(self):
        self.connect.wait()
        self.connect.clear()
        return JSProxy(self.send)

class JSProxy(object):
    def __init__(self, sendfunc, name="window"):
        self.send = sendfunc
        self.name = name
        self.attrs = self.send(method='query', params=[self.name])[0]
    
    def __getattr__(self, attr):
        assert attr in self.attrs
        if self.attrs[attr][0] in ["object", "function"]:
            return JSProxy(self.send, "%s.%s"%(self.name, attr))
        else:
            return self.attrs[attr][1]

    def __setattr__(self, attr, value):
        if not hasattr(self, "attrs") or attr not in self.attrs:
            return super(JSProxy, self).__setattr__(attr, value)

        assert self.attrs[attr] not in ["object", "function"]
        resp = self.send(method='set', params=["%s.%s"%(self.name, attr), value])
        if isinstance(resp[0], dict) and "error" in resp[0]:
            raise Exception(resp[0]['error'])
        else:
            return resp

    def __repr__(self):
        return "<JS: %s>"%self.name

    def __dir__(self):
        return list(self.attrs)

    def __call__(self, *args):
        resp = self.send(method='run', params=[self.name, args])
        if isinstance(resp[0], dict) and "error" in resp[0]:
            raise Exception(resp[0]['error'])
        else:
            return resp

    def __getitem__(self, idx):
        assert not isinstance(idx, (slice, list, tuple, np.ndarray))
        resp = self.send(method='index', params=[self.name, args])
        if isinstance(resp[0], dict) and "error" in resp[0]:
            raise Exception(resp[0]['error'])
        else:
            return resp

if __name__ == "__main__":
    app = WebApp([], 8888)
    app.start()
    app.join()
