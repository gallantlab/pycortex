import os
import struct
import mimetypes
import multiprocessing as mp

import sys
cwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(cwd, ".."))

import numpy as np
import tornado.web
import tornado.ioloop
from tornado import websocket

import db

def memoize(func):
    cache = {}
    def mfunc(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return mfunc

@memoize
def get_binary_pts(subj, types):
    types = ("fiducial",) + types + ("flat",)
    pts = []
    for t in types:
        pt, polys, norm = db.surfs.getVTK(subj, t, hemisphere="both", merge=True, nudge=True)
        pts.append(pt)

    #flip the flats to be on the X-Z plane
    flatpts = np.zeros_like(pts[-1])
    flatpts[:,[0,2]] = pts[-1][:,:2]
    flatpts[:,1] = pts[-2].min(0)[1]
    pts[-1] = flatpts*.66

    header = struct.pack('2I', len(types), pts[0].size)
    ptstr = ''.join([p.astype(np.float32).tostring() for p in pts])
    return header+ptstr+polys.astype(np.uint32).tostring()


class MainHandler(tornado.web.RequestHandler):
    def get(self, path):
        if path == '':
            self.write(open("mixer.html").read())
        elif os.path.isfile(path):
            self.set_header("Content-Type", mimetypes.guess_type(path)[0])
            self.write(open(path).read())
        else:
            self.write_error(404)

class BinarySurface(tornado.web.RequestHandler):
    def get(self, subj):
        self.set_header("Content-Type", "text/plain")
        data = get_binary_pts(subj, ("inflated",))
        self.write(data)

    def post(self, subj, hemi):
        self.set_header("Content-Type", "text/plain")
        types = self.get_argument("types").split(",")
        print "loading %r"%types
        data = get_binary_pts(subj, tuple(types))
        self.write(data)

class ClientSocket(websocket.WebSocketHandler):
    def initialize(self, sockets):
        self.sockets = sockets

    def open(self):
        self.sockets.append(self)
        print "WebSocket opened"

    def on_close(self):
        print "WebSocket closed"
        self.sockets.remove(self)


class WebApp(mp.Process):
    def __init__(self, port):
        super(WebApp, self).__init__()
        self._pipe, self.pipe = os.pipe()
        self.port = port

    def run(self):
        self.sockets = []
        application = tornado.web.Application([
            (r"/surfaces/(\w+)/?(\w+)?/?", BinarySurface),
            (r"/wsconnect/", ClientSocket, dict(sockets=self.sockets)),
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

if __name__ == "__main__":
    proc = WebApp(8888)
    proc.start()
    proc.join()