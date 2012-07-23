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
        else:
            print "Found in cache: %r"%(args,)
        return cache[args]
    return mfunc

@memoize
def get_binary_pts(subj, surftype, getPolys=False, compress=True):
    print "Retrieving suject %r, type %r, polys=%r"%(subj, surftype, getPolys)
    left, right = db.surfs.getVTK(subj, surftype, hemisphere="both", 
        merge=False, nudge=surftype != "fiducial")
    data = ""
    for pts, polys, norms in left, right:
        if not getPolys:
            polys = np.array([])
        minmax = pts.min(0).tolist() + (pts.max(0) - pts.min(0)).tolist()
        header = struct.pack('3I6f', compress, len(pts), len(polys), *minmax)
        if compress:
            pts -= pts.min(0)
            pts /= pts.max(0)
            pts *= np.iinfo(np.uint16).max
            pts = pts.astype(np.uint16)
        data += header+pts.tostring()+polys.astype(np.uint32).tostring()
    return data

class BinarySurface(tornado.web.RequestHandler):
    def get(self, subj, surftype):
        self.set_header("Content-Type", "application/octet-stream")
        self.write(get_binary_pts(subj, surftype, self.get_argument("polys")))

    def post(self, subj, hemi):
        self.set_header("Content-Type", "application/octet-stream")
        types = self.get_argument("types").split(",")
        print "loading %r"%types
        data = get_binary_pts(subj, tuple(types))
        self.write(data)


def embedData(*args):
    assert all([len(a) == len(args[0]) for a in args])
    assert all([a.dtype == args[0].dtype for a in args])
    shape = (np.ceil(len(args[0]) / 256.), 256)
    outstr = "";
    for data in args:
        if (data.dtype == np.uint8):
            mm = 0,0
            outmap = np.zeros((np.prod(shape), 4), dtype=np.uint8)
            outmap[:,-1] = 255
            outmap[:len(data),:data.shape[1]] = data
        else:
            outmap = np.zeros(shape, dtype=np.float32)
            mm = data.min(), data.max()
            outmap.ravel()[:len(data)] = (data - data.min()) / (data.max() - data.min())

        outstr += struct.pack('2f', mm[0], mm[1])+outmap.tostring()
        
    return struct.pack('3I', len(args), shape[1], shape[0])+outstr


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
