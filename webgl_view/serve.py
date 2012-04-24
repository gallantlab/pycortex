import os
import tornado.ioloop
import tornado.web
import struct
import mimetypes
import zlib

import sys
cwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(cwd, ".."))

import numpy as np

import db

def get_binary_pts(subj, types, hemi):
    types = ("fiducial",) + types + ("flat",)
    pts = []
    for t in types:
        pt, polys, norm = db.surfs.getVTK(subj, t, hemisphere=hemi)
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
    def get(self, subj, hemi):
        self.set_header("Content-Type", "text/plain")
        data = get_binary_pts(subj, ("inflated",), hemi or "both")
        self.write(data)

    def post(self, subj, hemi):
        self.set_header("Content-Type", "text/plain")
        types = self.get_argument("types").split(",")
        print "loading %r"%types
        data = get_binary_pts(subj, tuple(types), hemi or "both")
        self.write(data)

application = tornado.web.Application([
    (r"/surfaces/(\w+)/?(\w+)?/?", BinarySurface),
    (r"/(.*)", MainHandler),
], gzip=True)

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()