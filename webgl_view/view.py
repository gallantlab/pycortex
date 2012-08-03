import os
import json
import random
import mimetypes
import webbrowser
import multiprocessing as mp
import numpy as np

from tornado import web, template

from .. import utils

import serve

loader = template.Loader(serve.cwd, autoescape=None)
html = loader.load("mixer.html")

#Raw volume movie:      [[3, 4], t, z, y, x]
#Raw volume image:      [[3, 4], z, y, x]
#Raw cortical movie:    [[3, 4], t, vox]
#Raw cortical image:    [[3, 4], vox]
#Regular volume movie:  [t, z, y, x]
#Regular volume image:  [z, y, x]
#Regular cortical movie: [t, vox]
#Regular cortical image: [vox]

def _normalize_data(data, mask):
    if not isinstance(data, dict):
        data = dict(data0=data)

    return dict([(name, _fixarray(array, mask)) for name, array in data.items()])

def _fixarray(data, mask):
    if isinstance(data, str):
        import nibabel
        data = nibabel.load(data).get_data().T

    raw = data.dtype.type == np.uint8
    assert raw and data.shape[0] in [3, 4] or not raw
    shape = data.shape[1:] if raw else data.shape
    movie = len(shape) in [2, 4]
    volume = len(shape) in [3, 4]

    if raw:
        if volume:
            if movie:
                return data[:, :, mask].transpose(1, 2, 0)
            else:
                return data[:, mask].T
        else: #cortical
            if movie:
                return data.transpose(1,2,0)
            else:
                return data.T
    else: #regular
        if volume:
            if movie:
                return data[:, mask]
            else:
                return data[mask]
        else: #cortical
            return data

def make_movie(stimidx, outfile):
    pass

def show(data, subject, xfmname, types=("inflated",), recache=False):
    ctmfile = utils.get_ctmpack(subject, xfmname, types, method='raw', level=0, recache=recache)
    mask = utils.get_cortical_mask(subject, xfmname)
    jsondat = json.dumps(_normalize_data(data, mask), cls=serve.NPEncode)
    savesvg = mp.Array('c', 8192)

    class CTMHandler(web.RequestHandler):
        def get(self, path):
            fpath = os.path.split(ctmfile)[0]
            if path == '':
                self.set_header("Content-Type", "application/json")
                self.write(open(ctmfile).read())
            else:
                mtype = mimetypes.guess_type(os.path.join(fpath, path))[0]
                if mtype is None:
                    mtype = "application/octet-stream"
                self.set_header("Content-Type", mtype)
                self.write(open(os.path.join(fpath, path)).read())

    class MixerHandler(web.RequestHandler):
        def get(self):
            self.set_header("Content-Type", "text/html")
            self.write(html.generate(data=jsondat))

        def post(self):
            print "saving file to %s"%savesvg.value
            with open(savesvg.value, "w") as svgfile:
                svgfile.write(self.get_argument("svg"))

    class JSMixer(serve.JSProxy):
        def addData(self, **kwargs):
            Proxy = serve.JSProxy(self.send, "window.viewer.addData")
            return Proxy(_normalize_data(kwargs, mask))

        def saveflat(self, filename, height=1024):
            Proxy = serve.JSProxy(self.send, "window.viewer.saveflat")
            savesvg.value = filename
            return Proxy(height, "mixer.html")

    class WebApp(serve.WebApp):
        def get_client(self):
            self.c_evt.wait()
            self.c_evt.clear()
            return JSMixer(self.send, "window.viewer")

    server = WebApp([
            (r'/ctm/(.*)', CTMHandler),
            (r'/mixer.html', MixerHandler),
        ], 
        random.randint(1024, 65536))
    server.start()

    webbrowser.open("http://%s:%d/mixer.html"%(serve.hostname, server.port))

    client = server.get_client()
    client.server = server
    return client