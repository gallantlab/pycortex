import os
import re
import glob
import json
import shutil
import random
import mimetypes
import webbrowser
import multiprocessing as mp
import numpy as np
from scipy.stats import scoreatpercentile

from tornado import web, template

from .. import utils

import serve

loader = template.Loader(serve.cwd)
html = loader.load("mixer.html")

name_parse = re.compile(r".*/(\w+).png")
colormaps = glob.glob(os.path.join(serve.cwd, "resources/colormaps/*.png"))
colormaps = [(name_parse.match(cm).group(1), serve.make_base64(cm)) for cm in sorted(colormaps)]

def _normalize_data(data, mask):
    if not isinstance(data, dict):
        data = dict(data0=data)

    json = dict()
    for name, dat in data.items():
        json[name] = dict( __class__="Dataset")
        if isinstance(dat, dict):
            data = _fixarray(dat['data'], mask)
            json[name]['stim'] = dat['stim']
            json[name]['delay'] = dat['delay'] if 'delay' in dat else 0
        elif isinstance(dat, np.ndarray):
            data = _fixarray(dat, mask)

        json[name]['data'] = data
        json[name]['min'] = scoreatpercentile(data.ravel(), 1)
        json[name]['max'] = scoreatpercentile(data.ravel(), 99)

    return json

def _make_bindat(json, fmt="%s.bin"):
    newjs, bindat = dict(), dict()
    for name, data in json.items():
        newjs[name] = dict(data)
        newjs[name]['data'] = fmt%name
        bindat[name] = serve.make_binarray(data['data'])

    return newjs, bindat

def _fixarray(data, mask):
    if isinstance(data, str):
        import nibabel
        data = nibabel.load(data).get_data().T
    if data.dtype == np.float:
        data = data.astype(np.float32)

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

def make_movie(stim, outfile, fps=15, size="640x480"):
    import shlex
    import subprocess as sp
    cmd = "ffmpeg -r {fps} -i {infile} -b 4800k -g 30 -s {size} -vcodec libtheora {outfile}.ogv"
    fcmd = cmd.format(infile=stim, size=size, fps=fps, outfile=outfile)
    sp.call(shlex.split(fcmd))

def make_static(outpath, data, subject, xfmname, types=("inflated",), recache=False, cmap="RdBu_r", template="static.html"):
    print "You'll probably need nginx to view this, since file:// paths don't handle xsrf correctly"
    outpath = os.path.abspath(os.path.expanduser(outpath)) # To handle ~ expansion
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    #Create a new mg2 compressed CTM and move it into the outpath
    ctmfile = utils.get_ctmpack(subject, xfmname, types, method='mg2', level=9, recache=recache)
    oldpath, fname = os.path.split(ctmfile)
    fname, ext = os.path.splitext(fname)
    for ext in ['json','ctm', 'svg']:
        newfile = os.path.join(outpath, "%s.%s"%(fname, ext))
        if not os.path.exists(newfile):
            print "copying %s file"%ext
            shutil.copy2(os.path.join(oldpath, "%s.%s"%(fname, ext)), newfile)
        else:
            print "%s file already exists"%ext
    ctmfile = os.path.split(ctmfile)[1]

    #Generate the data binary objects and save them into the outpath
    mask = utils.get_cortical_mask(subject, xfmname)
    json, sdat = _make_bindat(_normalize_data(data, mask))
    for name, dat in sdat.items():
        with open(os.path.join(outpath, "%s.bin"%name), "w") as binfile:
            binfile.write(dat)
    
    #Parse the html file and paste all the js and css files directly into the html
    import htmlembed
    template = loader.load(template)
    html = template.generate(ctmfile=ctmfile, data=json, colormaps=colormaps, default_cmap=cmap, python_interface=False)
    htmlembed.embed(html, os.path.join(outpath, "index.html"))


def show(data, subject, xfmname, types=("inflated",), recache=False, cmap="RdBu_r"):
    '''Data can be a dictionary of arrays. Alternatively, the dictionary can also contain a 
    sub dictionary with keys [data, stim, delay].

    Data array can be a variety of shapes:
    Raw volume movie:      [[3, 4], t, z, y, x]
    Raw volume image:      [[3, 4], z, y, x]
    Raw cortical movie:    [[3, 4], t, vox]
    Raw cortical image:    [[3, 4], vox]
    Regular volume movie:  [t, z, y, x]
    Regular volume image:  [z, y, x]
    Regular cortical movie: [t, vox]
    Regular cortical image: [vox]
    '''
    ctmfile = utils.get_ctmpack(subject, xfmname, types, method='raw', level=0, recache=recache)
    mask = utils.get_cortical_mask(subject, xfmname)
    jsondat, bindat = _make_bindat(_normalize_data(data, mask), fmt='data/%s/')

    savesvg = mp.Array('c', 8192)
    queue = mp.Queue()
    
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

    class DataHandler(web.RequestHandler):
        def get(self, path):
            path = path.strip("/")
            if not queue.empty():
                d = queue.get()
                print "Got new data: %r"%d.keys()
                bindat.update(d)

            if path in bindat:
                self.write(bindat[path])
            else:
                self.set_status(404)
                self.write_error(404)

    class MixerHandler(web.RequestHandler):
        def get(self):
            self.set_header("Content-Type", "text/html")
            self.write(html.generate(data=jsondat, colormaps=colormaps, default_cmap=cmap, python_interface=True))

        def post(self):
            print "saving file to %s"%savesvg.value
            with open(savesvg.value, "w") as svgfile:
                svgfile.write(self.get_argument("svg"))

    class JSMixer(serve.JSProxy):
        def addData(self, **kwargs):
            Proxy = serve.JSProxy(self.send, "window.viewer.addData")
            json, data = _make_bindat(_normalize_data(kwargs, mask), fmt='data/%s/')
            queue.put(data)
            return Proxy(json)

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
            (r'/data/(.*)', DataHandler),
            (r'/mixer.html', MixerHandler),
        ], 
        random.randint(1024, 65536))
    server.start()
    print "Started server on port %d"%server.port
    webbrowser.open("http://%s:%d/mixer.html"%(serve.hostname, server.port))

    client = server.get_client()
    client.server = server
    return client