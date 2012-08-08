import os
import re
import glob
import json
import shutil
import binascii
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
static_template = loader.load("static_template.html")
try:
    colormaps
except NameError:
    name_parse = re.compile(r".*/(\w+).png")
    colormaps = []
    for cmap in sorted(glob.glob(os.path.join(serve.cwd, "resources/colormaps/*.png"))):
        name = name_parse.match(cmap).group(1)
        data = binascii.b2a_base64(open(cmap).read()).strip()
        colormaps.append((name, "data:image/png;base64,"+data))

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

def _embed_css(cssfile):
    csspath = os.path.split(cssfile)
    with open(cssfile) as fp:
        css = fp.read()
        cssparse = re.compile(r'(.*?){(.*?)}', re.S)
        urlparse = re.compile(r'url\((.*?)\)')
        for selector, content in cssparse.findall(css):
            for line in content.split(';'):
                attr, val = line.strip().split(':')
                url = urlparse.search(val)
                if url:
                    imgfile = os.path.join(csspath, url.group(1))
                    with open(imgfile) as img:
                        imgdat = "data:image/png;base64,"+binascii.b2a_base64(img.read()).strip()

def make_movie(stim, outfile, fps=15, size="640x480"):
    import shlex
    import subprocess as sp
    cmd = "ffmpeg -r {fps} -i {infile} -b 4800k -g 30 -s {size} -vcodec libtheora {outfile}.ogv"
    fcmd = cmd.format(infile=stim, size=size, fps=fps, outfile=outfile)
    sp.call(shlex.split(fcmd))

def make_static(outpath, data, subject, xfmname, stimmovie=None, cmap="RdBu_r"):
    import html5lib
    print "You'll probably need nginx to view this, since file:// paths don't handle xsrf correctly"
    outpath = os.path.abspath(outpath) # To handle ~ expansion

    #Create a new mg2 compressed CTM and move it into the outpath
    ctmfile = utils.get_ctmpack(subject, xfmname, types, method='mg2', level=9, recache=True)
    fname, ext = os.path.splitext(ctmfile)
    for ext in ['json','ctm', 'svg']:
        shutil.move("%s.%s"%(fname, ext), outpath)
    ctmfile = os.path.split(ctmfile)[1]

    #Generate the data binary objects and save them into the outpath
    mask = utils.get_cortical_mask(subject, xfmname)
    jsondat = dict()
    for name, array in _normalize_data(data, mask).items():
        with open(os.path.join(outpath, "%s.bin"%name), "w") as binfile:
            binfile.write(serve.make_bindat(array))
        jsondat[name] = name

    #Parse the html file and paste all the js and css files directly into the html
    with open(os.path.join(outpath, "index.html"), "w") as htmlfile:
        html = static_template.generate(ctmfile=ctmfile, data=jsondat, colormaps=colormaps, default_cmap=cmap)
        parser = html5lib.HTMLParser(tree=html5lib.treebuilders.getTreeBuilder("dom"))
        dom = parser.parse(html)
        for script in dom.getElementsByTagName("script"):
            src = script.getAttribute("src")
            if len(src) > 0:
                with open(os.path.join(serve.cwd, src)) as jsfile:
                    stext = dom.createTextNode(jsfile.read())
                    script.removeAttribute("src")
                    script.appendChild(stext)

        
        for css in dom.getElementsByTagName("link"):
            if (css.getAttribute("type") == "text/css"):
                href = css.getAttribute("href")
                csspath, cssfile = os.path.split(os.path.join(serve.cwd, href))
                with open(os.path.join(serve.cwd, href)) as cssfile:
                    ncss = dom.createElement("style")
                    ncss.setAttribute("type", "text/css")
                    ncss.appendChild(dom.createTextNode())
                    css.parentNode.insertBefore(ncss, css)
                    css.parentNode.removeChild(css)




def show(data, subject, xfmname, types=("inflated",), recache=False, cmap="RdBu_r"):
    ctmfile = utils.get_ctmpack(subject, xfmname, types, method='raw', level=0, recache=recache)
    mask = utils.get_cortical_mask(subject, xfmname)
    data = _normalize_data(data, mask)
    bindat = dict([(name, serve.make_bindat(array)) for name, array in data.items()])
    jsondat = dict([(name, "data/%s/"%name) for name in data.keys()])

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

    class DataHandler(web.RequestHandler):
        def get(self, path):
            path = path.strip("/")
            if path in bindat:
                self.write(bindat[path])
            else:
                self.set_status(404)
                self.write_error(404)

    class MixerHandler(web.RequestHandler):
        def get(self):
            self.set_header("Content-Type", "text/html")
            self.write(html.generate(data=jsondat, colormaps=colormaps, default_cmap=cmap))

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