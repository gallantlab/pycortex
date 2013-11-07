import os
import re
import glob
import json
import shutil
import random
import functools
import binascii
import mimetypes
import webbrowser
import multiprocessing as mp
import numpy as np

from tornado import web
from .FallbackLoader import FallbackLoader

from .. import utils, options, volume, dataset
from ..db import surfs

from . import serve
from .data import Package

name_parse = re.compile(r".*/(\w+).png")
try:
    cmapdir = options.config.get('webgl', 'colormaps')
except:
    cmapdir = os.path.join(options.config.get("basic", "filestore"), "colormaps")
colormaps = glob.glob(os.path.join(cmapdir, "*.png"))
colormaps = [(name_parse.match(cm).group(1), serve.make_base64(cm)) for cm in sorted(colormaps)]

viewopts = dict(voxlines="false", voxline_color="#FFFFFF", voxline_width='.01' )

def make_static(outpath, data, types=("inflated",), recache=False, cmap="RdBu_r", template="static.html", layout=None, anonymize=False, **kwargs):
    """Creates a static instance of the webGL MRI viewer that can easily be posted 
    or shared. 

    Parameters
    ----------
    outpath : string
        The directory where the static viewer will be saved. Will be created if it
        doesn't already exist.
    data : Dataset object
        Dataset object containing all the data you wish to plot
    types : tuple, optional
        Types of surfaces to include. Fiducial and flat surfaces are automatically
        included. Default ("inflated",)
    recache : bool, optional
        Whether to recreate CTM and SVG files for surfaces. Default False
    cmap : string, optional
        Name of default colormap used to show data. Default "RdBu_r"
    template : string, optional
        Name of template HTML file. Default "static.html"
    anonymize : bool, optional
        Whether to rename CTM and SVG files generically, for public distribution.
        Default False
    **kwargs : dict, optional
        All additional keyword arguments are passed to the template renderer.

    You'll probably need nginx to view this, since file:// paths don't handle xsrf correctly
    """
    outpath = os.path.abspath(os.path.expanduser(outpath)) # To handle ~ expansion
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        os.makedirs(os.path.join(outpath, "data"))

    data = dataset.normalize(data)
    if not isinstance(data, dataset.Dataset):
        data = dataset.Dataset(data=data)

    surfs.auxfile = data

    package = Package(data)
    subjects = list(package.subjects)

    ctmargs = dict(method='mg2', level=9, recache=recache)
    ctms = dict((subj, utils.get_ctmpack(subj, types, **ctmargs)) for subj in subjects)
    surfs.auxfile = None

    if layout is None:
        layout = [None, (1,1), (2,1), (3,1), (2,2), (3,2), (3,2), (3,3), (3,3), (3,3)][len(subjects)]

    ## Rename files to anonymize?
    submap = dict()
    for i, (subj, ctmfile) in enumerate(ctms.items()):
        oldpath, fname = os.path.split(ctmfile)
        fname, ext = os.path.splitext(fname)
        if anonymize:
            newfname = "S%d"%i
            submap[subj] = newfname
        else:
            newfname = fname
        ctms[subj] = newfname+".json"

        for ext in ['json','ctm', 'svg']:
            newfile = os.path.join(outpath, "%s.%s"%(newfname, ext))
            if os.path.exists(newfile):
                os.unlink(newfile)
            
            shutil.copy2(os.path.join(oldpath, "%s.%s"%(fname, ext)), newfile)

            if ext == "json" and anonymize:
                ## change filenames in json
                nfh = open(newfile)
                jsoncontents = nfh.read()
                nfh.close()
                
                ofh = open(newfile, "w")
                ofh.write(jsoncontents.replace(fname, newfname))
                ofh.close()

    if len(submap) == 0:
        submap = None

    #Process the data
    metadata = package.metadata(fmt="data/{name}_{frame}.png")
    images = package.images
    #Write out the PNGs
    for name, imgs in images.items():
        impath = os.path.join(outpath, "data", "{name}_{frame}.png")
        for i, img in enumerate(imgs):
            with open(impath.format(name=name, frame=i), "wb") as binfile:
                binfile.write(img)

    #Copy any stimulus files
    stimpath = os.path.join(outpath, "stim")
    for name, view in data:
        if 'stim' in view.attrs and os.path.exists(view.attrs['stim']):
            if not os.path.exists(stimpath):
                os.makedirs(stimpath)
            shutil.copy2(view.attrs['stim'], stimpath)
    
    #Parse the html file and paste all the js and css files directly into the html
    from . import htmlembed
    if os.path.exists(template):
        ## Load locally
        templatedir, templatefile = os.path.split(os.path.abspath(template))
        rootdirs = [templatedir, serve.cwd]
    else:
        ## Load system templates
        templatefile = template
        rootdirs = [serve.cwd]
        
    loader = FallbackLoader(rootdirs)
    tpl = loader.load(templatefile)
    kwargs.update(viewopts)
    html = tpl.generate(
        data=json.dumps(metadata), 
        colormaps=colormaps, 
        default_cmap=cmap, 
        python_interface=False, 
        layout=layout,
        subjects=ctms,
        **kwargs)
    htmlembed.embed(html, os.path.join(outpath, "index.html"), rootdirs)
    surfs.auxfile = None

def show(data, types=("inflated",), recache=False, cmap='RdBu_r', layout=None, autoclose=True, open_browser=True, port=None, pickerfun=None, **kwargs):
    """Display a dynamic viewer using the given dataset

    Optional attributes that affect the display:
    cmap
    vmin / vmax
    filter: ['nearest', 'trilinear', 'nearlin']
    stim: a filename for the stimulus (preferably OGV)
    delay: time in seconds to delay the data with respect to stimulus
    rate: volumes per second
    """
    data = dataset.normalize(data)
    if not isinstance(data, dataset.Dataset):
        data = dataset.Dataset(data=data)

    html = FallbackLoader([serve.cwd]).load("mixer.html")
    surfs.auxfile = data

    package = Package(data)
    metadata = json.dumps(package.metadata())
    images = package.images
    subjects = list(package.subjects)
    #Extract the list of stimuli, for special-casing
    stims = dict()
    for name, view in data:
        if 'stim' in view.attrs and os.path.exists(view.attrs['stim']):
            sname = os.path.split(view.attrs['stim'])[1]
            stims[sname] = view.attrs['stim']
    
    kwargs.update(dict(method='mg2', level=9, recache=recache))
    ctms = dict((subj, utils.get_ctmpack(subj, types, **kwargs)) for subj in subjects)
    subjectjs = dict((subj, "/ctm/%s/"%subj) for subj in subjects)
    surfs.auxfile = None

    if layout is None:
        layout = [None, (1,1), (2,1), (3,1), (2,2), (3,2), (3,2), (3,3), (3,3), (3,3)][len(subjects)]

    saveevt = mp.Event()
    saveimg = mp.Array('c', 8192)
    queue = mp.Queue()

    linear = lambda x, y, m: (1.-m)*x + m*y
    mixes = dict(
        linear=linear,
        smoothstep=(lambda x, y, m: linear(x,y,3*m**2 - 2*m**3)), 
        smootherstep=(lambda x, y, m: linear(x, y, 6*m**5 - 15*m**4 + 10*m**3))
    )

    class CTMHandler(web.RequestHandler):
        def get(self, path):
            subj, path = path.split('/')
            if path == '':
                self.set_header("Content-Type", "application/json")
                self.write(open(ctms[subj]).read())
            else:
                fpath = os.path.split(ctms[subj])[0]
                mtype = mimetypes.guess_type(os.path.join(fpath, path))[0]
                if mtype is None:
                    mtype = "application/octet-stream"
                self.set_header("Content-Type", mtype)
                self.write(open(os.path.join(fpath, path)).read())

    class DataHandler(web.RequestHandler):
        def get(self, path):
            path = path.strip("/")
            try:
                d = queue.get(True, 0.1)
                print("Got new data: %r"%list(d.keys()))
                images.update(d)
            except:
                pass

            try:
                dataname, frame = path.split('/')
            except ValueError:
                dataname = path
                frame = 0

            if dataname in images:
                self.set_header("Content-Type", "image/png")
                self.write(images[dataname][int(frame)])
            else:
                self.set_status(404)
                self.write_error(404)

    class StimHandler(serve.StaticFileHandler):
        def initialize(self):
            pass

        def get(self, path):
            if path not in stims:
                self.set_status(404)
                self.write_error(404)
            else:
                self.root, fname = os.path.split(stims[path])
                super(StimHandler, self).get(fname)

    class MixerHandler(web.RequestHandler):
        def get(self):
            self.set_header("Content-Type", "text/html")
            generated = html.generate(
                data=metadata, 
                colormaps=colormaps, 
                default_cmap=cmap, 
                python_interface=True, 
                layout=layout,
                subjects=subjectjs,
                **viewopts)
            self.write(generated)

        def post(self):
            print("saving file to %s"%saveimg.value)
            data = self.get_argument("svg", default=None)
            png = self.get_argument("png", default=None)
            with open(saveimg.value, "wb") as svgfile:
                if png is not None:
                    data = png[22:].strip()
                    try:
                        data = binascii.a2b_base64(data)
                    except:
                        print("Error writing image!")
                        data = png
                svgfile.write(data)
            saveevt.set()

    if pickerfun is None:
        pickerfun = lambda a,b: None

    class JSLocalMixer(serve.JSLocal):
        def addData(self, **kwargs):
            Proxy = serve.JSProxy(self.send, "window.viewers.addData")
            json, data = _make_bindat(_normalize_data(kwargs, pfunc), fmt='data/%s/')
            queue.put(data)
            return Proxy(json)

    class JSMixer(serve.JSProxy):
        def addData(self, **kwargs):
            Proxy = serve.JSProxy(self.send, "window.viewers.addData")
            metadata, images = _convert_dataset(Dataset(**kwargs), path='/data/', fmt='%s_%d.png')
            queue.put(images)
            return Proxy(metadata)

        def saveIMG(self, filename):
            Proxy = serve.JSProxy(self.send, "window.viewers.saveIMG")
            saveimg.value = filename
            return Proxy("mixer.html")

        def makeMovie(self, animation, filename="brainmovie%07d.png", offset=0, fps=30, shape=(1920, 1080), mix="linear"):
            state = dict()
            anim = []
            for f in sorted(animation, key=lambda x:x['idx']):
                if f['idx'] == 0:
                    self.setState(f['state'], f['value'])
                    state[f['state']] = dict(idx=f['idx'], val=f['value'])
                else:
                    if f['state'] not in state:
                        state[f['state']] = dict(idx=0, val=self.getState(f['state'])[0])
                    start = dict(idx=state[f['state']]['idx'], state=f['state'], value=state[f['state']]['val'])
                    end = dict(idx=f['idx'], state=f['state'], value=f['value'])
                    state[f['state']]['idx'] = f['idx']
                    state[f['state']]['val'] = f['value']
                    if start['value'] != end['value']:
                        anim.append((start, end))

            print(anim)
            self.resize(*shape)
            for i, sec in enumerate(np.arange(0, anim[-1][1]['idx'], 1./fps)):
                for start, end in anim:
                    if start['idx'] < sec < end['idx']:
                        idx = (sec - start['idx']) / (end['idx'] - start['idx'])
                        if start['state'] == 'frame':
                            func = mixes['linear']
                        else:
                            func = mixes[mix]
                            
                        val = func(np.array(start['value']), np.array(end['value']), idx)
                        if isinstance(val, np.ndarray):
                            self.setState(start['state'], val.ravel().tolist())
                        else:
                            self.setState(start['state'], val)
                saveevt.clear()
                self.saveIMG(filename%(i+offset))
                saveevt.wait()

    class PickerHandler(web.RequestHandler):
        def initialize(self, server):
            self.client = JSLocalMixer(server.srvsend, server.srvresp)

        def get(self):
            pickerfun(self.client, int(self.get_argument("voxel")), int(self.get_argument("vertex")))

    class WebApp(serve.WebApp):
        disconnect_on_close = autoclose
        def get_client(self):
            self.c_evt.wait()
            self.c_evt.clear()
            return JSMixer(self.send, "window.viewers")

        def get_local_client(self):
            return JSMixer(self.srvsend, "window.viewers")

    if port is None:
        port = random.randint(1024, 65536)
        
    srvdict = dict()
    server = WebApp([
            (r'/ctm/(.*)', CTMHandler),
            (r'/data/(.*)', DataHandler),
            (r'/stim/(.*)', StimHandler),
            (r'/mixer.html', MixerHandler),
            (r'/picker', PickerHandler, srvdict),
            (r'/', MixerHandler),
        ], port)
    srvdict['server'] = server
    server.start()
    print("Started server on port %d"%server.port)
    if open_browser:
        webbrowser.open("http://%s:%d/mixer.html"%(serve.hostname, server.port))

        client = server.get_client()
        client.server = server
        return client
