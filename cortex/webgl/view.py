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
from FallbackLoader import FallbackLoader

from .. import utils, options, volume, dataset
from ..db import surfs

from . import serve

name_parse = re.compile(r".*/(\w+).png")
try:
    cmapdir = options.config.get('webgl', 'colormaps')
except:
    cmapdir = os.path.join(options.config.get("basic", "filestore"), "colormaps")
colormaps = glob.glob(os.path.join(cmapdir, "*.png"))
colormaps = [(name_parse.match(cm).group(1), serve.make_base64(cm)) for cm in sorted(colormaps)]

viewopts = dict(voxlines="false", voxline_color="#FFFFFF", voxline_width='.01' )

def _package_data(braindata, submap=None):
    from scipy.stats import scoreatpercentile
    package = dict(__class__="Dataset")

    #Fill in extra metadata
    xfm = surfs.getXfm(braindata.subject, braindata.xfmname, 'coord')
    package['subject'] = braindata.subject
    package['raw'] = braindata.raw
    package['xfm'] = list(np.array(xfm.xfm).ravel())
    package['lmin'] = float(braindata.data.min())
    package['lmax'] = float(braindata.data.max())
    package['vmin'] = float(scoreatpercentile(braindata.data.ravel(), 1))
    package['vmax'] = float(scoreatpercentile(braindata.data.ravel(), 99))

    if submap is not None:
        package['subject'] = submap[braindata.subject]

    if not braindata.movie:
        voldat = braindata.volume[np.newaxis]
    else:
        voldat = braindata.volume
    if braindata.raw:
        voldat = voldat.astype(np.uint8)
    else:
        voldat = voldat.astype(np.float32)

    package['data'] = []
    for vol in voldat:
        im, package['mosaic'] = volume.mosaic(vol, show=False)
        package['data'].append(im)

    #Overwrite generated metadata
    package.update(braindata.attrs)
    #include only the filename for any stimuli
    if 'stim' in package:
        package['stim'] = os.path.join("stim", os.path.split(package['stim'])[1])

    return package

def _convert_dataset(data, fmt="%s_%d.png", submap=None):
    metadata, images = dict(__order__=[]), dict()
    for name, braindata in data:
        metadata['__order__'].append(name)
        if isinstance(braindata, dataset.VertexData):
            raise TypeError('Sorry, vertex data is currently not supported for webgl...')
        package = _package_data(braindata, submap=submap)
        for i, data in enumerate(package['data']):
            images[fmt%(name, i)] = _pack_png(data)

        frames = range(len(package['data']))
        package['data'] = [os.path.join("data", fmt%(name, i)) for i in frames]
        metadata[name] = package

    return metadata, images

def _pack_png(mosaic):
    import Image
    import cStringIO
    buf = cStringIO.StringIO()
    if mosaic.dtype not in (np.float32, np.uint8):
        raise TypeError

    im = Image.frombuffer('RGBA', mosaic.shape[:2], mosaic.data, 'raw', 'RGBA', 0, 1)
    im.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()

def make_static(outpath, data, types=("inflated",), recache=False, cmap="RdBu_r", template="static.html", layout=None, anonymize=False, **kwargs):
    """
    Creates a static instance of the webGL MRI viewer that can easily be posted 
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

    surfs.auxfile = dataset
    subjects = list(set([ds.subject for name, ds in dataset]))
    kwargs.update(dict(method='mg2', level=9, recache=recache))
    ctms = dict((subj, utils.get_ctmpack(subj, types, **kwargs)) for subj in subjects)
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

    #Process the dataset
    metadata, images = _convert_dataset(data, submap=submap)
    jsmeta = json.dumps(metadata, cls=serve.NPEncode)
    #Write out the PNGs
    for name, img in list(images.items()):
        with open(os.path.join(outpath, "data", name), "wb") as binfile:
            binfile.write(img)
    #Copy any stimulus files
    stimpath = os.path.join(outpath, "stim")
    for name, ds in data:
        if 'stim' in ds.attrs and os.path.exists(ds.attrs['stim']):
            if not os.path.exists(stimpath):
                os.makedirs(stimpath)
            shutil.copy2(ds.attrs['stim'], stimpath)
    
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
        data=jsmeta, 
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

    stims = dict()
    for name, ds in data:
        if 'stim' in ds.attrs and os.path.exists(ds.attrs['stim']):
            sname = os.path.split(ds.attrs['stim'])[1]
            stims[sname] = ds.attrs['stim']

    subjects = list(set([ds.subject for name, ds in data]))
    kwargs.update(dict(method='mg2', level=9, recache=recache))
    ctms = dict((subj, utils.get_ctmpack(subj, types, **kwargs)) for subj in subjects)
    subjectjs = dict((subj, "/ctm/%s/"%subj) for subj in subjects)
    surfs.auxfile = None

    if layout is None:
        layout = [None, (1,1), (2,1), (3,1), (2,2), (3,2), (3,2), (3,3), (3,3), (3,3)][len(subjects)]

    metadata, images = _convert_dataset(data)
    jsmeta = json.dumps(metadata, cls=serve.NPEncode)

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
            if not queue.empty():
                d = queue.get()
                print("Got new data: %r"%list(d.keys()))
                images.update(d)

            if path in images:
                self.set_header("Content-Type", "image/png")
                self.write(images[path])
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
                data=jsmeta, 
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
    
    class PickerHandler(web.RequestHandler):
        def get(self):
            pickerfun(int(self.get_argument("voxel")), int(self.get_argument("vertex")))

    class JSMixer(serve.JSProxy):
        def addData(self, **kwargs):
            Proxy = serve.JSProxy(self.send, "window.viewer.addData")
            metadata, images = _convert_dataset(Dataset(**kwargs), path='/data/', fmt='%s_%d.png')
            queue.put(images)
            return Proxy(metadata)

        def saveflat(self, filename, height=1024):
            Proxy = serve.JSProxy(self.send, "window.viewer.saveflat")
            saveimg.value = filename
            return Proxy(height, "mixer.html")

        def saveIMG(self, filename):
            Proxy = serve.JSProxy(self.send, "window.viewer.saveIMG")
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
                            self.setState(start['state'], list(val))
                        else:
                            self.setState(start['state'], val)
                saveevt.clear()
                self.saveIMG(filename%(i+offset))
                saveevt.wait()

    class WebApp(serve.WebApp):
        disconnect_on_close = autoclose
        def get_client(self):
            self.c_evt.wait()
            self.c_evt.clear()
            return JSMixer(self.send, "window.viewers")

    if port is None:
        port = random.randint(1024, 65536)
        
    server = WebApp([
            (r'/ctm/(.*)', CTMHandler),
            (r'/data/(.*)', DataHandler),
            (r'/stim/(.*)', StimHandler),
            (r'/mixer.html', MixerHandler),
            (r'/', MixerHandler),
            (r'/picker', PickerHandler)
        ], port)
    server.start()
    print("Started server on port %d"%server.port)
    if open_browser:
        webbrowser.open("http://%s:%d/mixer.html"%(serve.hostname, server.port))

        client = server.get_client()
        client.server = server
        return client
