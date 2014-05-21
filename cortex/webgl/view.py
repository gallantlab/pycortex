import os
import glob
import json
import shutil
import random
import functools
import binascii
import mimetypes
import threading
import webbrowser
import numpy as np

from tornado import web
from .FallbackLoader import FallbackLoader

from .. import utils, options, volume, dataset
from ..database import db

from . import serve
from .data import Package
from ConfigParser import NoOptionError

try:
    cmapdir = options.config.get('webgl', 'colormaps')
    if not os.path.exists(cmapdir):
        raise Exception("Colormap directory (%s) does not exits"%cmapdir)
except NoOptionError:
    cmapdir = os.path.join(options.config.get("basic", "filestore"), "colormaps")
    if not os.path.exists(cmapdir):
        raise Exception("Colormap directory was not defined in the config file and the default (%s) does not exits"%cmapdir)


colormaps = glob.glob(os.path.join(cmapdir, "*.png"))
colormaps = [(os.path.splitext(os.path.split(cm)[1])[0], serve.make_base64(cm)) for cm in sorted(colormaps)]

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

    db.auxfile = data

    package = Package(data)
    subjects = list(package.subjects)

    ctmargs = dict(method='mg2', level=9, recache=recache)
    ctms = dict((subj, utils.get_ctmpack(subj, types, **ctmargs)) for subj in subjects)
    db.auxfile = None

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
            srcfile = os.path.join(oldpath, "%s.%s"%(fname, ext))
            newfile = os.path.join(outpath, "%s.%s"%(newfname, ext))
            if os.path.exists(newfile):
                os.unlink(newfile)
            
            if os.path.exists(srcfile):
                shutil.copy2(srcfile, newfile)

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
        subjects=json.dumps(ctms),
        **kwargs)
    htmlembed.embed(html, os.path.join(outpath, "index.html"), rootdirs)

def show(data, types=("inflated",), recache=False, cmap='RdBu_r', layout=None, autoclose=True, open_browser=True, port=None, pickerfun=None, **kwargs):
    """Display a dynamic viewer using the given dataset
    """
    data = dataset.normalize(data)
    if not isinstance(data, dataset.Dataset):
        data = dataset.Dataset(data=data)

    html = FallbackLoader([serve.cwd]).load("mixer.html")
    db.auxfile = data

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
    subjectjs = json.dumps(dict((subj, "/ctm/%s/"%subj) for subj in subjects))
    db.auxfile = None

    if layout is None:
        layout = [None, (1,1), (2,1), (3,1), (2,2), (3,2), (3,2), (3,3), (3,3), (3,3)][len(subjects)]

    linear = lambda x, y, m: (1.-m)*x + m*y
    mixes = dict(
        linear=linear,
        smoothstep=(lambda x, y, m: linear(x,y,3*m**2 - 2*m**3)), 
        smootherstep=(lambda x, y, m: linear(x, y, 6*m**5 - 15*m**4 + 10*m**3))
    )

    post_lock = threading.Lock()
    post_name = None

    if pickerfun is None:
        pickerfun = lambda a,b: None

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

    class StimHandler(web.StaticFileHandler):
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
            data = self.get_argument("svg", default=None)
            png = self.get_argument("png", default=None)
            post_lock.acquire()
            with open(post_name, "wb") as svgfile:
                if png is not None:
                    data = png[22:].strip()
                    try:
                        data = binascii.a2b_base64(data)
                    except:
                        print("Error writing image!")
                        data = png
                svgfile.write(data)
            post_lock.release()

    class JSMixer(serve.JSProxy):
        def _setView(self,**kwargs):
            """Low-level command: sets one view parameter at a time.

            Settable keyword args: 
            altitude, azimuth, target, mix, radius

            NOTE: args must be lists instead of scalars, e.g. `azimuth`=[90]
            Could be resolved, but this is a hidden function, called by 
            higher-level functions that load .json files, which have the parameters
            in lists by default. So it's annoying either way.
            """
            props = ['altitude','azimuth','target','mix','radius']
            for k in kwargs.keys():
                if not k in props:
                    print('Unknown parameter %s!'%k)
                    continue
                self.setState(k,kwargs[k][0])
        def _getView(self):
            """Low-level command: returns a dict of current view parameters"""
            props = ['altitude','azimuth','target','mix','radius']
            # db.save_view()
            view = {}
            for p in props:
                view[p] = self.getState(p)[0]
            return view

        def save_view(self,subject,name):
            """Saves current view parameters to a .json file

            Parameters
            ----------
            fName : string
                name for view to store

            Notes
            -----
            Equivalent to call to cortex.db.save_view(subject,vw,name)
            
            To adjust view in javascript console:
            # Set BG to alpha:
            viewers.<subject>.renderer.setClearColor(0,0)

            # One hemisphere off:
            viewers.<subject>.meshes.left.visible = false

            See Also
            --------
            methods get_view, _setView, _getView
            """
            # Check for existence of view? 
            db.save_view(self,subject,name)

        def get_view(self,subject,name):
            """Sets current view parameters to those stored in a .json file

            Parameters
            ----------
            subject : pycortex subject ID
            name : string
                name of saved view to re-load

            Notes
            -----
            Equivalent to call to cortex.db.get_view(subject,vw,name)

            Further modifications possible in JavaScript console:
            # Set BG to alpha:
            viewers.<subject>.renderer.setClearColor(0,0)

            # One hemisphere off:
            viewers.<subject>.meshes.left.visible = false

            See Also
            --------
            methods save_view, _setView, _getView
            """
            view = db.get_view(self,subject,name)
            

        def addData(self, **kwargs):
            Proxy = serve.JSProxy(self.send, "window.viewers.addData")
            new_meta, new_ims = _convert_dataset(Dataset(**kwargs), path='/data/', fmt='%s_%d.png')
            metadata.update(new_meta)
            images.update(new_ims)
            return Proxy(metadata)

        def saveIMG(self, filename,size=None):
            """Saves currently displayed view to a .png image file

            Parameters
            ----------
            filename : string
                duh.
            size : tuple (x,y) 
                size (in pixels) of image to save. Resizes whole window.
            """
            if not size is None:
                self.resize(*size)
            post_lock.acquire()
            post_name = filename
            post_lock.release()

            Proxy = serve.JSProxy(self.send, "window.viewers.saveIMG")
            return Proxy("mixer.html")

        def makeMovie(self, animation, filename="brainmovie%07d.png", offset=0, fps=30, size=(1920, 1080), interpolation="linear"):
            """Renders movie frames for animation of mesh movement

            Makes an animation (for example, a transition between inflated and 
            flattened brain or a rotating brain) of a cortical surface. Takes a 
            list of dictionaries (`animation`) as input, and uses the values in
            the dictionaries as keyframes for the animation.

            Mesh display parameters that can be animated include 'elevation',
            'azimuth','mix','radius','target' (more?)


            Parameters
            ----------
            animation : list of dicts
                Each dict should have keys `idx`, `state`, and `value`.
                `idx` is the time (in seconds) at which you want to set `state` to `value`
                `state` is the parameter to animate (e.g. 'altitude','azimuth')
                `value` is the value to set for `state`
            filename : string path name
                Must contain '%d' (or some variant thereof) to account for frame
                number, e.g. '/some/directory/brainmovie%07d.png'
            offset : int
                Frame number for first frame rendered. Useful for concatenating
                animations.
            fps : int
                Frame rate of resultant movie
            size : tuple (x,y)
                Size (in pixels) of resulting movie
            interpolation : {"linear","smoothstep","smootherstep"}
                Interpolation method for values between keyframes.

            Example
            -------
            # Called after a call of the form: js_handle = cortex.webgl.show(DataViewObject)
            # Start with left hemisphere view
            js_handle._setView(azimuth=[90],altitude=[90.5],mix=[0])
            # Initialize list
            animation = []
            # Append 5 key frames for a simple rotation
            for az,idx in zip([90,180,270,360,450],[0,.5,1.0,1.5,2.0]):
                animation.append({'state':'azimuth','idx':idx,'value':[az]})
            # Animate! (use default settings)
            js_handle.makeMovie(animation)
            """

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
            #import ipdb
            #ipdb.set_trace()
            self.resize(*size)
            for i, sec in enumerate(np.arange(0, anim[-1][1]['idx']+1./fps, 1./fps)):
                for start, end in anim:
                    if start['idx'] < sec <= end['idx']:
                        idx = (sec - start['idx']) / (end['idx'] - start['idx'])
                        if start['state'] == 'frame':
                            func = mixes['linear']
                        else:
                            func = mixes[interpolation]
                            
                        val = func(np.array(start['value']), np.array(end['value']), idx)
                        #import ipdb
                        #ipdb.set_trace()
                        if isinstance(val, np.ndarray):
                            self.setState(start['state'], val.ravel().tolist())
                        else:
                            self.setState(start['state'], val)
                saveevt.clear()
                self.saveIMG(filename%(i+offset))
                saveevt.wait()

    class PickerHandler(web.RequestHandler):
        def get(self):
            pickerfun(int(self.get_argument("voxel")), int(self.get_argument("vertex")))

    class WebApp(serve.WebApp):
        disconnect_on_close = autoclose
        def get_client(self):
            self.connect.wait(5)
            self.connect.clear()
            return JSMixer(self.send, "window.viewers")

        def get_local_client(self):
            return JSMixer(self.srvsend, "window.viewers")

    if port is None:
        port = random.randint(1024, 65536)
        
    server = WebApp([
            (r'/ctm/(.*)', CTMHandler),
            (r'/data/(.*)', DataHandler),
            (r'/stim/(.*)', StimHandler),
            (r'/mixer.html', MixerHandler),
            (r'/picker', PickerHandler),
            (r'/', MixerHandler),
        ], port)
    server.start()
    print("Started server on port %d"%server.port)
    if open_browser:
        webbrowser.open("http://%s:%d/mixer.html"%(serve.hostname, server.port))
        client = server.get_client()
        client.server = server
        return client

    return server
