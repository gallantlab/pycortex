import os
import glob
import copy
import json
import Queue
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
colormaps = [(os.path.splitext(os.path.split(cm)[1])[0], serve.make_base64(cm))
             for cm in sorted(colormaps)]

def make_static(outpath, data, types=("inflated",), recache=False, cmap="RdBu_r",
                template="static.html", layout=None, anonymize=False,
                html_embed=True, overlays_visible=('rois', 'sulci'), labels_visible=('rois', ),
                overlay_file=None, copy_ctmfiles=True, title='Brain', **kwargs):
    """
    Creates a static webGL MRI viewer in your filesystem so that it can easily 
    be posted publically for sharing or just saved for later viewing.

    Parameters
    ----------
    outpath : string
        The directory where the static viewer will be saved. Will be created if it
        doesn't already exist.
    data : Dataset object or implicit Dataset
        Dataset object containing all the data you wish to plot. Can be any type
        of implicit dataset, such as a single Volume, Vertex, etc. object or a
        dictionary of Volume, Vertex. etc. objects.
    recache : bool, optional
        Force recreation of CTM and SVG files for surfaces. Default False
    template : string, optional
        Name of template HTML file. Default 'static.html'
    anonymize : bool, optional
        Whether to rename CTM and SVG files generically, for public distribution.
        Default False
    overlays_visible : tuple, optional
        The listed overlay layers will be set visible by default. Layers not listed
        here will be hidden by default (but can be enabled in the viewer GUI).
        Default ('rois', 'sulci')
    labels_visible : tuple, optional
        Labels for the listed layers will be set visible by default. Labels for
        layers not listed here will be hidden by default (but can be enabled in
        the viewer GUI). Default ('rois', )
    **kwargs
        All additional keyword arguments are passed to the template renderer.

    Other parameters
    ----------------
    types : tuple, optional
        Types of surfaces to include in addition to the original (fiducial, pial,
        and white matter) and flat surfaces. Default ('inflated', )
    cmap : string, optional
        Name of default colormap. Default 'RdBu_r'
        TODO: WHAT THE FUCK GOOD DOES THIS DO ANYMORE?
    overlay_file : str, optional
        Totally replace the overlays.svg file for this subject with the given
        file (if not None). Default None.
    html_embed : bool, optional
        Whether to embed the webgl resources in the html output.  Default 'True'.
        If 'False', the webgl resources must be served by your web server.
    copy_ctmfiles : bool, optional
        Whether to copy the CTM files to the static directory.  Default 'True'.
        In some use cases, the same CTM data will be used in many static views. To
        avoid duplication of files, set to 'False'.  (The datastore cache must
        then be served with your web server).
    title : str, optional
        The title that is displayed on the viewer website when it is loaded in
        a browser.

    Notes
    -----
    You will need a real web server to view this, since `file://` paths
    don't handle xsrf correctly
    """
    if overlay_file is not None:
        raise NotImplementedError("External overlay_file not supported yet, sorry!")
    
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
    ctms = dict((subj, utils.get_ctmpack(subj, types, **ctmargs))
                for subj in subjects)
    package.reorder(ctms)

    db.auxfile = None
    if layout is None:
        layout = [None, (1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 2), (3, 3), (3, 3), (3, 3)][len(subjects)]

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

        for ext in ['json', 'ctm', 'svg']:
            srcfile = os.path.join(oldpath, "%s.%s"%(fname, ext))
            newfile = os.path.join(outpath, "%s.%s"%(newfname, ext))
            if os.path.exists(newfile):
                os.unlink(newfile)

            if os.path.exists(srcfile) and copy_ctmfiles:
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

    # Put together all view options
    my_viewopts = dict(options.config.items('webgl_viewopts'))
    my_viewopts['overlays_visible'] = overlays_visible
    my_viewopts['labels_visible'] = labels_visible
    for sec in options.config.sections():
        if 'paths' in sec or 'labels' in sec:
            my_viewopts[sec] = dict(options.config.items(sec))

    html = tpl.generate(data=json.dumps(metadata),
                        colormaps=colormaps,
                        default_cmap=cmap,
                        python_interface=False,
                        leapmotion=True,
                        layout=layout,
                        subjects=json.dumps(ctms),
                        viewopts=json.dumps(my_viewopts),
                        title=title,
                        **kwargs)
    desthtml = os.path.join(outpath, "index.html")
    if html_embed:
        htmlembed.embed(html, desthtml, rootdirs)
    else:
        with open(desthtml, "w") as htmlfile:
            htmlfile.write(html)


def show(data, types=("inflated", ), recache=False, cmap='RdBu_r', layout=None,
         autoclose=True, open_browser=True, port=None, pickerfun=None, template="mixer.html",
         overlays_visible=('rois', 'sulci'), labels_visible=('rois', ), overlay_file=None,
         title='Brain', **kwargs):
    """
    Creates a webGL MRI viewer that is dynamically served by a tornado server
    running inside the current python process.

    Parameters
    ----------
    data : Dataset object or implicit Dataset
        Dataset object containing all the data you wish to plot. Can be any type
        of implicit dataset, such as a single Volume, Vertex, etc. object or a
        dictionary of Volume, Vertex. etc. objects.
    autoclose : bool, optional
        If True, the tornado server will automatically be destroyed when the last
        web client has disconnected. If False, the server will stay open, 
        allowing more connections. Default True
    open_browser : bool, optional
        If True, uses the webbrowser library to open the viewer in the default
        local browser. Default True
    port : int or None, optional
        The port that will be used by the server. If None, a random port will be
        selected from the range 1024-65536. Default None
    pickerfun : funcion or None, optional
        Should be a function that takes two arguments, a voxel index and a vertex
        index. Is called whenever a location on the surface is clicked in the 
        viewer. This can be used to print information about individual voxels or
        vertices, plot receptive fields, or many other uses. Default None
    recache : bool, optional
        Force recreation of CTM and SVG files for surfaces. Default False
    template : string, optional
        Name of template HTML file. Default 'mixer.html'
    overlays_visible : tuple, optional
        The listed overlay layers will be set visible by default. Layers not listed
        here will be hidden by default (but can be enabled in the viewer GUI).
        Default ('rois', 'sulci')
    labels_visible : tuple, optional
        Labels for the listed layers will be set visible by default. Labels for
        layers not listed here will be hidden by default (but can be enabled in
        the viewer GUI). Default ('rois', )
    **kwargs
        All additional keyword arguments are passed to the template renderer.

    Other parameters
    ----------------
    types : tuple, optional
        Types of surfaces to include in addition to the original (fiducial, pial,
        and white matter) and flat surfaces. Default ('inflated', )
    cmap : string, optional
        Name of default colormap. Default 'RdBu_r'
        TODO: WHAT THE FUCK GOOD DOES THIS DO ANYMORE?
    overlay_file : str or None, optional
        Totally replace the overlays.svg file for this subject with the given
        file (if not None). Default None.
    title : str, optional
        The title that is displayed on the viewer website when it is loaded in
        a browser.
    layout : (int, int), optional
        The layout of the viewer subwindows for showing multiple subjects.
        Default None, which selects the layout based on the number of subjects.
    """
    if overlay_file is not None:
        raise NotImplementedError("External overlay_file not supported yet, sorry!")
    
    data = dataset.normalize(data)
    if not isinstance(data, dataset.Dataset):
        data = dataset.Dataset(data=data)

    html = FallbackLoader([os.path.split(os.path.abspath(template))[0], serve.cwd]).load(template)
    db.auxfile = data

    #Extract the list of stimuli, for special-casing
    stims = dict()
    for name, view in data:
        if 'stim' in view.attrs and os.path.exists(view.attrs['stim']):
            sname = os.path.split(view.attrs['stim'])[1]
            stims[sname] = view.attrs['stim']

    package = Package(data)
    metadata = json.dumps(package.metadata())
    images = package.images
    subjects = list(package.subjects)

    ctmargs = dict(method='mg2', level=9, recache=recache)
    ctms = dict((subj, utils.get_ctmpack(subj, types, **ctmargs))
                for subj in subjects)
    package.reorder(ctms)

    subjectjs = json.dumps(dict((subj, "/ctm/%s/"%subj) for subj in subjects))
    db.auxfile = None

    if layout is None:
        layout = [None, (1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 2), (3, 3), (3, 3), (3, 3)][len(subjects)]

    linear = lambda x, y, m: (1.-m)*x + m*y
    mixes = dict(
        linear=linear,
        smoothstep=(lambda x, y, m: linear(x, y, 3*m**2 - 2*m**3)),
        smootherstep=(lambda x, y, m: linear(x, y, 6*m**5 - 15*m**4 + 10*m**3))
    )

    post_name = Queue.Queue()

    # Put together all view options
    my_viewopts = dict(options.config.items('webgl_viewopts'))
    my_viewopts['overlays_visible'] = overlays_visible
    my_viewopts['labels_visible'] = labels_visible
    for sec in options.config.sections():
        if 'paths' in sec or 'labels' in sec:
            my_viewopts[sec] = dict(options.config.items(sec))

    if pickerfun is None:
        pickerfun = lambda a, b: None

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
                dataimg = images[dataname][int(frame)]
                if dataimg[1:6] == "NUMPY":
                    self.set_header("Content-Type", "application/octet-stream")
                else:
                    self.set_header("Content-Type", "image/png")

                if 'Range' in self.request.headers:
                    self.set_status(206)
                    rangestr = self.request.headers['Range'].split('=')[1]
                    start, end = [ int(i) if len(i) > 0 else None for i in rangestr.split('-') ]

                    clenheader = 'bytes %s-%s/%s' % (start, end or len(dataimg), len(dataimg) )
                    self.set_header('Content-Range', clenheader)
                    self.set_header('Content-Length', end-start+1)
                    self.write(dataimg[start:end+1])
                else:
                    self.write(dataimg)
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

    class StaticHandler(web.StaticFileHandler):
        def initialize(self):
            self.root = ''

    class MixerHandler(web.RequestHandler):
        def get(self):
            self.set_header("Content-Type", "text/html")
            generated = html.generate(data=metadata,
                                      colormaps=colormaps,
                                      default_cmap=cmap,
                                      python_interface=True,
                                      leapmotion=True,
                                      layout=layout,
                                      subjects=subjectjs,
                                      viewopts=json.dumps(my_viewopts),
                                      title=title,
                                      **kwargs)
                                      #overlays_visible=json.dumps(overlays_visible),
                                      #labels_visible=json.dumps(labels_visible),
                                      #**viewopts)
            self.write(generated)

        def post(self):
            data = self.get_argument("svg", default=None)
            png = self.get_argument("png", default=None)
            with open(post_name.get(), "wb") as svgfile:
                if png is not None:
                    data = png[22:].strip()
                    try:
                        data = binascii.a2b_base64(data)
                    except:
                        print("Error writing image!")
                        data = png
                svgfile.write(data)

    class JSMixer(serve.JSProxy):
        # Enumerated list of properties settable for views. There may be a way to get this
        # from the javascript object, but I (ML) don't know how.
        # Old properites (some of which still need re-implementing in javascript)
        # view_props = ['altitude', 'azimuth', 'target', 'mix', 'radius', 'pivot',
        #     'visL', 'visR', 'alpha', 'rotationR', 'rotationL', 'projection',
        #     'volume_vis', 'frame', 'slices']
        @property
        def view_props(self):
            _camera_props = ['camera.%s'%k for k in self.ui.camera._controls.attrs.keys()]
            _subject = self.ui.surface._folders.attrs.keys()[0]
            _surface = getattr(self.ui.surface, _subject)
            _surface_props = ['surface.{subject}.%s'%k for k in _surface._controls.attrs.keys()]
            #view_props = _camera_props + _surface_props
            return _camera_props + _surface_props
            #['camera.altitude', 'camera.azimuth', 'camera.target', 'camera.radius', 
            #'surface.{subject}.unfold', 'surface.{subject}.pivot', 'surface.{subject}.depth',
            #'surface.{subject}.shift', 'surface.{subject}.left', 'surface.{subject}.right']
            #'surface.{subject}.specularity', 'frame', 'bg_alpha'
            #'visL', 'visR', 'alpha', 'rotationR', 'rotationL', 'projection', 
            #'volume_vis', 'frame', 'slices']

        def _set_view(self, **kwargs):
            """Low-level command: sets view parameters in the current viewer

            Sets each the state of each keyword argument provided. View parameters
            that can be set include all parameters in the data.gui in the html view.

            # OLD delete me :
            #altitude, azimuth, target, mix, radius, visL, visR, pivot,
            #(L/R hemisphere visibility), alpha (background alpha),
            #rotationL, rotationR (L/R hemisphere rotation, [x, y, z])

            Notes
            -----
            [the following is no longer true, I think - but check on this before deleting]
            Args must be lists instead of scalars, e.g. `azimuth`=[90]
            This could be changed, but this is a hidden function, called by
            higher-level functions that load .json files, which have the
            parameters in lists by default. So it's annoying either way.
            """
            # Set unfolding level first, as it interacts with other arguments
            subject_list = self.ui.surface._folders.attrs.keys()
            for subject in subject_list:
                if 'surface.{subject}.unfold' in kwargs:
                    unfold = kwargs.pop('surface.{subject}.unfold')
                    self.ui.set('surface.{subject}.unfold'.format(subject=subject), unfold)
                for k, v in kwargs.items():
                    if not k in self.view_props:
                        print('Unknown parameter %s!'%k)
                        continue
                    else:
                        self.ui.set(k.format(subject=subject) if '{subject}' in k else k, v)

        def _capture_view(self, time=None):
            """Low-level command: returns a dict of current view parameters

            Retrieves the following view parameters from current viewer:

            altitude, azimuth, target, mix, radius, visL, visR, alpha,
            rotationR, rotationL, projection, pivot

            `time` appends a 'time' key into the view (for use in animations)

            If multiple subjects are present, only retrieves view for first subject.
            """
            view = {}
            subject = self.ui.surface._folders.attrs.keys()[0]
            for p in self.view_props:
                try:
                    view[p] = self.ui.get(p.format(subject=subject) if '{subject}' in p else p)[0]
                except Exception as err:
                    # TO DO: Fix this hack with an error class in serve.py & catch it here 
                    msg = "Cannot read property 'undefined'"
                    if err.message[:len(msg)] != msg:
                        raise err
            if not time is None:
                view['time'] = time
            return view

        def save_view(self, subject, name, is_overwrite=False):
            """Saves current view parameters to pycortex database

            Parameters
            ----------
            subject : string
                pycortex subject id
            name : string
                name for view to store
            is_overwrite: bool
                whether to overwrite an extant view (default : False)

            Notes
            -----
            Equivalent to call to cortex.db.save_view(subject, vw, name)
            For a list of the view parameters saved, see viewer._capture_view

            See Also
            --------
            viewer methods get_view, _set_view, _capture_view
            """
            db.save_view(self, subject, name, is_overwrite)

        def get_view(self, subject, name):
            """Get saved view from pycortex database.

            Retrieves named view from pycortex database and sets current
            viewer parameters to retrieved values.

            Parameters
            ----------
            subject : string
                pycortex subject ID
            name : string
                name of saved view to re-load

            Notes
            -----
            Equivalent to call to cortex.db.get_view(subject, vw, name)
            For a list of the view parameters set, see viewer._capture_view

            See Also
            --------
            viewer methods save_view, _set_view, _capture_view
            """
            view = db.get_view(self, subject, name)

        def addData(self, **kwargs):
            Proxy = serve.JSProxy(self.send, "window.viewers.addData")
            new_meta, new_ims = _convert_dataset(Dataset(**kwargs), path='/data/', fmt='%s_%d.png')
            metadata.update(new_meta)
            images.update(new_ims)
            return Proxy(metadata)

        def getImage(self, filename, size=(1920, 1080)):
            """Saves currently displayed view to a .png image file

            Parameters
            ----------
            filename : string
                duh.
            size : tuple (x, y)
                size (in pixels) of image to save.
            """
            post_name.put(filename)
            Proxy = serve.JSProxy(self.send, "window.viewer.getImage")
            return Proxy(size[0], size[1], "mixer.html")

        def makeMovie(self, animation, filename="brainmovie%07d.png", offset=0,
                      fps=30, size=(1920, 1080), interpolation="linear"):
            """Renders movie frames for animation of mesh movement

            Makes an animation (for example, a transition between inflated and
            flattened brain or a rotating brain) of a cortical surface. Takes a
            list of dictionaries (`animation`) as input, and uses the values in
            the dictionaries as keyframes for the animation.

            Mesh display parameters that can be animated include 'elevation',
            'azimuth', 'mix', 'radius', 'target' (more?)


            Parameters
            ----------
            animation : list of dicts
                Each dict should have keys `idx`, `state`, and `value`.
                `idx` is the time (in seconds) at which you want to set `state` to `value`
                `state` is the parameter to animate (e.g. 'altitude', 'azimuth')
                `value` is the value to set for `state`
            filename : string path name
                Must contain '%d' (or some variant thereof) to account for frame
                number, e.g. '/some/directory/brainmovie%07d.png'
            offset : int
                Frame number for first frame rendered. Useful for concatenating
                animations.
            fps : int
                Frame rate of resultant movie
            size : tuple (x, y)
                Size (in pixels) of resulting movie
            interpolation : {"linear", "smoothstep", "smootherstep"}
                Interpolation method for values between keyframes.

            Example
            -------
            # Called after a call of the form: js_handle = cortex.webgl.show(DataViewObject)
            # Start with left hemisphere view
            js_handle._setView(azimuth=[90], altitude=[90.5], mix=[0])
            # Initialize list
            animation = []
            # Append 5 key frames for a simple rotation
            for az, idx in zip([90, 180, 270, 360, 450], [0, .5, 1.0, 1.5, 2.0]):
                animation.append({'state':'azimuth', 'idx':idx, 'value':[az]})
            # Animate! (use default settings)
            js_handle.makeMovie(animation)
            """
            # build up two variables: State and Anim.
            # state is a dict of all values being modified at any time
            state = dict()
            # anim is a list of transitions between keyframes
            anim = []
            setfunc = self.ui.set
            for f in sorted(animation, key=lambda x:x['idx']):
                if f['idx'] == 0:
                    setfunc(f['state'], f['value'])
                    state[f['state']] = dict(idx=f['idx'], val=f['value'])
                else:
                    if f['state'] not in state:
                        state[f['state']] = dict(idx=0, val=self.getState(f['state'])[0])
                    start = dict(idx=state[f['state']]['idx'],
                                 state=f['state'],
                                 value=state[f['state']]['val'])
                    end = dict(idx=f['idx'], state=f['state'], value=f['value'])
                    state[f['state']]['idx'] = f['idx']
                    state[f['state']]['val'] = f['value']
                    if start['value'] != end['value']:
                        anim.append((start, end))

            for i, sec in enumerate(np.arange(0, anim[-1][1]['idx']+1./fps, 1./fps)):
                for start, end in anim:
                    if start['idx'] < sec <= end['idx']:
                        idx = (sec - start['idx']) / float(end['idx'] - start['idx'])
                        if start['state'] == 'frame':
                            func = mixes['linear']
                        else:
                            func = mixes[interpolation]

                        val = func(np.array(start['value']), np.array(end['value']), idx)
                        if isinstance(val, np.ndarray):
                            setfunc(start['state'], val.ravel().tolist())
                        else:
                            setfunc(start['state'], val)
                self.getImage(filename%(i+offset), size=size)

        def _get_anim_seq(self, keyframes, fps=30, interpolation='linear'):
            """Convert a list of keyframes to a list of EVERY frame in an animation.

            Utility function called by make_movie; separated out so that individual
            frames of an animation can be re-rendered, or for more control over the
            animation process in general.

            """
            # Misc. setup
            fr = 0
            a = np.array
            func = mixes[interpolation]
            skip_props = ['surface.{subject}.right', 'surface.{subject}.left', ] #'projection', 
            # Get keyframes
            keyframes = sorted(keyframes, key=lambda x:x['time'])
            # Normalize all time to frame rate
            fs = 1./fps
            for k in range(len(keyframes)):
                t = keyframes[k]['time']
                t = np.round(t/fs)*fs
                keyframes[k]['time'] = t
            allframes = []
            for start, end in zip(keyframes[:-1], keyframes[1:]):
                t0 = start['time']
                t1 = end['time']
                tdif = float(t1-t0)
                # Check whether to continue frame sequence to endpoint
                use_endpoint = keyframes[-1]==end
                nvalues = np.round(tdif/fs)
                if use_endpoint:
                    nvalues +=1
                fr_time = np.linspace(0, 1, nvalues, endpoint=use_endpoint)
                # Interpolate between values
                for t in fr_time:
                    frame = {}
                    for prop in start.keys():
                        if prop=='time':
                            continue
                        if (prop in skip_props) or (start[prop] is None):
                            frame[prop] = start[prop]
                            continue
                        val = func(a(start[prop]), a(end[prop]), t)
                        if isinstance(val, np.ndarray):
                            frame[prop] = val.tolist()
                        else:
                            frame[prop] = val
                    allframes.append(frame)
            return allframes

        def make_movie_views(self, animation, filename="brainmovie%07d.png", offset=0,
                      fps=30, size=(1920, 1080), alpha=1, interpolation="linear"):
            """Renders movie frames for animation of mesh movement

            Makes an animation (for example, a transition between inflated and
            flattened brain or a rotating brain) of a cortical surface. Takes a
            list of dictionaries (`animation`) as input, and uses the values in
            the dictionaries as keyframes for the animation.

            Mesh display parameters that can be animated include 'elevation',
            'azimuth', 'mix', 'radius', 'target' (more?)


            Parameters
            ----------
            animation : list of dicts
                This is a list of keyframes for the animation. Each keyframe should be
                a dict in the form captured by the ._capture_view method. NOTE: every
                view must include all view parameters. Additionally, there should be
                one extra key/value pair for "time". The value for time should be
                in seconds. The list of keyframes is sorted by time before applying,
                so they need not be in order in the input.
            filename : string path name
                Must contain '%d' (or some variant thereof) to account for frame
                number, e.g. '/some/directory/brainmovie%07d.png'
            offset : int
                Frame number for first frame rendered. Useful for concatenating
                animations.
            fps : int
                Frame rate of resultant movie
            size : tuple (x, y)
                Size (in pixels) of resulting movie
            interpolation : {"linear", "smoothstep", "smootherstep"}
                Interpolation method for values between keyframes.

            Notes
            -----
            Make sure that all values that will be modified over the course
            of the animation are initialized (have some starting value) in the first
            frame.

            Example
            -------
            # Called after a call of the form: js_handle = cortex.webgl.show(DataViewObject)
            # Start with left hemisphere view
            js_handle._setView(azimuth=[90], altitude=[90.5], mix=[0])
            # Initialize list
            animation = []
            # Append 5 key frames for a simple rotation
            for az, t in zip([90, 180, 270, 360, 450], [0, .5, 1.0, 1.5, 2.0]):
                animation.append({'time':t, 'azimuth':[az]})
            # Animate! (use default settings)
            js_handle.make_movie(animation)
            """
            import time
            allframes = self._get_anim_seq(animation, fps, interpolation)
            for fr, frame in enumerate(allframes):
                self._set_view(**frame)
                # Set background alpha, color??
                #self.renderer.setClearColor([0,0,0], alpha)
                #self.saveIMG(filename%(fr+offset+1), size=size)
                self.getImage(filename%(fr+offset+1), size=size)
                time.sleep(0.01)

    class PickerHandler(web.RequestHandler):
        def get(self):
            pickerfun(int(self.get_argument("voxel")), int(self.get_argument("vertex")))

    class WebApp(serve.WebApp):
        disconnect_on_close = autoclose
        def get_client(self):
            self.connect.wait()
            self.connect.clear()
            return JSMixer(self.send, "window.viewer")

        def get_local_client(self):
            return JSMixer(self.srvsend, "window.viewer")

    if port is None:
        port = random.randint(1024, 65536)

    server = WebApp([
            (r'/ctm/(.*)', CTMHandler),
            (r'/data/(.*)', DataHandler),
            (r'/stim/(.*)', StimHandler),
            (r'/mixer.html', MixerHandler),
            (r'/picker', PickerHandler),
            (r'/', MixerHandler),
            (r'/static/(.*)', StaticHandler),
        ], port)
    server.start()
    print("Started server on port %d"%server.port)
    if open_browser:
        webbrowser.open("http://%s:%d/mixer.html"%(serve.hostname, server.port))
        client = server.get_client()
        client.server = server
        return client

    return server
