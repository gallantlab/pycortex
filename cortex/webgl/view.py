import binascii
import copy
import functools
import glob
import hmac
import json
import mimetypes
import os
import re
import secrets
import shutil
import sys
import threading
import time
from typing import Union, Any, Callable, Optional, ParamSpec, cast
import warnings
import webbrowser
from configparser import NoOptionError

# Now assumes python 3
from queue import Queue

import numpy as np
from tornado import web

from .. import dataset, options, utils, volume
from ..database import db
from . import serve
from .data import Package
from .FallbackLoader import FallbackLoader
from .interpolation import Interpolation, build_channels, evaluate

try:
    cmapdir = options.config.get('webgl', 'colormaps')
    if not os.path.exists(cmapdir):
        raise Exception("Colormap directory (%s) does not exist"%cmapdir)
except NoOptionError:
    cmapdir = os.path.join(options.config.get("basic", "filestore"), "colormaps")
    if not os.path.exists(cmapdir):
        raise Exception("Colormap directory was not defined in the config file and the default (%s) does not exist"%cmapdir)

domain_name = options.config.get("webgl", "domain_name")

colormaps = glob.glob(os.path.join(cmapdir, "*.png"))
colormaps = [(os.path.splitext(os.path.split(cm)[1])[0], serve.make_base64(cm))
             for cm in sorted(colormaps)]


def _has_flatmap(subject: str) -> bool:
    """Whether `subject` has a flat surface, without raising if it does not."""
    try:
        return hasattr(getattr(db, subject).surfaces, "flat")
    except Exception:
        return False


def _quickflat_size(subject: str, height: int = 1024) -> Optional[list[int]]:
    """The pixel size ``cortex.quickflat.make_png`` writes by default.

    ``make_png`` resizes the figure to the flatmap image and saves it at `dpi`,
    so the png comes out exactly as many pixels as that image. The width follows
    from the flat surface's bounding box, which is the one thing here that
    varies by subject.

    Reproduces the arithmetic of ``quickflat.utils._make_flatmask`` rather than
    calling it, because that function rasterizes the surface outline with PIL to
    build a mask this does not need -- and would cache a mask the viewer may
    never use.

    Returns
    -------
    list of int or None
        ``[width, height]``, or None if the subject has no flat surface or it
        could not be read.
    """
    if not _has_flatmap(subject):
        return None
    try:
        pts, _ = db.get_surf(subject, "flat", merge=True, nudge=True)
        span = pts.max(0) - pts.min(0)
        if span[1] <= 0:
            return None
        return [int((height / span[1]) * span[0]), int(height)]
    except Exception as err:
        warnings.warn("Could not work out the quickflat size for %s: %s"
                      % (subject, err))
        return None


def _load_saved_views(subjects: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """The views each of `subjects` offers, defaults overlaid with saved ones.

    Every subject gets the standard anatomical views from
    ``cortex.export.save_views.default_subject_views`` -- dorsal, ventral, the
    two lateral views, their inflated counterparts, and flat -- so that a
    subject with an empty (or missing) views/ directory still has them. A view
    stored in the filestore under one of those names replaces the default,
    which is how a subject whose anatomy needs a different angle, or who wants
    a different framing, overrides one.

    `subjects` is the list of subjects the viewer is actually displaying, so a
    viewer never reads (nor ships to the browser) views belonging to unrelated
    subjects in the filestore.

    Returns
    -------
    dict
        ``{subject: {view_name: {prop: value}}}``. The keys within each view keep
        the literal ``{subject}`` placeholder that ``JSMixer._capture_view``
        writes; the javascript side substitutes it per subject when the view is
        applied, so one saved view still works in a multi-subject viewer.
    """
    from ..export.save_views import default_subject_views

    saved: dict[str, dict[str, dict[str, Any]]] = {}
    for subj in subjects:
        saved[subj] = dict(default_subject_views(_has_flatmap(subj)))
        viewdir = os.path.join(db.filestore, subj, "views")
        # Glob *.json rather than using db.get_paths()['views'], which strips any
        # extension off any file in the directory (so notes.tar.gz would show up
        # as a view named "notes.tar").
        for path in sorted(glob.glob(os.path.join(viewdir, "*.json"))):
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                with open(path) as fp:
                    view = json.load(fp)
            except (ValueError, OSError) as err:
                warnings.warn("Skipping unreadable view %s: %s" % (path, err))
                continue
            if not isinstance(view, dict):
                warnings.warn("Skipping view %s: expected a dict of view "
                              "parameters, got %s" % (path, type(view).__name__))
                continue
            saved[subj][name] = view
    return saved


def make_static(
    outpath,
    data,
    recache=False,
    template="static.html",
    anonymize=False,
    overlays_available=None,
    overlays_visible=("rois", "sulci"),
    labels_visible=("rois",),
    types=("inflated",),
    html_embed=True,
    copy_ctmfiles=True,
    title="Brain",
    layout=None,
    overlay_file=None,
    curvature_brightness=None,
    curvature_contrast=None,
    curvature_smoothness=None,
    surface_specularity=None,
    **kwargs,
):
    """
    Creates a static webGL MRI viewer in your filesystem so that it can easily
    be posted publicly for sharing or just saved for later viewing.

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
    overlays_available : tuple, optional
        Overlays available in the viewer. If None, then all overlay layers of the
        svg file will be potentially available in the viewer (whether initially
        visible or not). This provides the option to include, e.g., only a subset
        of layers for a given static viewer.
    overlays_visible : tuple, optional
        The listed overlay layers will be set visible by default. Layers not listed
        here will be hidden by default (but can be enabled in the viewer GUI).
        Default ('rois', 'sulci')
    labels_visible : tuple, optional
        Labels for the listed layers will be set visible by default. Labels for
        layers not listed here will be hidden by default (but can be enabled in
        the viewer GUI). Default ('rois', )

    Other parameters
    ----------------
    types : tuple, optional
        Types of surfaces to include in addition to the original (fiducial, pial,
        and white matter) and flat surfaces. Default ('inflated', )
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
    layout : None or list of (int, int)
        The layout of the viewer subwindows for showing multiple subjects, passed to
        the template generator.
        Default to None, corresponding to no subwindows.
    overlay_file : str or None, optional
        Custom overlays.svg file to use instead of the default one for this
        subject (if not None). Default None.
    curvature_brightness : float or None, optional
        Brightness of curvature overlay. Default None, which uses the value
        specified in the config file.
    curvature_contrast : float or None, optional
        Contrast of curvature overlay. Default None, which uses the value
        specified in the config file.
    curvature_smoothness : float or None, optional
        Smoothness of curvature overlay. Default None, which uses the value
        specified in the config file.
    surface_specularity : float or None, optional
        Specularity of surfaces visualized with the WebGL viewer. 
        Default None, which uses the value specified in the config file under
        `webgl_viewopts.specularity`.
    **kwargs
        All additional keyword arguments are passed to the template renderer.

    Notes
    -----
    You will need a real web server to view this, since `file://` paths
    don't handle xsrf correctly
    """

    outpath = os.path.abspath(os.path.expanduser(outpath))  # To handle ~ expansion
    os.makedirs(os.path.join(outpath, "data"), exist_ok=True)

    data = dataset.normalize(data)
    if not isinstance(data, dataset.Dataset):
        data = dataset.Dataset(data=data)

    db.auxfile = data

    package = Package(data)
    subjects = list(package.subjects)

    ctmargs = dict(
        method="mg2",
        level=9,
        recache=recache,
        external_svg=overlay_file,
        overlays_available=overlays_available,
    )
    ctms = dict((subj, utils.get_ctmpack(subj, types, **ctmargs)) for subj in subjects)
    package.reorder(ctms)

    db.auxfile = None

    ## Rename files to anonymize
    submap = dict()
    for i, (subj, ctmfile) in enumerate(ctms.items()):
        oldpath, fname = os.path.split(ctmfile)
        fname, ext = os.path.splitext(fname)
        if anonymize:
            newfname = "S%d" % i
            submap[subj] = newfname
        else:
            newfname = fname
        ctms[subj] = newfname + ".json"

        for ext in ["json", "ctm", "svg"]:
            srcfile = os.path.join(oldpath, "%s.%s" % (fname, ext))
            newfile = os.path.join(outpath, "%s.%s" % (newfname, ext))
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
    if anonymize:
        old_subjects = sorted(list(ctms.keys()))
        ctms = dict(("S%d" % i, ctms[k]) for i, k in enumerate(old_subjects))
    if len(submap) == 0:
        submap = None

    # Process the data
    metadata = package.metadata(fmt="data/{name}_{frame}.png", submap=submap)
    images = package.images
    # Write out the PNGs
    for name, imgs in images.items():
        impath = os.path.join(outpath, "data", "{name}_{frame}.png")
        for i, img in enumerate(imgs):
            with open(impath.format(name=name, frame=i), "wb") as binfile:
                binfile.write(img)

    # Copy any stimulus files
    stimpath = os.path.join(outpath, "stim")
    for name, view in data:
        if "stim" in view.attrs and os.path.exists(view.attrs["stim"]):
            if not os.path.exists(stimpath):
                os.makedirs(stimpath)
            shutil.copy2(view.attrs["stim"], stimpath)

    # Parse the html file and paste all the js and css files directly into the html
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
    my_viewopts = dict(options.config.items("webgl_viewopts"))
    my_viewopts["overlays_visible"] = overlays_visible
    my_viewopts["labels_visible"] = labels_visible
    my_viewopts["brightness"] = (
        options.config.get("curvature", "brightness")
        if curvature_brightness is None
        else curvature_brightness
    )
    my_viewopts["contrast"] = (
        options.config.get("curvature", "contrast")
        if curvature_contrast is None
        else curvature_contrast
    )
    my_viewopts["smoothness"] = (
        options.config.get("curvature", "webgl_smooth")
        if curvature_smoothness is None
        else curvature_smoothness
    )
    my_viewopts["specularity"] = (
        options.config.get("webgl_viewopts", "specularity")
        if surface_specularity is None
        else surface_specularity
    )

    for sec in options.config.sections():
        if "paths" in sec or "labels" in sec:
            my_viewopts[sec] = dict(options.config.items(sec))

    # Views saved in the filestore, for the "camera > views" menu. Only the
    # subjects this viewer displays are read.
    my_viewopts["saved_views"] = _load_saved_views(subjects)
    my_viewopts["quickflat_size"] = {subj: _quickflat_size(subj)
                                     for subj in subjects}

    html = tpl.generate(
        data=json.dumps(metadata),
        colormaps=colormaps,
        default_cmap="RdBu_r",
        python_interface=False,
        leapmotion=True,
        layout=layout,
        subjects=json.dumps(ctms),
        viewopts=json.dumps(my_viewopts),
        title=title,
        **kwargs,
    )
    desthtml = os.path.join(outpath, "index.html")
    if html_embed:
        htmlembed.embed(html, desthtml, rootdirs)
    else:
        # tpl.generate returns bytes, like everything tornado templates render.
        with open(desthtml, "wb") as htmlfile:
            htmlfile.write(html)


def show(
    data: Union[dataset.Dataset, dataset.Dataview],
    autoclose: Optional[bool]=None,
    open_browser: Optional[bool]=None,
    port: Optional[int]=None,
    pickerfun: Optional[Callable[[tuple[int, int, int], int, str], None]]=None,
    recache: bool=False,
    template: str="mixer.html",
    overlays_available: Optional[tuple[str, ...]]=None,
    overlays_visible: Optional[tuple[str, ...]]=("rois", "sulci"),
    labels_visible: Optional[tuple[str, ...]]=("rois",),
    types: Optional[tuple[str, ...]]=("inflated",),
    overlay_file: Optional[str]=None,
    curvature_brightness: Optional[float]=None,
    curvature_contrast: Optional[float]=None,
    curvature_smoothness: Optional[float]=None,
    surface_specularity: Optional[float]=None,
    title: str="Brain",
    layout: Optional[str]=None,
    display_url: bool=True,
    movie_dir: Optional[str]=None,
    **kwargs,
):
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
        The port that will be used by the server. If None, a free ephemeral
        port is assigned by the operating system. Default None
    pickerfun : function or None, optional
        Should be a function that takes three arguments, a 3-D voxel vector, a
        vertex index, and the hemisphere ("left" or "right"). Is called whenever
        a location on the surface is clicked in the viewer. This can be used to
        print information about individual voxels or vertices, plot receptive
        fields, or many other uses. Default None
    recache : bool, optional
        Force recreation of CTM and SVG files for surfaces. Default False
    template : string, optional
        Name of template HTML file. Default 'mixer.html'
    overlays_available : tuple, optional
        Overlays available in the viewer. If None, then all overlay layers of the
        svg file will be potentially available in the viewer (whether initially
        visible or not). 
    overlays_visible : tuple, optional
        The listed overlay layers will be set visible by default. Layers not listed
        here will be hidden by default (but can be enabled in the viewer GUI).
        Default ('rois', 'sulci')
    labels_visible : tuple, optional
        Labels for the listed layers will be set visible by default. Labels for
        layers not listed here will be hidden by default (but can be enabled in
        the viewer GUI). Default ('rois', )

    Other parameters
    ----------------
    types : tuple, optional
        Types of surfaces to include in addition to the original (fiducial, pial,
        and white matter) and flat surfaces. Default ('inflated', )
    overlay_file : str or None, optional
        Custom overlays.svg file to use instead of the default one for this
        subject (if not None). Default None.
    curvature_brightness : float or None, optional
        Brightness of curvature overlay. Default None, which uses the value
        specified in the config file.
    curvature_contrast : float or None, optional
        Contrast of curvature overlay. Default None, which uses the value
        specified in the config file.
    curvature_smoothness : float or None, optional
        Smoothness of curvature overlay. Default None, which uses the value
        specified in the config file.
    surface_specularity : float or None, optional
        Specularity of surfaces visualized with the WebGL viewer. 
        Default None, which uses the value specified in the config file under
        `webgl_viewopts.specularity`.
    title : str, optional
        The title that is displayed on the viewer website when it is loaded in
        a browser.
    layout : None or list of (int, int), optional
        The layout of the viewer subwindows for showing multiple subjects, passed to
        the template generator.
        Default None, corresponding to no subwindows.
    display_url : bool, optional
        If True and ``open_browser=False``, display an IPython widget with a URL
        link to access the viewer. Set to False to suppress this display message,
        which can be useful in contexts like Marimo notebooks or programmatic
        headless viewers. Default True
    movie_dir : str or None, optional
        Root directory that the viewer's animation panel may render frames into.
        The folder typed into the panel is interpreted relative to this root, and
        the server refuses to write anywhere outside it. Default None, meaning
        the current working directory.
    **kwargs
        All additional keyword arguments are passed to the template renderer.
    """

    # populate default webshow args
    if autoclose is None:
        autoclose = options.config.get('webshow', 'autoclose', fallback='true') == 'true'
    if open_browser is None:
        open_browser = options.config.get('webshow', 'open_browser', fallback='true') == 'true'

    data = dataset.normalize(data)
    if not isinstance(data, dataset.Dataset):
        data = dataset.Dataset(data=data)

    html = FallbackLoader([os.path.split(os.path.abspath(template))[0], serve.cwd]).load(template)
    db.auxfile = data

    #Extract the list of stimuli, for special-casing
    stims: dict[str, str] = dict()
    for name, view in data:
        if 'stim' in view.attrs and os.path.exists(view.attrs['stim']):
            sname = os.path.split(view.attrs['stim'])[1]
            stims[sname] = view.attrs['stim']

    package = Package(data)
    # Keep the metadata as a plain dict (rather than a JSON string) so that
    # JSMixer.addData can merge newly added dataviews into it at runtime. It
    # is serialized to JSON on demand, when the mixer page is generated.
    metadata = package.metadata()
    images = package.images
    subjects = list(package.subjects)

    ctmargs = dict(method='mg2', level=9, recache=recache,
        external_svg=overlay_file, overlays_available=overlays_available)
    ctms = dict((subj, utils.get_ctmpack(subj, types, **ctmargs))
                for subj in subjects)
    package.reorder(ctms)

    subjectjs = json.dumps(dict((subj, "ctm/%s/"%subj) for subj in subjects))
    db.auxfile = None


    linear = lambda x, y, m: (1.-m)*x + m*y
    mixes = dict(
        linear=linear,
        smoothstep=(lambda x, y, m: linear(x, y, 3*m**2 - 2*m**3)),
        smootherstep=(lambda x, y, m: linear(x, y, 6*m**5 - 15*m**4 + 10*m**3))
    )

    post_name: Queue[str] = Queue()

    # Put together all view options
    my_viewopts: dict[str, Any] = dict(options.config.items('webgl_viewopts'))
    my_viewopts['overlays_visible'] = overlays_visible
    my_viewopts['labels_visible'] = labels_visible
    my_viewopts["brightness"] = (
        options.config.get("curvature", "brightness")
        if curvature_brightness is None
        else curvature_brightness
    )
    my_viewopts["contrast"] = (
        options.config.get("curvature", "contrast")
        if curvature_contrast is None
        else curvature_contrast
    )
    my_viewopts["smoothness"] = (
        options.config.get("curvature", "webgl_smooth")
        if curvature_smoothness is None
        else curvature_smoothness
    )
    my_viewopts["specularity"] = (
        options.config.get("webgl_viewopts", "specularity")
        if surface_specularity is None
        else surface_specularity
    )

    for sec in options.config.sections():
        if 'paths' in sec or 'labels' in sec:
            my_viewopts[sec] = dict(options.config.items(sec))

    # Views saved in the filestore, for the "camera > views" menu. Only the
    # subjects this viewer displays are read.
    my_viewopts['saved_views'] = _load_saved_views(subjects)

    # So the animation panel can say what render size reproduces the png
    # quickflat.make_png writes by default, for animations using the flat view.
    my_viewopts['quickflat_size'] = {subj: _quickflat_size(subj)
                                     for subj in subjects}

    # Where the animation panel is allowed to write rendered frames. The browser
    # sends a path relative to this root and MovieHandler refuses anything that
    # resolves outside it; see MovieHandler below.
    movie_root = os.path.realpath(os.getcwd() if movie_dir is None else movie_dir)
    movie_token = secrets.token_urlsafe(32)
    my_viewopts['movie_post'] = dict(url="movie", token=movie_token,
                                     root=movie_root)

    if pickerfun is None:
        pickerfun = lambda *a: None

    class CTMHandler(web.RequestHandler):
        def get(self, path: str):
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
                self.write(open(os.path.join(fpath, path), 'rb').read())

    class DataHandler(web.RequestHandler):
        def get(self, path: str):
            path = path.strip("/")
            frame: Union[int, str]
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

        def get(self, path: str):
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
            generated = html.generate(data=json.dumps(metadata),
                                      colormaps=colormaps,
                                      default_cmap="RdBu_r",
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

    class MovieHandler(web.RequestHandler):
        """Writes one animation frame rendered by the viewer's animation panel.

        Kept separate from MixerHandler.post, which pairs uploads with filenames
        by the order they were pushed onto `post_name`: a browser-driven render
        loop has no way to keep that queue in step, so each frame carries its own
        destination instead.

        The destination is always resolved underneath `movie_root` (see the
        `movie_dir` argument of show). The server binds all interfaces and serves
        the page unauthenticated, so the token below only keeps unrelated local
        processes out -- `movie_root` is what stops this from being an arbitrary
        file-write primitive.
        """
        def post(self):
            # Compare as bytes: compare_digest rejects non-ASCII str outright,
            # which would turn a hostile token into a 500 instead of a 403.
            sent = self.get_argument("token", "").encode("utf-8", "replace")
            if not hmac.compare_digest(sent, movie_token.encode("utf-8")):
                self.set_status(403)
                self.finish("Bad or missing token")
                return

            name = self.get_argument("name", "frame")
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
                self.set_status(400)
                self.finish("Invalid frame name: use letters, digits, '_', '-' and '.'")
                return

            try:
                frame = int(self.get_argument("frame"))
            except (TypeError, ValueError):
                self.set_status(400)
                self.finish("Invalid or missing frame number")
                return

            dest = os.path.realpath(os.path.join(movie_root,
                                                 self.get_argument("dir", "")))
            if dest != movie_root and not dest.startswith(movie_root + os.sep):
                self.set_status(403)
                self.finish("Refusing to write outside %s" % movie_root)
                return

            png = self.get_argument("png", default="")
            try:
                data = binascii.a2b_base64(png[png.index(",") + 1:].strip())
            except (ValueError, binascii.Error):
                self.set_status(400)
                self.finish("Could not decode png data")
                return

            try:
                os.makedirs(dest, exist_ok=True)
                fname = os.path.join(dest, "%s_%05d.png" % (name, frame))
                with open(fname, "wb") as fp:
                    fp.write(data)
            except OSError as err:
                self.set_status(500)
                self.finish("Could not write frame: %s" % err)
                return

            self.write(dict(path=fname))

    P = ParamSpec('P')

    class JSMixer(serve.JSProxy[P]):
        @property
        def view_props(self) -> list[str]:
            """An enumerated list of settable properties for views. 
            There may be a way to get this from the javascript object, 
            but I (ML) don't know how.

            There may be additional properties we want to set in views
            and animations; those must be added here.

            Old property list that used to be settable before webgl refactor:
            view_props = ['altitude', 'azimuth', 'target', 'mix', 'radius', 'pivot',
                'visL', 'visR', 'alpha', 'rotationR', 'rotationL', 'projection',
                'volume_vis', 'frame', 'slices']
            """
            camera = getattr(self.ui, "camera")
            _camera_props = ['camera.%s' % k for k in camera._controls.attrs.keys()]
            surface = getattr(self.ui, "surface")
            _subject = list(surface._folders.attrs.keys())[0]
            _surface = getattr(surface, _subject)
            _surface_props = ['surface.{subject}.%s'%k for k in _surface._controls.attrs.keys()]
            _curvature_props = ['surface.{subject}.curvature.brightness',
                                'surface.{subject}.curvature.contrast',
                                'surface.{subject}.curvature.smoothness']
            _lighting_props = ['surface.{subject}.lighting.topleft_lighting',
                               'surface.{subject}.lighting.uniform_illumination',
                               'surface.{subject}.lighting.specularity']
            return _camera_props + _surface_props + _curvature_props + _lighting_props

        # Lighting controls used to sit directly in the surface menu; they now
        # live in its lighting sub-folder. Keep the old names working, both for
        # user code and for views saved to the database before the move.
        _legacy_props = {
            'surface.{subject}.specularity':
                'surface.{subject}.lighting.specularity',
            'surface.{subject}.uniform_illumination':
                'surface.{subject}.lighting.uniform_illumination',
        }

        def _set_view(self, **kwargs):
            """Low-level command: sets view parameters in the current viewer

            Sets each the state of each keyword argument provided. View parameters
            that can be set include all parameters in the data.gui in the html view.

            """
            # Set unfolding level first, as it interacts with other arguments
            assert isinstance(self.ui, serve.JSProxy)
            surface: serve.JSProxy[P] = getattr(self.ui, "surface")
            subject_list = cast(serve.JSProxy[P], surface._folders).attrs.keys()
            # Better to only self.view_props once; it interacts with javascript, 
            # don't want to do that too often, it leads to glitches.
            vw_props = copy.copy(self.view_props)
            for old_key, new_key in self._legacy_props.items():
                if old_key in kwargs and new_key not in kwargs:
                    kwargs[new_key] = kwargs.pop(old_key)
            for subject in subject_list:
                if 'surface.{subject}.unfold' in kwargs:
                    unfold = kwargs.pop('surface.{subject}.unfold')
                    self.ui.set('surface.{subject}.unfold'.format(subject=subject), unfold)
                for k, v in kwargs.items():
                    if not k in vw_props:
                        print('Unknown parameter %s!'%k)
                        continue
                    else:
                        self.ui.set(k.format(subject=subject) if '{subject}' in k else k, v)
                        # Wait for webgl. Wait for it. .... WAAAAAIIIT.
                        time.sleep(0.03)

        def fit_flat_view(self) -> Optional[dict[str, Any]]:
            """Frame the flattened surface the way ``quickflat`` frames it.

            Points the camera at the middle of the flat surface and backs it off
            until the flatmap is as large as fits, which is what
            ``cortex.quickflat.make_png`` does with the bounds of the image it
            writes. The framing follows the shape of the frame, so it fills one
            exactly at the flatmap's own aspect ratio -- the subject's quickflat
            size (``cortex.webgl.view._quickflat_size``, which the viewer's
            animation panel fills into its render form) -- and fits inside any
            other shape rather than being cropped to it. ``getImage`` re-frames
            for the image it writes, so rendering a flat view at the quickflat
            size reproduces ``make_png``'s png whatever the window's shape.

            Only meaningful once the surface is flat and square-on to the
            camera, which is the pose the ``flat`` view sets.

            Returns
            -------
            dict or None
                The framing that was applied, as ``{"target": [x, y, z],
                "radius": r}``, or None if the subject has no flat surface.

            See Also
            --------
            getImage : re-frames what this framed for the image it writes.

            Notes
            -----
            Applied on request rather than by ``_set_view``, so that setting the
            flat view from python -- as :func:`cortex.export.save_3d_views` does
            -- keeps whatever framing the caller asked for.
            """
            resp = self.send(method="run",
                             params=["window.viewer.fitFlatView", []])
            return resp[0] if isinstance(resp, list) and len(resp) > 0 else None

        def _capture_view(self, frame_time=None):
            """Low-level command: returns a dict of current view parameters

            Retrieves the following view parameters from current viewer:

            altitude, azimuth, target, mix, radius, visL, visR, alpha,
            rotationR, rotationL, projection, pivot

            Parameters
            ----------
            frame_time : scalar
                time (in seconds) to specify for this frame.
            
            Notes
            -----
            If multiple subjects are present, only retrieves view for first subject.
            """
            view = {}
            subject = list(self.ui.surface._folders.attrs.keys())[0]
            for p in self.view_props:
                try:
                    view[p] = self.ui.get(p.format(subject=subject) if '{subject}' in p else p)[0]
                    # Wait for webgl.
                    time.sleep(0.03)
                except Exception as err:
                    # TO DO: Fix this hack with an error class in serve.py & catch it here
                    print(err) #msg = "Cannot read property 'undefined'"
                    #if err.message[:len(msg)] != msg:
                    #    raise err
            # A flat pose records no camera angle, since a flattened surface
            # ignores it (see FLAT_INERT_PROPS). An animation would otherwise
            # have something spurious to interpolate towards on the way in,
            # spinning the brain as it flattens and leaving the folded angle
            # overwritten on the way out. Mirrors vt.captureView in
            # resources/js/viewtools.js, so the two still interchange.
            from ..export.save_views import FLAT_INERT_PROPS

            if (view.get('surface.{subject}.unfold', 0) >= 0.999
                    and not view.get('surface.{subject}.allow_tilt')):
                for prop in FLAT_INERT_PROPS:
                    view.pop(prop, None)

            if frame_time is not None:
                view['time'] = frame_time
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
            """
            view = db.get_view(self, subject, name)

        def retrieve_new_views(self) -> dict[str, dict[str, Any]]:
            """Get views saved through the viewer's GUI.

            Returns the views created with the "save view" button in the viewer's
            camera menu. These live only in the browser until they are retrieved,
            which keeps them separate from the views that were loaded out of the
            filestore when the viewer started.

            Returns
            -------
            dict of str to dict
                Maps the name typed into the viewer to a dict of view parameters,
                in the same format as ``_capture_view``, so they can be passed
                straight to ``_set_view``. Use ``save_new_views`` to make them
                permanent.

            See Also
            --------
            save_new_views : write these views into the pycortex filestore.

            Notes
            -----
            If several subjects are displayed, only the first one's viewer is
            queried, mirroring the behavior of ``_capture_view``.
            """
            # One round trip, rather than the three that walking the proxy
            # attribute by attribute would cost (each level is a `query`).
            resp = self.send(method="run",
                             params=["window.viewer.getNewViews", []])
            val = resp[0] if isinstance(resp, list) and len(resp) > 0 else None
            if isinstance(val, dict) and "error" in val:
                raise Exception(val["error"])
            # `send` returns [None] when the browser does not answer in time.
            return cast(dict[str, dict[str, Any]], val) if isinstance(val, dict) else {}

        def save_new_views(self, subject: Optional[str]=None,
                           names: Optional[list[str]]=None,
                           is_overwrite: bool=False) -> dict[str, str]:
            """Store views saved through the viewer's GUI in the filestore.

            Writes each view created with the viewer's "save view" button to
            ``<filestore>/<subject>/views/<name>.json``, where the rest of
            pycortex looks for saved views: they show up in the camera > views
            menu of every viewer opened for that subject from then on, and can
            be applied with ``get_view``.

            A view that has been written is no longer "new". It moves into the
            running viewer's views menu and out of ``retrieve_new_views``, so
            calling this twice does not rewrite the same files.

            Parameters
            ----------
            subject : str or None, optional
                pycortex subject id to save the views under. Default None,
                meaning the first subject the viewer is displaying.
            names : list of str or None, optional
                Save only these views. Default None, meaning every view
                currently held in the viewer.
            is_overwrite : bool, optional
                Whether to replace views of the same name that are already in
                the filestore (default False).

            Returns
            -------
            dict of str to str
                Maps each saved view's name to the file it was written to.

            Raises
            ------
            KeyError
                If `names` mentions a view the viewer does not have.
            ValueError
                If a view name cannot be used as a filename.
            IOError
                If a view is already stored under that name and `is_overwrite`
                is False.

            See Also
            --------
            retrieve_new_views : get the same views without storing them.

            Examples
            --------
            >>> handle = cortex.webgl.show(volume)   # doctest: +SKIP
            >>> # ... position the brain and press "save view" in the viewer
            >>> handle.save_new_views()             # doctest: +SKIP
            {'lateral': '/path/to/filestore/S1/views/lateral.json'}
            """
            if subject is None:
                subject = subjects[0]

            new_views = self.retrieve_new_views()
            if names is None:
                names = sorted(new_views)
            else:
                missing = [n for n in names if n not in new_views]
                if len(missing) > 0:
                    raise KeyError(
                        "The viewer has no view named %s. Views already stored "
                        "in the filestore cannot be re-saved; the viewer holds "
                        "%s." % (", ".join(repr(n) for n in missing),
                                 ", ".join(repr(n) for n in sorted(new_views))
                                 or "nothing"))

            viewdir = os.path.join(db.filestore, subject, "views")
            # db.save_view leaves this to get_paths, which makes it a latent
            # FileNotFoundError for a subject imported without a views dir.
            os.makedirs(viewdir, exist_ok=True)

            # Check everything before writing anything, so that a name clash
            # partway through does not leave some views stored and some not.
            # The names come from a text field in the browser, so they also
            # have to be prevented from escaping the views directory.
            paths = {}
            for name in names:
                # A leading '.' is what makes '..' (and hidden files) possible,
                # so only that is kept out of the first position.
                if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9 _.-]*", name) is None:
                    raise ValueError(
                        "Cannot save the view named %r: a view name must start "
                        "with a letter, digit or '_' and contain only letters, "
                        "digits, spaces, '_', '-' and '.'" % name)
                path = os.path.join(viewdir, name + ".json")
                if os.path.exists(path) and not is_overwrite:
                    raise IOError(
                        "Refusing to over-write the extant view %s. If you want "
                        "to do this, set is_overwrite=True!" % path)
                paths[name] = path

            for name in names:
                with open(paths[name], "w") as fp:
                    json.dump(new_views[name], fp)
                # Now that it is on disk it belongs with the loaded views.
                self.send(method="run",
                          params=["window.viewer.promoteNewView", [name]])

            return paths

        def addData(self, **kwargs):
            """Add (or replace) dataviews in the running viewer.

            This makes it possible to push new data to an already open
            viewer, without restarting the server::

                client = cortex.webshow(volume)
                client.addData(second=other_volume)

            Parameters
            ----------
            kwargs : dict of str to Dataview
                Named dataviews to add to the viewer. A name that is already
                displayed replaces the corresponding dataview. The viewer
                switches to the first of the newly added dataviews, mirroring
                the behavior of the initial page load.

            Returns
            -------
            The response of the javascript ``viewer.addData`` call.

            Notes
            -----
            All new dataviews must belong to a subject that was already
            present when the viewer was created: the surfaces (and the vertex
            re-ordering they imply) are baked into the page at startup.
            """
            Proxy = serve.JSProxy(self.send, "window.viewer.addData")

            new_data = dataset.Dataset(**kwargs)
            new_package = Package(new_data)
            unknown = set(new_package.subjects) - set(subjects)
            if len(unknown) > 0:
                raise ValueError(
                    "Cannot add data for subject(s) %s: the viewer was "
                    "started with subject(s) %s, and surfaces cannot be "
                    "added to a running viewer."
                    % (", ".join(sorted(unknown)), ", ".join(sorted(subjects))))

            # Vertex data has to be reordered to match the vertex order of the
            # CTM files that were generated when the viewer was started.
            new_package.reorder(ctms)
            new_metadata = new_package.metadata()

            # Serve the images of the new dataviews, and make the new
            # dataviews part of the metadata used to (re)generate the page, so
            # that reloading the viewer shows everything that was added.
            images.update(new_package.images)
            new_names = set(view["name"] for view in new_metadata["views"])
            metadata["views"] = [view for view in metadata["views"]
                                 if view["name"] not in new_names]
            metadata["views"].extend(new_metadata["views"])
            metadata["data"].update(new_metadata["data"])
            metadata["images"].update(new_metadata["images"])

            # Forget the brains that no dataview refers to anymore, so that
            # repeatedly refreshing the same dataview does not pile up unused
            # image buffers in the server.
            referenced = set()
            for view in metadata["views"]:
                for brain in view["data"]:
                    # 2D dataviews refer to a pair of brains.
                    referenced.update(brain if isinstance(brain, list) else [brain])
            for brain in set(metadata["data"]) - referenced:
                del metadata["data"][brain]
                del metadata["images"][brain]
                images.pop(brain, None)

            for name, view in new_data:
                if 'stim' in view.attrs and os.path.exists(view.attrs['stim']):
                    stims[os.path.split(view.attrs['stim'])[1]] = view.attrs['stim']

            # Only the new dataviews are sent over: the javascript side keeps
            # the ones it already knows about.
            return Proxy(new_metadata)

        def getImage(self, filename: str, size: tuple[int, int]=(1920, 1080)):
            """Saves currently displayed view to a .png image file

            Parameters
            ----------
            filename : string
                duh.
            size : tuple (x, y)
                size (in pixels) of image to save.

            Notes
            -----
            A flatmap that ``fit_flat_view`` (or the viewer's own ``flat`` view)
            has framed is re-framed for the image being written, since the
            framing follows the shape of the frame and `size` need not have the
            shape of the window. That is what makes a framed flat view rendered
            at the subject's quickflat size come out as the png
            ``cortex.quickflat.make_png`` writes. A camera that is sitting
            anywhere else -- which is every camera this method has not been
            asked to frame -- is left exactly where it is.
            """
            self.send(method="run", params=["window.viewer.refitFlatView",
                                            [size[0] / size[1]]])
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

            Parameters
            ----------
            keyframes : list of dicts
                Each holds a 'time' in seconds plus view properties, in the form
                ``_capture_view`` returns. A keyframe may also carry an
                'interpolation' key naming its own
                :class:`~cortex.webgl.interpolation.Interpolation` mode, which
                is how the browser's animation panel stores per-keyframe
                smoothing.
            fps : int, optional
                Frame rate the times are quantized to. Default 30.
            interpolation : str, optional
                Either one of the three whole-animation easings, 'linear',
                'smoothstep' or 'smootherstep', or the name of one of the eight
                per-keyframe modes in
                :class:`~cortex.webgl.interpolation.Interpolation` -- in which
                case it supplies the mode for keyframes that do not name one of
                their own. Default 'linear'.

            Returns
            -------
            list of dicts
                One view dict per frame of the animation.

            Notes
            -----
            The per-keyframe modes interpolate each property across the whole
            keyframe list rather than between neighbouring pairs, because the
            tangent at a keyframe depends on the keyframes on both sides of it.
            The arithmetic is shared with the browser through
            ``cortex/webgl/interpolation.py`` and its javascript twin, so an
            animation built in the viewer renders the same way here.
            """
            if interpolation not in mixes or any(
                    'interpolation' in frame for frame in keyframes):
                return self._get_smoothed_anim_seq(keyframes, fps, interpolation)

            # Misc. setup
            fr = 0
            a = np.array
            func = mixes[interpolation]
            #skip_props = ['surface.{subject}.right', 'surface.{subject}.left', ] #'projection',
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
                nvalues = np.round(tdif/fs).astype(int)
                if use_endpoint:
                    nvalues += 1
                fr_time = np.linspace(0, 1, nvalues, endpoint=use_endpoint)
                # Interpolate between values
                for t in fr_time:
                    frame = {}
                    # The union of the two: a property only one of them carries
                    # is the one keyframe of the pair that constrains it, so it
                    # holds that value across the segment. Flat keyframes carry
                    # no camera angle (see FLAT_INERT_PROPS), which is what this
                    # is for.
                    for prop in list(start.keys()) + [
                            p for p in end.keys() if p not in start]:
                        if prop=='time':
                            continue
                        if prop not in start:
                            frame[prop] = end[prop]
                            continue
                        if (start[prop] is None) or (prop not in end) or (start[prop] == end[prop]) or isinstance(start[prop], (bool, str)):
                            frame[prop] = start[prop]
                            continue
                        val = func(a(start[prop]), a(end[prop]), t)
                        if isinstance(val, np.ndarray):
                            frame[prop] = val.tolist()
                        else:
                            frame[prop] = val
                    allframes.append(frame)
            return allframes

        def _get_smoothed_anim_seq(self, keyframes, fps=30,
                                   interpolation='Bezier'):
            """``_get_anim_seq`` for the per-keyframe interpolation modes.

            Kept separate from the pairwise path above rather than replacing
            it: 'smoothstep' and 'smootherstep' ease a whole segment and have
            no per-keyframe equivalent, and leaving that code untouched is the
            cheapest guarantee that existing animations still render frame for
            frame as they did.

            The frame times are generated exactly as the pairwise path
            generates them, so the two produce the same number of frames for
            the same keyframes; only the values differ.

            One value differs in kind rather than degree: 'camera.azimuth' is
            unwrapped before it is interpolated, so a spin takes the short way
            around and 350 -> 10 degrees crosses zero instead of running all
            the way back. That matches the viewer, whose own playback has always
            done this (Viewer._animInterp in resources/js/mriview.js), and it is
            what makes an animation laid out in the panel render the same way
            here. The pairwise path is left alone and still runs the long way.
            """
            if not keyframes:
                return []
            if interpolation in mixes:
                # 'linear' reaches here when a keyframe names its own mode. The
                # two spellings mean the same curve, so map it across; the other
                # two legacy easings have no per-keyframe form and are rejected.
                if interpolation != 'linear':
                    raise ValueError(
                        "interpolation=%r eases a whole segment and cannot be "
                        "combined with per-keyframe modes; use one of %s"
                        % (interpolation,
                           ", ".join(m.value for m in Interpolation)))
                interpolation = Interpolation.Linear
            try:
                default_mode = Interpolation(interpolation)
            except ValueError:
                raise ValueError(
                    "Unknown interpolation %r; expected one of %s, or one of "
                    "the whole-animation easings %s"
                    % (interpolation,
                       ", ".join(m.value for m in Interpolation),
                       ", ".join(sorted(mixes)))) from None

            # Quantize to the frame grid on copies. The pairwise path rewrites
            # the caller's dicts in place; there is no reason to inherit that.
            fs = 1. / fps
            frames = [dict(frame) for frame in keyframes]
            for frame in frames:
                frame['time'] = np.round(frame['time'] / fs) * fs
            frames.sort(key=lambda frame: frame['time'])

            channels = build_channels(frames, time_key='time',
                                      default_mode=default_mode)

            allframes = []
            for start, end in zip(frames[:-1], frames[1:]):
                t0, t1 = start['time'], end['time']
                use_endpoint = end is frames[-1]
                nvalues = np.round((t1 - t0) / fs).astype(int)
                if use_endpoint:
                    nvalues += 1
                for t in np.linspace(0, 1, nvalues, endpoint=use_endpoint):
                    allframes.append(evaluate(channels, t0 + t * (t1 - t0)))
            return allframes

        def make_movie_views(self, animation, filename="brainmovie%07d.png",
            offset=0, fps=30, size=(1920, 1080), alpha=1, frame_sleep=0.05,
            frame_start=0, interpolation="linear"):
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
            interpolation : str
                How values between keyframes are found. Either one of the
                whole-animation easings "linear", "smoothstep" or
                "smootherstep", which blend each pair of neighbouring
                keyframes, or one of the per-keyframe modes named by
                :class:`~cortex.webgl.interpolation.Interpolation` -- "Bezier",
                "CubicHermite", "Linear", "BezierInHoldOut",
                "CubicHermiteInHoldOut", "LinearInHoldOut",
                "LinearInBezierOut", "LinearInCubicHermiteOut" -- which fit a
                curve through the whole keyframe list and so carry velocity
                smoothly through the interior keyframes. Default "linear".

            Notes
            -----
            Make sure that all values that will be modified over the course
            of the animation are initialized (have some starting value) in the first
            frame.

            An individual keyframe may override `interpolation` by carrying its
            own "interpolation" key, which is how animations built in the
            viewer's animation panel store per-keyframe smoothing. The two
            implementations share their arithmetic, so such an animation renders
            here exactly as it played in the browser.

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
            allframes = self._get_anim_seq(animation, fps, interpolation)
            for fr, frame in enumerate(allframes[frame_start:], frame_start):
                self._set_view(**frame)
                time.sleep(frame_sleep)
                self.getImage(filename%(fr+offset+1), size=size)
                time.sleep(frame_sleep)

    class PickerHandler(web.RequestHandler):
        def get(self):
            voxel_arg = self.get_argument("voxel", None)
            if voxel_arg is None:
                self.set_status(400)
                self.finish("Missing 'voxel' query parameter")
                return
            parts = voxel_arg.split(",")
            if len(parts) != 3:
                self.set_status(400)
                self.finish("Invalid 'voxel' query parameter: expected 3 comma-separated integers")
                return
            try:
                voxel: tuple[int, int, int] = tuple(int(i) for i in parts)
                vertex: int = int(self.get_argument("vertex"))
            except (TypeError, ValueError):
                self.set_status(400)
                self.finish("Invalid 'voxel' or 'vertex' query parameter")
                return
            hemi: str = self.get_argument("hemi")
            pickerfun(voxel, vertex, hemi)

    class WebApp(serve.WebApp):
        disconnect_on_close = autoclose
        def get_client(self):
            self.connect.wait()
            self.connect.clear()
            return JSMixer(self.send, "window.viewer")

        def get_local_client(self):
            return JSMixer(self.srvsend, "window.viewer")

    if port is None:
        # Let the OS assign a guaranteed-free ephemeral port (WebApp binds
        # port 0 and reads the real port back). This avoids the random-port
        # collisions that made headless/CI runs intermittently hang.
        port = 0

    server = WebApp([(r'/ctm/(.*)', CTMHandler),
                     (r'/data/(.*)', DataHandler),
                     (r'/stim/(.*)', StimHandler),
                     (r'/mixer.html', MixerHandler),
                     (r'/movie', MovieHandler),
                     (r'/picker', PickerHandler),
                     (r'/', MixerHandler),
                     (r'/static/(.*)', StaticHandler)],
                    port)

    server.start()
    print("Started server on port %d"%server.port)
    url = "http://%s%s:%d/mixer.html"%(serve.hostname, domain_name, server.port)
    if open_browser:
        webbrowser.open(url)
        client = server.get_client()
        client.server = server
        return client
    elif display_url:
        try:
            from IPython.display import HTML, display
            display(HTML('Open viewer: <a href="{0}" target="_blank">{0}</a>'.format(url)))
        except:
            pass

    return server
