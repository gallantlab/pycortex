import os

import re
import copy
import shlex
import tempfile
import itertools
import numpy as np
import subprocess as sp
from matplotlib.path import Path
from scipy.spatial import cKDTree
from builtins import zip, str

from distutils.version import LooseVersion

from lxml import etree
from lxml.builder import E

from .options import config
from .testing_utils import INKSCAPE_VERSION

svgns = "http://www.w3.org/2000/svg"
inkns = "http://www.inkscape.org/namespaces/inkscape"
sodins = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"

parser = etree.XMLParser(remove_blank_text=True, huge_tree=True)

cwd = os.path.abspath(os.path.split(__file__)[0])

class SVGOverlay(object):
    """Object to represent all vector graphic overlays (rois, sulci, etc) stored in an svg file

    This object facilitates interaction with the information in the overlays.svg files
    that exist for each subject in the pycortex database.

    Parameters
    ----------
    svgfile : string
        svg file to read in. Must be formatted like the overlays.svg files in pycortex's
        filestore
    coords : array-like
        (Unclear...)
    overlays_available : list or tuple
        list of layers of svg file to extract. If None, extracts all overlay layers 
        (i.e. all layers that do not contain images)
    """
    def __init__(self, svgfile, coords=None, overlays_available=None):
        self.svgfile = svgfile
        self.overlays_available = overlays_available
        self.reload()
        if coords is not None:
            self.set_coords(coords)

    def reload(self):
        """Initial load of data from svgfile

        Strips out `data` layer of svg file, saves only layers consisting of vector paths.
        """
        self.svg = scrub(self.svgfile, overlays_available=self.overlays_available)
        w = float(self.svg.getroot().get("width"))
        h = float(self.svg.getroot().get("height"))
        self.svgshape = w, h

        # Grab relevant layers
        self.layers = dict()
        
        for layer in self.svg.getroot().findall("{%s}g"%svgns):
            layer = Overlay(self, layer)
            self.layers[layer.name] = layer

    def set_coords(self, coords):
        """Unclear what this does. James??"""
        # Normalize coordinates 0-1
        if np.any(coords.max(0) > 1) or np.any(coords.min(0) < 0):
            coords -= coords.min(0)
            coords /= coords.max(0)
        # Renormalize coordinates to shape of svg
        self.coords = coords * self.svgshape
        # Update of scipy (0.16+) means that cKDTree hangs / takes absurdly long to compute with new default
        # balanced_tree=True. Seems only to be true on Mac OS, for whatever reason. Possibly a broken
        # C library, unclear. Setting balanced_tree=False seems to resolve the issue, thus going with that for now
        # See http://stackoverflow.com/questions/31819778/scipy-spatial-ckdtree-running-slowly
        try:
            # not compatible with scipy version < 0.16
            self.kdt = cKDTree(self.coords, balanced_tree=False)
        except:
            # Older call signature
            self.kdt = cKDTree(self.coords)

        for layer in self:
            for name in layer.labels.elements:
                for element in layer.labels.elements[name]:
                    x, y = float(element.get("x")), float(element.get("y"))
                    dist, idx = self.kdt.query((x, self.svgshape[1]-y))
                    if idx >= len(self.kdt.data):
                        idx = 0
                    element.attrib['data-ptidx'] = str(idx)

    def __getattr__(self, attr):
        return self.layers[attr]

    def __dir__(self):
        return list(self.layers.keys()) + ['svg', 'svgfile', 'svgshape']

    def __repr__(self):
        return "<SVGOverlay with layers [%s]>"%(','.join(self.layers.keys()))

    def __iter__(self):
        return iter(self.layers.values())

    def add_layer(self, name):
        """Add a layer to the svgfile on which this object is based

        Adds a new layer named `name` to the svgfile by the SVGOverlay object, and
        overwrites the original file (incorporating the new layer).
        """
        svg = etree.parse(self.svgfile, parser=parser)
        layer = _make_layer(svg.getroot(), name)
        shapes = _make_layer(layer, "shapes")
        shapes.attrib['id'] = "%s_shapes"%name
        shapes.attrib['clip-path'] = "url(#edgeclip)"
        labels = _make_layer(layer, "labels")
        labels.attrib['id'] = "%s_labels"%name
        with open(self.svgfile, "wb") as fp:
            #try:
            fp.write(etree.tostring(svg, pretty_print=True)) # python2.X
            #except:
            #    fp.write(etree.tostring(svg, encoding=str, pretty_print=True)) # python3.X
        self.reload()

    def toxml(self, pretty=True):
        """Return a string xml version of the SVGOverlay object"""
        return etree.tostring(self.svg, pretty_print=pretty)

    def get_svg(self, filename=None, layers=['rois'], labels=True, with_ims=None):
        """Returns a new SVG file with images embedded

        Parameters
        ----------
        filename : string
            File path to which to write new svg
        layers : list
            List of layer names to show
        labels : boolean
            Whether labels should be visible or not
        with_ims : list
            list of images to incorporate into new svg file. The first image
            listed will be on the uppermost layer, the last will be lowest.
        """
        outsvg = self.svg

        if with_ims is not None:
            if isinstance(with_ims, (list, tuple)):
                with_ims = zip(range(len(with_ims)), with_ims)

            datalayer = _make_layer(outsvg.getroot(), "data")
            for imnum, im in reversed(list(with_ims)):  # need list() with zip for python 3.5 compatibility
                imlayer = _make_layer(datalayer, "image_%d" % imnum)
                img = E.image(
                    {"{http://www.w3.org/1999/xlink}href":"data:image/png;base64,%s"%str(im,'utf-8')},
                    id="image_%d"%imnum, x="0", y="0",
                    width=str(self.svgshape[0]),
                    height=str(self.svgshape[1]),
                    )
                imlayer.append(img)
                outsvg.getroot().insert(0, imlayer)

        for layer in self:
            if layer.name in layers:
                layer.visible = True
                layer.labels.visible = labels
                for name_, shape_ in layer.shapes.items():
                    shape_.visible = True
                    # Set visibility of labels (by setting text alpha to 0)
                    # This could be less baroque, but text elements currently
                    # do not have individually settable visibility / style params
                    tmp_style = copy.deepcopy(layer.labels.text_style)
                    tmp_style['fill-opacity'] = '1' if labels else '0'
                    tmp_style_str = ';'.join(['%s:%s'%(k,v) for k, v in tmp_style.items() if v != 'None'])
                    for i in range(len(layer.labels.elements[name_])):
                        layer.labels.elements[name_][i].set('style', tmp_style_str)
            else:
                layer.visible = False
                layer.labels.visible = False

        with open(filename, "wb") as outfile:
            outfile.write(etree.tostring(outsvg))
        print('Saved SVG to: %s'%filename)

    def get_texture(self, layer_name, height, name=None, background=None, labels=True,
        shape_list=None, **kwargs):
        """Renders a specific layer of this svgobject as a png

        Parameters
        ----------
        layer_name : string
            Name of layer of svg file to be rendered
        height : scalar
            Height of image to be generated
        name : string
            If `background` is specified, provides a name for the background image
        background : idkwtf
            An image? Unclear.
        labels : boolean
            Whether to render labels for paths in the svg file
        shape_list : list
            list of string names for path/shape elements in this layer to be rendered
            (any elements not on this list will be set to invisible, if this list is
            provided)
        kwargs : keyword arguments
            keywords to specify display properties of svg path objects, e.g. {'stroke':'white',
            'stroke-width':2} etc. See inkscape help for names for properties. This function
            is used by quickflat.py, which provides dictionaries to map between more matplotlib-
            like properties (linecolor->stroke, linewidth->stroke-width) for an easier-to-use API.

        Returns
        -------
        image : array
            Rendered image of svg layer with specified parameters

        Notes
        -----
        missing bits=32 keyword input argument, did not seeme necessary to specify
        png bits.
        """
        # Give a more informative error in case we don't have inkscape
        # installed
        if INKSCAPE_VERSION is None:
            raise RuntimeError(
                "Inkscape doesn't seem to be installed on this system."
                "SVGOverlay.get_texture requires inkscape."
                "Please make sure that inkscape is installed and that is "
                "accessible from the terminal.")

        import matplotlib.pyplot as plt
        # Set the size of the texture
        if background is not None:
            img = E.image(
                {"{http://www.w3.org/1999/xlink}href":"data:image/png;base64,%s"%background},
                id="image_%s"%name, x="0", y="0",
                width=str(self.svgshape[0]),
                height=str(self.svgshape[1]),
            )
            self.svg.getroot().insert(0, img)
        if height is None:
            height = self.svgshape[1]
        height = int(height)
        #label_defaults = _parse_defaults(layer+'_labels')
        
        # separate kwargs starting with "label-"
        label_kwargs = {k[6:]:v for k, v in kwargs.items() if k[:6] == "label-"}
        kwargs = {k:v for k, v in kwargs.items() if k[:6] != "label-"}
        for layer in self:
            if layer.name == layer_name:
                layer.visible = True
                layer.labels.visible = labels
                for name_, shape_ in layer.shapes.items():
                    # honor visibility set in the svg
                    if shape_list is not None:
                        shape_.visible = name_ in shape_list
                    # Set visibility of labels (by setting text alpha to 0)
                    # This could be less baroque, but text elements currently
                    # do not have individually settable visibility / style params
                    tmp_style = copy.deepcopy(layer.labels.text_style)
                    tmp_style['fill-opacity'] = '1' if shape_.visible else '0'
                    tmp_style.update(label_kwargs)
                    tmp_style_str = ';'.join(['%s:%s'%(k,v) for k, v in tmp_style.items() if v != 'None'])
                    for i in range(len(layer.labels.elements[name_])):
                        layer.labels.elements[name_][i].set('style', tmp_style_str)
                layer.set(**kwargs)
            else:
                layer.visible = False
                layer.labels.visible = False

        pngfile = name
        if name is None:
            png = tempfile.NamedTemporaryFile(suffix=".png")
            pngfile = png.name

        inkscape_cmd = config.get('dependency_paths', 'inkscape')
        if LooseVersion(INKSCAPE_VERSION) < LooseVersion('1.0'):
            cmd = "{inkscape_cmd} -z -h {height} -e {outfile} /dev/stdin"
        else:
            cmd = "{inkscape_cmd} -h {height} --export-filename {outfile} " \
                  "/dev/stdin"
        cmd = cmd.format(inkscape_cmd=inkscape_cmd, height=height, outfile=pngfile)
        proc = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = proc.communicate(etree.tostring(self.svg))
        
        # print stderr, except the warning "Format autodetect failed."
        if hasattr(stderr, 'decode'):
            stderr = stderr.decode()
        for line in stderr.split('\n'):
            if line != '' and 'Format autodetect failed.' not in line:
                print(line)

        if background is not None:
            self.svg.getroot().remove(img)

        if name is None:
            png.seek(0)
            try:
                im = plt.imread(png)
            except SyntaxError as e:
                raise RuntimeError(f"Error reading image from {pngfile}: {e}"
                                   f" (inkscape version: {INKSCAPE_VERSION})"
                                   f" (inkscape command: {inkscape_cmd})"
                                   f" (stdout: {stdout})"
                                   f" (stderr: {stderr})")
            return im

class Overlay(object):
    """Class to represent a single layer of an SVG file
    """
    def __init__(self, svgobject, layer):
        self.svgobject = svgobject
        self.svg = svgobject.svg
        self.layer = layer
        self.name = layer.attrib['{%s}label'%inkns]
        self.layer.attrib['class'] = 'display_layer'

        # Check to see if the layer is locked, to see if we need to override the style
        locked = '{%s}insensitive'%sodins
        self.shapes = dict()
        for layer_ in _find_layer(layer, "shapes").findall("{%s}g"%svgns):
            override = locked not in layer_.attrib or layer_.attrib[locked] == "false"
            shape = Shape(layer_, self.svgobject.svgshape[1], override_style=override)
            self.shapes[shape.name] = shape

        self.labels = Labels(self)

    def __repr__(self):
        return "<svg layer with shapes [%s]>"%(','.join(self.shapes.keys()))

    def __getitem__(self, name):
        return self.shapes[name]

    @property
    def visible(self):
        # assume visible if "style" property is not set
        if 'style' not in self.layer.attrib:
            return True
        else:
            return 'none' not in self.layer.attrib['style']

    @visible.setter
    def visible(self, value):
        style = "display:inline;" if value else "display:none;"
        self.layer.attrib['style'] = style

    def set(self, **kwargs):
        for shape in list(self.shapes.values()):
            shape.set(**kwargs)

    def get_mask(self, name):
        return self.shapes[name].get_mask(self.svgobject.coords)

    def add_shape(self, name, pngdata=None, add_path=True):
        """Adds projected data for defining a new ROI to the saved overlays.svg file in a new layer"""
        # self.svg deletes the images -- we want to save those, so let's load it again
        svg = etree.parse(self.svgobject.svgfile, parser=parser)
        imglayer = _find_layer(svg, "data")
        if add_path:
            layer = _find_layer(svg, self.name)
            _make_layer(_find_layer(layer, "shapes"), name)

        # Hide all the other layers in the image
        for layer in imglayer.findall(".//{%s}g"%svgns):
            layer.attrib["style"] = "display:hidden;"

        layer = _make_layer(imglayer, "img_%s"%name)
        layer.append(E.image(
            {"{http://www.w3.org/1999/xlink}href":"data:image/png;base64,%s"%pngdata},
            id="image_%s"%name, x="0", y="0",
            width=str(self.svgobject.svgshape[0]),
            height=str(self.svgobject.svgshape[1]),
        ))

        with open(self.svgobject.svgfile, "wb") as xml:
            #try:
            xml.write(etree.tostring(svg, pretty_print=True)) # python2.X
            #except:
            #    xml.write(etree.tostring(svg, encoding=str, pretty_print=True)) # python3.X

class Labels(object):
    def __init__(self, overlay):
        self.overlay = overlay
        self.layer = _find_layer(self.overlay.layer, "labels")
        # This should be layer-specific,and read from different fields in the options.cfg file
        # if possible, falling back to overlay_text if the speific layer defaults don't exist.
        # Don't have time to figure out how to get the layer name here (rois, sulci, etc)
        self.text_style = dict(config.items("overlay_text"))

        text_style = self.text_style.items()
        text_style = ';'.join(['%s:%s'%(k,v) for k, v in text_style if v != 'None'])

        #generate a list of labels that should be in the layer
        self.elements = dict()
        for shape in self.overlay.shapes.values():
            self.elements[shape.name] = shape.get_labelpos()

        # match up existing labels with their respective paths
        def close(pt, x, y):
            try:
                xx, yy = pt[0], pt[1]
            except IndexError:  # when loading overlay from a dataset pack
                xx, yy = float(pt.get('x')), float(pt.get('y'))
            return np.sqrt((xx - x)**2 + (yy-y)**2) < 250
        for text in self.layer.findall(".//{%s}text"%svgns):
            x = float(text.get('x'))
            y = float(text.get('y'))
            #check this element against all known paths
            for name in self.elements.keys():
                if text.text == name:
                    for i, pos in enumerate(self.elements[name]):
                        if close(pos, x, y):
                            self.elements[name][i] = text

        #add missing elements
        self.override = []
        for name in self.elements.keys():
            for i, pos in enumerate(self.elements[name]):
                if isinstance(pos, np.ndarray):
                    text = etree.SubElement(self.layer, "{%s}text"%svgns)
                    text.text = name
                    text.attrib["x"] = str(pos[0])
                    text.attrib["y"] = str(pos[1])
                    text.attrib['style'] = text_style
                    self.elements[name][i] = text
                    self.override.append(text)

    def set(self, override=False, **kwargs):
        self.text_style.update(kwargs)
        text_style = self.text_style.items()
        text_style = ';'.join(['%s:%s'%(k,v) for k, v in text_style if v != 'None'])

        labels = self.override
        if override:
            labels = self.labels.findall(".//{%s}text"%svgns)

        for element in labels:
            element.attrib['style'] = text_style

    @property
    def visible(self):
        return self.text_style['display'] != "none"

    @visible.setter
    def visible(self, value):
        if value:
            self.text_style['display'] = 'inline'
        else:
            self.text_style['display'] = 'none'
        self.set()

class Shape(object):
    def __init__(self, layer, height, override_style=True):
        self.layer = layer
        self.height = height
        self.name = layer.attrib['{%s}label'%inkns]
        self.paths = layer.findall('{%s}path'%svgns)

        #default style
        self.style = dict(config.items("overlay_paths"))

        locked = '{%s}insensitive'%sodins
        if not override_style or locked in layer.attrib:
            self._get_style()

        self.set()

    def _get_style(self):
        # populate the style dictionary with the first path that has a style tag
        for path in self.paths:
            if 'style' in path.attrib:
                style = dict(s.split(':') for s in path.attrib['style'].split(";"))
                self.style.update(style)
                break

    def set(self, **kwargs):
        self.style.update(**kwargs)
        style = ';'.join(['%s:%s'%(k,v) for k, v in self.style.items() if v != "None"])
        for path in self.paths:
            path.attrib['style'] = style

    def get_labelpos(self):
        labels = []
        for path in self.paths:
            pos = _parse_svg_pts(path.attrib['d'])
            labels.append(_center_pts(pos))

        return labels

    def get_mask(self, vts):
        """get list of vertices inside this roi"""
        if len(self.splines)==0:
            # No splines defined for this (ROI). Wut.
            import warnings
            warnings.warn("Requested layer in svg file (%s) contains no splines"%self.name)
            return []
        # Annoying: The paths created are upside-down wrt vertex coordinates. So flip them.
        verts_upside_down = copy.copy(vts)
        verts_upside_down[:, 1] = self.height - verts_upside_down[:, 1]
        verts_in_any_path = [p.contains_points(verts_upside_down) for p in self.splines]
        vert_idx_list = np.hstack([np.nonzero(v)[0] for v in verts_in_any_path])
        return  vert_idx_list

    @property
    def splines(self):
        return [gen_path(p) for p in self.paths]

    @property
    def visible(self):
        # assume visible if "style" property is not set
        if "style" not in self.layer.attrib:
            return True
        else:
            return 'none' not in self.layer.attrib['style']

    @visible.setter
    def visible(self, value):
        style = "display:inline;" if value else "display:none;"
        self.layer.attrib['style'] = style

###################################################################################
# SVG Helper functions
###################################################################################
def _find_layer_names(svg):
    layers = svg.findall("{%s}g[@{%s}label]"%(svgns, inkns))
    layer_names = [l.get("{%s}label"%inkns) for l in layers]
    return layer_names

def _find_layer(svg, label):
    layers = [l for l in svg.findall("{%s}g[@{%s}label]"%(svgns, inkns)) if l.get("{%s}label"%inkns) == label]
    if len(layers) < 1:
        raise ValueError("Cannot find layer %s"%label)
    return layers[0]

def _make_layer(parent, name):
    layer = etree.SubElement(parent, "{%s}g"%svgns)
    layer.attrib['id'] = name
    layer.attrib['style'] = "display:inline;"
    layer.attrib["{%s}label"%inkns] = name
    layer.attrib["{%s}groupmode"%inkns] = "layer"
    return layer

try:
    from shapely.geometry import Polygon

    def _center_pts(pts):
        '''Fancy label position generator, using erosion to get label coordinate'''
        min = pts.min(0)
        pts -= min
        max = pts.max(0)
        max[max == 0] = 1        
        pts /= max

        #probably don't need more than 20 points, reduce detail of the polys
        if len(pts) > 20:
            pts = pts[::len(pts)//20]

        try:
            if len(pts) < 3:
                raise RuntimeError()
            poly = Polygon([tuple(p) for p in pts])
            last_i = None
            for i in np.linspace(0,1,100):
                if poly.buffer(-i).is_empty:
                    if last_i is None:
                        raise RuntimeError()
                    a = list(poly.buffer(-last_i).centroid.coords)[0] * max + min
                    return a
                last_i = i

            import warnings
            warnings.warn("Unable to find zero centroid.")
            return list(poly.buffer(-100).centroid.coords)[0] * max + min
        except RuntimeError:
            return np.nanmean(pts, 0) * max + min

except (ImportError, OSError):
    import warnings
    warnings.warn("Cannot find shapely, using simple label placement.")

    def _center_pts(pts):
        return np.nanmean(pts, 0)


def _labelpos(pts):
    if pts.ndim < 3:
        return _center_pts(pts)

    ptm = pts.copy().astype(float)
    ptm -= ptm.mean(0)
    u, s, v = np.linalg.svd(ptm, full_matrices=False)
    sp = np.diag(s)
    sp[-1,-1] = 0
    try:
        x, y = _center_pts(np.dot(ptm, np.dot(v.T, sp))[:,:2])
    except Exception as e:
        print(e)

    sp = np.diag(1./(s+np.finfo(float).eps))
    pt = np.dot(np.dot(np.array([x,y,0]), sp), v)
    return pt + pts.mean(0)

def _split_multipath(pathstr):
    """Appropriately splits an SVG path with multiple sub-paths.
    """
    # m is absolute path, M is relative path (or vice versa?)
    if not pathstr[0] in ["m","M"]:
        raise ValueError("Bad path format: %s" % pathstr)
    import re
    subpaths = [sp for sp in re.split('[Mm]',pathstr) if len(sp)>0]
    headers = re.findall('[Mm]',pathstr)
    for subpath,header in zip(subpaths,headers):
        # Need further parsing of multi-path strings? perhaps no.
        yield (header + subpath).strip()

def scrub(svgfile, overlays_available=None):
    """Remove data layers from an svg object prior to rendering

    Returns etree-parsed svg object
    """
    svg = etree.parse(svgfile, parser=parser)
    try:
        layers_to_remove = ['data']
        if overlays_available is not None:
            overlays_to_remove = [x for x in _find_layer_names(svg) if x not in overlays_available]
            layers_to_remove = overlays_to_remove
        for layer in layers_to_remove:
            rmnode = _find_layer(svg, layer)
            rmnode.getparent().remove(rmnode)
    except ValueError:
        # Seems sketch - should catch this? 
        pass
    svgtag = svg.getroot()
    svgtag.attrib['id'] = "svgoverlay"
    inkver = "{%s}version"%inkns
    if inkver in svgtag.attrib:
        del svgtag.attrib[inkver]
    try:
        for tagname in ["{%s}namedview"%sodins, "{%s}metadata"%svgns]:
            for tag in svg.findall(".//%s"%tagname):
                tag.getparent().remove(tag)
    except:
        import traceback
        traceback.print_exc()

    return svg

def make_svg(pts, polys):
    from .polyutils import trace_poly, boundary_edges
    pts = pts.copy()
    pts -= pts.min(0)
    pts *= 1024 / pts.max(0)[1]
    pts[:,1] = 1024 - pts[:,1]
    path = ""
    left, right = trace_poly(boundary_edges(polys))

    for poly in [left, right]:
        path +="M%f %f L"%tuple(pts[poly.pop(0), :2])
        path += ', '.join(['%f %f'%tuple(pts[p, :2]) for p in poly])
        path += 'Z '

    w, h = pts.max(0)[:2]
    with open(os.path.join(cwd, "svgbase.xml")) as fp:
        svg = fp.read().format(width=w, height=h, clip=path)

    return svg

def get_overlay(subject, svgfile, pts, polys, remove_medial=False, 
                overlays_available=None, modify_svg_file=True, **kwargs):
    """Return a python represent of the overlays present in `svgfile`.

    Parameters
    ----------
    subject: str
        Name of the subject.
    svgfile: str
        File name with the overlays (.svg).
    pts: array of shape (n_vertices, 3)
        Coordinates of all vertices, as returned by for example by
        cortex.db.get_surf.
    polys: arrays of shape (n_polys, 3)
        Indices of the vertices of all polygons, as returned for example by
        cortex.db.get_surf.
    remove_medial: bool
        Whether to remove duplicate vertices. If True, the function also
        returns an array with the unique vertices.
    overlays_available: tuple or None
        Overlays to keep in the result. If None, then all overlay layers of
        the SVG file will be available in the result. If None, also add 3 empty
        layers named 'sulci', 'cutouts', and 'display' (if not already
        present).
    modify_svg_file: bool
        Whether to modify the SVG file when overlays_available=None, which can
        add layers 'sulci', 'cutouts', and 'display' (if not already present).
        If False, the SVG file will not be modified.
    **kwargs
        Other keyword parameters are given to the SVGOverlay constructor.
    
    Returns
    -------
    svg : SVGOverlay instance.
        Object with the overlays.
    valid : array of shape (n_vertices, )
        Indices of all vertices (without duplicates).
        Only returned if remove_medial is True.
    """
    cullpts = pts[:,:2]
    if remove_medial:
        valid = np.unique(polys)
        cullpts = cullpts[valid]

    if not os.path.exists(svgfile):
        # Overlay file does not exist yet! We need to create and populate it
        # I think this should be an entirely separate function, and it should
        # be made clear when this file is created - opening a git issue on 
        # this soon...ML
        print("Create new file: %s" % (svgfile, ))
        with open(svgfile, "wb") as fp:
            fp.write(make_svg(pts.copy(), polys).encode())

        svg = SVGOverlay(svgfile, coords=cullpts, **kwargs)

        ## Add default layers
        from .database import db
        import io
        from . import quickflat
        import binascii

        # Curvature
        for layer_name, cmap in zip(['curvature', 'sulcaldepth', 'thickness'], ['gray', 'RdBu_r', 'viridis']):
            try:
                curv = db.get_surfinfo(subject, layer_name)
            except:
                print("Failed to import svg layer for %s, continuing"%layer_name)
                continue
            curv.cmap = cmap
            vmax = np.abs(curv.data).max()
            curv.vmin = -vmax
            curv.vmax = vmax
            fp = io.BytesIO()
            quickflat.make_png(fp, curv, height=1024, with_rois=False, with_labels=False)
            fp.seek(0)
            svg.rois.add_shape(layer_name, binascii.b2a_base64(fp.read()).decode('utf-8'), False)

    else:
        if not modify_svg_file:
            # To avoid modifying the svg file, we copy it in a temporary file
            import shutil
            svg_tmp = tempfile.NamedTemporaryFile(suffix=".svg")
            svgfile_tmp = svg_tmp.name
            shutil.copy2(svgfile, svgfile_tmp)
            svgfile = svgfile_tmp

        svg = SVGOverlay(svgfile, 
                         coords=cullpts, 
                         overlays_available=overlays_available,
                         **kwargs)
    
    if overlays_available is None:
        # Assure all layers are present
        # (only if some set of overlays is not specified)
        # NOTE: this actually modifies the svg file.
        # Use allow_change=False to avoid modifying the svg file.
        for layer in ['sulci', 'cutouts', 'display']:
            if layer not in svg.layers:
                svg.add_layer(layer)

    if remove_medial:
        return svg, valid

    return svg

## From svg.path (https://github.com/regebro/svg.path/blob/master/src/svg/path/parser.py)
COMMANDS = set('MmZzLlHhVvCcSsQqTtAa')
UPPERCASE = set('MZLHVCSQTA')

COMMAND_RE = re.compile("([MmZzLlHhVvCcSsQqTtAa])")
FLOAT_RE = re.compile("[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

def _tokenize_path(pathdef):
    for x in COMMAND_RE.split(pathdef):
        if x in COMMANDS:
            yield x
        for token in FLOAT_RE.findall(x):
            yield token

def _parse_svg_pts(datastr):
    data = list(_tokenize_path(datastr))
    #data = data.replace(",", " ").split()
    if data.pop(0).lower() != "m":
        raise ValueError("Unknown path format")
    #offset = np.array([float(x) for x in data[1].split(',')])
    offset = np.array([float(x) for x in [data.pop(0), data.pop(0)]])
    mode = "l"
    pts = [[offset[0], offset[1]]]

    def canfloat(n):
        try:
            float(n)
            return True
        except ValueError:
            return False

    lastlen = len(data)
    while len(data) > 0:
        #print mode, data
        if not canfloat(data[0]):
            mode = data.pop(0)
            continue
        if mode == "l":
            offset += list([float(x) for x in [data.pop(0), data.pop(0)]])
        elif mode == "L":
            offset = np.array(list([float(x) for x in [data.pop(0), data.pop(0)]]))
        elif mode == "h":
            offset += list([float(x) for x in [data.pop(0), 0]])
        elif mode == 'H':
            offset = np.array(list([float(x) for x in [data.pop(0), 0]]))
        elif mode == "v":
            offset += list([float(x) for x in [0, data.pop(0)]])
        elif mode == "V":
            offset = np.array(list([float(x) for x in [0, data.pop(0)]]))
        elif mode == "c":
            data = data[4:]
            offset += list([float(x) for x in [data.pop(0), data.pop(0)]])
        elif mode == "C":
            data = data[4:]
            offset = np.array(list([float(x) for x in [data.pop(0), data.pop(0)]]))
        #support multi-part paths, by only using one label for the whole path
        elif mode == 'm' :
            offset += list([float(x) for x in [data.pop(0), data.pop(0)]])
        elif mode == 'M' :
            offset = list([float(x) for x in [data.pop(0), data.pop(0)]])

        ## Check to see if nothing has happened, and, if so, fail
        if len(data) == lastlen:
            raise ValueError("Error parsing path.")
        else:
            lastlen = len(data)

        pts.append([offset[0],offset[1]])

    return np.array(pts)

def import_roi(roifile, outfile):
    """Convert rois.svg file (from previous versions of pycortex) to overlays.svg"""
    import warnings
    warnings.warn("Converting rois.svg to overlays.svg")
    svg = etree.parse(roifile, parser=parser)

    label_layer = None
    for layer in svg.findall("{%s}g[@{%s}label]"%(svgns, inkns)):
        name = layer.get("{%s}label"%inkns)
        if name == "data":
            #maintain data layer, do not process
            pass
        elif name == "roilabels": #label layer
            label_layer = layer
            layer.getparent().remove(layer)
        else:
            parent = _make_layer(layer.getparent(), name)
            layer.getparent().remove(layer)
            layer.attrib['id'] = '%s_shapes'%name
            layer.attrib['{%s}label'%inkns] = 'shapes'
            layer.attrib['clip-path'] = "url(#edgeclip)"
            parent.append(layer)
            labels = _make_layer(parent, "labels")
            labels.attrib['id'] = '%s_labels'%name

    if label_layer is not None:
        rois = _find_layer(svg, "rois")
        labels = _find_layer(rois, 'labels')
        rois.remove(labels)

        label_layer.attrib['id'] = 'rois_labels'
        label_layer.attrib['{%s}label'%inkns] = 'labels'
        rois.append(label_layer)


    with open(outfile, "wb") as fp:
        fp.write(etree.tostring(svg, pretty_print=True))

    # Final check for all layers
    svgo = SVGOverlay(outfile)
    for new_layer in ['sulci', 'cutouts', 'display']:
        if new_layer not in svgo.layers:
            svgo.add_layer(new_layer)

def gen_path(path):
    mdict = dict(m=Path.MOVETO, l=Path.LINETO, h=Path.LINETO, v=Path.LINETO)
    verts, codes = [], []
    mode, pen = None, np.array([0.,0.])

    it = iter(path.get('d').strip().split(' '))
    run = True
    while run:
        try:
            cmd = next(it)
            if len(cmd) == 1:
                mode = cmd
                if cmd.lower() == 'z':
                    verts.append([0,0])
                    codes.append(Path.CLOSEPOLY)
            elif mode.lower() == 'c':
                p1 = [float(ss) for ss in cmd.split(',')]
                p2 = [float(ss) for ss in next(it).split(',')]
                p3 = [float(ss) for ss in next(it).split(',')]
                if mode == 'c':
                    verts.append(pen + p1)
                    verts.append(pen + p2)
                    verts.append(pen + p3)
                    pen += p3
                else:
                    verts.append(p1)
                    verts.append(p2)
                    verts.append(p3)
                    pen = np.array(p3)
                codes.append(Path.CURVE4)
                codes.append(Path.CURVE4)
                codes.append(Path.CURVE4)
            else:
                if mode.lower() == 'h':
                    val = [float(cmd), 0]
                elif mode.lower() == 'v':
                    val = [0, float(cmd)]
                else:
                    val = [float(cc) for cc in cmd.split(',')]
                codes.append(mdict[mode.lower()])
                if mode.lower() == mode:
                    pen += val
                    verts.append(pen.tolist())
                else:
                    pen = np.array(val)
                    verts.append(val)

                if mode == 'm':
                    mode = 'l'
                elif mode == 'M':
                    mode = 'L'

        except StopIteration:
            run = False
    return Path(verts, codes=codes)
