import tempfile
import subprocess as sp
from xml.dom.minidom import parse as xmlparse

import numpy as np
import traits.api as traits
from scipy.interpolate import griddata
from matplotlib.pyplot import imread

from db import options
default_lw = options['line_width'] if 'line_width' in options else 3.

class ROI(traits.HasTraits):
    name = traits.Str

    hide = traits.Bool(False)
    linewidth = traits.Float(default_lw)
    linecolor = traits.Tuple((0.,0.,0.,1.))
    roifill = traits.Tuple((0.,0.,0.,0.2))

    def __init__(self, parent, xml, **kwargs):
        super(ROI, self).__init__(**kwargs)
        self.parent = parent
        self.name = xml.getAttribute("inkscape:label")
        self.paths = xml.getElementsByTagName("path")
        pts = [ self._parse_svg_pts(path.getAttribute("d")) for path in self.paths]
        self.coords = [ griddata(parent.tcoords, np.arange(len(parent.tcoords)), p, "nearest") for p in pts ]
        self.hide = xml.hasAttribute("style") and "display:none" in xml.attributes['style'].value

        self.linewidth = self.parent.linewidth
        self.linecolor = self.parent.linecolor
        self.roifill = self.parent.roifill
    
    def _parse_svg_pts(self, data):
        data = data.split()
        if data[0].lower() != "m":
            raise ValueError("Unknown path format")
        offset = np.array([float(x) for x in data[1].split(',')])
        data = data[2:]
        mode = "l"
        pts = [[offset[0], offset[1]]]
        while len(data) > 0:
            d = data.pop(0)
            if isinstance(d, (unicode, str)) and len(d) == 1:
                mode = d
                continue
            if mode == "l":
                offset += map(float, d.split(','))
            elif mode == "L":
                offset = np.array(map(float, d.split(',')))
            elif mode == "c":
                data.pop(0)
                offset += map(float, data.pop(0).split(','))
            elif mode == "C":
                data.pop(0)
                offset = np.array(map(float, data.pop(0).split(',')))
            pts.append([offset[0],offset[1]])

        pts = np.array(pts)
        pts /= self.parent.svgshape
        pts[:,1] = 1-pts[:,1]
        return pts
    
    @traits.on_trait_change("linewidth, linecolor, roifill, hide")
    def update_attribs(self):
        style = "fill:{fill}; fill-opacity:{fo};stroke-width:{lw}px;"+\
                    "stroke-linecap:butt;stroke-linejoin:miter;"+\
                    "stroke:{lc};stroke-opacity:{lo}; {hide}"
        roifill = np.array(self.roifill)*255
        linecolor = np.array(self.linecolor)*255
        hide = "display:none;" if self.hide else ""
        style = style.format(
            fill="rgb(%d,%d,%d)"%tuple(roifill[:-1]), fo=self.roifill[-1],
            lc="rgb(%d,%d,%d)"%tuple(linecolor[:-1]), lo=self.linecolor[-1], 
            lw=self.linewidth, hide=hide)
        for path in self.paths:
            path.setAttribute("style", style)
    
    def get_labelpos(self, pts, norms, fancy=True):
        if fancy:
            labels = []
            for coord in self.coords:
                try:
                    labels.append((_labelpos(pts[coord]), norms[coord].mean(0)))
                except:
                    labels.append((pts[coord].mean(0), norms[coord].mean(0)))
            return labels
        return [(pts[coord].mean(0), norms[coord].mean(0)) for coord in self.coords]

class ROIpack(traits.HasTraits):
    svg = traits.Instance("xml.dom.minidom.Document")
    svgfile = traits.Str

    linewidth = traits.Float(default_lw)
    linecolor = traits.Tuple((0.,0.,0.,1.))
    roifill = traits.Tuple((0.,0.,0.,0.2))

    def __init__(self, tcoords, svgfile):
        super(ROIpack, self).__init__()
        if np.any(tcoords.max(0) > 1) or np.any(tcoords.min(0) < 0):
            tcoords -= tcoords.min(0)
            tcoords /= tcoords.max(0)
        self.tcoords = tcoords
        self.svgfile = svgfile
        self.reload()

    def reload(self):
        self.svg = xmlparse(self.svgfile)
        svgdoc = self.svg.getElementsByTagName("svg")[0]
        w = float(svgdoc.getAttribute("width"))
        h = float(svgdoc.getAttribute("height"))
        self.svgshape = w, h

        #Remove the base images -- we don't need to render them for the texture
        rmnode = _find_layer(self.svg, "data")
        rmnode.parentNode.removeChild(rmnode)

        #Set up the ROI dict
        self.rois = {}
        for r in _find_layer(self.svg, "rois").getElementsByTagName("g"):
            roi = ROI(self, r)
            self.rois[roi.name] = roi

        self.update_style()

    def add_roi(self, name, pngdata):
        #self.svg deletes the images -- we want to save those, so let's load it again
        svg = xmlparse(self.svgfile)
        imglayer = _find_layer(svg, "data")
        _make_layer(_find_layer(svg, "rois"), name)

        #Hide all the other layers in the image
        for layer in imglayer.getElementsByTagName("g"):
            layer.setAttribute("style", "display:hidden;")

        layer = _make_layer(imglayer, name, prefix="img_")
        img = svg.createElement("image")
        img.setAttribute("id", "image_%s"%name)
        img.setAttribute("x", "0")
        img.setAttribute("y", "0")
        img.setAttribute("width", str(self.svgshape[0]))
        img.setAttribute("height", str(self.svgshape[1]))
        img.setAttribute("xlink:href", "data:image/png;base64,%s"%pngdata)
        layer.appendChild(img)

        with open(self.svgfile, "w") as xml:
            xml.write(svg.toprettyxml())

    @traits.on_trait_change("linewidth, linecolor, roifill")
    def update_style(self):
        for roi in self.rois.values():
            roi.set(linewidth=self.linewidth, linecolor=self.linecolor, roifill=self.roifill)

    def get_texture(self, texres, bits=32):
        '''Renders the current roimap as a png'''
        #set the current size of the texture
        w, h = self.svgshape
        dpi = texres / h * 72
        cmd = "convert -depth 8 -density {dpi} - png{bits}:-".format(dpi=dpi, bits=bits)
        convert = sp.Popen(cmd.split(), stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        raw = convert.communicate(self.svg.toxml())
        
        pngfile = tempfile.NamedTemporaryFile(suffix=".png")
        pngfile.write(raw[0])
        pngfile.flush()
        return pngfile

    def get_labelpos(self, pts, norms, fancy=True):
        return dict([(name, roi.get_labelpos(pts, norms, fancy)) for name, roi in self.rois.items()])

    def get_roi(self, roiname):
        state = dict()
        for name, roi in self.rois.items():
            #Store what the ROI style so we can restore
            state[name] = dict(linewidth=roi.linewidth, roifill=roi.roifill, hide=roi.hide)
            if name == roiname:
                roi.set(linewidth=0, roifill=(0,0,0,1), hide=False)
            else:
                roi.hide = True
        
        im = self.get_texture(self.svgshape[1], bits=8)
        im.seek(0)
        imdat = imread(im)[::-1,:,0]
        idx = (self.tcoords*(np.array(self.svgshape)-1)).round().astype(int)[:,::-1]
        roiidx = np.nonzero(imdat[tuple(idx.T)] == 0)[0]

        #restore the old roi settings
        for name, roi in self.rois.items():
            roi.set(**state[name])

        return roiidx
    
    @property
    def names(self):
        return self.rois.keys()

    def __getitem__(self, name):
        return self.rois[name]

###################################################################################
# SVG Helper functions
###################################################################################
def _find_layer(svg, label):
    layers = [l for l in svg.getElementsByTagName("g") if l.getAttribute("inkscape:label") == label]
    assert len(layers) > 0, "Cannot find layer %s"%label
    return layers[0]

def _make_layer(parent, name, prefix=""):
    layer = parent.ownerDocument.createElement("g")
    layer.setAttribute("id", "%s%s"%(prefix,name))
    layer.setAttribute("style", "display:inline;")
    layer.setAttribute("inkscape:label", "%s%s"%(prefix,name))
    layer.setAttribute("inkscape:groupmode", "layer")
    parent.appendChild(layer)
    return layer

try:
    from shapely.geometry import Polygon
    def _center_pts(pts):
        '''Fancy label position generator, using erosion to get label coordinate'''
        min = pts.min(0)
        pts -= min
        max = pts.max(0)
        pts /= max

        poly = Polygon([tuple(p) for p in pts])
        for i in np.linspace(0,1,100):
            if poly.buffer(-i).is_empty:
                return list(poly.buffer(-last_i).centroid.coords)[0] * max + min
            last_i = i

        print "unable to find zero centroid..."
        return list(poly.buffer(-100).centroid.coords)[0] * max + min

except ImportError:
    print "Cannot find shapely, using simple label placement"
    def _center_pts(pts):
        return pts.mean(0)

def _labelpos(pts):
    ptm = pts.copy().astype(float)
    ptm -= ptm.mean(0)
    u, s, v = np.linalg.svd(ptm, full_matrices=False)
    sp = np.diag(s)
    sp[-1,-1] = 0
    try:
        x, y = _center_pts(np.dot(ptm, np.dot(v.T, sp))[:,:2])
    except Exception as e:
        print e

    sp = np.diag(1./(s+np.finfo(float).eps))
    pt = np.dot(np.dot(np.array([x,y,0]), sp), v)
    return pt + pts.mean(0)


def test():
    import db
    pts, polys, norms = db.surfs.getVTK("JG", "flat")
    return ROIpack(pts[:,:2], "/home/james/code/mritools_store/overlays/JG_rois.svg")