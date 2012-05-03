import tempfile
import subprocess as sp
from xml.dom.minidom import parse as xmlparse

import Image
import numpy as np
import traits.api as traits
from scipy.interpolate import griddata

class ROIpack(traits.HasTraits):
    svg = traits.Instance("xml.dom.minidom.Document")
    svgfile = traits.Str
    rois = traits.Dict

    linewidth = traits.Float(5)
    labelsize = traits.Int(24)
    linewidth = traits.Float(5)
    linecolor = traits.Tuple((0.,0.,0.,1.))
    roifill = traits.Tuple((0.,0.,0.,0.2))

    def __init__(self, flat, svgfile):
        self.flat = flat.copy()
        self.svgfile = svgfile
        flat -= flat.min(0)
        flat /= flat.max(0)
        self.tcoords = flat

        svg = xmlparse(self.svgfile)
        svgdoc = svg.getElementsByTagName("svg")[0]
        w = float(svgdoc.getAttribute("width"))
        h = float(svgdoc.getAttribute("height"))
        self.svgshape = w, h

        #Remove the base images -- we don't need to render them for the texture
        rmnode = _find_layer(svg, "data")
        rmnode.parentNode.removeChild(rmnode)

        #Set up the ROI dict
        rois = {}
        cidx = np.arange(len(self.tcoords))
        for r in _find_layer(svg, "rois").getElementsByTagName("g"):
            if not r.hasAttribute("style") or "display:none" not in r.attributes['style'].value:
                name = r.getAttribute("inkscape:label")
                paths = r.getElementsByTagName("path")
                pts = [ self._parse_svg_pts(path.getAttribute("d")) for path in paths]
                coords = [ griddata(self.tcoords, cidx, p, "nearest") for p in pts ]
                rois[name] = zip(paths, coords)
        
        #use traits callbacks to update the lines and textures
        self.rois = rois
        self.svg = svg

    def _parse_svg_pts(self, data):
        data = data.split()
        assert data[0].lower() == "m", "Unknown path format"
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
        pts /= self.svgshape
        pts[:,1] = 1-pts[:,1]
        return pts

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

    @traits.on_trait_change("rois, linewidth, linecolor, roifill")
    def update_rois(self):
        for name, paths in self.rois.items():
            style = "fill:{fill}; fill-opacity:{fo};stroke-width:{lw}px;"+\
                    "stroke-linecap:butt;stroke-linejoin:miter;"+\
                    "stroke:{lc};stroke-opacity:{lo}; {hide}"
            roifill = np.array(self.roifill)*255
            linecolor = np.array(self.linecolor)*255
            style = style.format(
                fill="rgb(%d,%d,%d)"%tuple(roifill[:-1]), fo=self.roifill[-1],
                lc="rgb(%d,%d,%d)"%tuple(linecolor[:-1]), lo=self.linecolor[-1], 
                lw=self.linewidth, hide="")

            for path, coords in paths:
                path.setAttribute("style", style)

    def get_texture(self, texres):
        '''Renders the current roimap as a texture map'''
        #set the current size of the texture
        w, h = self.svgshape
        cmd = "convert -depth 8 -density {dpi} - png32:-".format(dpi=texres / h * 72)
        convert = sp.Popen(cmd.split(), stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        raw = convert.communicate(self.svg.toxml())
        pngfile = tempfile.NamedTemporaryFile(suffix=".png")
        pngfile.write(raw[0])
        pngfile.flush()
        return pngfile

    def get_labelpos(self, pts, fancy=True):
        labels = dict()
        for name, paths in self.rois.items():
            if fancy:
                labels[name] = [_labelpos(pts[coords]) for path, coords in paths]
            else:
                labels[name] = [pts[coords].mean(0) for path, coords in paths]
        return labels

    def get_roi(self):
        im = self.get_texture(self.svgshape[1])
        im.seek(0)
        imdat = np.array(Image(im))
        idx = (self.tcoords*(np.array(self.svgshape)-1)).round().astype(int)[:,::-1]
        return np.nonzero(imdat[idx] == 0)[0]

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
    except:
        print pts, u, s, v

    sp = np.diag(1./(s+np.finfo(float).eps))
    pt = np.dot(np.dot(np.array([x,y,0]), sp), v)
    return pt + pts.mean(0)


def test():
    import db
    pts, polys, norms = db.surfs.getVTK("JG", "flat")
    return ROIpack(pts[:,:2], "/auto/k2/share/mritools_store/overlays/JG_rois.svg")