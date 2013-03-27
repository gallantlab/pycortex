import os
import shlex
import tempfile
import subprocess as sp

import numpy as np
from scipy.spatial import cKDTree

from lxml import etree
from lxml.builder import E

from cortex.options import config

svgns = "http://www.w3.org/2000/svg"
inkns = "http://www.inkscape.org/namespaces/inkscape"
sodins = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
parser = etree.XMLParser(remove_blank_text=True)

cwd = os.path.abspath(os.path.split(__file__)[0])

class ROIpack(object):
    def __init__(self, tcoords, svgfile, callback=None, 
            linewidth=float(config.get("rois", "line_width")),
            linecolor=tuple(map(float, config.get("rois", "line_color").split(','))),
            roifill=tuple(map(float, config.get("rois", "fill_color").split(','))),
            shadow=float(config.get("rois", "shadow"))):
        if np.any(tcoords.max(0) > 1) or np.any(tcoords.min(0) < 0):
            tcoords -= tcoords.min(0)
            tcoords /= tcoords.max(0)

        self.tcoords = tcoords
        self.svgfile = svgfile
        self.callback = callback
        self.kdt = cKDTree(tcoords)
        self.linewidth = linewidth
        self.linecolor = linecolor
        self.roifill = roifill
        self.shadow = shadow
        self.reload()

    def reload(self):
        self.svg = scrub(self.svgfile)
        w = float(self.svg.getroot().get("width"))
        h = float(self.svg.getroot().get("height"))
        self.svgshape = w, h

        #Set up the ROI dict
        self.rois = {}
        for r in _find_layer(self.svg, "rois").findall("{%s}g"%svgns):
            roi = ROI(self, r)
            self.rois[roi.name] = roi

        self.set()
        self.labels = self.setup_labels()

    def add_roi(self, name, pngdata, add_path=True):
        #self.svg deletes the images -- we want to save those, so let's load it again
        svg = etree.parse(self.svgfile)
        imglayer = _find_layer(svg, "data")
        if add_path:
            _make_layer(_find_layer(svg, "rois"), name)

        #Hide all the other layers in the image
        for layer in imglayer.findall(".//{%s}g"%svgns):
            layer.attrib["style"] = "display:hidden;"

        layer = _make_layer(imglayer, "img_%s"%name)
        layer.append(E.image(
            {"{http://www.w3.org/1999/xlink}href":"data:image/png;base64,%s"%pngdata},
            id="image_%s"%name, x="0", y="0",
            width=str(self.svgshape[0]),
            height=str(self.svgshape[1]),
        ))

        with open(self.svgfile, "w") as xml:
            xml.write(etree.tostring(svg, pretty_print=True))

    def set(self, linewidth=None, linecolor=None, roifill=None, shadow=None):
        if linewidth is not None:
            self.linewidth = linewidth
        if linecolor is not None:
            self.linecolor = linecolor
        if roifill is not None:
            self.roifill = roifill
        if shadow is not None:
            self.shadow = shadow
            self.svg.find("//{%s}feGaussianBlur"%svgns).attrib["stdDeviation"] = str(shadow)

        for roi in list(self.rois.values()):
            roi.set(linewidth=self.linewidth, linecolor=self.linecolor, roifill=self.roifill)

        try:
            if self.callback is not None:
                self.callback()
        except:
            print("cannot callback")

    def get_texture(self, texres, name=None, background=None, labels=True, bits=32):
        '''Renders the current roimap as a png'''
        #set the current size of the texture
        w, h = self.svgshape
        dpi = texres / h * 72

        if background is not None:
            img = E.image(
                {"{http://www.w3.org/1999/xlink}href":"data:image/png;base64,%s"%background},
                id="image_%s"%name, x="0", y="0",
                width=str(self.svgshape[0]),
                height=str(self.svgshape[1]),
            )
            self.svg.getroot().insert(0, img)

        if labels:
            self.labels.attrib['style'] = "display:inline;"
        else:
            self.labels.attrib['style'] = "display:none;"

        pngfile = name
        if name is None:
            png = tempfile.NamedTemporaryFile(suffix=".png")
            pngfile = png.name

        cmd = "convert -background none -density {dpi} SVG:- PNG{bits}:{outfile}"
        cmd = cmd.format(dpi=dpi, outfile=pngfile, bits=bits)
        proc = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=sp.PIPE)
        proc.communicate(etree.tostring(self.svg))
        
        if background is not None:
            self.svg.getroot().remove(img)

        if name is None:
            png.seek(0)
            return png

    def get_labelpos(self, pts=None, norms=None, fancy=True):
        return dict([(name, roi.get_labelpos(pts, norms, fancy)) for name, roi in list(self.rois.items())])

    def get_ptidx(self):
        return dict([(name, roi.get_ptidx()) for name, roi in list(self.rois.items())])

    def get_roi(self, roiname):
        import Image
        shadow = self.shadow
        self.set(shadow=0)

        state = dict()
        for name, roi in list(self.rois.items()):
            #Store what the ROI style so we can restore
            state[name] = dict(linewidth=roi.linewidth, roifill=roi.roifill, hide=roi.hide)
            if name == roiname:
                roi.set(linewidth=0, roifill=(0,0,0,1), hide=False)
            else:
                roi.set(hide=True)
        
        im = self.get_texture(self.svgshape[1], labels=False, bits=8)
        imdat = np.array(Image.open(im))[::-1]
        idx = (self.tcoords*(np.array(self.svgshape)-1)).round().astype(int)[:,::-1]
        roiidx = np.nonzero(imdat[tuple(idx.T)] == 1)[0]

        #restore the old roi settings
        for name, roi in list(self.rois.items()):
            roi.set(**state[name])

        self.set(shadow=shadow)
        return roiidx
    
    @property
    def names(self):
        return list(self.rois.keys())

    def __getitem__(self, name):
        return self.rois[name]

    def setup_labels(self, size="16pt", color="#FFFFFF"):
        try:
            layer = _find_layer(self.svg, "roilabels")
        except AssertionError:
            layer = _make_layer(self.svg.getroot(), "roilabels")

        labelpos, candidates = [], []
        for roi in list(self.rois.values()):
            for i, pos in enumerate(roi.get_labelpos()):
                labelpos.append(pos)
                candidates.append((roi, i))

        w, h = self.svgshape
        nolabels = set(candidates)
        txtstyle = "font-family:sans;font-size:%s;font-weight:bold;font-style:italic;fill:%s;text-anchor:middle;"%(size, color)
        for text in layer.findall(".//{%s}text"%svgns):
            x = float(text.get('x'))
            y = float(text.get('y'))
            text.attrib['style'] = txtstyle
            text.attrib['data-ptidx'] = str(self.kdt.query((x / w, 1-(y / h)))[1])
            pts, cand = [], []
            for p, c in zip(labelpos, candidates):
                if c[0].name == text.text:
                    pts.append((p[0]*w, (1-p[1])*h))
                    cand.append(c)
            d, idx = cKDTree(pts).query((x,y))
            nolabels.remove(cand[idx])

        for roi, i in nolabels:
            x, y = roi.get_labelpos()[i]
            text = etree.SubElement(layer, "{%s}text"%svgns)
            text.text = roi.name
            text.attrib["x"] = str(x*w)
            text.attrib["y"] = str((1-y)*h)
            text.attrib['filter'] = "url(#dropshadow)"
            text.attrib['style'] = txtstyle
            text.attrib['data-ptidx'] = str(self.kdt.query((x, y))[1])

        return layer

    def toxml(self, pretty=True):
        return etree.tostring(self.svg, pretty_print=pretty)


class ROI(object):
    def __init__(self, parent, xml):
        self.parent = parent
        self.name = xml.get("{%s}label"%inkns)
        self.paths = xml.findall(".//{%s}path"%svgns)
        pts = [ self._parse_svg_pts(path.get("d")) for path in self.paths]
        self.coords = [ self.parent.kdt.query(p)[1] for p in pts ]
        self.hide = "style" in xml.attrib and "display:none" in xml.get("style")
        self.set(linewidth=self.parent.linewidth, linecolor=self.parent.linecolor, roifill=self.parent.roifill)
    
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
            if isinstance(d, str) and len(d) == 1:
                mode = d
                continue
            if mode == "l":
                offset += list(map(float, d.split(',')))
            elif mode == "L":
                offset = np.array(list(map(float, d.split(','))))
            elif mode == "c":
                data.pop(0)
                offset += list(map(float, data.pop(0).split(',')))
            elif mode == "C":
                data.pop(0)
                offset = np.array(list(map(float, data.pop(0).split(','))))
            pts.append([offset[0],offset[1]])

        pts = np.array(pts)
        pts /= self.parent.svgshape
        pts[:,1] = 1-pts[:,1]
        return pts
    
    def set(self, linewidth=None, linecolor=None, roifill=None, shadow=None, hide=None):
        if linewidth is not None:
            self.linewidth = linewidth
        if linecolor is not None:
            self.linecolor = linecolor
        if roifill is not None:
            self.roifill = roifill
        if hide is not None:
            self.hide = hide

        style = "fill:{fill}; fill-opacity:{fo};stroke-width:{lw}px;"+\
                    "stroke-linecap:butt;stroke-linejoin:miter;"+\
                    "stroke:{lc};stroke-opacity:{lo};{hide}"
        roifill = np.array(self.roifill)*255
        linecolor = np.array(self.linecolor)*255
        hide = "display:none;" if self.hide else ""
        style = style.format(
            fill="rgb(%d,%d,%d)"%tuple(roifill[:-1]), fo=roifill[-1],
            lc="rgb(%d,%d,%d)"%tuple(linecolor[:-1]), lo=linecolor[-1], 
            lw=self.linewidth, hide=hide)

        for path in self.paths:
            path.attrib["style"] = style
            if self.parent.shadow > 0:
                path.attrib["filter"] = "url(#dropshadow)"
            elif "filter" in path.attrib:
                del path.attrib['filter']
    
    def get_labelpos(self, pts=None, norms=None, fancy=True):
        if pts is None:
            pts = self.parent.tcoords

        if fancy:
            labels = []
            for coord in self.coords:
                try:
                    if norms is None:
                        labels.append(_labelpos(pts[coord]))
                    else:
                        labels.append((_labelpos(pts[coord]), norms[coord].mean(0)))
                except:
                    if norms is None:
                        labels.append(pts[coord].mean(0))
                    else:
                        labels.append((pts[coord].mean(0), norms[coord].mean(0)))
            return labels

        if norms is None:
            return [pts[coord].mean(0) for coord in self.coords]

        return [(pts[coord].mean(0), norms[coord].mean(0)) for coord in self.coords]

    def get_ptidx(self):
        return self.coords

###################################################################################
# SVG Helper functions
###################################################################################
def _find_layer(svg, label):
    layers = [l for l in svg.findall("//{%s}g[@{%s}label]"%(svgns, inkns)) if l.get("{%s}label"%inkns) == label]
    assert len(layers) > 0, "Cannot find layer %s"%label
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
        pts /= max

        poly = Polygon([tuple(p) for p in pts])
        for i in np.linspace(0,1,100):
            if poly.buffer(-i).is_empty:
                return list(poly.buffer(-last_i).centroid.coords)[0] * max + min
            last_i = i

        print("unable to find zero centroid...")
        return list(poly.buffer(-100).centroid.coords)[0] * max + min

except ImportError:
    print("Cannot find shapely, using simple label placement")
    def _center_pts(pts):
        return pts.mean(0)

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

def scrub(svgfile):
    svg = etree.parse(svgfile, parser=parser)
    rmnode = _find_layer(svg, "data")
    rmnode.getparent().remove(rmnode)
    svgtag = svg.getroot()
    svgtag.attrib['id'] = "svgroi"
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
    from .polyutils import trace_both
    pts -= pts.min(0)
    pts *= 1024 / pts.max(0)[1]
    pts[:,1] = 1024 - pts[:,1]
    path = ""
    for poly in trace_both(pts, polys):
        path +="M%f %f L"%tuple(pts[poly.pop(0), :2])
        path += ', '.join(['%f %f'%tuple(pts[p, :2]) for p in poly])
        path += 'Z '

    w, h = pts.max(0)[:2]
    with open(os.path.join(cwd, "svgbase.xml")) as fp:
        svg = fp.read().format(width=w, height=h, clip=path)

    return svg

def get_roipack(subject, remove_medial=False):
    from .db import surfs
    flat, polys = surfs.getSurf(subject, "flat", merge=True, nudge=True)
    if remove_medial:
        valid = np.unique(polys)
        flat = flat[valid]
    svgfile = surfs.getFiles(subject)['rois']
    if not os.path.exists(svgfile):
        with open(svgfile, "w") as fp:
            fp.write(svgroi.make_svg(flat.copy(), polys))
    rois = ROIpack(flat[:,:2], svgfile)
    if remove_medial:
        return rois, valid
        
    return rois