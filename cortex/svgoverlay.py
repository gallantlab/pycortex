import os

import re
import copy
import shlex
import tempfile
import itertools
import numpy as np
import subprocess as sp
from .svgsplines import LineSpline, QuadBezSpline, CubBezSpline, ArcSpline

from scipy.spatial import cKDTree

from lxml import etree
from lxml.builder import E

from cortex.options import config

svgns = "http://www.w3.org/2000/svg"
inkns = "http://www.inkscape.org/namespaces/inkscape"
sodins = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
parser = etree.XMLParser(remove_blank_text=True, huge_tree=True)

cwd = os.path.abspath(os.path.split(__file__)[0])

class SVGOverlay(object):
    def __init__(self, svgfile, coords=None):
        self.svgfile = svgfile
        self.reload()
        if coords is not None:
            self.set_coords(coords)

    def reload(self):
        self.svg = scrub(self.svgfile)
        w = float(self.svg.getroot().get("width"))
        h = float(self.svg.getroot().get("height"))
        self.svgshape = w, h

        #grab relevant layers
        self.layers = dict()
        for layer in self.svg.getroot().findall("{%s}g"%svgns):
            layer = Overlay(self, layer)
            self.layers[layer.name] = layer

    def set_coords(self, coords):
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
        svg = etree.parse(self.svgfile, parser=parser)
        layer = _make_layer(svg.getroot(), name)
        shapes = _make_layer(layer, "shapes")
        shapes.attrib['id'] = "%s_shapes"%name
        labels = _make_layer(layer, "labels")
        labels.attrib['id'] = "%s_labels"%name
        with open(self.svgfile, "w") as fp:
            fp.write(etree.tostring(svg, pretty_print=True))
        self.reload()

    def toxml(self, pretty=True):
        return etree.tostring(self.svg, pretty_print=pretty)

    def get_svg(self, filename=None, labels=True, with_ims=None): # This did nothing - why?:, **kwargs):
        """Returns an SVG with the included images."""
        self.labels.visible = labesl
        
        outsvg = copy.deepcopy(self.svg)
        if with_ims is not None:
            if isinstance(with_ims, (list, tuple)):
                with_ims = zip(range(len(with_ims)), with_ims)

            datalayer = _make_layer(outsvg.getroot(), "data")
            for imnum,im in reversed(with_ims):
                imlayer = _make_layer(datalayer, "image_%d" % imnum)
                img = E.image(
                    {"{http://www.w3.org/1999/xlink}href":"data:image/png;base64,%s"%im},
                    id="image_%d"%imnum, x="0", y="0",
                    width=str(self.svgshape[0]),
                    height=str(self.svgshape[1]),
                    )
                imlayer.append(img)
                outsvg.getroot().insert(0, imlayer)
        
        if filename is None:
            return etree.tostring(outsvg)
        else:
            with open(filename, "w") as outfile:
                outfile.write(etree.tostring(outsvg))
        
    def get_texture(self, layer_name, texres, name=None, background=None, labels=True, bits=32, **kwargs):
        '''Renders a specific layer of this svgobject as a png

        '''
        #set the size of the texture
        w, h = self.svgshape
        dpi = texres / h * 72 # 72 is screen resolution assumption for svg files

        if background is not None:
            img = E.image(
                {"{http://www.w3.org/1999/xlink}href":"data:image/png;base64,%s"%background},
                id="image_%s"%name, x="0", y="0",
                width=str(self.svgshape[0]),
                height=str(self.svgshape[1]),
            )
            self.svg.getroot().insert(0, img)

        for layer in self:
            if layer.name==layer_name:
                layer.visible = True
                if len(kwargs)>0:
                    print('Setting: %r'%repr(kwargs))
                layer.set(**kwargs)
            else:
                layer.visible = False
            layer.labels.visible = labels

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
            #im = plt.imread(png)
            return png

class Overlay(object):
    def __init__(self, svgobject, layer):
        self.svgobject = svgobject
        self.svg = svgobject.svg
        self.layer = layer
        self.name = layer.attrib['{%s}label'%inkns]
        self.layer.attrib['class'] = 'display_layer'

        #check to see if the layer is locked, to see if we need to override the style
        locked = '{%s}insensitive'%sodins
        self.shapes = dict()
        for layer in _find_layer(layer, "shapes").findall("{%s}g"%svgns):
            override = locked not in layer.attrib or layer.attrib[locked] == "false"
            shape = Shape(layer, self.svgobject.svgshape[1], override_style=override)
            self.shapes[shape.name] = shape

        self.labels = Labels(self)

    def __repr__(self):
        return "<svg layer with shapes [%s]>"%(','.join(self.shapes.keys()))

    def __getitem__(self, name):
        return self.shapes[name]

    @property
    def visible(self):
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
        """Adds projected data for defining a new ROI to the saved rois.svg file in a new layer"""
        #self.svg deletes the images -- we want to save those, so let's load it again
        svg = etree.parse(self.svgobject.svgfile, parser=parser)
        imglayer = _find_layer(svg, "data")
        if add_path:
            layer = _find_layer(svg, self.name)
            _make_layer(_find_layer(layer, "shapes"), name)

        #Hide all the other layers in the image
        for layer in imglayer.findall(".//{%s}g"%svgns):
            layer.attrib["style"] = "display:hidden;"

        layer = _make_layer(imglayer, "img_%s"%name)
        layer.append(E.image(
            {"{http://www.w3.org/1999/xlink}href":"data:image/png;base64,%s"%pngdata},
            id="image_%s"%name, x="0", y="0",
            width=str(self.svgobject.svgshape[0]),
            height=str(self.svgobject.svgshape[1]),
        ))

        with open(self.svgobject.svgfile, "w") as xml:
            xml.write(etree.tostring(svg, pretty_print=True))

class Labels(object):
    def __init__(self, overlay):
        self.overlay = overlay
        self.layer = _find_layer(self.overlay.layer, "labels")
        self.text_style = dict(config.items("overlay_text"))
        
        text_style = self.text_style.items()
        text_style = ';'.join(['%s:%s'%(k,v) for k, v in text_style if v != 'None'])

        #generate a list of labels that should be in the layer
        self.elements = dict()
        for shape in self.overlay.shapes.values():
            self.elements[shape.name] = shape.get_labelpos() 

        #match up existing labels with their respective paths
        def close(pt, x, y):
            return np.sqrt((pt[0] - x)**2 + (pt[1]-y)**2) < 250
        for text in self.layer.findall(".//{%s}text"%svgns):
            x = float(text.get('x'))
            y = float(text.get('y'))
            #check this element against all known paths
            for name in self.elements.keys():
                if text.text == name:
                    for i, pos in enumerate(self.elements[name]):
                        if close(pos, x, y):
                            self.elements[key][i] = text

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

    ###
    # The get_mask function takes in an roi's name and returns an array indicating if every vertex is or isn't in that roi
    # The way it works is that it collapses all of the x-values of the vertex coordinates approximately around the roi to the same
    # small value, making a vertical line left of the roi. Then, it stretches the line to the right again, but stops the coordinates if they
    # hit either an roi boundary or the original vertex position. In other words, it increases the x-values of the coordinates to either
    # those of the the nearest spline path or the original vertex coordinate, whichever has the closer x-value.
    # This way, it keeps track of how many boundaries it hit, starting from the outside, going inward toward the the original vertex coordinate.
    # An odd number of boundaries found before the vertex means 'outside' the region or 'False' in the array, and an even number of
    # boundaries found before the vertex means 'inside' the region or 'True' in the array
    #
    # This is all implemented with 1d and nd arrays manipulations, so the math is very algebraic.
    #
    # NOTE: Look into replacing these with matplotlib functions. 
    # http://matplotlib.org/1.2.1/api/path_api.html#matplotlib.path.Path.contains_points
    # For parsing svg files to matplotlib paths, see: 
    # https://github.com/rougier/LinuxMag-HS-2014/blob/master/matplotlib/firefox.py
    ###
    def get_mask(self, vts):
        all_splines = self.splines #all_splines is a list of generally two roi paths, one for each hemisphere

        vts_inside_region = np.zeros(vts.shape[0],dtype=bool) # ultimately what gets returned

        for splines in all_splines: #retrieves path splines for each hemisphere separately
            x0s = np.min(vts[:,0])*.98*np.ones(vts.shape[0])

            # Only checks the vertices in a bounding box around the spline path.
            # The splines are always within a convex shape whose corners are
            # their svg command's end point and control points, so the box is their
            # min and max X and Y coordinates.
            beforeSplineRegionX = vts[:,0] < np.min([float(sp_i.smallestX()) for sp_i in splines])
            beforeSplineRegionY = vts[:,1] < np.min([float(sp_i.smallestY()) for sp_i in splines])
            afterSplineRegionX = vts[:,0] > np.max([float(sp_i.biggestX()) for sp_i in splines])
            afterSplineRegionY = vts[:,1] > np.max([float(sp_i.biggestY()) for sp_i in splines])

            found_vtxs = np.zeros(vts.shape[0],dtype=bool)
            found_vtxs[beforeSplineRegionX] = True
            found_vtxs[beforeSplineRegionY] = True
            found_vtxs[afterSplineRegionX] = True
            found_vtxs[afterSplineRegionY] = True
            
            vt_isx = np.vstack([x0s,vts[:,1]]).T #iterable coords, same x-value as each other, but at their old y-value positions
 
            vtx_is = vt_isx[~found_vtxs]

            splines_xs = [] # stores the roi's splines
            for i in range(len(splines)):
                splines_xs.append(splines[i].allSplineXGivenY(vtx_is)) # gets all the splines' x-values for each y-value in the line we're checking

            small_vts = vts[~found_vtxs,:]
            small_vts_inside_region = vts_inside_region[~found_vtxs]
            small_found_vtxs = found_vtxs[~found_vtxs]

            # keeps stretching the vertical line to the right until all the points find their original vertex again            
            while sum(small_found_vtxs) != len(small_found_vtxs):
                closest_xs = np.Inf*np.ones(vtx_is.shape[0]) # starting marker for all vts are at Inf

                for i in range(len(splines_xs)):
                    spline_i_xs = splines_xs[i]
                    if len(spline_i_xs.shape) == 1: # Line splines
                        isGreaterThanVtx = spline_i_xs > vtx_is[:,0]
                        isLessThanClosestX = spline_i_xs < closest_xs
                        closest_xs[isGreaterThanVtx*isLessThanClosestX] = spline_i_xs[isGreaterThanVtx*isLessThanClosestX]
                    else: # all other splines
                        for j in range(spline_i_xs.shape[1]):
                            isGreaterThanVtx = spline_i_xs[:,j] > vtx_is[:,0]
                            isLessThanClosestX = spline_i_xs[:,j] < closest_xs
                            closest_xs[isGreaterThanVtx*isLessThanClosestX] = spline_i_xs[isGreaterThanVtx*isLessThanClosestX,j]                    
            
                # checks if it's found the boundary or the original vertex
                # it forgets about all the points in the line who've found their original vertex
                # if it found a boundary, then flip the 'inside' flag to 'outside', and vice versa
                
                small_found_vtxsx = small_vts[~small_found_vtxs,0]<closest_xs
                small_found_vtxs[~small_found_vtxs] = small_found_vtxsx

                small_vts_inside_region[~small_found_vtxs] = True - small_vts_inside_region[~small_found_vtxs]
                vtx_is[~small_found_vtxsx,0] = closest_xs[~small_found_vtxsx]
                vtx_is = vtx_is[~small_found_vtxsx,:]        
                
                for i in range(len(splines_xs)):
                    if len(splines_xs[i].shape) == 1:
                        splines_xs[i] = splines_xs[i][~small_found_vtxsx]
                    else:
                        splines_xs[i] = splines_xs[i][~small_found_vtxsx,:]

            vts_inside_region[~found_vtxs] = small_vts_inside_region # reverts shape back from small bounding box to whole brain shape

            if sum(vts_inside_region) == len(vts_inside_region):
                break

        return np.nonzero(vts_inside_region)[0] # output indices of vertices that are inside the roi
    @property
    def splines(self):
        path_strs = [list(_tokenize_path(path.get('d'))) for path in self.paths]

        COMMANDS = set('MmZzLlHhVvCcSsQqTtAa')
        splines = []
        ###  
        # this is for the svg path parsing (https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths)
        # the general format is that there is a state machine that keeps track of which command (path_ind)
        # that it's listening to while parsing over the appropriately sized (param_len) groups of
        # coordinates for that command 
        ###
        for path in path_strs:
            path_splines = []
            first_coord = np.zeros(2) #array([0,0])
            prev_coord = np.zeros(2) #array([0,0])
            isFirstM = True# inkscape may create multiple starting commands to move to the spline's starting coord, this just treats those as one commend


            for path_ind in range(len(path)):
                if path_ind == 0 and path[path_ind].lower() != 'm':
                    raise ValueError('Unknown path format!')
                
                elif path[path_ind].lower() == 'm': 
                    param_len = 2
                    p_j = path_ind + 1 # temp index
                    
                    while p_j < len(path) and len(COMMANDS.intersection(path[p_j])) == 0:
                        old_prev_coord = np.zeros(2)
                        old_prev_coord[0] = prev_coord[0]
                        old_prev_coord[1] = self.height - prev_coord[1]

                        if path[path_ind] == 'M':
                            prev_coord[0] = float(path[p_j])
                            prev_coord[1] = self.height - float(path[p_j+1])
                        else:
                            prev_coord[0] += float(path[p_j])
                            if isFirstM:
                                prev_coord[1] = self.height - float(path[p_j+1])
                            else:
                                prev_coord[1] -= self.height - float(path[p_j+1])

                            # this conditional is for recognizing and storing the last coord in the first M command(s)
                            # as the official first coord in the spline path for any 'close path (ie, z)' command 
                        if isFirstM == True:
                            first_coord[0] = prev_coord[0]
                            first_coord[1] = prev_coord[1]
                            isFirstM = False
                        else:
                            path_splines.append(LineSpline(old_prev_coord,prev_coord))

                        p_j += param_len
                        
                elif path[path_ind].lower() == 'z':
                    path_splines.append(LineSpline(prev_coord, first_coord))                    
                    prev_coord[0] = first_coord[0]
                    prev_coord[1] = first_coord[1]
                    
                elif path[path_ind].lower() == 'l':
                    param_len = 2
                    p_j = path_ind + 1
                    next_coord = np.zeros(2)
                                       
                    while p_j < len(path) and len(COMMANDS.intersection(path[p_j])) == 0:
                        if path[path_ind] == 'L':
                            next_coord[0] = float(path[p_j])
                            next_coord[1] = float(path[p_j+1])

                        else:
                            next_coord[0] = prev_coord[0] + float(path[p_j])
                            next_coord[1] = prev_coord[1] - float(path[p_j+1])
                        
                        path_splines.append(LineSpline(prev_coord, next_coord))
                        prev_coord[0] = next_coord[0]
                        prev_coord[1] = next_coord[1]
                        p_j += param_len

                elif path[path_ind].lower() == 'h':
                    param_len = 1
                    p_j = path_ind + 1
                    next_coord = np.zeros(2)
                                       
                    while p_j < len(path) and len(COMMANDS.intersection(path[p_j])) == 0:
                        if path[path_ind] == 'H':
                            next_coord[0] = float(path[p_j])
                            next_coord[1] = prev_coord[1]
                        else:
                            next_coord[0] = prev_coord[0] + float(path[p_j])
                            next_coord[1] = prev_coord[1]
                        
                        path_splines.append(LineSpline(prev_coord, next_coord))
                        prev_coord[0] = next_coord[0]
                        prev_coord[1] = next_coord[1]
                        p_j += param_len
                
                elif path[path_ind].lower() == 'v':
                    param_len = 1
                    p_j = path_ind + 1
                    next_coord = np.zeros(2)

                    while p_j < len(path) and len(COMMANDS.intersection(path[p_j])) == 0:
                        if path[path_ind] == 'V':
                            next_coord[0] = prev_coord[0]
                            next_coord[1] = float(path[p_j])
                        else:
                            next_coord[0] = prev_coord[0]
                            next_coord[1] = prev_coord[1] - float(path[p_j])
                        
                        path_splines.append(LineSpline(prev_coord, next_coord))
                        prev_coord[0] = next_coord[0]
                        prev_coord[1] = next_coord[1]
                        p_j += param_len
                
                elif path[path_ind].lower() == 'c':
                    param_len = 6
                    p_j = path_ind + 1
                    ctl1_coord = np.zeros(2)
                    ctl2_coord = np.zeros(2)
                    end_coord = np.zeros(2)

                    while p_j < len(path) and len(COMMANDS.intersection(path[p_j])) == 0:
                        if path[path_ind] == 'C':
                            ctl1_coord[0] = float(path[p_j])
                            ctl1_coord[1] = float(path[p_j+1])

                            ctl2_coord[0] = float(path[p_j+2])
                            ctl2_coord[1] = float(path[p_j+3])

                            end_coord[0] = float(path[p_j+4])
                            end_coord[1] = float(path[p_j+5])

                        else:
                            ctl1_coord[0] = prev_coord[0] + float(path[p_j])
                            ctl1_coord[1] = prev_coord[1] - float(path[p_j+1])
                            
                            ctl2_coord[0] = prev_coord[0] + float(path[p_j+2])
                            ctl2_coord[1] = prev_coord[1] - float(path[p_j+3])

                            end_coord[0] = prev_coord[0] + float(path[p_j+4])
                            end_coord[1] = prev_coord[1] - float(path[p_j+5])

                        path_splines.append(CubBezSpline(prev_coord, ctl1_coord, ctl2_coord, end_coord))

                        prev_coord[0] = end_coord[0]
                        prev_coord[1] = end_coord[1]
                        p_j += param_len
                
                elif path[path_ind].lower() == 's':
                    param_len = 4
                    p_j = path_ind + 1
                    ctl1_coord = np.zeros(2)
                    ctl2_coord = np.zeros(2)
                    end_coord = np.zeros(2)

                    while p_j < len(path) and len(COMMANDS.intersection(path[p_j])) == 0:
                        ctl1_coord = prev_coord - path_splines[len(path_splines)-1].c2 + prev_coord

                        if path[path_ind] == 'S':
                            ctl2_coord[0] = float(path[p_j])
                            ctl2_coord[1] = float(path[p_j+1])

                            end_coord[0] = float(path[p_j+2])
                            end_coord[1] = float(path[p_j+3])

                        else:
                            ctl2_coord[0] = prev_coord[0] + float(path[p_j])
                            ctl2_coord[1] = prev_coord[1] - float(path[p_j+1])
                            
                            end_coord[0] = prev_coord[0] + float(path[p_j+2])
                            end_coord[1] = prev_coord[1] - float(path[p_j+3])
                        
                        path_splines.append(CubBezSpline(prev_coord, ctl1_coord, ctl2_coord, end_coord))
                        prev_coord[0] = end_coord[0]
                        prev_coord[1] = end_coord[1]
                        p_j += param_len

                elif path[path_ind].lower() == 'q':
                    param_len = 4
                    p_j = path_ind + 1
                    ctl_coord = np.zeros(2)
                    end_coord = np.zeros(2)

                    while p_j < len(path) and len(COMMANDS.intersection(path[p_j])) == 0:
                        if path[path_ind] == 'Q':
                            ctl_coord[0] = float(path[p_j])
                            ctl_coord[1] = float(path[p_j+1])

                            end_coord[0] = float(path[p_j+2])
                            end_coord[1] = float(path[p_j+3])
                        else:
                            ctl_coord[0] = prev_coord[0] + float(path[p_j])
                            ctl_coord[1] = prev_coord[1] - float(path[p_j+1])

                            end_coord[0] = prev_coord[0] + float(path[p_j+2])
                            end_coord[1] = prev_coord[1] - float(path[p_j+3])
                                        
                        path_splines.append(QuadBezSpline(prev_coord, ctl_coord, end_coord))
                        prev_coord[0] = end_coord[0]
                        prev_coord[1] = end_coord[1]
                        p_j += param_len
                
                elif path[path_ind].lower() == 't':
                    param_len = 2
                    p_j = path_ind + 1
                    ctl_coord = np.zeros(2)
                    end_coord = np.zeros(2)

                    while p_j < len(path) and len(COMMANDS.intersection(path[p_j])) == 0:
                        ctl_coord = prev_coord - path_splines[len(path_splines)-1].c + prev_coord

                        if path[path_ind] == 'T':
                            end_coord[0] = float(path[p_j])
                            end_coord[1] = float(path[p_j+1])
                        else:
                            end_coord[0] = prev_coord[0] + float(path[p_j])
                            end_coord[1] = prev_coord[1] - float(path[p_j+1])

                        path_splines.append(QuadBezSpline(prev_coord, ctl_coord, end_coord))
                        prev_coord[0] = end_coord[0]
                        prev_coord[1] = end_coord[1]
                        p_j += param_len
                
                # NOTE: This is *NOT* functional. Arcspline parsing saves to an incomplete ArcSpline class
                elif path[path_ind].lower() == 'a': 
                    param_len = 7
                    p_j = path_ind + 1
                    end_coord = np.zeros(2)

                    while p_j < len(path) and len(COMMANDS.intersection(path[p_j])) == 0:
                        rx = float(path[p_j])
                        ry = float(path[p_j+1])
                        x_rot = float(path[p_j+2])
                        large_arc_flag = int(path[p_j+3])
                        sweep_flag = int(path[p_j+4])

                        if path[path_ind] == 'A':
                            end_coord[0] = float(path[p_j+5])
                            end_coord[1] = float(path[p_j+6])
                        else:
                            end_coord[0] = prev_coord[0] + float(path[p_j+5])
                            end_coord[1] = prev_coord[1] - float(path[p_j+6])

                        path_splines.append(ArcSpline(prev_coord, rx, ry, x_rot, large_arc_flag, sweep_flag, end_coord))

                        prev_coord[0] = end_coord[0]
                        prev_coord[1] = end_coord[1]
                        p_j += param_len

            splines.append(path_splines)

        return splines

    @property
    def visible(self):
        return 'none' not in self.layer.attrib['style']
    @visible.setter
    def visible(self, value):
        style = "display:inline;" if value else "display:none;"
        self.layer.attrib['style'] = style

###################################################################################
# SVG Helper functions
###################################################################################
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
        pts /= max

        #probably don't need more than 20 points, reduce detail of the polys
        if len(pts) > 20:
            pts = pts[::len(pts)/20]

        try:
            poly = Polygon([tuple(p) for p in pts])
            for i in np.linspace(0,1,100):
                if poly.buffer(-i).is_empty:
                    return list(poly.buffer(-last_i).centroid.coords)[0] * max + min
                last_i = i

            print("unable to find zero centroid...")
            return list(poly.buffer(-100).centroid.coords)[0] * max + min
        except:
            print("Shapely error")
            return np.nanmean(pts, 0)

except (ImportError, OSError):
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

def scrub(svgfile):
    """Remove data layers from an svg object prior to rendering

    Returns etree-parsed svg object
    """
    svg = etree.parse(svgfile, parser=parser)
    try:
        rmnode = _find_layer(svg, "data")
        rmnode.getparent().remove(rmnode)
    except ValueError:
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
    polyiter = trace_poly(boundary_edges(polys))
    for poly in [polyiter.next(), polyiter.next()]:
        path +="M%f %f L"%tuple(pts[poly.pop(0), :2])
        path += ', '.join(['%f %f'%tuple(pts[p, :2]) for p in poly])
        path += 'Z '

    w, h = pts.max(0)[:2]
    with open(os.path.join(cwd, "svgbase.xml")) as fp:
        svg = fp.read().format(width=w, height=h, clip=path)

    return svg

def get_overlay(svgfile, pts, polys, remove_medial=False, **kwargs):
    cullpts = pts[:,:2]
    if remove_medial:
        valid = np.unique(polys)
        cullpts = cullpts[valid]

    if not os.path.exists(svgfile):
        with open(svgfile, "w") as fp:
            fp.write(make_svg(pts.copy(), polys))

    svg = SVGOverlay(svgfile, coords=cullpts, **kwargs)
    
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
    offset = np.array(map(float, [data.pop(0), data.pop(0)]))
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
            offset += list(map(float, [data.pop(0), data.pop(0)]))
        elif mode == "L":
            offset = np.array(list(map(float, [data.pop(0), data.pop(0)])))
        elif mode == "c":
            data = data[4:]
            offset += list(map(float, [data.pop(0), data.pop(0)]))
        elif mode == "C":
            data = data[4:]
            offset = np.array(list(map(float, [data.pop(0), data.pop(0)])))
        #support multi-part paths, by only using one label for the whole path
        elif mode == 'm' :
            offset += list(map(float, [data.pop(0), data.pop(0)]))
        elif mode == 'M' :
            offset = list(map(float, [data.pop(0), data.pop(0)]))

        ## Check to see if nothing has happened, and, if so, fail
        if len(data) == lastlen:
            raise ValueError("Error parsing path.")
        else:
            lastlen = len(data)

        pts.append([offset[0],offset[1]])

    return np.array(pts)

def import_roi(roifile, outfile):
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

    with open(outfile, "w") as fp:
        fp.write(etree.tostring(svg, pretty_print=True))
