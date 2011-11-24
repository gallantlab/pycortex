import os
import json
import binascii
import tempfile
import cStringIO
import subprocess as sp
import multiprocessing as mp
from xml.dom.minidom import parse as xmlparse

import Image
import numpy as np
from scipy.interpolate import interp1d, griddata

try:
    from traits.api import HasTraits, Instance, Array, Float, Int, Str, Bool, Dict, Range, Any, Color,Enum, Callable, on_trait_change
    from traitsui.api import View, Item, HGroup, Group, VGroup, ImageEnumEditor, ColorEditor

    from tvtk.api import tvtk
    from tvtk.pyface.scene import Scene

    from mayavi import mlab
    from mayavi.core.ui import lut_manager
    from mayavi.core.api import PipelineBase, Source, Filter, Module
    from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
    from mayavi.sources.array_source import ArraySource
except ImportError:
    from enthought.traits.api import HasTraits, Instance, Array, Float, Int, Str, Bool, Dict, Any, Range, Color,Enum, Callable, on_trait_change
    from enthought.traits.ui.api import View, Item, HGroup, Group, VGroup, ImageEnumEditor, ColorEditor, Handler

    from enthought.tvtk.api import tvtk
    from enthought.tvtk.pyface.scene import Scene

    from enthought.mayavi import mlab
    from enthought.mayavi.core.ui import lut_manager
    from enthought.mayavi.core.api import PipelineBase, Source, Filter, Module
    from enthought.mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
    from enthought.mayavi.sources.array_source import ArraySource

import db

_top = np.vstack([np.tile(255,[3,128]), np.arange(0,256,2)])
_bottom = np.vstack([np.tile(np.arange(256)[-2::-2], [3,1]), [np.tile(255,128)]])
clear_white_black = np.vstack([top.T, bottom.T])

cwd = os.path.split(os.path.abspath(__file__))[0]
options = json.load(open(os.path.join(cwd, "defaults.json")))
default_texres = options['texture_res'] if 'texure_res' in options else 1024.
default_lw = options['line_width'] if 'line_width' in options else 5.
default_labelsize = options['label_size'] if 'label_size' in options else 16
default_renderheight = options['renderheight'] if 'renderheight' in options else 1024.

class Mixer(HasTraits):
    points = Instance(interp1d)
    polys = Array(shape=(None, 3))
    xfm = Array(shape=(4,4))
    data = Array
    mix = Range(0., 1., value=1)

    figure = Instance(MlabSceneModel, ())
    data_src = Instance(Source)
    surf = Instance(Module)

    colormap = Enum(*lut_manager.lut_mode_list())
    fliplut = Bool

    svg = Instance("xml.dom.minidom.Document")
    svgfile = Str
    rois = Dict
    roilabels = Dict

    tex = Instance(ArraySource, ())
    texres = Float(default_texres)

    showrois = Bool(False)
    showlabels = Bool(False)
    labelsize = Int(default_labelsize)
    linewidth = Float(default_lw)
    roifill = Str("none")

    def __init__(self, points, polys, xfm, data=None, svgfile=None, **kwargs):
        #special init function must be used because points must be set before data can be set
        super(Mixer, self).__init__(polys=polys, xfm=xfm, **kwargs)
        self.points = points
        if data is not None:
            self.data = data
        if svgfile is not None:
            self.svgfile = svgfile

    def _data_src_default(self):
        pts = self.points(1)
        src = mlab.pipeline.triangular_mesh_source(
            pts[:,0], pts[:,1], pts[:,2],
            self.polys, figure=self.figure.mayavi_scene)
        #Set the texture coordinates
        pts -= pts.min(0)
        pts /= pts.max(0)
        src.data.point_data.t_coords = pts[:,[0,2]]
        return src

    def _surf_default(self):
        n = mlab.pipeline.poly_data_normals(self.data_src, figure=self.figure.mayavi_scene)
        surf = mlab.pipeline.surface(n, figure=self.figure.mayavi_scene)
        surf.actor.texture.interpolate = True
        surf.actor.texture.repeat = False
        surf.actor.texture.lookup_table = tvtk.LookupTable(
            table=clear_white_black, range=(-1,1))
        surf.actor.enable_texture = self.showrois
        return surf

    @on_trait_change("figure.activated")
    def _start(self):
        #initialize the figure
        self.figure.scene.background = (0,0,0)
        self.figure.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        self.figure.scene.render_window.stereo_type = "anaglyph"
        self.figure.camera.view_up = [0,0,1]
        self.reset_view()

        #Add traits callbacks to update label visibility and positions
        self.figure.camera.on_trait_change(self._fix_label_vis, "position")

        self.data_src
        self.surf
        self.colormap = "RdBu"
        self.fliplut = True
    
    def _update_label_pos(self):
        '''Creates and/or updates the position of the text to match the surface'''
        for name, labels in self.roilabels.items():
            for t, pts in labels:
                tpos = self._lookup_tex_world(pts).mean(0)
                t.set(x_position=tpos[0], y_position=tpos[1], z_position=tpos[2])
    
    def _fix_label_vis(self):
        '''Use backface culling behind the focal_point to hide labels behind the brain'''
        if self.showlabels:
            self.figure.disable_render = True
            fpos = self.figure.camera.focal_point
            vec = self.figure.camera.position - fpos
            for name, labels in self.roilabels.items():
                for t, pts in labels:
                    tpos = np.array((t.x_position, t.y_position, t.z_position))
                    t.visible = np.dot(vec, tpos - fpos) >= -1e-4
            self.figure.disable_render = False
    
    def _mix_changed(self):
        self.figure.disable_render = True
        self.data_src.data.points.from_array(self.points(self.mix))
        self.figure.renderer.reset_camera_clipping_range()
        self._update_label_pos()
        self.figure.disable_render = False
        self.figure.render()
        #def func():
        #    self.data_src.data.points = self.points(self.mix)
        #    GUI.invoke_later(self.data_src.data.update)
        #threading.Thread(target=func).start()
    
    def _points_changed(self):
        pts = self.points(0)
        wpts = np.append(pts, np.ones((len(pts),1)), axis=-1).T
        self.coords = np.dot(self.xfm, wpts)[:3].T.round().astype(int)
    
    def _data_changed(self):
        '''Trait callback for transforming the data and applying it to data'''
        scalars = np.array([self.data.T[tuple(p)] for p in self.coords])
        self.data_src.mlab_source.scalars = scalars
    
    def _tex_changed(self):
        self.figure.disable_render = True
        self.surf.actor.texture_source_object = self.tex
        #Enable_Texture doesn't actually reflect whether it's visible or not unless you flip it!
        self.surf.actor.enable_texture = not self.showrois
        self.surf.actor.enable_texture = self.showrois
        self.disable_render = False
    
    def _showrois_changed(self):
        self.surf.actor.enable_texture = self.showrois    
    
    def _showlabels_changed(self):
        self.figure.disable_render = True
        for name, labels in self.roilabels.items():
            for l, pts in labels:
                l.visible = self.showlabels
        self.figure.disable_render = False
    
    def _labelsize_changed(self):
        for name, labels in self.roilabels.items():
            for l, pts in labels:
                l.property.font_size = self.labelsize
    
    @on_trait_change("colormap, fliplut")
    def _update_colors(self):
        self.surf.parent.scalar_lut_manager.lut_mode = self.colormap
        self.surf.parent.scalar_lut_manager.reverse_lut = self.fliplut
    
    def _lookup_tex_world(self, pts):
        tcoords = self.data_src.data.point_data.t_coords.to_array()
        pos = self.data_src.data.points.to_array()
        return griddata(tcoords, pos, pts, method="nearest")
    
    def reset_view(self, center=True):
        '''Sets the view so that the flatmap is centered'''
        #set up the flatmap view
        self.mix = 1
        ptmax = self.data_src.data.points.to_array().max(0)
        ptmin = self.data_src.data.points.to_array().min(0)
        size = ptmax-ptmin
        focus = size / 2 + ptmin
        if center:
            focus[[0,2]] = 0
        
        x, y = self.figure.get_size()
        h = y / float(x) * size[0] / 2
        h /= np.tan(np.radians(self.figure.camera.view_angle / 2))
        campos = focus - [0, h, 0]
        self.figure.camera.position, self.figure.camera.focal_point = campos, focus
        self.figure.renderer.reset_camera_clipping_range()
        self.figure.render()
    
    def saveflat(self, filename=None, height=default_renderheight):
        #Save the current view to restore
        startmix = self.mix
        lastpos = self.figure.camera.position, self.figure.camera.focal_point

        #Turn on offscreen rendering
        mlab.options.offscreen = True
        x, y = self.figure.get_size()
        ptmax = self.data_src.data.points.to_array().max(0)
        ptmin = self.data_src.data.points.to_array().min(0)
        size = ptmax-ptmin
        aspect = size[0] / size[-1]
        width = height * aspect
        self.figure.set_size((width, height))
        self.figure.interactor.update_size(int(width), int(height))
        if 'use_offscreen' not in options or options['use_offscreen']:
            print "Using offscreen rendering"
            self.figure.off_screen_rendering = True
        if filename is None:
            self.reset_view(center=False)
            tf = tempfile.NamedTemporaryFile()
            self.figure.save_png(tf.name)
            pngdata = binascii.b2a_base64(tf.read())
        else:
            self.reset_view(center=True)
            self.figure.save_png(filename)

        #Restore the last view, turn off offscreen rendering
        self.figure.interactor.update_size(x, y)
        self.figure.set_size((x,y))
        self.mix = startmix
        self.figure.camera.position, self.figure.camera.focal_point = lastpos
        self.figure.renderer.reset_camera_clipping_range()
        if 'use_offscreen' not in options or options['use_offscreen']:
            self.figure.off_screen_rendering = False

        if filename is None:
            return (width, height), pngdata

    def add_roi(self, name):
        assert self.svgfile is not None, "Cannot find current ROI svg"
        #self.svg deletes the images -- we want to save those, so let's load it again
        svg = xmlparse(self.svgfile)
        imglayer = _find_layer(svg, "data")
        _make_layer(_find_layer(svg, "rois"), name)

        #Hide all the other layers in the image
        for layer in imglayer.getElementsByTagName("g"):
            layer.setAttribute("style", "display:hidden;")

        show = self.showrois, self.showlabels
        self.showrois, self.showlabels = False, False
        (width, height), pngdata = self.saveflat()
        self.showrois, self.showlabels = show

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
            
        sp.call(["inkscape",self.svgfile])
        self._svgfile_changed()
    
    def import_rois(self, filename):
        self.svgfile = filename
    
    def _svgfile_changed(self):
        svg = xmlparse(self.svgfile)
        svgdoc = svg.getElementsByTagName("svg")[0]
        w = float(svgdoc.getAttribute("width"))
        h = float(svgdoc.getAttribute("height"))
        self.svgshape = w, h

        #Remove the base images -- we don't need to render them for the texture
        rmnode = _find_layer(svg, "data")
        rmnode.parentNode.removeChild(rmnode)

        def make_path_pos(path):
            pts = _parse_svg_pts(path.getAttribute("d"))
            pts /= self.svgshape
            pts[:,1] = 1 - pts[:,1]
            return pts

        #Set up the ROI dict
        rois = _find_layer(svg, "rois")
        rois = dict([(r.getAttribute("inkscape:label"), r.getElementsByTagName("path"))
            for r in rois.getElementsByTagName("g")])
        
        #use traits callbacks to update the lines and textures
        self.rois = rois
        self.svg = svg
        self._create_roilabels()
    
    def _create_roilabels(self):
        self.figure.disable_render = True
        #Delete the existing roilabels, if there are any
        for name, roi in self.roilabels.items():
            for l in roi:
                l.remove()

        self.roilabels = {}
        for name, paths in self.rois.items():
            self.roilabels[name] = []
            for path in paths:
                pts = _parse_svg_pts(path.getAttribute("d"))
                pts /= self.svgshape
                pts[:,1] = 1 - pts[:,1]

                tpos = self._lookup_tex_world(pts).mean(0)
                txt = mlab.text(tpos[0], tpos[1], name, z=tpos[2], 
                        figure=self.figure.mayavi_scene, name=name)
                txt.set(visible=self.showlabels)
                txt.property.set(color=(0,0,0), bold=True, justification="center", 
                    vertical_justification="center", font_size=self.labelsize)
                txt.actor.text_scale_mode = "none"
                self.roilabels[name].append((txt, pts))
    
    @on_trait_change("rois, linewidth, roifill")
    def update_rois(self):
        for name, paths in self.rois.items():
            style = "fill:{fill};stroke:#000000;stroke-width:{lw}px;"+\
                    "stroke-linecap:butt;stroke-linejoin:miter;"+\
                    "stroke-opacity:1"
            style = style.format(fill=self.roifill, lw=self.linewidth)

            for i, path in enumerate(paths):
                #Set the fill and stroke
                path.setAttribute("style", style)
    
    @on_trait_change("svg, texres, linewidth, roifill")
    def update_texture(self):
        '''Updates the current texture as found in self.svg. 
        Converts it to PNG and applies it to the image'''
        #set the current size of the texture
        w, h = self.svgshape
        cmd = "convert -density {dpi} - png:-".format(dpi=self.texres / h * 72)
        convert = sp.Popen(cmd.split(), stdin=sp.PIPE, stdout=sp.PIPE)
        tex = cStringIO.StringIO(convert.communicate(self.svg.toxml())[0])
        tex = np.asarray(Image.open(tex)).astype(float).swapaxes(0,1)[:,::-1]
        self.tex = ArraySource(scalar_data=1. - tex[...,0] / 255.)

    view = View(
        HGroup(
            Group(
                Item("figure", editor=SceneEditor(scene_class=MayaviScene)),
                "mix",
                show_labels=False),
            Group(
                Item('colormap',
                     editor=ImageEnumEditor(values=lut_manager.lut_mode_list(),
                     cols=6, path=lut_manager.lut_image_dir)),
                "fliplut", "_", "showlabels", "showrois"),
        show_labels=False),
        resizable=True, title="Mixer")

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

def _parse_svg_pts(data):
    data = data.split()
    assert data[0] == "m", "Unknown path format"
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

    return np.array(pts)

def _get_surf_interp(subject, types=('inflated',), hemisphere="both"):
    types = ("fiducial",) + types + ("flat",)
    pts = []
    for t in types:
        pt, polys, norm = db.surfs.getVTK(subject, t, hemisphere=hemisphere)
        pts.append(pt)

    #flip the flats to be on the X-Z plane
    flatpts = np.zeros_like(pts[-1])
    flatpts[:,[0,2]] = pts[-1][:,:2]
    flatpts[:,1] = pts[-2].min(0)[1]
    pts[-1] = flatpts

    interp = interp1d(np.linspace(0,1,len(pts)), pts, axis=0)
    return interp, polys

svg_format = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="{width}" height="{height}"
   id="svg2" version="1.1"
   inkscape:version="0.48.1 r9760"
   sodipodi:docname="rois.svg">
  <defs id="defs4" />
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.35"
     inkscape:cx="932.14286"
     inkscape:cy="448.57143"
     inkscape:document-units="px"
     inkscape:current-layer="roilayer"
     showgrid="false"
     inkscape:window-width="1918"
     inkscape:window-height="1163"
     inkscape:window-x="0"
     inkscape:window-y="35"
     inkscape:window-maximized="0" />
  <metadata id="metadata7">
    <rdf:RDF>
      <cc:Work rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g inkscape:label="data" inkscape:groupmode="layer"
     id="datalayer" transform="translate(0,0)"
     style="display:inline">
  </g>
  <g inkscape:label="rois" inkscape:groupmode="layer"
     id="roilayer" transform="translate(0,0)"
     style="display:inline" />
</svg>"""

def show(data, subject, xfm, types=('inflated',), hemisphere="both"):
    '''View epi data, transformed into the space given by xfm. 
    Types indicates which surfaces to add to the interpolater. Always includes fiducial and flat'''
    interp, polys = _get_surf_interp(subject, types, hemisphere)

    if hasattr(data, "get_affine"):
        #this is a nibabel file -- it has the nifti headers intact!
        if isinstance(xfm, str):
            xfm = db.surfs.getXfm(subject, xfm, xfmtype="magnet")
            assert xfm is not None, "Cannot find transform by this name!"
            xfm = np.dot(np.linalg.inv(data.get_affine()), xfm[0])
        data = data.get_data()
    else:
        xfm = db.surfs.getXfm(subject, xfm, xfmtype="coord")
        assert xfm is not None, "Cannot find coord transform, please provide a nifti!"
        xfm = xfm[0]
    assert xfm.shape == (4, 4), "Not a transform matrix!"
    
    overlay = os.path.join(options['file_store'], "overlays", "%s_rois.svg"%subject)
    if not os.path.exists(overlay):
        #Can't find the roi overlay, create a new one!
        pts = interp(1)
        size = pts.max(0) - pts.min(0)
        aspect = size[0] / size[-1]
        with open(overlay, "w") as xml:
            xml.write(svg_format.format(width=aspect * 1024, height=1024))
        
    m = Mixer(points=interp, polys=polys, xfm=xfm, data=data, svgfile=overlay)
    m.edit_traits()
    return m

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Display epi data on various surfaces, \
        allowing you to interpolate between the surfaces")
    parser.add_argument("epi", type=str)
    parser.add_argument("--transform", "-T", type=str)
    parser.add_argument("--surfaces", nargs="*")