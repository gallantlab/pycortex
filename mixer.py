import os
import binascii
import tempfile
import cStringIO
import threading
import traceback
import subprocess as sp

import numpy as np
from scipy.interpolate import interp1d

import svgroi

try:
    from traits.api import HasTraits, Instance, Array, Float, Int, Str, Bool, Dict, Range, Any, Color,Enum, Callable, Tuple, Button, on_trait_change
    from traitsui.api import View, Item, HGroup, Group, VGroup, ImageEnumEditor, ColorEditor

    from tvtk.api import tvtk
    from tvtk.pyface.scene import Scene

    from pyface.api import GUI

    from mayavi import mlab
    from mayavi.core.ui import lut_manager
    from mayavi.core.api import PipelineBase, Source, Filter, Module
    from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
    from mayavi.sources.image_reader import ImageReader
    from mayavi.sources.array_source import Source

except ImportError:
    from enthought.traits.api import HasTraits, Instance, Array, Float, Int, Str, Bool, Dict, Any, Range, Color,Enum, Callable, Tuple, Button, on_trait_change
    from enthought.traits.ui.api import View, Item, HGroup, Group, VGroup, ImageEnumEditor, ColorEditor

    from enthought.tvtk.api import tvtk
    from enthought.tvtk.pyface.scene import Scene

    from enthought.pyface.api import GUI

    from enthought.mayavi import mlab
    from enthought.mayavi.core.ui import lut_manager
    from enthought.mayavi.core.api import PipelineBase, Source, Filter, Module
    from enthought.mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
    from enthought.mayavi.sources.image_reader import ImageReader

cwd = os.path.split(os.path.abspath(__file__))[0]
from db import options
default_texres = options['texture_res'] if 'texure_res' in options else 1024.
default_labelsize = options['label_size'] if 'label_size' in options else 24
default_renderheight = options['renderheight'] if 'renderheight' in options else 1024.
default_labelhide = options['labelhide'] if 'labelhide' in options else True
default_cmap = options['colormap'] if 'colormap' in options else "RdBu"

class Mixer(HasTraits):
    points = Any
    polys = Array(shape=(None, 3))
    xfm = Array(shape=(4,4))
    data = Array
    tcoords = Array
    mix = Range(0., 1., value=1)
    nstops = Int(3)

    figure = Instance(MlabSceneModel, ())
    data_src = Instance(Source)
    surf = Instance(Module)

    colormap = Enum(*lut_manager.lut_mode_list())
    fliplut = Bool
    show_colorbar = Bool

    #tex = Instance(ImageReader, ())
    tex = Instance(Source, ())
    texres = Float(default_texres)
    rois = Any
    roilabels = Dict
    labelsize = Int(default_labelsize)

    showrois = Bool(False)
    showlabels = Bool(False)

    reset_btn = Button(label="Reset View")

    def __init__(self, points, polys, xfm, data=None, svgfile=None, **kwargs):
        #special init function must be used because points must be set before data can be set
        self.xfm = xfm
        self.points = points
        super(Mixer, self).__init__(polys=polys, xfm=xfm, **kwargs)
        
        if data is not None:
            self.data = data

        if svgfile is not None:
            self.rois = svgroi.ROIpack(self.tcoords, svgfile)
            
    def _data_src_default(self):
        pts = self.points(1)
        src = mlab.pipeline.triangular_mesh_source(
            pts[:,0], pts[:,1], pts[:,2],
            self.polys, figure=self.figure.mayavi_scene)
        #Set the texture coordinates

        if len(self.tcoords) == 0:
            pts -= pts.min(0)
            pts /= pts.max(0)
            self.tcoords = pts[:,[0,2]]
        src.data.point_data.t_coords = self.tcoords
        
        return src

    def _surf_default(self):
        n = mlab.pipeline.poly_data_normals(self.data_src, figure=self.figure.mayavi_scene)
        surf = mlab.pipeline.surface(n, figure=self.figure.mayavi_scene)
        surf.actor.texture.interpolate = True
        surf.actor.texture.repeat = False
        surf.actor.enable_texture = self.showrois
        surf.module_manager.scalar_lut_manager.scalar_bar.title = None
        return surf

    @on_trait_change("figure.activated")
    def _start(self):
        #initialize the figure
        self.figure.render_window.set(alpha_bit_planes=1, stereo_type="anaglyph", multi_samples=0)
        self.figure.renderer.set(use_depth_peeling=1, maximum_number_of_peels=100, occlusion_ratio=0.1)
        self.figure.scene.background = (0,0,0)
        self.figure.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        self.figure.camera.view_up = [0,0,1]
        self.reset_view()

        if hasattr(self.figure.mayavi_scene, "on_mouse_pick"):
            def picker(picker):
                print self.coords[picker.point_id]
            self.picker = self.figure.mayavi_scene.on_mouse_pick(picker)
            self.picker.tolerance = 0.005

        #Add traits callbacks to update label visibility and positions
        self.figure.camera.on_trait_change(self._fix_label_vis, "position")

        self.data_src
        self.surf
        self.colormap = default_cmap
        self.fliplut = True
        self.figure.camera.focal_point = self.data_src.data.points.to_array().mean(0)
    
    def _update_label_pos(self):
        '''Creates and/or updates the position of the text to match the surface'''
        currender = self.figure.scene.disable_render
        self.figure.scene.disable_render = True
        for name, (interp, labels) in self.roilabels.items():
            for t, (pos, norm) in zip(labels, interp(self.mix)):
                t.set(x_position=pos[0], y_position=pos[1], z_position=pos[2], norm=tuple(norm))
        self.figure.scene.disable_render = currender
    
    def _fix_label_vis(self):
        '''Use backface culling behind the focal_point to hide labels behind the brain'''
        if self.showlabels and self.mix != 1:
            flipme = []
            fpos = self.figure.camera.focal_point
            for name, (interp, labels) in self.roilabels.items():
                for t in labels:
                    tpos = np.array((t.x_position, t.y_position, t.z_position))
                    cam = self.figure.camera.position
                    state = np.dot(cam-tpos, t.norm) >= 1e-4 and np.dot(cam-fpos, tpos-fpos) >= -1
                    if t.visible != state:
                        flipme.append(t)
            
            if len(flipme) > 0:
                if default_labelhide:
                    self.figure.scene.disable_render = True
                for t in flipme:
                    t.visible = not t.visible
                if default_labelhide:
                    self.figure.scene.disable_render = False
    
    def _mix_changed(self):
        pts = self.points(self.mix)
        self.figure.scene.disable_render = True
        self.data_src.data.points.from_array(pts)
        self.figure.renderer.reset_camera_clipping_range()
        self._update_label_pos()
        self.figure.camera.focal_point = pts.mean(0)
        #self.figure.render()
        self.figure.scene.disable_render = False
        #self.figure.render()
        '''
        def func():
            def update():
                pass
            GUI.invoke_later(update)
        threading.Thread(target=func).start()
        '''
    
    def _points_changed(self):
        pts = self.points(0)
        wpts = np.append(pts, np.ones((len(pts),1)), axis=-1).T
        self.coords = np.dot(self.xfm, wpts)[:3].T.round().astype(int)
    
    def _data_changed(self):
        '''Trait callback for transforming the data and applying it to data'''
        coords = np.array([np.clip(c, 0, l-1) for c, l in zip(self.coords.T, self.data.T.shape)]).T
        scalars = self.data.T[tuple(coords.T)]
        if self.data.dtype == np.uint8 and len(self.data.shape) > 3:
            print "setting raw color data..."
            vtk_data = tvtk.UnsignedCharArray()
            vtk_data.from_array(scalars)
            vtk_data.name = "scalars"
            self.data_src.data.point_data.scalars = vtk_data
        else:
            self.data_src.mlab_source.scalars = scalars

    
    def _tex_changed(self):
        self.figure.scene.disable_render = True
        self.surf.actor.texture_source_object = self.tex
        #Enable_Texture doesn't actually reflect whether it's visible or not unless you flip it!
        self.surf.actor.enable_texture = not self.showrois
        self.surf.actor.enable_texture = self.showrois
        self.disable_render = False
    
    def _showrois_changed(self):
        self.surf.actor.enable_texture = self.showrois    
    
    def _showlabels_changed(self):
        self.figure.scene.disable_render = True
        for name, (interp, labels) in self.roilabels.items():
            for l in labels:
                l.visible = self.showlabels
        self.figure.scene.disable_render = False
    
    def _labelsize_changed(self):
        self.figure.scene.disable_render = True
        for name, labels in self.roilabels.items():
            for l, pts in labels:
                l.property.font_size = self.labelsize
        self.figure.scene.disable_render = False
    
    def _show_colorbar_changed(self):
        self.surf.module_manager.scalar_lut_manager.show_legend = self.show_colorbar
    
    @on_trait_change("colormap, fliplut")
    def _update_colors(self):
        self.surf.parent.scalar_lut_manager.lut_mode = self.colormap
        self.surf.parent.scalar_lut_manager.reverse_lut = self.fliplut
    
    def data_to_points(self, arr):
        '''Maps the given 3D data array [arr] to vertices on the mesh.
        '''
        return np.array([arr.T[tuple(p)] for p in self.coords])

    def lindata_to_points(self, linarr, mask):
        '''Maps the given 1D data array [linarr] to vertices on the mesh, but first
        maps the 1D data into 3D space via the given [mask].

        Parameters
        ----------
        linarr : (N,) array, float
            A vector containing a floating point value for each voxel.
        mask : (Z,Y,X) array, binary
            A 3D mask that is True wherever a voxel value should be mapped to
            the surface.

        Returns
        -------
        pointdata : (M,) array, float
            A new vector that contains, for each vertex, the value of the voxel
            that vertex lies inside.
        '''
        datavol = mask.copy().astype(linarr.dtype)
        datavol[mask>0] = linarr
        return self.data_to_points(datavol)
    
    @on_trait_change("reset_btn")
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
    
    def get_curvature(self):
        '''Compute the curvature at each vertex on the surface and return it.
        The curvature is NEGATIVE for vertices where the surface is concave,
        e.g. inside sulci. The curvature is POSITIVE for vertices where the
        surface is convex, e.g. on gyri.
        '''
        currender = self.figure.scene.disable_render
        self.figure.scene.disable_render = True
        curmix = float(self.mix)
        self.mix = 0
        #smooth = mlab.pipeline.user_defined(self.data_src, filter="SmoothPolyDataFilter")
        curve = mlab.pipeline.user_defined(self.data_src, filter="Curvatures")
        curve.filter.curvature_type = "mean"
        #self.data_src.mlab_source.scalars = curve.filter.get_output().point_data.scalars.to_array()
        curvature = -1 * curve.filter.get_output().point_data.scalars.to_array()
        self.mix = curmix
        self.figure.scene.disable_render = currender

        return curvature

    def show_curvature(self, thresh=False):
        '''Replace the current data with surface curvature. By default this
        function sets the data range to (-3..3), which works well for most
        cases.

        If [thresh] is set to True, curvature will be thresholded.
        '''
        currender = self.figure.scene.disable_render
        self.figure.scene.disable_render = True
        ## Load the curvature onto the surface
        curv = self.get_curvature()
        if thresh:
            curv[curv>0] = 1
            curv[curv<0] = -1
        self.data_src.mlab_source.scalars = curv
        ## Set the colormap to gray
        self.colormap = "gray"
        ## Set the data range appropriately
        self.surf.module_manager.scalar_lut_manager.data_range = (-3, 3)
        self.figure.scene.disable_render = currender
    
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
        '''Opens Inkscape and adds currently displayed data as a new image layer.
        When Inkscape closes, the SVG overlay is reloaded.
        '''
        ## First get a PNG of the current scene w/out labels
        last_rois = self.showrois
        self.showrois = False
        (w,h),pngdata = self.saveflat()
        self.showrois = last_rois

        ## Then call add_roi of our SVG object to add the new image layer
        self.rois.add_roi(name, pngdata)

        ## Then open inkscape
        sp.call(["inkscape", self.rois.svgfile])

        ## Finally update the ROI overlay based on the new svg
        self.rois.reload()
        self.update_texture()
    
    @on_trait_change("rois, texres")
    def update_texture(self):
        texfile = self.rois.get_texture(self.texres)
        self.tex = ImageReader(file_list=[texfile.name])
    
    @on_trait_change("rois")
    def _create_roilabels(self):
        #Delete the existing roilabels, if there are any
        self.figure.scene.disable_render = True

        for name, (interp, labels) in self.roilabels.items():
            for l in labels:
                l.remove()
                
        mixes = np.linspace(0, 1, self.nstops)
        interps = dict([(name,[]) for name in self.rois.names])
        for mix in mixes:
            pts = self.points(mix)
            self.data_src.data.points.from_array(pts)
            norms = self.surf.parent.parent.outputs[0].point_data.normals.to_array()
            for name, posnorm in self.rois.get_labelpos(pts, norms).items():
                interps[name].append(posnorm)
        
        self.roilabels = dict()
        for name, pos in interps.items():
            interp = interp1d(mixes, pos, axis=0)
            self.roilabels[name] = interp, []

            for pos, norm in interp(self.mix):
                txt = mlab.text(pos[0], pos[1], name, z=pos[2], 
                        figure=self.figure.mayavi_scene, name=name)
                txt.set(visible=self.showlabels)
                txt.property.set(color=(0,0,0), bold=True, justification="center", 
                    vertical_justification="center", font_size=self.labelsize)
                txt.actor.text_scale_mode = "none"
                txt.add_trait("norm", tuple)
                txt.norm = tuple(norm)
                self.roilabels[name][1].append(txt)

        self.data_src.data.points.from_array(self.points(self.mix))
        self.figure.scene.disable_render = False
            
    
    def load_colormap(self, cmap):
        if cmap.max() <= 1:
            cmap = cmap.copy() * 255
        if cmap.shape[-1] < 4:
            cmap = np.hstack([cmap, 255*np.ones((len(cmap), 1))])
        self.surf.module_manager.scalar_lut_manager.lut.table = cmap
        self.figure.render()
    
    def show(self):
        return mlab.show()

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
                "fliplut", "show_colorbar", "_", "showlabels", "showrois",
                "reset_btn"
                ),
        show_labels=False),
        resizable=True, title="Mixer")
