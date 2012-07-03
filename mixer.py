import os
import binascii
import tempfile
import cStringIO
import threading
import traceback
import cPickle
import subprocess as sp

import numpy as np
from scipy.interpolate import interp1d

import svgroi
from db import options

try:
    from traits.api import HasTraits, Instance, Array, Float, Int, Str, Bool, Dict, Range, Any, Color,Enum, Callable, Tuple, Button, on_trait_change, List
    from traitsui.api import View, Item, HGroup, Group, VGroup, ImageEnumEditor, ColorEditor, InstanceEditor, ListEditor

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
    from enthought.traits.api import HasTraits, Instance, Array, Float, Int, Str, Bool, Dict, Any, Range, Color,Enum, Callable, Tuple, Button, on_trait_change, List
    from enthought.traits.ui.api import View, Item, HGroup, Group, VGroup, ImageEnumEditor, ColorEditor, InstanceEditor, ListEditor

    from enthought.tvtk.api import tvtk
    from enthought.tvtk.pyface.scene import Scene

    from enthought.pyface.api import GUI

    from enthought.mayavi import mlab
    from enthought.mayavi.core.ui import lut_manager
    from enthought.mayavi.core.api import PipelineBase, Source, Filter, Module
    from enthought.mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
    from enthought.mayavi.sources.image_reader import ImageReader

cwd = os.path.split(os.path.abspath(__file__))[0]
default_texres = options['texture_res'] if 'texure_res' in options else 1024.
default_labelsize = options['label_size'] if 'label_size' in options else 24
default_renderheight = options['renderheight'] if 'renderheight' in options else 1024.
default_labelhide = options['labelhide'] if 'labelhide' in options else True
default_cmap = options['colormap'] if 'colormap' in options else "RdBu"
default_svfile = options['saved_views'] if 'saved_views' in options else None

class DataPack(HasTraits):
    name = Str

    def __init__(self, data, **kwargs):
        super(DataPack, self).__init__(**kwargs)
        self.ldat, self.rdat = data

    def set(self, srcs):
        left, right = srcs
        left.mlab_source.scalars = self.ldat
        right.mlab_source.scalars = self.rdat

    view = View(Item("name", show_label=False))

class SavedView(HasTraits):
    name = Str
    desc = Str

    def __init__(self, view, mix, pivot=0, **kwargs):
        super(SavedView, self).__init__(**kwargs)
        self.view = view, mix, pivot

    def set(self, mixer):
        view, mix, pivot = self.view
        mixer.set(mix=mix, pivot=pivot)
        mlab.view(*view, figure=mixer.figure)

    view = View('name', 'desc')

class Mixer(HasTraits):
    points = Any
    polys = Any
    coords = Any
    data = Array
    tcoords = Any
    mix = Range(0., 1., value=1)
    pivot = Range(-180., 180., value=-180.)
    nstops = Int(3)

    figure = Instance(MlabSceneModel, ())
    data_srcs = Any
    surfs = Any

    colormap = Enum(*lut_manager.lut_mode_list())
    fliplut = Bool
    show_colorbar = Bool
    vmin = Float
    vmax = Float

    #tex = Instance(ImageReader, ())
    tex = Instance(Source, ())
    texres = Float(default_texres)
    rois = Any
    roilabels = Dict
    labelsize = Int(default_labelsize)

    showrois = Bool(False)
    showlabels = Bool(False)

    reset_btn = Button(label="Reset View")
    datavars = List(DataPack)
    datapack = Instance(DataPack)

    saved_views = List
    viewpoint = Instance(SavedView)

    def __init__(self, points, polys, coords, data=None, dataname=None, svgfile=None, **kwargs):
        super(Mixer, self).__init__(points=points, polys=polys, coords=coords, **kwargs)
        if data is not None:
            self.data = data

        self.pivinterp = None
        if len(points) > 1:
            pint = np.zeros(self.nstops)
            pint[-1] = -180
            self.pivinterp = interp1d(np.linspace(0, 1, self.nstops), pint)

        if svgfile is not None:
            self.rois = svgroi.ROIpack(np.vstack(self.tcoords), svgfile)

        if default_svfile is not None and os.path.exists(os.path.join(cwd, default_svfile)):
            self.saved_views = cPickle.load(open(os.path.join(cwd, options['saved_views'])))
        self.update_crange()

    def _vmin_default(self):
        vmin = np.finfo(self.data.dtype).max
        for surf in self.surfs:
            lut = surf.module_manager.scalar_lut_manager
            if lut.data_range[0] < vmin:
                vmin = lut.data_range[0]
        return vmin

    def _vmax_default(self):
        vmax = np.finfo(self.data.dtype).min
        for surf in self.surfs:
            lut = surf.module_manager.scalar_lut_manager
            if lut.data_range[1] > vmax:
                vmax = lut.data_range[1]
        return vmax
    
    def _data_srcs_default(self):
        sources = []
        for points, polys, tcoords in zip(self.points, self.polys, self.tcoords):
            pts = points(1)
            src = mlab.pipeline.triangular_mesh_source(
                pts[:,0], pts[:,1], pts[:,2],
                polys, figure=self.figure.mayavi_scene)

            src.data.point_data.t_coords = tcoords
            sources.append(src)

        return sources

    def _surfs_default(self):
        surfs = []
        
        for data_src in self.data_srcs:
            t = mlab.pipeline.transform_data(data_src, figure=self.figure.mayavi_scene)
            t.widget.enabled = 0
            n = mlab.pipeline.poly_data_normals(t, figure=self.figure.mayavi_scene)
            n.filter.splitting = 0
            surf = mlab.pipeline.surface(n, figure=self.figure.mayavi_scene)
            surf.actor.texture.interpolate = True
            surf.actor.texture.repeat = False
            surf.actor.enable_texture = self.showrois
            lut = surf.module_manager.scalar_lut_manager
            lut.scalar_bar.title = ""
            lut.use_default_range = False
            surfs.append(surf)

        return surfs

    @on_trait_change("figure.activated")
    def _start(self):
        #initialize the figure
        self.figure.render_window.set(alpha_bit_planes=1, stereo_type="anaglyph", multi_samples=0)
        self.figure.renderer.set(use_depth_peeling=1, maximum_number_of_peels=100, occlusion_ratio=0.1)
        self.figure.scene.background = (0,0,0)
        self.figure.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        
        if hasattr(self.figure.mayavi_scene, "on_mouse_pick"):
            def picker(picker):
                for surf, coord in zip(self.surfs, self.coords):
                    if picker.actor == surf.actor.actor:
                        print coord[picker.point_id]

            self.picker = self.figure.mayavi_scene.on_mouse_pick(picker)
            self.picker.tolerance = 0.005

        #Add traits callbacks to update label visibility and positions
        self.figure.camera.on_trait_change(self._fix_label_vis, "position")

        self.data_srcs
        self.surfs
        for surf in self.surfs:
            surf.parent.parent.parent.widget.enabled = False
        self.colormap = default_cmap
        self.fliplut = True
        
        self.figure.reset_zoom()
        self.reset_view()
        self.figure.camera.view_up = [0,0,1]
        self.figure.render()

    def picker_to_coord(self, picker):
        '''Convenience function for handling pickers on multiple surfaces.
        You give this function a [picker] object (e.g. the one passed to
        a picker function) and it gives you back an (x,y,z) tuple of the
        voxel in EPI space that was picked.
        '''
        for surf, coord in zip(self.surfs, self.coords):
            if picker.actor == surf.actor.actor:
                return coord[picker.point_id]
    
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
        self.viewpoint = None
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
        self.figure.scene.disable_render = True
        self.viewpoint = None
        for data_src, points in zip(self.data_srcs, self.points):
            data_src.data.points.from_array(points(self.mix))
        
        if self.pivinterp is not None:
            self.pivot = float(self.pivinterp(self.mix))

        self._update_label_pos()
        #self.figure.renderer.reset_camera_clipping_range()
        self.figure.reset_zoom()
        self.figure.camera.view_up = [0,0,1]
        self.figure.scene.disable_render = False

    def _pivot_changed(self):
        '''Pivots the brain halves away from each other by pivot degrees'''
        left, right = self.data_srcs
        lxfm, rxfm, txfm = np.eye(4), np.eye(4), np.eye(4)
        p = np.radians(self.pivot/2)
        rot = np.array([
                [np.cos(p),-np.sin(p), 0, 0 ],
                [np.sin(p), np.cos(p), 0, 0 ],
                [   0,         0,      1, 0 ],
                [   0,         0,      0, 1]])
        if self.pivot > 0:
            lmove = left.data.points.to_array()[:,1].max()
            rmove = right.data.points.to_array()[:,1].max()
        elif self.pivot < 0:
            lmove = left.data.points.to_array()[:,1].min()
            rmove = right.data.points.to_array()[:,1].min()
        lrot = rot.copy()
        lrot[[0,1], [1,0]] = -lrot[[0,1], [1,0]]

        if self.pivot != 0:
            lxfm[1,-1] = -lmove
            txfm[1,-1] = lmove
            lxfm = np.dot(txfm, np.dot(lrot, lxfm))
            rxfm[1,-1] = -rmove
            txfm[1,-1] = rmove
            rxfm = np.dot(txfm, np.dot(rot, rxfm))
        try:
            left.children[0].transform.matrix.from_array(lxfm)
            right.children[0].transform.matrix.from_array(rxfm)
            left.children[0].update_pipeline()
            right.children[0].update_pipeline()
        except:
            pass
    
    def _data_changed(self):
        '''Trait callback for transforming the data and applying it to the surface.
        The data can be a 3D volume in the EPI space.
        Or a 4D volume where the first dimension is of length 3, in which case the
        data will be treated as color values for each voxel.
        '''
        pack = []
        for data_src, coords in zip(self.data_srcs, self.coords):
            coords = np.array([np.clip(c, 0, l-1) for c, l in zip(coords.T, self.data.T.shape)]).T
            scalars = self.data.T[tuple(coords.T)]
            if self.data.dtype == np.uint8 and len(self.data.shape) > 3:
                print "setting raw color data..."
                vtk_data = tvtk.UnsignedCharArray()
                vtk_data.from_array(scalars)
                vtk_data.name = "scalars"
                pack.append(vtk_data)
                #data_src.data.point_data.scalars = vtk_data
            else:
                pack.append(scalars)
                #data_src.mlab_source.scalars = scalars
        self.datapack = DataPack(pack, name="data%d"%len(self.datavars))
        self.datavars.append(self.datapack)

    def _datapack_changed(self):
        self.figure.disable_render = True
        self.datapack.set(self.data_srcs)
        self.figure.disable_render = False

    def _viewpoint_changed(self):
        self.figure.disable_render = True
        if self.viewpoint is not None:
            self.viewpoint.set(self)
        self.figure.disable_render = True
    
    def _tex_changed(self):
        self.figure.scene.disable_render = True
        for surf in self.surfs:
            surf.actor.texture_source_object = self.tex
            #Enable_Texture doesn't actually reflect whether it's visible or not unless you flip it!
            surf.actor.enable_texture = not self.showrois
            surf.actor.enable_texture = self.showrois
        self.figure.disable_render = False
    
    def _showrois_changed(self):
        self.figure.disable_render = True
        for surf in self.surfs:
            surf.actor.enable_texture = self.showrois
        self.figure.disable_render = False
    
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
        for surf in self.surfs:
            surf.module_manager.scalar_lut_manager.show_legend = self.show_colorbar

    @on_trait_change("vmin, vmax")
    def update_crange(self):
        for surf in self.surfs:
            lut = surf.module_manager.scalar_lut_manager
            lut.data_range = self.vmin, self.vmax
    
    @on_trait_change("colormap, fliplut")
    def _update_colors(self):
        self.figure.disable_render = True
        for surf in self.surfs:
            surf.parent.scalar_lut_manager.lut_mode = self.colormap
            surf.parent.scalar_lut_manager.reverse_lut = self.fliplut
        self.figure.disable_render = False
    
    @on_trait_change("reset_btn")
    def reset_view(self, center=True):
        '''Sets the view so that the flatmap is centered'''
        #set up the flatmap view
        self.mix = 1
        pts = np.vstack([surf.parent.parent.outputs[0].points.to_array() for surf in self.surfs])
        size = pts.max(0)-pts.min(0)
        focus = size / 2 + pts.min(0)
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
        curves = []
        for data_src in self.data_srcs:
            #smooth = mlab.pipeline.user_defined(self.data_src, filter="SmoothPolyDataFilter")
            curve = mlab.pipeline.user_defined(data_src, filter="Curvatures")
            curve.filter.curvature_type = "mean"
            #self.data_src.mlab_source.scalars = curve.filter.get_output().point_data.scalars.to_array()
            curvature = -1 * curve.filter.get_output().point_data.scalars.to_array()
            curves.append(curvature)
        self.mix = curmix
        self.figure.scene.disable_render = currender

        return curves

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
        for curv, data_src in zip(self.get_curvature(), self.data_srcs):
            if thresh:
                curv[curv>0] = 1
                curv[curv<0] = -1
            data_src.mlab_source.scalars = curv
            ## Set the colormap to gray
            self.colormap = "gray"
        ## Set the data range appropriately
        self.surf.module_manager.scalar_lut_manager.data_range = (-3, 3)
        self.figure.scene.disable_render = currender
    
    def saveflat(self, filename=None, height=default_renderheight):
        #Save the current view to restore
        startmix = self.mix
        lastpos = self.figure.camera.position, self.figure.camera.focal_point
        self.mix = 1
        x, y = self.figure.get_size()
        pts = np.vstack([surf.parent.parent.outputs[0].points.to_array() for surf in self.surfs])
        ptmax = pts.max(0)
        ptmin = pts.min(0)
        size = ptmax-ptmin
        aspect = size[0] / size[-1]
        width = height * aspect
        self.figure.set_size((width, height))
        self.figure.interactor.update_size(int(width), int(height))
        if 'use_offscreen' not in options or options['use_offscreen']:
            print "Using offscreen rendering"
            mlab.options.offscreen = True
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
        startmix = self.mix

        for name, (interp, labels) in self.roilabels.items():
            for l in labels:
                l.remove()
                
        mixes = np.linspace(0, 1, self.nstops)
        interps = dict([(name,[]) for name in self.rois.names])
        for mix in mixes:
            #self.mix = mix
            allpts, allnorms = [], []
            for data_src, points in zip(self.data_srcs, self.points):
                data_src.data.points.from_array(points(mix))

            if self.pivinterp is not None:
                self.pivot = float(self.pivinterp(mix))

            for data_src, surf, points in zip(self.data_srcs, self.surfs, self.points):
                pts = surf.parent.parent.outputs[0].points.to_array()
                norms = surf.parent.parent.outputs[0].point_data.normals.to_array()
                allpts.append(pts)
                allnorms.append(norms)

            for name, posnorm in self.rois.get_labelpos(np.vstack(allpts), np.vstack(allnorms)).items():
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

        self.mix = startmix
        self.figure.scene.disable_render = False

    def load_colormap(self, cmap):
        if cmap.max() <= 1:
            cmap = cmap.copy() * 255
        if cmap.shape[-1] < 4:
            cmap = np.hstack([cmap, 255*np.ones((len(cmap), 1))])

        for surf in self.surfs:
            surf.module_manager.scalar_lut_manager.lut.table = cmap
        self.figure.render()

    def save_view(self, name="Saved View"):
        vp = SavedView(mlab.view(figure=self.figure), self.mix, self.pivot, name=name)
        self.saved_views.append(vp)
        self.set(viewpoint=vp, trait_change_notify=False)
        if 'saved_views' in options:
            cPickle.dump(self.saved_views, open(os.path.join(cwd, options['saved_views']), 'w'))

    view = View(
        HGroup(
            Group(
                Item("figure", editor=SceneEditor(scene_class=MayaviScene)),
                Group("mix", "pivot", show_labels=True),
                show_labels=False),
            Group(
                Item('colormap',
                     editor=ImageEnumEditor(values=lut_manager.lut_mode_list(),
                     cols=6, path=lut_manager.lut_image_dir)),
                "fliplut", Item("show_colorbar", label="colorbar"), "vmin", "vmax", "_", 
                Item('datapack', editor=InstanceEditor(name="datavars", editable=False), width=100),
                Item('datavars', editor=ListEditor(style='custom')),
                "_",
                "showlabels", "showrois", Item("reset_btn", show_label=False),
                Item('viewpoint', editor=InstanceEditor(name='saved_views', editable=False), 
                    visible_when="len(saved_views) > 0"),
            ),
        show_labels=False),
        resizable=True, title="Mixer")
