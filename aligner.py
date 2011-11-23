#/usr/bin/env python
import os
import numpy as np
import nibabel

try:
    from traits.api import HasTraits, List, Instance, Array, Bool, Dict, Range, Float, Enum, Color, on_trait_change
    from traitsui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor

    from tvtk.api import tvtk
    from tvtk.pyface.scene import Scene

    from mayavi import mlab
    from mayavi.core.ui import lut_manager
    from mayavi.core.api import PipelineBase, Source, Filter, Module
    from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
except ImportError:
    from enthought.traits.api import HasTraits, List, Instance, Array, Bool, Dict, Float, Enum, Range, Color, on_trait_change
    from enthought.traits.ui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor

    from enthought.tvtk.api import tvtk
    from enthought.tvtk.pyface.scene import Scene

    from enthought.mayavi import mlab
    from enthought.mayavi.core.ui import lut_manager
    from enthought.mayavi.core.api import PipelineBase, Source, Filter, Module
    from enthought.mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene

class RotationWidget(HasTraits):
    radius = Float(value=1)
    angle = Float(value=0)
    pos = Array(value=[0,0,0])
    enabled = Bool(value=True)

    def __init__(self, figure, callback, **traits):
        self._t = np.linspace(0, 2*np.pi, 32)

        super(RotationWidget, self).__init__(**traits)
        self.callback = callback
        self.figure = figure

        self.center = tvtk.HandleWidget()
        self.center.set_representation(tvtk.SphereHandleRepresentation())
        self.center.representation.world_position = self.pos
        self.center.priority = 1

        self.edge = tvtk.HandleWidget()
        self.edge.set_representation(tvtk.SphereHandleRepresentation())
        self.edge.representation.world_position = self.pos + (self.radius,0,0)
        self.edge.priority = 1

        self.circle = mlab.pipeline.line_source(*self._gen_circle(), figure=figure)
        self.tube = mlab.pipeline.tube(self.circle, figure=figure)
        self.surf = mlab.pipeline.surface(self.tube, figure=figure, color=(1,1,1))
        self.tube.filter.radius = 1

        figure.scene.add_widgets([self.center, self.edge])
        def startmove(obj, evt):
            self.startmove = (self.pos, self.angle, self.radius)
        def endmove(obj, evt):
            if hasattr(self.callback, "__call__"):
                self.callback(  self.pos - self.startmove[0], 
                                self.angle - self.startmove[1], 
                                self.radius / self.startmove[2])
        self.edge.add_observer("StartInteractionEvent", startmove)
        self.edge.add_observer("InteractionEvent", self._move_edge)
        self.edge.add_observer("EndInteractionEvent", endmove)
        self.center.add_observer("StartInteractionEvent", startmove)
        self.center.add_observer("InteractionEvent", self._move_center)
        self.center.add_observer("EndInteractionEvent", endmove)
    
    def move(self, pos=(0,0,0), angle=0, radius=1):
        if hasattr(self.callback, "__call__"):
            self.callback(pos, angle, radius)
        self.set(pos=self.pos+pos, angle=self.angle+angle, radius=radius*self.radius)
    
    def _move_center(self, obj=None, evt=None):
        self.pos = self.center.representation.world_position
    
    def _move_edge(self, obj=None, evt=None):
        c = self.center.representation.world_position
        r = self.edge.representation.world_position
        self.edge.representation.world_position = r[0], r[1], c[2]
        r -= c

        angle = np.arctan2(r[1], r[0])
        self.set(angle=angle, radius=np.sqrt(np.sum(r**2)))
    
    def _gen_circle(self):
        t = self._t+self.angle
        x = self.radius*np.cos(t) + self.pos[0]
        y = self.radius*np.sin(t) + self.pos[1]
        return x, y, np.repeat(self.pos[2], len(t))
    
    @on_trait_change("pos,angle,radius")
    def _set_circle(self):
        if hasattr(self, "circle"):
            self.center.representation.world_position = self.pos
            rpos = map(lambda f: self.radius*f(self.angle), [np.cos, np.sin])
            self.edge.representation.world_position = self.pos + (rpos+[0])
            
            #self.circle.mlab_source.set(dict(zip(("x", "y", "z"), self._gen_circle())))
            self.circle.data.points = np.array(self._gen_circle()).T
    
    @on_trait_change("enabled")
    def _enable(self):
        if self.enabled:
            self.center.on()
            self.edge.on()
        else:
            self.center.off()
            self.edge.off()
        
        self.circle.visible = self.enabled

class ThreeDScene(MayaviScene):
    aligner = Instance("Align")
    state = Array(value=[1,1,1])

    def OnKeyDown(self, evt):
        key = chr(evt.GetKeyCode() % 256)
        focus = self.aligner.scene3d.camera.focal_point
        if key == "1":
            self.aligner.scene3d.camera.position = focus + [0, self.state[1]*500, 0]
            self.state[1] = -self.state[1]
        elif key == "3":
            self.aligner.scene3d.camera.position = focus + [self.state[0]*500, 0, 0]
            self.state[0] = -self.state[0]
        elif key == "5":
            self.aligner.scene3d.parallel_projection = not self.aligner.scene3d.parallel_projection
        elif key == "7":
            self.aligner.scene3d.camera.position = focus + [0, 1e-6, self.state[2]*500]
            self.state[2] = -self.state[2]
        elif key == "\x1a" and evt.CmdDown() and hasattr(self.aligner, "_last_transform"):
            self.aligner.xfm.transform.set_matrix(self.aligner._last_transform.ravel())
            self.aligner.xfm.widget.set_transform(self.aligner.xfm.transform)
            self.aligner.xfm.update_pipeline()
            self.aligner.update_slab()
        else:
            super(ThreeDScene, self).OnKeyDown(evt)
        self.aligner.scene3d.renderer.reset_camera_clipping_range()
        self.aligner.scene3d.render()

class FlatScene(Scene):
    handle = Instance(RotationWidget)
    aligner = Instance("Align")
    invert = Bool(value=False)

    def OnKeyDown(self, evt):
        i = -1 if self.invert else 1
        key = evt.GetKeyCode()
        moves = {314:(-2,0,0), 315:(0,2,0), 316:(2,0,0), 317:(0,-2,0)}
        if self.invert:
            moves = {314:(0,-2,0), 315:(2,0,0), 316:(0,2,0), 317:(-2,0,0)}
        
        mult = (1,.25)[evt.ShiftDown()]
        if key in moves:
            self.handle.move(np.array(moves[key])*mult)
        elif key == 366:
            self.handle.move(angle=np.pi / 120.*i*mult)
        elif key == 367:
            self.handle.move(angle=-np.pi / 120.*i*mult)
        elif chr(key % 256) == "h":
            for o in self.aligner.outlines:
                o.visible = not o.visible
        else:
            super(FlatScene, self).OnKeyDown(evt)         

################################################################################
# The object implementing the dialog
class Align(HasTraits):
    # The position of the view
    position = Array(shape=(3,))
    brightness = Range(-1., 1., value=0.)
    contrast = Range(0., 2., value=1.)
    opacity = Range(0., 1.)
    colormap = Enum(*lut_manager.lut_mode_list())
    fliplut = Bool
    outlines = List
    ptcolor = Color(value="navy")

    # The 4 views displayed
    scene3d = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    epi_src = Instance(Source)
    surf_src = Instance(Source)
    xfm = Instance(Filter)
    surf = Instance(Module)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    # The cursors on each view:
    cursors = Dict()

    disable_render = Bool

    _axis_names = dict(x=0, y=1, z=2)

    flip_fb = Bool
    flip_lr = Bool
    flip_ud = Bool

    #---------------------------------------------------------------------------
    # Object interface
    #---------------------------------------------------------------------------
    def __init__(self, pts, polys, epi, xfm=None, xfmtype='magnet', **traits):
        '''
        Parameters
        ----------
        xfm : array_like, optional
            The initial 4x4 rotation matrix into magnet space 
            (epi with slice affine)
        '''
        nii = nibabel.load(epi)
        epi = nii.get_data()
        self.affine = nii.get_affine()
        base = nii.get_header().get_base_affine()
        self.base = base
        self.origin = base[:3, -1]
        self.spacing = np.diag(base)[:3]
        if xfm is None:
            self.startxfm = np.dot(base, np.linalg.inv(self.affine))
        else:
            self.startxfm = np.dot(np.dot(base, np.linalg.inv(self.affine)), xfm)
        self.center = self.spacing*nii.get_shape() / 2 + self.origin

        self.epi = epi - epi.min()
        self.epi /= self.epi.max()
        self.epi *= 2
        self.epi -= 1

        self.pts, self.polys = pts, polys

        super(Align, self).__init__(**traits)
        # Force the creation of the image_plane_widgets:
        self.ipw_3d_x
        self.ipw_3d_y
        self.ipw_3d_z

    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _position_default(self):
        return np.abs(self.origin)

    def _epi_src_default(self):
        sf = mlab.pipeline.scalar_field(self.epi,
                            figure=self.scene3d.mayavi_scene,
                            name='EPI')
        sf.origin = self.origin
        sf.spacing = self.spacing
        return sf
    
    def _surf_src_default(self):
        return mlab.pipeline.triangular_mesh_source(
                            self.pts[:,0], self.pts[:,1], self.pts[:,2], self.polys,
                            figure=self.scene3d.mayavi_scene,
                            name='Cortex')
    
    def _surf_default(self):
        smooth = mlab.pipeline.poly_data_normals(self.xfm, figure=self.scene3d.mayavi_scene)
        smooth.filter.splitting = False
        surf = mlab.pipeline.surface(smooth, figure=self.scene3d.mayavi_scene)
        surf.actor.mapper.scalar_visibility = 0
        return surf

    def _xfm_default(self):
        xfm = mlab.pipeline.transform_data(self.surf_src, figure=self.scene3d.mayavi_scene)
        def savexfm(info, evt):
            self._last_transform = xfm.transform.matrix.to_array()
        xfm.widget.add_observer("StartInteractionEvent", savexfm)
        xfm.widget.add_observer("EndInteractionEvent", self.update_slab)
        xfm.transform.set_matrix(self.startxfm.ravel())
        xfm.widget.set_transform(xfm.transform)
        return xfm

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(self.epi_src,
                        figure=self.scene3d.mayavi_scene,
                        plane_orientation='%s_axes' % axis_name,
                        name='Cut %s' % axis_name)
        ipw.ipw.texture_interpolate = 0
        ipw.ipw.reslice_interpolate = 'nearest_neighbour'
        ipw.ipw.color_map.output_format = 'rgb'
        slab = mlab.pipeline.user_defined(self.surf, filter='GeometryFilter', 
                            figure=self.scene3d.mayavi_scene)
        slab.filter.extent_clipping = True
        slab.filter.point_clipping = True
        surf = mlab.pipeline.surface(slab, 
                            color=(1,1,1), 
                            figure=self.scene3d.mayavi_scene, 
                            representation='points')
        surf.actor.property.point_size = 5
        setattr(self, "slab_%s"%axis_name, slab)

        return ipw

    def _ipw_3d_x_default(self):
        return self.make_ipw_3d('x')

    def _ipw_3d_y_default(self):
        return self.make_ipw_3d('y')

    def _ipw_3d_z_default(self):
        return self.make_ipw_3d('z')
    

    #---------------------------------------------------------------
    # Set up side views
    #---------------------------------------------------------------
    def make_side_view(self, axis_name):
        ipw_3d = getattr(self, 'ipw_3d_%s' % axis_name)
        scene = getattr(self, 'scene_%s' % axis_name)
        scene.scene.parallel_projection = True
        
        side_src = ipw_3d.ipw._get_reslice_output()
        ipw = mlab.pipeline.image_plane_widget( side_src,
                            plane_orientation='z_axes',
                            figure=scene.mayavi_scene,
                            name='Cut view %s' % axis_name,
                            )
        ipw.ipw.left_button_action = 0
        ipw.ipw.texture_interpolate = 0
        ipw.ipw.reslice_interpolate = 'nearest_neighbour'
        ipw.parent.scalar_lut_manager.use_default_range = False
        ipw.parent.scalar_lut_manager.default_data_range = [-1, 1]
        ipw.parent.scalar_lut_manager.data_range = [-1, 1]
        setattr(self, 'ipw_%s' % axis_name, ipw)

        pts = mlab.pipeline.scalar_scatter( *np.random.randn(3, 10),
            figure=scene.mayavi_scene)
        glyph = mlab.pipeline.glyph(pts, scale_mode='none', color=(1,1,1), mode='2dsquare',
            figure=scene.mayavi_scene)
        glyph.glyph.glyph_source.glyph_source.filled = True
        setattr(self, "outline_%s"%axis_name, pts)
        self.outlines.append(pts)

        # Extract the spacing of the side_src to convert coordinates
        # into indices
        spacing = side_src.spacing

        x, y, z = self.position
        cursor = mlab.points3d(x, y, z,
                            mode='axes',
                            color=(0, 0, 0),
                            scale_factor=2*max(self.epi[0].shape),
                            figure=scene.mayavi_scene,
                            name='Cursor view %s' % axis_name,
                        )
        self.cursors[axis_name] = cursor

        # Add a callback on the image plane widget interaction to
        # move the others
        this_axis_number = self._axis_names[axis_name]
        def move_view(obj, evt):
            # Disable rendering on all scene
            position = list(obj.GetCurrentCursorPosition()*spacing)[:2]
            position.insert(this_axis_number, self.position[this_axis_number])
            # We need to special case y, as the view has been rotated.
            if axis_name is 'y':
                position = position[::-1]
            
            self.position = position

        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)

        # 2D interaction: only pan and zoom
        scene.scene.interactor.interactor_style = tvtk.InteractorStyleImage()
        def focusfunc(vtkobj, i):
            scene.scene_editor.control.SetFocusFromKbd()
        scene.scene.interactor.add_observer("MouseMoveEvent", focusfunc)
        scene.scene.background = (0, 0, 0)

        # Some text:
        mlab.text(0.01, 0.8, axis_name, width=0.08)

        # Choose a view that makes sense
        center = side_src.whole_extent[1::2] * spacing / 2.
        width = (side_src.whole_extent[1::2] * spacing)[:2]
        width = np.min(width) * 0.5

        rotaxis = ['rotate_x', 'rotate_y', 'rotate_z']
        def handlemove(pos, angle, radius):
            signs = np.sign(np.diag(self.xfm.transform.matrix.to_array()))
            if this_axis_number == 1:
                trans = np.insert(pos[:2][::-1], this_axis_number, 0)
            else:
                trans = np.insert(pos[:2], this_axis_number, 0)
            trans *= signs[:-1]
            rot = np.degrees(angle)*-signs[this_axis_number]
            scale = np.repeat(radius, 3)
            scale[this_axis_number] = 1

            self.xfm.transform.translate(trans)
            getattr(self.xfm.transform, rotaxis[this_axis_number])(rot)
            self.xfm.transform.scale(scale)
            self.xfm.widget.set_transform(self.xfm.filter.transform)
            self.xfm.update_pipeline()
            self.update_slab()

        handle = RotationWidget(scene.mayavi_scene, handlemove, radius=width, pos=center)
        setattr(self, "handle_%s"%axis_name, handle)

        views = dict(x=(0, 0), y=(90, 180), z=(0, 0))
        mlab.view(*views[axis_name],
                  focalpoint=center,
                  figure=scene.mayavi_scene)
        scene.scene.camera.parallel_scale = width * 1.2
    
    #---------------------------------------------------------------------------
    # Scene activation callbacks
    #---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        self.scene3d.scene.renderer.use_depth_peeling = True
        self.scene3d.mlab.view(40, 50)
        # Interaction properties can only be changed after the scene
        # has been created, and thus the interactor exists
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            ipw.ipw.interaction = 0
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()

        self.ipw_3d_z.parent.scalar_lut_manager.use_default_range = False
        self.ipw_3d_z.parent.scalar_lut_manager.default_data_range = [-1, 1]
        self.ipw_3d_z.parent.scalar_lut_manager.data_range = [-1, 1]

        self.scene3d.scene_editor.aligner = self

        self.opacity = 0.1
        self.xfm.widget.enabled = False
        self.update_position()
        self.colormap = "RdYlGn"

    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        self.make_side_view('x')
        self.scene_x.scene_editor.handle = self.handle_x
        self.scene_x.scene_editor.aligner = self

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        self.make_side_view('y')
        self.scene_y.scene_editor.handle = self.handle_y
        self.scene_y.scene_editor.invert = True
        self.scene_y.scene_editor.aligner = self

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        self.make_side_view('z')
        self.scene_z.scene_editor.handle = self.handle_z
        self.scene_z.scene_editor.aligner = self

    #---------------------------------------------------------------------------
    # Traits callback
    #---------------------------------------------------------------------------
    @on_trait_change('position')
    def update_position(self):
        """ Update the position of the cursors on each side view, as well
            as the image_plane_widgets in the 3D view.
        """
        self.disable_render = True
        origin = self.origin * np.sign(self.spacing)
        offset = np.abs(self.spacing) / 2

        for axis_name, axis_number in self._axis_names.iteritems():
            ipw3d = getattr(self, 'ipw_3d_%s' % axis_name)
            ipw3d.ipw.slice_position = self.position[axis_number]+origin[axis_number] + offset[axis_number]
            
            p = list(self.position + offset)
            p.pop(axis_number)
            if axis_name is 'y':
                p = p[::-1]
            p.append(0)
            self.cursors[axis_name].parent.parent.data.points = [p]

        self.update_slab()
        # Finally re-enable rendering
        self.disable_render = False
    
    def update_slab(self, obj=None, evt=None):
        self.disable_render = True
        origin = self.origin * np.sign(self.spacing)
        limit = self.epi_src.scalar_data.shape * abs(self.spacing) + origin
        xfmpts = self.xfm.outputs[0].points.to_array()
        
        for axis_name, axis_number in self._axis_names.items():
            ipw3d = getattr(self, 'ipw_3d_%s' % axis_name)
            slab = getattr(self, 'slab_%s' % axis_name)
            outline = getattr(self, 'outline_%s' % axis_name)

            gap = abs(self.spacing[axis_number]) / 2
            pos = ipw3d.ipw.slice_position
            lim = zip(origin, limit)
            lim[axis_number] = (pos-gap, pos+gap)
            slab.filter.extent = reduce(lambda x, y:x+y, lim)

            mask = np.unique(slab.outputs[0].polys.data).astype(int)
            idx = dict(x=[1, 2], y=[2, 0], z=[0, 1])[axis_name]
            pts = xfmpts[mask][:,idx] - origin[idx]
            pts = np.hstack([pts, [1,-1,1][axis_number]*np.ones((len(mask), 1))])
            outline.data.points = pts

        self.disable_render = False

    @on_trait_change('disable_render')
    def _render_enable(self):
        for scene in (self.scene3d, self.scene_x, self.scene_y, self.scene_z):
            scene.scene.disable_render = self.disable_render
    
    @on_trait_change("brightness,contrast")
    def update_brightness(self):
        self.epi_src.scalar_data = (self.epi*self.contrast)+self.brightness
    
    @on_trait_change("opacity")
    def update_opacity(self):
        self.surf.actor.property.opacity = self.opacity
    
    @on_trait_change("colormap, fliplut")
    def update_colormap(self):
        self.ipw_3d_z.parent.scalar_lut_manager.lut_mode = self.colormap
        self.ipw_x.parent.scalar_lut_manager.lut_mode = self.colormap
        self.ipw_y.parent.scalar_lut_manager.lut_mode = self.colormap
        self.ipw_z.parent.scalar_lut_manager.lut_mode = self.colormap
        self.ipw_3d_z.parent.scalar_lut_manager.reverse_lut = self.fliplut
        self.ipw_x.parent.scalar_lut_manager.reverse_lut = self.fliplut
        self.ipw_y.parent.scalar_lut_manager.reverse_lut = self.fliplut
        self.ipw_z.parent.scalar_lut_manager.reverse_lut = self.fliplut
    
    @on_trait_change("flip_ud")
    def update_flipud(self):
        #self.epi_src.scalar_data = self.epi_src.scalar_data[:,:,::-1]
        flip = np.eye(4)
        flip[2,2] = -1
        mat = self.xfm.transform.matrix.to_array()
        self.set_xfm(np.dot(mat, flip))
    
    @on_trait_change("flip_lr")
    def update_fliplr(self):
        #self.epi_src.scalar_data = self.epi_src.scalar_data[::-1]
        flip = np.eye(4)
        flip[0,0] = -1
        mat = self.xfm.transform.matrix.to_array()
        self.set_xfm(np.dot(mat, flip))
    
    @on_trait_change("flip_fb")
    def update_flipfb(self):
        #self.epi_src.scalar_data = self.epi_src.scalar_data[:,::-1]
        flip = np.eye(4)
        flip[1,1] = -1
        mat = self.xfm.transform.matrix.to_array()
        self.set_xfm(np.dot(mat, flip))
    
    @on_trait_change("ptcolor")
    def update_ptcolor(self):
        self.disable_render = True
        for a in self._axis_names.keys():
            slab = getattr(self, "slab_%s"%a)
            outline = getattr(self, "outline_%s"%a)
            color = tuple(map(lambda x:x/255., self.ptcolor.asTuple()))
            slab.children[0].children[0].actor.property.color = color
            outline.children[0].children[0].actor.property.color = color
        self.disable_render = False
    
    def get_xfm(self, xfmtype="magnet"):
        if xfmtype in ["anat->epicoord", "coord"]:
            ibase = np.linalg.inv(self.base)
            xfm = self.xfm.transform.matrix.to_array()
            return np.dot(ibase, xfm)
        elif xfmtype in ["anat->epibase", "base"]:
            return self.xfm.transform.matrix.to_array()
        elif xfmtype in ['anat->magnet', "magnet"]:
            ibase = np.linalg.inv(self.base)
            xfm = self.xfm.transform.matrix.to_array()
            return np.dot(self.affine, np.dot(ibase, xfm))
    
    def set_xfm(self, matrix, xfmtype='magnet'):
        assert xfmtype in "magnet coord base".split(), "Unknown transform type"
        if xfmtype == "coord":
            matrix = np.dot(self.base, matrix)
        elif xfmtype == "magnet":
            iaff = np.linalg.inv(self.affine)
            matrix = np.dot(self.base, np.dot(iaff, matrix))

        self.xfm.transform.set_matrix(matrix.ravel())
        self.xfm.widget.set_transform(self.xfm.transform)
        self.xfm.update_pipeline()
        self.update_slab()

    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    view = View(HGroup(
                  Group(
                       Item('scene_y',
                            editor=SceneEditor(scene_class=FlatScene)),
                       Item('scene_z',
                            editor=SceneEditor(scene_class=FlatScene)),
                       show_labels=False,
                  ),
                  Group(
                       Item('scene_x',
                            editor=SceneEditor(scene_class=FlatScene)),
                       Item('scene3d',
                            editor=SceneEditor(scene_class=ThreeDScene)),
                       show_labels=False,
                  ),
                  Group("brightness", "contrast", "_", "opacity", "_",
                        Item('colormap',
                            editor=ImageEnumEditor(values=lut_manager.lut_mode_list(),
                                              cols=6,
                                              path=lut_manager.lut_image_dir)),
                        "fliplut",
                        "_", "flip_ud", "flip_lr", "flip_fb", 
                        "_", Item('ptcolor', editor=ColorEditor())

                  )
                ), 
                resizable=True,
                title='Aligner'
            )

def align(subject, xfmname, epi=None, xfm=None):
    import db
    data = db.surfs.getXfm(subject, xfmname, xfmtype='magnet')
    if data is None:
        data = db.surfs.getXfm(subject, xfmname, xfmtype='coord')
        if data is not None:
            dbxfm, epi = data
        assert epi is not None, "Unknown transform"
        data = db.surfs.getVTK(subject, 'fiducial')
        assert data is not None, "Cannot find subject"
        m = Align(data[0], data[1], epi, xfm=xfm)
        m.configure_traits()
    else:
        dbxfm, epi = data
        data = db.surfs.getVTK(subject, 'fiducial')
        assert data is not None, "Cannot find subject"
        m = Align(data[0], data[1], epi, xfm=dbxfm if xfm is None else xfm)
        m.configure_traits()
    
    magnet = m.get_xfm("magnet")
    shortcut = m.get_xfm("coord")
    epi = os.path.abspath(epi)
    resp = raw_input("Save? (Y/N) ")
    if resp.lower().strip() in ["y", "yes"]:
        print "Saving..."
        db.surfs.loadXfm(subject, xfmname, magnet, xfmtype='magnet', epifile=epi, override=True)
        db.surfs.loadXfm(subject, xfmname, shortcut, xfmtype='coord', epifile=epi, override=True)
        print "Complete!"
    else:
        print "Cancelled... %s"%resp
    
    return magnet

################################################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Align a fiducial surface to an epi image")
    parser.add_argument("--epi", type=str,
        help="Epi image to align to (in nifti format). Not required if using the database")
    
    group = parser.add_mutually_exclusive_group(required=True)
    
    #these next two arguments must be mutually exclusive, whether we draw from database or not
    group.add_argument("--subject", "-S", dest="subject", type=str, 
        help="Subject name (draws from database)")
    group.add_argument("--fiducials", "-F", nargs=2, 
        help="Pair of fiducial VTK's. Mutually exclusive with --subject!")

    #following only applies without the database
    parser.add_argument("--transform", "-T", dest="transform", type=str,
        help="Initial transform without database. Must be in magnet space.") #optional
    parser.add_argument("--out", "-o", dest="out", type=str,
        help="Output file without database. Transform will be in magnet space") #mandatory

    #following only applies with the database
    parser.add_argument("--name", type=str,
        help="Transform name within the database")

    args = parser.parse_args()

    if args.subject is not None:
        assert args.name is not None, "Please provide the transform name for the database!"
        if args.fiducials is not None:
            print "Fiducials ignored -- drawing from database"
        xfm = None
        if args.transform is not None:
            xfm = np.loadtxt(args.transform)
        xfm = align(args.subject, args.name, epi=args.epi, xfm=xfm)
        if args.out is not None:
            np.savetxt(args.out, xfm, fmt="%.6f")
    else:
        assert args.fiducials is not None, "Please provide surfaces to align!"
        assert args.out is not None, "Please provide an output file!"
        assert args.epi is not None, "Please provide an epi file to align to!"
        import vtkutils
        pts, polys, norms = vtkutils .read(args.fiducials)
        xfm = None
        if args.transform is not None:
            xfm = np.loadtxt(args.transform)
        m = Align(pts, polys, args.epi, xfm=xfm)
        m.configure_traits()
        xfm = m.get_xfm("magnet")
        np.savetxt(args.out, xfm, fmt="%.6f")

    
    #/auto/k7/james/docdb/13611980401580143636.nii