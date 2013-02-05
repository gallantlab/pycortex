import types
import nibabel
import numpy as np

from traits.api import HasTraits, List, Instance, Array, Bool, Dict, Range, Float, Enum, Color, Int, on_trait_change, Button, DelegatesTo, Any
from traitsui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.ui import lut_manager
from mayavi.core.api import PipelineBase, Source, Filter, Module
from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene

from db import options
import utils

class RotationWidget(HasTraits):
    radius = Float(value=1)
    angle = Float(value=0)
    pos = Array(value=[0,0,0])
    enabled = Bool(value=True)
    constrain = Bool(value=False)

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
            if hasattr(self.callback, "__call__") and self._btn == 1:
                self.callback( self, self.pos - self.startmove[0], 
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
            self.callback(self, pos, angle, radius)
        self.set(pos=self.pos+pos, angle=self.angle+angle, radius=radius*self.radius)
    
    def _move_center(self, obj=None, evt=None):
        self.pos = self.center.representation.world_position
    
    def _move_edge(self, obj=None, evt=None):
        c = self.center.representation.world_position
        r = self.edge.representation.world_position

        r -= c

        angle = np.arctan2(r[1], r[0])
        radius = np.sqrt(np.sum(r**2))

        self.set(angle=angle, radius=radius)
    
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
        focus = self.aligner.scene_3d.camera.focal_point
        if key == "1":
            self.aligner.scene_3d.camera.position = focus + [0, self.state[1]*500, 0]
            self.state[1] = -self.state[1]
        elif key == "3":
            self.aligner.scene_3d.camera.position = focus + [self.state[0]*500, 0, 0]
            self.state[0] = -self.state[0]
        elif key == "5":
            self.aligner.scene_3d.parallel_projection = not self.aligner.scene_3d.parallel_projection
        elif key == "7":
            self.aligner.scene_3d.camera.position = focus + [0, 1e-6, self.state[2]*500]
            self.state[2] = -self.state[2]
        elif key == "\x1a" and evt.CmdDown() and len(self.aligner._undolist) > 0:
            self.aligner._redo = self.aligner._undolist.pop()
            self.aligner.xfm.transform.set_matrix(self.aligner._undolist[-1].ravel())
            self.aligner.xfm.widget.set_transform(self.aligner.xfm.transform)
            self.aligner.xfm.update_pipeline()
            self.aligner.update_slab()
        else:
            print "Unknown key, %s"%key
            super(ThreeDScene, self).OnKeyDown(evt)
        self.aligner.scene_3d.renderer.reset_camera_clipping_range()
        self.aligner.scene_3d.render()

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

    def OnButtonDown(self, evt):
        self.handle._btn = evt.Button

        if evt.ShiftDown():
            self.handle.constrain = True
        else:
            self.handle.constrain = False
        super(FlatScene, self).OnButtonDown(evt)

################################################################################

class Axis(HasTraits):
    axis = Int
    parent = Instance('Align')

    ipw_3d = Instance(PipelineBase)
    ipw = Instance(PipelineBase)
    cursor = Instance(Module)
    surf = Instance(Module)
    outline = Instance(Module)
    planes = List
    slab = Instance(PipelineBase)
    handle = Instance(RotationWidget)
    clip = Instance(tvtk.ClipPolyData)

    scene_3d = DelegatesTo('parent')
    position = DelegatesTo('parent')
    disable_render = DelegatesTo('parent')
    xfm = DelegatesTo('parent')

    def __init__(self, **kwargs):
        super(Axis, self).__init__(**kwargs)
        spacing = self.parent.spacing
        spacing.pop(self.axis)
        spacing.append(1)
        shape = list(self.parent.epi.shape)
        shape.pop(self.axis)
        if self.axis == 1:
            shape = shape[::-1]
        shape.append(0)
        center = shape * spacing / 2. + ((np.array(shape)+1)%2) * spacing / 2.
        width = (shape * spacing)[:2]
        width = np.min(width) * 0.5

        self.scene.scene_editor.handle = self.handle
        self.scene.scene_editor.aligner = self.parent
        self.scene.scene.background = (0, 0, 0)
        mlab.view(*([(0, 0), (90, 180), (0, 0)][self.axis]),
                  focalpoint=center,
                  figure=self.scene.mayavi_scene)
        self.scene.scene.parallel_projection = True
        self.scene.scene.camera.parallel_scale = width * 1.2
        # 2D interaction: only pan and zoom
        self.scene.scene.interactor.interactor_style = tvtk.InteractorStyleImage()
        def focusfunc(vtkobj, i):
            self.scene.scene_editor.control.SetFocusFromKbd()
        self.scene.scene.interactor.add_observer("MouseMoveEvent", focusfunc)

    @on_trait_change("scene_3d.activated")
    def activate_3d(self):
        self.surf
        self.ipw_3d
        self.ipw
        self.outline

    def _planes_default(self):
        clipnorms = [0, 0, 0]
        clipnorms[self.axis] = 1
        top = tvtk.Planes(normals=[list(clipnorms)], points=[list(clipnorms)])
        clipnorms[self.axis] = -1
        bot = tvtk.Planes(normals=[list(clipnorms)], points=[list(clipnorms)])
        return [top, bot]

    def _clip_default(self):
        cliptop = tvtk.ClipPolyData(clip_function=self.planes[0], inside_out=1)
        clipbot = tvtk.ClipPolyData(clip_function=self.planes[1], inside_out=1)
        cliptop.set_input(self.parent.surf.parent.parent.filter.output)
        clipbot.set_input(cliptop.output)
        clipbot.update()
        return clipbot

    def _slab_default(self):
        return mlab.pipeline.add_dataset(self.clip.output, figure=self.scene.mayavi_scene)

    def _surf_default(self):
        slab = mlab.pipeline.add_dataset(self.clip.output, figure=self.scene_3d.mayavi_scene)
        surf = mlab.pipeline.surface(slab, 
            color=(1,1,1), 
            figure=self.scene_3d.mayavi_scene, 
            representation='wireframe')
        surf.actor.property.line_width = 5
        return surf

    def _ipw_3d_default(self):
        space = list(np.abs(self.parent.spacing))
        shape = list(np.array(self.parent.epi.shape) / self.parent.padshape)
        space.pop(self.axis)
        shape.pop(self.axis)
        if self.axis == 1:
            space = space[::-1]
            shape = shape[::-1]
        space.append(1)
        shape = [(0,0), (shape[0],0), (0, shape[1]), (shape[0],shape[1])]
        origin = [space[0] / 2., space[1] / 2., 0]

        self.ipw_space = (space, shape)

        ipw = mlab.pipeline.image_plane_widget(self.parent.epi_src,
            figure=self.scene_3d.mayavi_scene,
            plane_orientation='%s_axes' % 'xyz'[self.axis],
            name='Cut %s' % self.axis)
        ipw.ipw.color_map.output_format = 'rgb'
        ipw.ipw.set(texture_interpolate=0, reslice_interpolate='nearest_neighbour')
        ipw.ipw.reslice.set(output_spacing=space, output_origin=origin)
        ipw.ipw.poly_data_algorithm.output.point_data.t_coords = shape
        ipw.ipw.interaction = 0
        return ipw

    def _ipw_default(self):
        side_src = self.ipw_3d.ipw.reslice_output
        spacing = side_src.spacing

        # Add a callback on the image plane widget interaction to
        # move the others
        def move_view(obj, evt):
            # Disable rendering on all scene
            cpos = obj.GetCurrentCursorPosition()
            position = list(cpos*spacing)[:2]
            position.insert(self.axis, self.position[self.axis])
            # We need to special case y, as the view has been rotated.
            if self.axis == 1:
                position = position[::-1]
            
            self.position = position

        ipw = mlab.pipeline.image_plane_widget( side_src,
            plane_orientation='z_axes',
            figure=self.scene.mayavi_scene,
            name='Cut view %s' % self.axis,
            )
        ipw.ipw.set(left_button_action=0, texture_interpolate=0, reslice_interpolate='nearest_neighbour')
        ipw.parent.scalar_lut_manager.set(use_default_range=False, default_data_range=[-1,1], data_range=[-1,1])
        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)
        return ipw

    def _cursor_default(self):
        return mlab.points3d(*self.position, mode='axes', color=(0, 0, 0), 
            scale_factor=2*max(self.parent.epi[0].shape), figure=self.scene.mayavi_scene,
            name='Cursor view %s' % self.axis)

    def _handle_default(self):
        spacing = self.ipw_3d.ipw.reslice_output.spacing
        shape = list(self.parent.epi.shape)
        shape.pop(self.axis)
        if self.axis == 1:
            shape = shape[::-1]
        shape.append(0)
        center = shape * spacing / 2. + ((np.array(shape)+1)%2) * spacing / 2.
        width = (shape * spacing)[:2]
        width = np.min(width) * 0.5

        def handlemove(handle, pos, angle, radius):
            inv = self.xfm.transform.homogeneous_inverse
            wpos = handle.center.representation.world_position
            wpos -= center
            scale = [radius, radius]
            if self.axis == 1:
                trans = np.insert(pos[:2][::-1], self.axis, 0)
                wpos = np.insert(wpos[:2][::-1], self.axis, self.ipw_3d.ipw.slice_position)
                scale = np.insert(scale[::-1], self.axis, 1)
            else:
                trans = np.insert(pos[:2], self.axis, 0)
                wpos = np.insert(wpos[:2], self.axis, self.ipw_3d.ipw.slice_position)
                scale = np.insert(scale, self.axis, 1)

            self.xfm.transform.post_multiply()
            self.xfm.transform.translate(-wpos)
            self.xfm.transform.rotate_wxyz(np.degrees(angle), *self.ipw_3d.ipw.normal)
            self.xfm.transform.scale(scale)
            self.xfm.transform.translate(wpos)
            self.xfm.transform.translate(trans)
            self.xfm.transform.pre_multiply()

            self.xfm.widget.set_transform(self.xfm.filter.transform)
            self.xfm.update_pipeline()
            self.parent.update_slabs()

            self.parent._undolist.append(self.xfm.transform.matrix.to_array())
            np.save("/tmp/last_xfm.npy", self.get_xfm())

        return RotationWidget(self.scene.scene.mayavi_scene, handlemove, radius=width, pos=center)

    def _disable_render_changed(self):
        self.scene.scene.disable_render = self.disable_render

    def _position_changed(self):
        """ Update the position of the cursors on each side view, as well
            as the image_plane_widgets in the 3D view.
        """
        self.disable_render = True
        origin = self.parent.origin * np.sign(self.parent.spacing)
        offset = np.abs(self.parent.spacing) / 2

        space, shape = self.ipw_space
        self.ipw_3d.ipw.slice_position = self.position[self.axis] + origin[self.axis]
        self.ipw_3d.ipw.reslice.set(output_spacing=space, output_origin=[space[0] / 2., space[1] / 2., 0])
        self.ipw_3d.ipw.poly_data_algorithm.output.point_data.t_coords = shape
        
        p = list(self.position + offset)
        p.pop(self.axis)
        if self.axis == 1:
            p = p[::-1]
        p.append(0)
        self.cursor.parent.parent.data.points = [p]

        origin, spacing = self.parent.origin, self.parent.spacing
        origin = origin * np.sign(spacing) - np.abs(spacing) / 2.
        limit = self.parent.epi_src.scalar_data.shape * abs(spacing) + origin
        
        gap = abs(spacing[self.axis]) / 2
        pos = self.ipw_3d.ipw.slice_position
        pts = [0, 0, 0]
        pts[self.axis] = pos+gap
        self.planes[0].points = [list(pts)]
        pts[self.axis] = pos-gap 
        self.planes[1].points = [list(pts)]
        self.surf.update_pipeline()

        self.disable_render = False

class XAxis(Axis):
    axis = 0
    scene = DelegatesTo('parent', 'scene_x')
    def _outline_default(self):
        origin, spacing = self.parent.origin, self.parent.spacing
        translate = origin * spacing - np.abs(spacing) / 2.
        xfm = mlab.pipeline.transform_data(self.slab, figure=self.scene.mayavi_scene)
        #xfm.filter.transform.translate(-translate)
        xfm.filter.transform.rotate_y(90)
        xfm.filter.transform.rotate_z(90)
        xfm.widget.enabled = False
        return mlab.pipeline.surface(xfm, figure=self.scene.mayavi_scene, color=(1,1,1))

class YAxis(Axis):
    axis = 1
    scene = DelegatesTo('parent', 'scene_y')
    def _outline_default(self):
        origin, spacing = self.parent.origin, self.parent.spacing
        translate = origin * spacing - np.abs(spacing) / 2.
        xfm = mlab.pipeline.transform_data(self.slab, figure=self.scene.mayavi_scene)
        #xfm.filter.transform.translate(-translate)
        xfm.filter.transform.rotate_x(-90)
        xfm.widget.enabled = False
        return mlab.pipeline.surface(xfm, figure=self.scene.mayavi_scene, color=(1,1,1))

class ZAxis(Axis):
    axis = 2
    scene = DelegatesTo('parent', 'scene_z')
    def _outline_default(self):
        origin, spacing = self.parent.origin, self.parent.spacing
        translate = origin * spacing - np.abs(spacing) / 2.
        translate[2] = 1
        xfm = mlab.pipeline.transform_data(self.slab, figure=self.scene.mayavi_scene)
        #xfm.filter.transform.translate(-translate)
        xfm.widget.enabled = False
        return mlab.pipeline.surface(xfm, figure=self.scene.mayavi_scene, color=(1,1,1))

class Align(HasTraits):
    # The position of the view
    position = Array(shape=(3,))

    brightness = Range(-1., 1., value=0.)
    contrast = Range(0., 3., value=1.)
    opacity = Range(0., 1.)
    colormap = Enum(*lut_manager.lut_mode_list())
    fliplut = Bool
    outlines = List
    ptcolor = Color(value=options['ptcolor'] if 'ptcolor' in options else 'blue')
    epi_filter = Enum(None, "median", "gradient")
    filter_strength = Range(1, 20, value=3)

    scene_3d = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    epi_src = Instance(Source)
    surf_src = Instance(Source)
    xfm = Instance(Filter)
    surf = Instance(Module)

    disable_render = Bool

    flip_fb = Bool
    flip_lr = Bool
    flip_ud = Bool

    save_callback = Instance(types.FunctionType)
    save_btn = Button(label="Save Transform")

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
        self.load_epi(epi, xfm, xfmtype)
        self.pts, self.polys = pts, polys
        self._undolist = []
        self._redo = None
        super(Align, self).__init__(**traits)

    def load_epi(self, epifilename, xfm=None, xfmtype="magnet"):
        """Loads the EPI image from the specified epifilename.
        """
        nii = nibabel.load(epifilename)
        self.epi_file = nii
        epi = nii.get_data().astype(float).squeeze()
        if epi.ndim>3:
            epi = epi[:,:,:,0]
        self.affine = nii.get_affine()
        base = nii.get_header().get_base_affine()
        self.base = base
        self.origin = base[:3, -1]
        self.spacing = np.diag(base)[:3]
        if xfm is None:
            self.startxfm = np.dot(base, np.linalg.inv(self.affine))
        elif xfmtype == "magnet":
            self.startxfm = np.dot(np.dot(base, np.linalg.inv(self.affine)), xfm)
        else:
            print "using xfmtype %s"%xfmtype
            self.startxfm = xfm

        self.center = self.spacing*nii.get_shape()[:3] / 2 + self.origin

        self.padshape = 2**(np.ceil(np.log2(np.array(epi.shape))))
        self.epi_orig = epi - epi.min()
        self.epi_orig /= self.epi_orig.max()
        self.epi_orig *= 2
        self.epi_orig -= 1
        self.epi = self.epi_orig.copy()

    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _position_default(self):
        return np.abs(self.origin) + ((np.array(self.epi.shape)+1)%2) * np.abs(self.spacing) / 2

    def _epi_src_default(self):
        sf = mlab.pipeline.scalar_field(self.epi,
                            figure=self.scene_3d.mayavi_scene,
                            name='EPI')
        sf.origin = self.origin
        sf.spacing = self.spacing
        return sf
    
    def _surf_src_default(self):
        return mlab.pipeline.triangular_mesh_source(
                            self.pts[:,0], self.pts[:,1], self.pts[:,2], self.polys,
                            figure=self.scene_3d.mayavi_scene,
                            name='Cortex')
    
    def _surf_default(self):
        smooth = mlab.pipeline.poly_data_normals(self.xfm, figure=self.scene_3d.mayavi_scene)
        smooth.filter.splitting = False
        surf = mlab.pipeline.surface(smooth, figure=self.scene_3d.mayavi_scene)
        surf.actor.mapper.scalar_visibility = 0
        return surf

    def _xfm_default(self):
        xfm = mlab.pipeline.transform_data(self.surf_src, figure=self.scene_3d.mayavi_scene)
        def savexfm(info, evt):
            self._undolist.append(xfm.transform.matrix.to_array())
            np.save("/tmp/last_xfm.npy", self.get_xfm())

        xfm.widget.add_observer("EndInteractionEvent", savexfm)
        #xfm.widget.add_observer("EndInteractionEvent", self.update_slab)
        xfm.transform.set_matrix(self.startxfm.ravel())
        xfm.widget.set_transform(xfm.transform)
        return xfm
    
    #---------------------------------------------------------------------------
    # Scene activation callbacks
    #---------------------------------------------------------------------------
    @on_trait_change('scene_3d.activated')
    def display_scene_3d(self):
        self.scene_3d.mlab.view(40, 50)
        self.scene_3d.scene.renderer.use_depth_peeling = True
        self.scene_3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene_3d.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        self.scene_3d.scene_editor.aligner = self

        self.opacity = 0.1
        self.xfm.widget.enabled = False
        self.colormap = options['colormap'] if 'colormap' in options else 'gray'

    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        self.x_axis = XAxis(parent=self)

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        self.y_axis = YAxis(parent=self)

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        self.z_axis = ZAxis(parent=self)

    #---------------------------------------------------------------------------
    # Traits callback
    #---------------------------------------------------------------------------

    def _save_btn_changed(self):
        if self.save_callback is not None:
            self.save_callback(self)

    @on_trait_change("colormap, fliplut")
    def update_colormap(self):
        for ax in [self.x_axis, self.y_axis, self.z_axis]:
            if ax.ipw_3d and ax.ipw:
                ax.ipw_3d.parent.scalar_lut_manager.set(lut_mode=self.colormap, reverse_lut=self.fliplut)
                ax.ipw.parent.scalar_lut_manager.set(lut_mode=self.colormap, reverse_lut=self.fliplut)

    @on_trait_change('disable_render')
    def _render_enable(self):
        self.scene_3d.scene.disable_render = self.disable_render
    
    @on_trait_change("brightness,contrast")
    def update_brightness(self):
        self.epi_src.scalar_data = (self.epi*self.contrast)+self.brightness
    
    @on_trait_change("opacity")
    def update_opacity(self):
        self.surf.actor.property.opacity = self.opacity
    
    @on_trait_change("flip_ud")
    def update_flipud(self):
        #self.epi_src.scalar_data = self.epi_src.scalar_data[:,:,::-1]
        flip = np.eye(4)
        flip[2,2] = -1
        mat = self.xfm.transform.matrix.to_array()
        self.set_xfm(np.dot(mat, flip), "base")
    
    @on_trait_change("flip_lr")
    def update_fliplr(self):
        #self.epi_src.scalar_data = self.epi_src.scalar_data[::-1]
        flip = np.eye(4)
        flip[0,0] = -1
        mat = self.xfm.transform.matrix.to_array()
        self.set_xfm(np.dot(mat, flip), "base")
    
    @on_trait_change("flip_fb")
    def update_flipfb(self):
        #self.epi_src.scalar_data = self.epi_src.scalar_data[:,::-1]
        flip = np.eye(4)
        flip[1,1] = -1
        mat = self.xfm.transform.matrix.to_array()
        self.set_xfm(np.dot(mat, flip), "base")
    
    @on_trait_change("epi_filter, filter_strength")
    def update_epifilter(self):
        if self.epi_filter is None:
            self.epi = self.epi_orig.copy()
        elif self.epi_filter == "median":
            fstr = np.floor(self.filter_strength / 2)*2+1
            self.epi = utils.detrend_volume_median(self.epi_orig.T, fstr).T
        elif self.epi_filter == "gradient":
            self.epi = utils.detrend_volume_gradient(self.epi_orig.T, self.filter_strength).T
        
        self.update_brightness()
    
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
                       Item('scene_3d',
                            editor=SceneEditor(scene_class=ThreeDScene)),
                       show_labels=False,
                  ),
                  Group(Item("save_btn", show_label=False, visible_when="save_callback is not None"),
                        "brightness", "contrast", "epi_filter", 
                        Item('filter_strength', visible_when="epi_filter is not None"),
                        "_", "opacity", "_",
                        Item('colormap',
                            editor=ImageEnumEditor(values=lut_manager.lut_mode_list(),
                                              cols=6,
                                              path=lut_manager.lut_image_dir)),
                        "fliplut",
                        "_", "flip_ud", "flip_lr", "flip_fb", 
                        "_", Item('ptcolor', editor=ColorEditor()),

                  )
                ), 
                resizable=True,
                title='Aligner'
            )

def get_aligner(subject, xfmname, epi=None, xfm=None, xfmtype="magnet"):
    import db
    data = db.surfs.getXfm(subject, xfmname, xfmtype='magnet')
    if data is None:
        data = db.surfs.getXfm(subject, xfmname, xfmtype='coord')
        if data is not None:
            dbxfm, epi = data
        else:
            dbxfm = None
        assert epi is not None, "Unknown transform"
    else:
        dbxfm, epi = data

    data = db.surfs.getVTK(subject, 'fiducial', merge=True)
    return Align(data[0], data[1], epi, xfm=dbxfm if xfm is None else xfm, xfmtype=xfmtype)
