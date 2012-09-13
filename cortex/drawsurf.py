import numpy as np
from matplotlib.pyplot import imread
import cairo

try:
    from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, Any, Color,Enum, Str, Callable, on_trait_change
    from traitsui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor

    from tvtk.api import tvtk
    from tvtk.pyface.scene import Scene

    from mayavi import mlab
    from mayavi.core.ui import lut_manager
    from mayavi.core.api import PipelineBase, Source, Filter, Module
    from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
    from mayavi.sources.array_source import ArraySource
except ImportError:
    from enthought.traits.api import HasTraits, Instance, Array, Bool, Dict, Any, Range, Color,Enum, Str, Callable, on_trait_change
    from enthought.traits.ui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor, Handler

    from enthought.tvtk.api import tvtk
    from enthought.tvtk.pyface.scene import Scene

    from enthought.mayavi import mlab
    from enthought.mayavi.core.ui import lut_manager
    from enthought.mayavi.core.api import PipelineBase, Source, Filter, Module
    from enthought.mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
    from enthought.mayavi.sources.array_source import ArraySource

import view

class MouseDownScene(MayaviScene):
    mousedown = Bool(value=False)
    move = Bool

#    def OnKeyDown(self, evt):
#        key = chr(evt.GetKeyCode()%256)
#        if key == "d":
#            print "toggle modes"
#            self.move = not self.move
#        return super(MouseDownScene, self).OnKeyDown(evt)

    def OnButtonDown(self, evt):
        self.mousedown = True
        if self.move:
            return super(MouseDownScene, self).OnButtonDown(evt)
    
    def OnButtonUp(self, evt):
        self.mousedown = False
        if self.move:
            return super(MouseDownScene, self).OnButtonUp(evt)

class DrawOnTex(view.Mixer):
    texpath = Str
    tex = Instance(ArraySource)

    csurf = Instance(cairo.ImageSurface, (cairo.FORMAT_ARGB32, 2048, 1024))
    cctx = Instance(cairo.Context)

    def _cctx_default(self):
        ctx = cairo.Context(self.csurf)
        ctx.set_line_width(5)
        return ctx

    def _tex_default(self):
        return ArraySource(scalar_data=np.zeros((2048,1024)))
    
    def _texpath_changed(self):
        self.tex.scalar_data = np.load(self.texpath)
        
    def _start_picker(self):
        self.last = None
        def callback(widget, evt):
            self.figure.scene_editor.control.SetFocusFromKbd()
            if widget.GetShiftKey() == 1:
                self.figure.scene_editor.move = False
            else:
                self.figure.scene_editor.move = True

            if self.figure.scene_editor.mousedown:
                pos = widget.GetEventPosition()
                self.figure.picker.pointpicker.pick(
                    (pos[0], pos[1], 0), 
                    self.figure.mayavi_scene.scene.renderer)
            else:
                if self.last is not None:
                    #Finished drawing, time to draw the stroke
                    self._draw()
                self.last = None
                
        self.figure.interactor.add_observer("MouseMoveEvent", callback)
        picker = self.figure.mayavi_scene.on_mouse_pick(self._pick)
        picker.tolerance = 0.005
        
    @on_trait_change("figure.activated")
    def _start(self):
        super(DrawOnTex, self)._start()
        self.surf
        self.figure.scene_editor.move = True
        self._start_picker()

    def _pick(self, picker):
        if picker.point_id != -1:
            u, v = self.data_src.data.point_data.t_coords[picker.point_id]
            if self.last is not None:
                self.cctx.line_to(u*2048,v*1024)
            else:
                self.cctx.move_to(u*2048,v*1024)
            self.last = u, v
    
    def _draw(self):
        print "Finalizing stroke"
        self.cctx.stroke()
        data = np.frombuffer(self.csurf.get_data(), np.uint8).reshape(1024, 2048, 4)
        self.tex.scalar_data = (data[...,-1] == 255).astype(int).T
        self.surf.update_pipeline()
    
    view = View(
        HGroup(
            Group(
                Item("figure", editor=SceneEditor(scene_class=MouseDownScene)),
                "mix",
                show_labels=False),
            Group(
                Item('colormap',
                     editor=ImageEnumEditor(values=lut_manager.lut_mode_list(),
                     cols=6, path=lut_manager.lut_image_dir)),
                "fliplut"),
        show_labels=False),
        resizable=True, title="Mixer")

def show(subject, types=('inflated',), hemisphere="both"):
    '''View epi data, transformed into the space given by xfm. 
    Types indicates which surfaces to add to the interpolater. Always includes fiducial and flat'''
    interp, polys = view._get_surf_interp(subject, xfm, types, hemisphere)
    m = PickPoints(points=interp, polys=polys)
    m.edit_traits()
    return m

if __name__ == "__main__":
    import cPickle
    from scipy.interpolate import interp1d
    x, y, polys = cPickle.load(open("../../Dropbox/mesh.pkl"))
    interp = interp1d(x, y)
    m = view.Mixer(points=interp, polys=polys)
    m.edit_traits()