try:
    from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, Any, Color,Enum, Callable, on_trait_change
    from traitsui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor

    from tvtk.api import tvtk
    from tvtk.pyface.scene import Scene

    from mayavi import mlab
    from mayavi.core.ui import lut_manager
    from mayavi.core.api import PipelineBase, Source, Filter, Module
    from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
except ImportError:
    from enthought.traits.api import HasTraits, Instance, Array, Bool, Dict, Any, Range, Color,Enum, Callable, on_trait_change
    from enthought.traits.ui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor, Handler

    from enthought.tvtk.api import tvtk
    from enthought.tvtk.pyface.scene import Scene

    from enthought.mayavi import mlab
    from enthought.mayavi.core.ui import lut_manager
    from enthought.mayavi.core.api import PipelineBase, Source, Filter, Module
    from enthought.mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene


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

class PickPoints(view.Mixer):
    def _start_picker(self):
        self.blah = []
        def callback(widget, evt):
            self.figure.scene_editor.control.SetFocusFromKbd()
            pos = widget.GetEventPosition()
            if widget.GetShiftKey() == 1:
                self.figure.scene_editor.move = False
            else:
                self.figure.scene_editor.move = True

            if self.figure.scene_editor.mousedown:
                self.figure.picker.pointpicker.pick(
                    (pos[0], pos[1], 0), 
                    self.figure.mayavi_scene.scene.renderer)
                
        self.figure.interactor.add_observer("MouseMoveEvent", callback)
        picker = self.figure.mayavi_scene.on_mouse_pick(self._pick)
        picker.tolerance = 0.01
        
    @on_trait_change("figure.activated")
    def _start(self):
        super(PickPoints, self)._start()
        self.figure.scene_editor.move = True
        self.figure.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        self._start_picker()

    def _pick(self, picker):
        if picker.point_id != -1:
            self.data_src.mlab_source.scalars[picker.point_id] = 0
    
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