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

import multiprocessing as mp
import numpy as np
from scipy.interpolate import interp1d, Rbf

class Mixer(HasTraits):
    points = Any
    polys = Array(shape=(None, 3))

    mix = Range(0., 1., value=1)
    figure = Instance(MlabSceneModel, ())
    data_src = Instance(Source)
    surf = Instance(Module)

    colormap = Enum(*lut_manager.lut_mode_list())
    fliplut = Bool

    def _data_src_default(self):
        pts = self.points(1)
        return mlab.pipeline.triangular_mesh_source(
            pts[:,0], pts[:,1], pts[:,2],
            self.polys, figure=self.figure.mayavi_scene)

    def _surf_default(self):
        n = mlab.pipeline.poly_data_normals(self.data_src, figure=self.figure.mayavi_scene)
        return mlab.pipeline.surface(n, figure=self.figure.mayavi_scene)

    @on_trait_change("figure.activated")
    def _start(self):
        self.figure.scene.background = (0,0,0)
        self.colormap = "RdBu"
        self.fliplut = True
        self.data_src
        self.surf
        self.figure.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        self.figure.scene.render_window.stereo_type = "anaglyph"

    #@on_trait_change('mix')
    def _mix_changed(self):
        self.data_src.data.points.from_array(self.points(self.mix))
        self.figure.renderer.reset_camera_clipping_range()
        self.figure.render()
        #def func():
        #    self.data_src.data.points = self.points(self.mix)
        #    GUI.invoke_later(self.data_src.data.update)
        #threading.Thread(target=func).start()
    
    @on_trait_change("colormap, fliplut")
    def _update_colors(self):
        self.surf.parent.scalar_lut_manager.lut_mode = self.colormap
        self.surf.parent.scalar_lut_manager.reverse_lut = self.fliplut
    
    def set(self, data):
        self.data_src.mlab_source.scalars = data

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
            "fliplut"),
    show_labels=False),
    resizable=True, title="Mixer")

def show(data, subject, xfm, types=('inflated',), hemisphere="both"):
    '''View epi data, transformed into the space given by xfm. 
    Types indicates which surfaces to add to the interpolater. Always includes fiducial and flat'''
    import db
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

    wpts = np.append(pts[0], np.ones((len(pts[0]),1)), axis=-1).T
    coords = np.dot(xfm, wpts)[:3].T.round().astype(int)
    scalars = np.array([data.T[tuple(p)] for p in coords])

    interp = interp1d(np.linspace(0,1,len(pts)), pts, axis=0)
    m = Mixer(points=interp, polys=polys)
    m.edit_traits()
    m.set(scalars)
    return m

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Display epi data on various surfaces, \
        allowing you to interpolate between the surfaces")
    parser.add_argument("epi", type=str)
    parser.add_argument("--transform", "-T", type=str)
    parser.add_argument("--surfaces", nargs="*")