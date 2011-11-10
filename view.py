try:
    from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, Any, Color, on_trait_change
    from traitsui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor

    from tvtk.api import tvtk
    from tvtk.pyface.scene import Scene

    from mayavi import mlab
    from mayavi.core.ui import lut_manager
    from mayavi.core.api import PipelineBase, Source, Filter, Module
    from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
except ImportError:
    from enthought.traits.api import HasTraits, Instance, Array, Bool, Dict, Any, Range, Color, on_trait_change
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

class Iron(HasTraits):
    points = Any
    polys = Array(shape=(None, 3))

    mix = Range(0., 1., value=0)
    figure = Instance(MlabSceneModel, ())
    data_src = Instance(Source)
    surf = Instance(Module)

    def _data_src_default(self):
        pts = self.points(0)
        return mlab.pipeline.triangular_mesh_source(
            pts[:,0], pts[:,1], pts[:,2],
            self.polys, figure=self.figure.mayavi_scene)

    def _surf_default(self):
        n = mlab.pipeline.poly_data_normals(self.data_src, figure=self.figure.mayavi_scene)
        return mlab.pipeline.surface(n, figure=self.figure.mayavi_scene)

    @on_trait_change("figure.activated")
    def _start(self):
        self.data_src
        self.surf
        self.figure.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        self.figure.scene.render_window.stereo_type = "anaglyph"

    #@on_trait_change('mix')
    def _mix_changed(self):
        self.data_src.data.points = self.points(self.mix)
        #def func():
        #    self.data_src.data.points = self.points(self.mix)
        #    GUI.invoke_later(self.data_src.data.update)
        #threading.Thread(target=func).start()
    
    def set(self, data):
        self.data_src.mlab_source.scalars = data

    view = View(Group(
        Item("figure", editor=SceneEditor(scene_class=MayaviScene)),
        "mix",
        show_labels=False
    ), resizable=True, title="Mixer")

def view(data, subject, xfm, types=('inflated',), hemisphere="both"):
    '''View epi data, transformed into the space given by xfm. 
    Types indicates which surfaces to add to the interpolater. Always includes fiducial and flat'''
    import db
    types = ("fiducial",) + types + ("flat",)
    pts = []
    for t in types:
        pt, polys, norm = db.flats.getVTK(subject, t, hemisphere=hemisphere)
        pts.append(pt)
    #flip the flats to be on the X-Z plane
    flatpts = np.zeros_like(pts[-1])
    flatpts[:,[0,2]] = pts[-1][:,:2]
    pts[-1] = flatpts

    if hasattr(data, "get_affine"):
        #this is a nibabel file -- it has the nifti headers intact!
        if isinstance(xfm, str):
            xfm = db.flats.getXfm(subject, xfm, xfmtype="magnet")
            assert xfm is not None, "Cannot find transform by this name!"
            xfm = np.dot(np.linalg.inv(data.get_affine()), xfm[0])
        data = data.get_data()
    else:
        xfm = db.flats.getXfm(subject, xfm, xfmtype="coord")
        assert xfm is not None, "Cannot find coord transform, please provide the nifti!"
        xfm = xfm[0]
    assert xfm.shape == (4, 4), "Not a transform matrix!"

    wpts = np.append(pts[0], np.zeros((len(pts[0]),1)), axis=-1).T
    scalars = np.array([data.T[tuple(p)] for p in np.dot(xfm, wpts)[:3].T])
    interp = interp1d(np.linspace(0,1,len(pts)), pts, axis=0)
    m = Iron(points=interp, polys=polys)
    m.edit_traits()
    m.set(scalars)