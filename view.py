try:
    from traits.api import HasTraits, Instance, Array, Bool, Dict, Range, Any, Color, on_trait_change
    from traitsui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor
    from traitsui.key_bindings import KeyBinding, KeyBindings

    from tvtk.api import tvtk
    from tvtk.pyface.scene import Scene

    from mayavi import mlab
    from mayavi.core.ui import lut_manager
    from mayavi.core.api import PipelineBase, Source, Filter, Module
    from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
except ImportError:
    from enthought.traits.api import HasTraits, Instance, Array, Bool, Dict, Any, Range, Color, on_trait_change
    from enthought.traits.ui.api import View, Item, HGroup, Group, ImageEnumEditor, ColorEditor, Handler
    from enthought.traits.ui.key_bindings import KeyBinding, KeyBindings

    from enthought.tvtk.api import tvtk
    from enthought.tvtk.pyface.scene import Scene

    from enthought.mayavi import mlab
    from enthought.mayavi.core.ui import lut_manager
    from enthought.mayavi.core.api import PipelineBase, Source, Filter, Module
    from enthought.mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene

import multiprocessing as mp
import numpy as np
from scipy.interpolate import interp1d, Rbf

class Bored(HasTraits):
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


class mpInterp(object):
    def __init__(self, data):
        cpus = mp.cpu_count()
        frames = np.linspace(0, 1, len(data))
        length = data.shape[1] / float(cpus)

        self.status = mp.Value('f', 0)
        self.runproc = []
        self.data = mp.Queue()

        self.procs = []
        for i in range(cpus):
            interp = interp1d(frames, data[:,i*length:(i+1)*length], axis=0)
            evt = mp.Event()
            def _run(status, run, data):
                while status.value >= 0:
                    run.wait()
                    data.put((i, interp(status.value)))
                    run.clear()
                    
            p = mp.Process(target=_run, args=(self.status, evt, self.data))
            p.start()
            self.procs.append(p)
            self.runproc.append(evt)
    
    def __call__(self, mix):
        self.status.value = mix
        for evt in self.runproc:
            evt.set()
        
        data = [[]]*len(self.procs)
        for _ in range(len(self.procs)):
            i, d = self.data.get()
            data[i] = d
        return np.vstack(data)
    
    def __del__(self):
        self.status.value = -1

if __name__ == "__main__":
    from utils.mri import vtk
    print "reading..."
    start_pts, tmp, norms = vtk.vtkread(['/auto/k5/nbilenko/MRI/caret/JG/Human.JG.L.Fiducial.vtk'])
    mid_pts,   tmp, norms = vtk.vtkread(['/auto/k5/nbilenko/MRI/caret/JG/Human.JG.L.Inflated.vtk'])
    #mid_pts1,   tmp = vtk.vtkread(['/auto/k5/nbilenko/MRI/caret/JG/Human.JG.L.VeryInflated.vtk'])
    #mid_pts2,   tmp = vtk.vtkread(['/auto/k5/nbilenko/MRI/caret/JG/Human.JG.L.Ellipsoidal.vtk'])
    end_pts,polys, norms = vtk.vtkread(['/auto/k5/nbilenko/MRI/caret/JG/Human.JG.L.FLAT.vtk'])
    print "done"
    flat_pts = np.zeros_like(end_pts)
    flat_pts[:, 1] = -end_pts[:,0]
    flat_pts[:, 2] = end_pts[:,1]
    flat_pts *= 0.5
    stackpts = np.array([start_pts, mid_pts, flat_pts])
    '''
    import cPickle
    #cPickle.dump((stackpts, polys), open("flats.pkl", "w"), 2)
    stackpts, polys = cPickle.load(open("flats.pkl"))
    '''
    print "interpolating..."
    #interp = mpInterp(stackpts)
    #interp = Rbf(linspace(0,1,len(stackpts)), stackpts, function)
    interp = interp1d(np.linspace(0, 1, len(stackpts)), stackpts, axis=0)
    print "starting..."
    m = Bored(points=interp, polys=polys)
    m.edit_traits()
    m.set(np.load("flatten_motor.npy"))







'''
class tInterp(object):
    def __init__(self, data, n=4):
        frames = range(len(data))
        length = data.shape[1] / float(n)

        self.status = 0
        self.runproc = []
        self.data = np.empty_like(data[0])
        self.done = []
        self.ready = threading.Event()

        self.threads = []
        for i in range(n):
            interp = interp1d(frames, data[:,i*length:(i+1)*length], axis=0)
            evt = threading.Event()
            thread = threading.Thread(target=self._run, args=(self.status, evt))
            thread.start()
            self.runproc.append(evt)
            self.threads.append(thread)
    
    def _run(self, status, evt):
        while status >= 0:
            evt.wait()
            self.data[i*length:(i+1)*length] = interp(status)
            if len(self.done) + 1 == len(self.threads):
                self.ready.set()
                self.done = []
            else:
                self.done.append(None)
            evt.clear()
    
    def __call__(self, mix):
        self.status = mix
        for evt in self.runproc:
            evt.set()
        
        self.ready.wait()
        self.ready.clear()
        return self.data
'''