import numpy as np
try:
    from enthought.mayavi.sources.vtk_file_reader import VTKFileReader
    from enthought.mayavi import mlab
    from enthought.tvtk.api import tvtk
except ImportError:
    from mayavi.sources.vtk_file_reader import VTKFileReader
    from mayavi import mlab
    from tvtk.api import tvtk

def get(vtk):
    vfr = VTKFileReader()
    vfr.initialize(vtk)
    return vfr

def read(vtks, offset=(0,0,0)):
    pts, polys, normals, llen = [], [], [], 0
    for i, fname in enumerate(vtks):
        vfr = get(fname)
        poly = vfr.outputs[0].polys.data.to_array().reshape(-1,4)[:,1:]
        polys.append(poly+llen)
        p = vfr.outputs[0].points.to_array()
        pts.append(p + (-1,1)[i%2]*np.array(offset))
        llen = len(p)
        if vfr.outputs[0].point_data.normals is not None:
            normals.append(vfr.outputs[0].point_data.normals.to_array())
    
    if len(normals) > 0:
        normals = np.vstack(normals)
    return np.vstack(pts), np.vstack(polys), normals

def show(vtks, offset=(0,0,0)):
    fig = mlab.figure()
    fig.scene.background = (0,0,0)
    fig.scene.interactor.interactor_style = \
                                tvtk.InteractorStyleTerrain()
    for i, vtk in enumerate(vtks):
        vfr = get(vtk)
        vfr.outputs[0].points = vfr.outputs[0].points.to_array() + (-1,1)[i%2]*np.array(offset)
        fig.parent.add_source(vfr)
        mlab.pipeline.surface(vfr, figure=fig)