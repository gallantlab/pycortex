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

def read(vtks, offset=None):
    pts, polys, normals, llen = [], [], [], 0
    for i, fname in enumerate(vtks):
        vfr = get(fname)
        poly = vfr.outputs[0].polys.data.to_array().reshape(-1,4)[:,1:]
        polys.append(poly+llen)
        p = vfr.outputs[0].points.to_array()
        if offset is not None:
            p += (-1,1)[i%2]*offset
        pts.append(p)
        llen = len(p)
        if vfr.outputs[0].point_data.normals is not None:
            normals.append(vfr.outputs[0].point_data.normals.to_array())
    
    if offset is None and len(pts) > 1:
        #Special case for loading overlapping brain halves with no offset defined
        pts[0][:,0] -= pts[0].max(0)[0]
        pts[1][:,0] -= pts[1].min(0)[0]
    
    if len(normals) > 0:
        normals = np.vstack(normals)
    return np.vstack(pts), np.vstack(polys), normals

def show(vtks, offset=None):
    fig = mlab.figure()
    fig.scene.background = (0,0,0)
    fig.scene.interactor.interactor_style = \
                                tvtk.InteractorStyleTerrain()
    pts = []
    fig.scene.disable_render = True
    for i, vtk in enumerate(vtks):
        vfr = get(vtk)
        p = vfr.outputs[0].points.to_array()
        if offset is not None:
            p += (-1,1)[i%2]*offset
        vfr.outputs[0].points = p
        fig.parent.add_source(vfr)
        mlab.pipeline.surface(vfr, figure=fig)
        pts.append(vfr.outputs[0])
    
    if offset is None and len(pts) > 1:
        p = pts[0].points.to_array()
        p[:,0] -= p.max(0)[0]
        pts[0].points = p
        
        p = pts[1].points.to_array()
        p[:,0] -= p.min(0)[0]
        pts[1].points = p
    
    fig.scene.disable_render = False
    fig.render()

    return pts