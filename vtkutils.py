import numpy as np

def get(vtk):
    try:
        from enthought.mayavi.sources.vtk_file_reader import VTKFileReader
    except ImportError:
        from mayavi.sources.vtk_file_reader import VTKFileReader
    vfr = VTKFileReader()
    vfr.initialize(vtk)
    return vfr

def read(vtk):
    normals = None
    vfr = get(vtk)
    polys = vfr.reader.output.polys.data.to_array().reshape(-1,4)[:,1:]
    pts = vfr.reader.output.points.to_array()
    if vfr.reader.output.point_data.normals is not None:
        normals = vfr.reader.output.point_data.normals.to_array()
    
    return pts, polys, normals

def show(vtks):
    try:
        from enthought.mayavi import mlab
        from enthought.tvtk.api import tvtk
    except ImportError:
        from mayavi import mlab
        from tvtk.api import tvtk

    fig = mlab.figure()
    fig.scene.background = (0,0,0)
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    fig.scene.disable_render = True
    pts = []
    for i, vtk in enumerate(vtks):
        vfr = get(vtk)
        fig.parent.add_source(vfr)
        mlab.pipeline.surface(vfr, figure=fig)
        pts.append(vfr.reader.output)
    
    if len(pts) > 1:
        p = pts[0].points.to_array()
        p[:,0] -= p.max(0)[0]
        pts[0].points = p
        
        p = pts[1].points.to_array()
        p[:,0] -= p.min(0)[0]
        pts[1].points = p
    
    fig.scene.disable_render = False
    fig.render()
    return fig
