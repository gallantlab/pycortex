import numpy as np
try:
    from enthought.mayavi.sources.vtk_file_reader import VTKFileReader
except ImportError:
    from mayavi.sources.vtk_file_reader import VTKFileReader

def vtkread(vtks):
    pts, polys, normals, llen = [], [], [], 0
    for fname in vtks:
        vfr = VTKFileReader()
        vfr.initialize(fname)
        poly = vfr.outputs[0].polys.data.to_array().reshape(-1,4)[:,1:]
        polys.append(poly+llen)
        pts.append(vfr.outputs[0].points.to_array())
        if vfr.outputs[0].point_data.normals is not None:
            normals.append(vfr.outputs[0].point_data.normals.to_array())
        llen = len(pts[-1])
    
    if len(normals) > 0:
        normals = np.vstack(normals)
    return np.vstack(pts), np.vstack(polys), normals