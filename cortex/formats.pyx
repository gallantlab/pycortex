import os
import glob
import struct
import numpy as np
from collections import OrderedDict

cimport cython
cimport numpy as np
from libc.string cimport strtok
from libc.stdlib cimport atoi, atof

np.import_array()

def read(str globname):
    readers = OrderedDict([('npz', read_npz), ('vtk', read_vtk), ('off', read_off)])
    for ext, func in readers.items():
        try:
            return func(globname+"."+ext)
        except IOError:
            pass
    raise IOError('No such surface file')

def read_off(str filename):
    pts, polys = [], []
    with open(filename) as fp:
        assert fp.readline()[:3] == 'OFF', 'Not an OFF file'
        npts, nface, nedge = map(int, fp.readline().split())
        print(npts, nface)
        for i in range(npts):
            pts.append([float(p) for p in fp.readline().split()])

        for i in range(nface):
            polys.append([int(i) for i in fp.readline().split()][1:])

    return np.array(pts), np.array(polys)

def read_npz(str filename):
    npz = np.load(filename)
    return npz['pts'], npz['polys']

@cython.boundscheck(False)
def read_vtk(str filename):
    cdef str vtk, line
    cdef bytes svtk
    cdef char *cstr = NULL, *cvtk = NULL
    cdef np.ndarray[np.float_t, ndim=2] pts = None
    cdef np.ndarray[np.uint32_t, ndim=2] polys = None
    cdef object _, sn, dtype, nel
    cdef int i, j, n

    with open(filename) as fp:
        vtk = fp.read()
        svtk = vtk.encode('UTF-8')
        cvtk = svtk

    cstr = strtok(cvtk, "\n")
    while pts is None or polys is None and cstr is not NULL:
        line = cstr.encode('UTF-8')
        if line.startswith("POINTS"):
            _, sn, dtype = line.split()
            n = int(sn)
            i = 0
            pts = np.empty((n, 3), dtype=float)
            while i < n:
                for j in range(3):
                    cstr = strtok(NULL, " \t\n")
                    pts[i, j] = atof(cstr)
                i += 1

        elif line.startswith("POLYGONS"):
            _, sn, nel = line.split()
            n = int(sn)
            i = 0
            polys = np.empty((n, 3), dtype=np.uint32)
            while i < n:
                cstr = strtok(NULL, " \t\n")
                if atoi(cstr) != 3:
                    raise ValueError('Only triangular VTKs are supported')
                for j in range(3):
                    cstr = strtok(NULL, " \t\n")
                    polys[i, j] = atoi(cstr)
                i += 1

        cstr = strtok(NULL, "\n")

    return pts, polys

def write_vtk(bytes filename, object pts, object polys, object norms=None):
    with open(filename, "w") as fp:
        fp.write("# vtk DataFile Version 3.0\nWritten by pycortex\nASCII\nDATASET POLYDATA\n")
        fp.write("POINTS %d float\n"%len(pts))
        np.savetxt(fp, pts, fmt='%0.12g')
        fp.write("\n")

        fp.write("POLYGONS %d %d\n"%(len(polys), 4*len(polys)))
        spolys = np.hstack((3*np.ones((len(polys),1), dtype=polys.dtype), polys))
        np.savetxt(fp, spolys, fmt='%d')
        fp.write("\n")

        if norms is not None and len(norms) == len(pts):
            fp.write("NORMALS Normals float")
            np.savetxt(fp, norms, fmt='%0.12g')

def write_off(bytes filename, object pts, object polys):
    spolys = np.hstack((3*np.ones((len(polys),1), dtype=polys.dtype), polys))
    with open(filename, 'w') as fp:
        fp.write('OFF\n')
        fp.write('%d %d 0\n'%(len(pts), len(polys)))
        np.savetxt(fp, pts, fmt='%f')
        np.savetxt(fp, spolys, fmt='%d')

def write_stl(bytes filename, object pts, object polys):
    dtype = np.dtype("3f4, 9f4, H")
    data = np.zeros((len(polys),), dtype=dtype)
    data['f1'] = pts[polys].reshape(-1, 9)
    with open(filename, 'w') as fp:
        fp.write(struct.pack('80xI', len(polys)))
        fp.write(data.tostring())
