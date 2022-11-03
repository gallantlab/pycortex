import os
import sys
import warnings
import glob
import struct
import numpy as np
from collections import OrderedDict

cimport cython
cimport numpy as np
from libc.string cimport strtok
from libc.stdlib cimport atoi, atof

np.import_array()

PY3 = sys.version_info[0] > 3


def read(globname):
    readers = OrderedDict([('gii', read_gii), ('npz', read_npz), ('vtk', read_vtk), ('off', read_off), ('stl', read_stl)])
    for ext, func in readers.items():
        try:
            return func(globname+"."+ext)
        except IOError:
            pass
    raise IOError('No such surface file')

def read_off(filename):
    pts, polys = [], []
    with open(filename) as fp:
        assert fp.readline()[:3] == 'OFF', 'Not an OFF file'
        #npts, nface, nedge = map(int, fp.readline().split())
        npts, nface, nedge = [int(x) for x in fp.readline().split()] # python3 compatible
        print(npts, nface)
        for i in range(npts):
            pts.append([float(p) for p in fp.readline().split()])

        for i in range(nface):
            polys.append([int(i) for i in fp.readline().split()][1:])

    return np.array(pts), np.array(polys)

def read_npz(filename):
    npz = np.load(filename)
    pts, polys = npz['pts'], npz['polys']
    npz.close()
    return pts, polys

def read_gii(filename):
    from nibabel import load
    with warnings.catch_warnings():
        # Note that cython < 0.28 will fail to compile in python 2 since
        # ResourceWarning does not exist in python 2.
        if PY3:  
            warnings.simplefilter('ignore', ResourceWarning)
        gii = load(filename)
    pts = gii.get_arrays_from_intent('pointset')[0].data
    polys = gii.get_arrays_from_intent('triangle')[0].data
    return pts, polys

@cython.boundscheck(False)
def read_stl(filename):
    cdef int i, j

    dtype = np.dtype("3f4, (3,3)f4, H")
    with open(filename, 'r') as fp:
        header = fp.read(80)
        if header[:5] == "solid":
            raise TypeError("Cannot read ASCII STL files")
        npolys, = struct.unpack('I', fp.read(4))
        data = np.fromstring(fp.read(), dtype=dtype)
        if npolys != len(data):
            raise ValueError('File invalid')

    idx = dict()
    polys = np.empty((npolys,3), dtype=np.uint32)
    points = []
    for i, pts in enumerate(data['f1']):
        for j, pt in enumerate(pts):
            if tuple(pt) not in idx:
                idx[tuple(pt)] = len(idx)
                points.append(tuple(pt))
            polys[i, j] = idx[tuple(pt)]

    return np.array(points), polys


def read_obj(filename, norm=False, uv=False):
    pts, polys = [], []
    n, t = None, None
    if norm:
        n = []
    if uv:
        t = []
    with open(filename) as fp:
        for line in fp:
            if line.startswith('v '):
                pts.append([float(s) for s in line[2:].split(" ")])
            elif norm and line.startswith('vn '):
                n.append([float(s) for s in line[3:].split(" ")])
            elif uv and line.startswith('vt '):
                t.append([float(s) for s in line[3:].split(" ")])
            elif line.startswith('f '):
                polys.append([int(l.split("/")[0])-1 for l in line[2:].split(" ")])

    if not norm and not uv:
        return np.array(pts), np.array(polys)
    else:
        return np.array(pts), np.array(polys), n, t


@cython.boundscheck(False)
def read_vtk(filename):
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
        line = cstr.decode('UTF-8')
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

def write_vtk(filename, object pts, object polys, object norms=None):
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

def write_off(filename, object pts, object polys):
    spolys = np.hstack((3*np.ones((len(polys),1), dtype=polys.dtype), polys))
    with open(filename, 'w') as fp:
        fp.write('OFF\n')
        fp.write('%d %d 0\n'%(len(pts), len(polys)))
        np.savetxt(fp, pts, fmt='%f')
        np.savetxt(fp, spolys, fmt='%d')

def write_stl(filename, object pts, object polys):
    dtype = np.dtype("3f4, 9f4, H")
    data = np.zeros((len(polys),), dtype=dtype)
    data['f1'] = pts[polys].reshape(-1, 9)
    with open(filename, 'wb') as fp:
        fp.write(struct.pack('80xI', len(polys)))
        fp.write(data.tostring())



def write_gii(filename, object pts, object polys):
    import nibabel
    from nibabel import gifti
    pts_darray = gifti.GiftiDataArray(pts.astype(np.float32), "pointset")
    polys_darray = gifti.GiftiDataArray(polys, "triangle")
    gii = gifti.GiftiImage(darrays=[pts_darray, polys_darray])
    nibabel.save(gii, filename)


def write_obj(filename, object pts, object polys, object colors=None):
    with open(filename, 'w') as fp:
        fp.write("o Object\n")
        if colors is not None:
            for pt, c in zip(pts, colors):
                fp.write("v %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f\n"%(pt[0], pt[1], pt[2], c[0], c[1], c[2]))
        else:
            for pt in pts:
                fp.write("v %0.6f %0.6f %0.6f\n"%tuple(pt))
        fp.write("s off\n")
        for f in polys:
            fp.write("f %d %d %d\n"%tuple((f+1)))
