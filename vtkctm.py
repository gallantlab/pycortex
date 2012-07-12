import os
import struct
import ctypes
import json
import tempfile
import numpy as np
from scipy.spatial import cKDTree

import mritools

cwd = os.path.split(os.path.abspath(__file__))[0]

class Mesh(ctypes.Structure):
    _fields_ = [
        ("npts", ctypes.c_uint32),
        ("npolys", ctypes.c_uint32),
        ("nelem", ctypes.c_uint32),
        ("pts", ctypes.POINTER(ctypes.c_float)),
        ("polys", ctypes.POINTER(ctypes.c_uint32))]

class Hemi(ctypes.Structure):
    _fields_ = [
        ("fiducial", ctypes.POINTER(Mesh)),
        ("flat", ctypes.POINTER(Mesh)),
        ("between", ctypes.POINTER(Mesh)*6),
        ("names", ctypes.c_char*1024*6),
        ("nbetween", ctypes.c_uint32),
        ("datamap", ctypes.POINTER(ctypes.c_float))
    ]

class Subject(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char*128),
        ("xfm", ctypes.c_float*16),
        ("left", Hemi),
        ("right", Hemi)]

lib = ctypes.cdll.LoadLibrary(os.path.join(cwd, "_vtkctm.so"))
lib.readVTK.restype = ctypes.POINTER(Mesh)
lib.readVTK.argtypes = [ctypes.c_char_p, ctypes.c_bool]
lib.readCTM.restype = ctypes.POINTER(Mesh)
lib.readCTM.argtypes = [ctypes.c_char_p, ctypes.c_bool]
lib.meshFree.argtypes = [ctypes.POINTER(Mesh)]

lib.newSubject.restype = ctypes.POINTER(Subject)
lib.newSubject.argtypes = [ctypes.c_char_p]
lib.hemiAddFid.argtypes = [ctypes.POINTER(Hemi), ctypes.c_char_p]
lib.hemiAddFlat.argtypes = [ctypes.POINTER(Hemi), ctypes.c_char_p]
lib.hemiAddSurf.argtypes = [ctypes.POINTER(Hemi), ctypes.c_char_p, ctypes.c_char_p]
lib.hemiAddMap.argtypes = [ctypes.POINTER(Hemi), np.ctypeslib.ndpointer(np.uint32)]
lib.saveCTM.argtypes = [ctypes.POINTER(Subject), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]

def readVTK(filename, readpoly=True):
    return _getmesh(lib.readVTK(filename, readpoly))

def readCTM(filename, readpoly=True):
    return _getmesh(lib.readCTM(filename, readpoly))

def _getmesh(mesh):
    mesh = mesh.contents;
    pts = np.array(mesh.pts[:mesh.npts*3],dtype=np.float32).reshape(-1, 3)
    polys = np.array(mesh.polys[:mesh.npolys*3], dtype=np.uint32).reshape(-1, 3)
    lib.meshFree(mesh)
    return pts, polys


class CTMfile(object):
    def __init__(self, subj, xfmname=None, shape=(31, 100, 100)):
        self.shape = shape
        self.xfmname = xfmname
        self.files = os.path.join(mritools.db.filestore, "surfaces", "{subj}_{type}_{hemi}.vtk")
        self.subj = lib.newSubject(subj)
        self.name = subj
        cont = self.subj.contents
        for h, hemi, datamap in zip(["lh", "rh"], [cont.left, cont.right], self.maps):
            fname = self.files.format(subj=self.name, type="fiducial", hemi=h)
            lib.hemiAddFid(ctypes.byref(hemi), fname)
            fname = self.files.format(subj=self.name, type="flat", hemi=h)
            lib.hemiAddFlat(ctypes.byref(hemi), fname)
            lib.hemiAddMap(ctypes.byref(hemi), datamap)

    @property
    def maps(self):
        indices = []
        mask = mritools.get_cortical_mask(self.name, self.xfmname, shape=self.shape)
        imask = mask.astype(np.uint32)
        imask[imask > 0] = np.arange(mask[mask > 0].sum())

        left, right = mritools.surfs.getCoords(self.name, self.xfmname)
        for coords in (left, right):
            idx = np.ravel_multi_index(coords.T, self.shape[::-1])
            indices.append(imask.T.ravel()[idx])
        return indices

    def addSurf(self, surf, name=None):
        cont = self.subj.contents
        for h, hemi in zip(["lh", "rh"], [cont.left, cont.right]):
            fname = self.files.format(subj=self.name, type=surf, hemi=h)
            lib.hemiAddSurf(ctypes.byref(hemi), fname, name)

    def save(self, filename):
        left = tempfile.NamedTemporaryFile()
        right = tempfile.NamedTemporaryFile()
        lib.saveCTM(self.subj, left.name, right.name, 0x203, 9)
        print "Save complete! Hunting down deleted polys"
        cont = self.subj.contents
        didx = []
        for fp, hemi in zip([left, right], [cont.left, cont.right]):
            rpts, rpolys = _getmesh(hemi.fiducial) #reference
            fpts, fpolys = _getmesh(hemi.flat) #flat
            #polygons which need to be removed
            dpolys  = set([tuple(p) for p in np.sort(rpolys, axis=1)])
            dpolys -= set([tuple(p) for p in np.sort(fpolys, axis=1)])
            dpolys = np.array(list(dpolys))

            pts, polys = readCTM(fp.name)
            polys = dict([(tuple(p), i) for i, p in enumerate(polys)])
            kdt = cKDTree(pts)
            diff, idx = kdt.query(rpts)
            #get the new point indices
            dpolys = np.sort(idx[dpolys], axis=1)
            dpolys = [polys[tuple(dpoly)] for dpoly in dpolys if tuple(dpoly) in polys]
            didx.append(np.array(sorted(dpolys), dtype=np.uint32))
            fp.seek(0)
        
        offsets = []
        with open(filename, "w") as fp:
            head = struct.pack('2I', len(didx[0]), len(didx[1]))
            fp.write(head)
            fp.write(didx[0].tostring())
            fp.write(didx[1].tostring())
            offsets.append(fp.tell())
            fp.write(left.read())
            offsets.append(fp.tell())
            fp.write(right.read())

        return offsets

    def __del__(self):
        lib.subjFree(self.subj)

def makePack(subj, xfm, types=("inflated",)):
    fname = "{subj}_{xfm}_[{types}].%s".format(subj=subj,xfm=xfm,types=','.join(types))
    ctm = CTMfile("JG", "20110909JG_nb")
    for t in types:
        ctm.addSurf(t)

    offsets = ctm.save(fname%"ctm")
    jsdat = dict(data=fname%"ctm", materials=[], offsets=offsets)
    json.dump(jsdat, open(fname%"json", "w"))

if __name__ == "__main__":
    makePack("JG", "20110909JG_nb", types=("inflated", "veryinflated"))
