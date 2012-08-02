import os
import sys
import struct
import ctypes
import json
import tempfile
import numpy as np
from scipy.spatial import cKDTree

from utils import get_cortical_mask, get_roipack
from db import surfs, filestore

cwd = os.path.split(os.path.abspath(__file__))[0]

class Mesh(ctypes.Structure):
    _fields_ = [
        ("npts", ctypes.c_uint32),
        ("npolys", ctypes.c_uint32),
        ("nelem", ctypes.c_uint32),
        ("pts", ctypes.POINTER(ctypes.c_float)),
        ("polys", ctypes.POINTER(ctypes.c_uint32))]

class MinMax(ctypes.Structure):
    _fields_ = [
        ("min", ctypes.c_float*3),
        ("max", ctypes.c_float*3),
    ]

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
lib.minmaxFree.argtypes = [ctypes.POINTER(MinMax)]

lib.newSubject.restype = ctypes.POINTER(Subject)
lib.newSubject.argtypes = [ctypes.c_char_p]
lib.hemiAddFid.argtypes = [ctypes.POINTER(Hemi), ctypes.c_char_p]
lib.hemiAddFlat.argtypes = [ctypes.POINTER(Hemi), ctypes.c_char_p]
lib.hemiAddSurf.argtypes = [ctypes.POINTER(Hemi), ctypes.c_char_p, ctypes.c_char_p]
lib.hemiAddMap.argtypes = [ctypes.POINTER(Hemi), np.ctypeslib.ndpointer(np.uint32)]

lib.saveCTM.restype = ctypes.POINTER(MinMax)
lib.saveCTM.argtypes = [ctypes.POINTER(Subject), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]

def readVTK(filename, readpoly=True):
    return _getmesh(lib.readVTK(filename, readpoly));

def readCTM(filename, readpoly=True):
    return _getmesh(lib.readCTM(filename, readpoly))

def _getmesh(mesh, free=True):
    mesh = mesh.contents;
    pts = np.array(mesh.pts[:mesh.npts*3],dtype=np.float32).reshape(-1, 3)
    polys = np.array(mesh.polys[:mesh.npolys*3], dtype=np.uint32).reshape(-1, 3)
    if free:
        lib.meshFree(mesh);
    return pts, polys

compformats = dict(
    raw=0x201,
    mg1=0x202,
    mg2=0x203)

class CTMfile(object):
    def __init__(self, subj, xfmname=None, **kwargs):
        self.name = subj
        self.xfmname = xfmname
        self.files = os.path.join(filestore, "surfaces", "{subj}_{type}_{hemi}.vtk")
        self.auxdat = kwargs

    def __enter__(self):
        self.subj = lib.newSubject(self.name)
        cont = self.subj.contents
        for h, hemi, datamap in zip(["lh", "rh"], [cont.left, cont.right], self.maps):
            fname = self.files.format(subj=self.name, type="fiducial", hemi=h)
            lib.hemiAddFid(ctypes.byref(hemi), fname)
            fname = self.files.format(subj=self.name, type="flat", hemi=h)
            lib.hemiAddFlat(ctypes.byref(hemi), fname)
            lib.hemiAddMap(ctypes.byref(hemi), datamap)
        return self

    def __exit__(self, type, value, traceback):
        lib.subjFree(self.subj)

    @property
    def maps(self):
        indices = []
        mask = get_cortical_mask(self.name, self.xfmname)
        imask = mask.astype(np.uint32)
        imask[imask > 0] = np.arange(mask[mask > 0].sum())

        left, right = surfs.getCoords(self.name, self.xfmname)
        for coords in (left, right):
            idx = np.ravel_multi_index(coords.T, mask.shape[::-1])
            indices.append(imask.T.ravel()[idx])
        return indices

    def addSurf(self, surf, name=None):
        cont = self.subj.contents
        for h, hemi in zip(["lh", "rh"], [cont.left, cont.right]):
            fname = self.files.format(subj=self.name, type=surf, hemi=h)
            lib.hemiAddSurf(ctypes.byref(hemi), fname, name)

    def save(self, filename, compmeth='mg2', complevel=9):
        left = tempfile.NamedTemporaryFile()
        right = tempfile.NamedTemporaryFile()
        minmax = lib.saveCTM(self.subj, left.name, right.name, compformats[compmeth], complevel)
        flatlims = [tuple(minmax.contents.min), tuple(minmax.contents.max)]
        lib.minmaxFree(minmax)
        print "Save complete! Hunting down deleted polys"

        cont = self.subj.contents
        didx = []
        ptidx = {}
        for fp, hemi, name in zip([left, right], [cont.left, cont.right], ["left", "right"]):
            rpts, rpolys = _getmesh(hemi.fiducial, False) #reference
            fpts, fpolys = _getmesh(hemi.flat, False) #flat
            #polygons which need to be removed
            dpolys  = set([tuple(p) for p in np.sort(rpolys, axis=1)])
            dpolys -= set([tuple(p) for p in np.sort(fpolys, axis=1)])
            dpolys = np.array(list(dpolys))

            pts, polys = readCTM(fp.name)
            polys = dict([(tuple(p), i) for i, p in enumerate(np.sort(polys, axis=1))])
            kdt = cKDTree(pts)
            diff, idx = kdt.query(rpts)
            #get the new point indices
            dpolys = np.sort(idx[dpolys], axis=1)
            dpolys = [polys[tuple(dpoly)] for dpoly in dpolys if tuple(dpoly) in polys]
            didx.append(np.array(sorted(dpolys), dtype=np.uint32))
            fp.seek(0)
            ptidx[name] = idx, len(rpts)
        
        offsets = []
        path, fname = os.path.split(filename)
        fname, ext = os.path.splitext(fname)

        with open(filename, "w") as fp:
            head = struct.pack('2I', len(didx[0]), len(didx[1]))
            fp.write(head)
            fp.write(didx[0].tostring())
            fp.write(didx[1].tostring())
            offsets.append(fp.tell())
            fp.write(left.read())
            offsets.append(fp.tell())
            fp.write(right.read())

        auxdat = dict(
            data=filename,
            offsets=offsets,
            materials=[],
            flatlims=flatlims)
        auxdat.update(self.auxdat)

        json.dump(auxdat, open(os.path.join(path, "%s.json"%fname), "w"))
        return ptidx
        

def get_pack(subj, xfm, types=("inflated",), method='mg1', level=1):
    ctmcache = os.path.join(filestore, "ctmcache")
    fname = os.path.join(ctmcache, "{subj}_{xfm}_[{types}]_{meth}_{lvl}.%s".format(
        subj=subj, xfm=xfm, types=','.join(types), 
        meth=method, lvl=level))

    if os.path.exists(fname%"json"):
        return fname%"json"

    print "No ctm found in cache, generating..."
    svgname = fname%"svg"
    kwargs = dict(rois=svgname, names=types)
    with CTMfile(subj, xfm, **kwargs) as ctm:
        for t in types:
            ctm.addSurf(t)

        ptidx = ctm.save(fname%"ctm", compmeth=method, complevel=level)

    print "Packing up SVG...",
    sys.stdout.flush()
    roipack = get_roipack(subj)
    with open(svgname, "w") as svgout:
        layer = roipack.make_text_layer()
        for element in layer.getElementsByTagName("p"):
            idx = int(element.getAttribute("data-ptidx"))
            if idx < ptidx['left'][1]:
                idx = ptidx['left'][0][idx]
            else:
                idx -= ptidx['left'][1]
                idx = ptidx['right'][0][idx] + ptidx['left'][1]
            element.setAttribute("data-ptidx", str(idx))
        roipack.svg.getElementsByTagName("svg")[0].appendChild(layer)
        svgout.write(roipack.svg.toxml())
    print "Done"

    return fname%"json"

if __name__ == "__main__":
    print get_pack("AH", "AH_huth2", types=("inflated", "superinflated"))
