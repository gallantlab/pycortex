import os
import sys
import struct
import ctypes
import json
import tempfile
import numpy as np
from scipy.spatial import cKDTree

from utils import get_cortical_mask, get_roipack, get_curvature
from db import surfs

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
        ("datamap", ctypes.POINTER(ctypes.c_float)),
        ("aux", ctypes.POINTER(ctypes.c_float))
    ]

class Subject(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char*128),
        ("xfm", ctypes.c_float*16),
        ("left", Hemi),
        ("right", Hemi)]

cwd = os.path.split(os.path.abspath(__file__))[0]
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
lib.hemiAddAux.argtypes = [ctypes.POINTER(Hemi), np.ctypeslib.ndpointer(np.float32), ctypes.c_uint16]
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
    def __init__(self, subject, xfmname):
        self.name = subject
        self.xfmname = xfmname
        self.files = surfs.getFiles(subject)
        self.mask = get_cortical_mask(self.name, self.xfmname)
        self.coords = surfs.getCoords(self.name, self.xfmname)

    def __enter__(self):
        self.subj = lib.newSubject(self.name)
        cont = self.subj.contents
        for h, hemi, datamap, drop in zip(["lh", "rh"], [cont.left, cont.right], self.maps, self.dropout):
            lib.hemiAddFid(ctypes.byref(hemi), self.files['surfs']['fiducial'][h])
            lib.hemiAddFlat(ctypes.byref(hemi), self.files['surfs']['flat'][h])
            lib.hemiAddMap(ctypes.byref(hemi), datamap)
            lib.hemiAddAux(ctypes.byref(hemi), drop.astype(np.float32), 0)
        return self

    def __exit__(self, type, value, traceback):
        lib.subjFree(self.subj)

    def _vox_to_idx(self, vox):
        values = []
        left, right = self.coords
        for coords in (left, right):
            idx = np.ravel_multi_index(coords.T, vox.shape[::-1])
            values.append(vox.T.ravel()[idx])
        return values

    @property
    def maps(self):
        imask = self.mask.astype(np.uint32)
        imask[imask > 0] = np.arange(self.mask[self.mask > 0].sum())
        return self._vox_to_idx(imask)

    @property
    def dropout(self):
        import nibabel
        xfm, ref = surfs.getXfm(self.name, self.xfmname)
        nib = nibabel.load(ref)
        data = nib.get_data().T
        norm = (data - data.min()) / (data.max() - data.min())
        return self._vox_to_idx(norm**20)

    def addSurf(self, surf):
        cont = self.subj.contents
        for h, hemi in zip(["lh", "rh"], [cont.left, cont.right]):
            lib.hemiAddSurf(ctypes.byref(hemi), self.files['surfs'][surf][h], None)

    def addCurv(self, **kwargs):
        cont = self.subj.contents
        curvs = get_curvature(self.name, **kwargs)
        for h, hemi, curv in zip(['lh', 'rh'], [cont.left, cont.right], curvs):
            lib.hemiAddAux(ctypes.byref(hemi), curv.astype(np.float32), 1)

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
        with open(filename, "w") as fp:
            head = struct.pack('2I', len(didx[0]), len(didx[1]))
            fp.write(head)
            fp.write(didx[0].tostring())
            fp.write(didx[1].tostring())
            offsets.append(fp.tell())
            fp.write(left.read())
            offsets.append(fp.tell())
            fp.write(right.read())

        return ptidx, dict(offsets=offsets, flatlims=flatlims)

def make_pack(outfile, subj, xfm, types=("inflated",), method='raw', level=0, **curvargs):
    fname, ext = os.path.splitext(outfile)
    with CTMfile(subj, xfm) as ctm:
        for t in types:
            ctm.addSurf(t)
        ctm.addCurv(**curvargs)
        ptidx, jsondat = ctm.save("%s.ctm"%fname, compmeth=method, complevel=level)

    svgname = "%s.svg"%fname
    jsondat.update(dict(
        rois=os.path.split(svgname)[1], 
        data=os.path.split("%s.ctm"%fname)[1],
        names=types,
        materials=[],
    ))
    json.dump(jsondat, open(outfile, "w"))

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

    return outfile
    

if __name__ == "__main__":
    print make_pack("/tmp/test.json", "AH", "AH_huth2", types=("inflated", "superinflated"), recache=True)
