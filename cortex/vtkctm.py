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
        self.curvs = np.load(surfs.getAnat(self.name, type='curvature'))

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
        hemis = []
        for coords in self.coords:
            idx = np.ravel_multi_index(coords.T, vox.shape[::-1], mode='clip')
            hemis.append(vox.T.ravel()[idx])
        return hemis

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
        rawdata = nib.get_data().T
        if len(rawdata.shape) > 3:
            rawdata = rawdata.mean(0)
        data = np.zeros(self.mask.shape, dtype=np.float32)
        data[self.mask > 0] = rawdata[self.mask]
        norm = (data - data.min()) / (data.max() - data.min())
        return self._vox_to_idx((1-norm)**20)

    def addSurf(self, surf):
        cont = self.subj.contents
        for h, hemi in zip(["lh", "rh"], [cont.left, cont.right]):
            lib.hemiAddSurf(ctypes.byref(hemi), self.files['surfs'][surf][h], None)

    def addCurv(self, **kwargs):
        cont = self.subj.contents
        for h, hemi, curv in zip(['lh', 'rh'], [cont.left, cont.right], [self.curvs['left'], self.curvs['right']]):
            lib.hemiAddAux(ctypes.byref(hemi), curv.astype(np.float32), 1)

    def save(self, filename, compmeth='mg2', complevel=9):
        cont = self.subj.contents
        allpts = []
        for hemi in [cont.left, cont.right]:
            rpts, rpolys = _getmesh(hemi.fiducial, False) #reference
            fpts, fpolys = _getmesh(hemi.flat, False)
            dpolys  = set([tuple(p) for p in np.sort(rpolys, axis=1)])
            dpolys -= set([tuple(p) for p in np.sort(fpolys, axis=1)])
            dpolys = np.array(list(dpolys))

            mwall = np.zeros(len(rpts), dtype=np.float32)
            mwall[dpolys.ravel()] = 1
            lib.hemiAddAux(ctypes.byref(hemi), mwall, 2)
            allpts.append(rpts)

        left = tempfile.NamedTemporaryFile()
        right = tempfile.NamedTemporaryFile()
        minmax = lib.saveCTM(self.subj, left.name, right.name, compformats[compmeth], complevel)
        flatlims = [tuple(minmax.contents.min), tuple(minmax.contents.max)]
        lib.minmaxFree(minmax)
        print "CTM saved...",
        sys.stdout.flush()

        #Find where the points went
        ptidx = []
        for fp, rpts in zip([left, right], allpts):
            pts, polys = readCTM(fp.name)
            kdt = cKDTree(pts)
            diff, idx = kdt.query(rpts)
            ptidx.append((idx, len(rpts)))
            fp.seek(0)
        
        offsets = [0]
        with open(filename, "w") as fp:
            fp.write(left.read())
            offsets.append(fp.tell())
            fp.write(right.read())

        print "Complete!"

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
    layer = roipack.setup_labels()
    with open(svgname, "w") as svgout:
        for element in layer.findall(".//{http://www.w3.org/2000/svg}text"):
            idx = int(element.attrib["data-ptidx"])
            if idx < ptidx[0][1]:
                idx = ptidx[0][0][idx]
            else:
                idx -= ptidx[0][1]
                idx = ptidx[1][0][idx] + ptidx[0][1]
            element.attrib["data-ptidx"] = str(idx)
        svgout.write(roipack.toxml())
    print "Done"

    return outfile

def downsample(subject, angle=20):
    from tvtk.api import tvtk
    outputs = []
    fpolys = [polys for pts, polys, norms in surfs.getVTK(subject, "flat")]
    for (pts, polys, _), fpoly in zip(surfs.getVTK(subject, "fiducial"), fpolys):
        pts = pts.astype(np.float32)
        pd = tvtk.PolyData(points=pts, polys=fpoly)
        decimate = tvtk.DecimatePro(input=pd)
        decimate.set(
            boundary_vertex_deletion = False,
            feature_angle = angle, 
            preserve_topology = True,
            splitting = False,
            target_reduction = 1)
        decimate.update()
        dpts = dict((tuple(sorted(p)), i) for i, p in enumerate(decimate.output.points.to_array()))
        mask = np.array([tuple(sorted(p)) in dpts for p in pts])

        cutfaces = set(tuple(sorted(f)) for f in polys) - set(tuple(sorted(f)) for f in fpoly)
        dpoly = np.array([ dpts[tuple(sorted(pts[f]))] for f in np.array(list(cutfaces)).ravel() ])

        outputs.append((mask, dpoly.reshape(-1, 3), decimate.output.polys.to_array().reshape(-1, 3)))

    return outputs

if __name__ == "__main__":
    print make_pack("/tmp/test.json", "AH", "AH_huth2", types=("inflated", "superinflated"), method='mg2', level=9)
