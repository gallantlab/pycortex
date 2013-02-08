import os
import json
import tempfile
import numpy as np
from scipy.spatial import cKDTree

from db import surfs
from utils import get_cortical_mask, get_mapper, get_roipack
from openctm import CTMfile

class BrainCTM(object):
    def __init__(self, subject, xfmname, base='fiducial'):
        self.subject = subject
        self.xfmname = xfmname
        self.files = surfs.getFiles(subject)
        self.types = []

        xfm, epifile = surfs.getXfm(subject, xfmname)
        import nibabel
        nib = nibabel.load(epifile)
        self.shape = nib.get_shape()[::-1]
        if len(self.shape) > 3:
            self.shape = self.shape[-3:]

        self.left, self.right = map(Hemi, surfs.getVTK(subject, base))

        #Find the flatmap limits
        left, right = surfs.getVTK(subject, "flat", nudge=True, merge=False)
        flatmerge = np.vstack([left[0][:,:2], right[0][:,:2]])
        fmin, fmax = flatmerge.min(0), flatmerge.max(0)
        self.flatlims = list(-fmin), list(fmax-fmin)
        self.left.setFlat(left[0])
        self.right.setFlat(right[0])

        #set medial wall
        for hemi, ptpoly in zip([self.left, self.right], [left, right]):
            fidpolys = set(tuple(f) for f in np.sort(hemi.polys, axis=1))
            flatpolys = set(tuple(f) for f in np.sort(ptpoly[1], axis=1))
            mwall = np.zeros(len(hemi.ctm))
            mwall[np.array(list(fidpolys - flatpolys))] = 1
            hemi.aux[:,2] = mwall

    def addSurf(self, typename):
        left, right = surfs.getVTK(self.subject, typename, nudge=False, merge=False)
        self.left.addSurf(left[0])
        self.right.addSurf(right[0])
        self.types.append(typename)

    def addDropout(self, projection='trilinear', power=20):
        import nibabel
        xfm, ref = surfs.getXfm(self.subject, self.xfmname)
        nib = nibabel.load(ref)
        rawdata = nib.get_data().T
        if rawdata.ndim > 3:
            rawdata = rawdata.mean(0)

        norm = (rawdata - rawdata.min()) / (rawdata.max() - rawdata.min())
        mapper = get_mapper(self.subject, self.xfmname, projection)
        left, right = mapper(norm**power)
        self.left.aux[:,0] = left
        self.right.aux[:,0] = right

    def addCurvature(self, **kwargs):
        npz = np.load(surfs.getAnat(self.subject, type='curvature', **kwargs))
        self.left.aux[:,1] = npz['left']
        self.right.aux[:,1] = npz['right']

    def addMap(self):
        mapper = get_mapper(self.subject, self.xfmname, 'nearest')
        mask = mapper.mask.astype(np.uint32)
        mask[mask > 0] = np.arange(mask.sum())
        self.left.aux[:, 3], self.right.aux[:,3] = mapper(mask)

    def save(self, path, method='mg2', **kwargs):
        ctmname = path+".ctm"
        svgname = path+".svg"
        jsname = path+".json"
        ptmapname = path+".npz"

        ##### Save CTM concatenation
        (lpts, _, _), lbin = self.left.save(method=method, **kwargs)
        (rpts, _, _), rbin = self.right.save(method=method, **kwargs)

        offsets = [0]
        with open(path+'.ctm', 'w') as fp:
            fp.write(lbin)
            offsets.append(fp.tell())
            fp.write(rbin)

        ##### Save the JSON descriptor
        json.dump(dict(rois=os.path.split(svgname)[1], data=os.path.split(ctmname)[1], names=self.types, 
            materials=[], offsets=offsets, flatlims=self.flatlims, shape=self.shape), open(jsname, 'w'))

        ##### Compute and save the index map
        if method != 'raw':
            ptmap, inverse = [], []
            for hemi, pts in zip([self.left, self.right], [lpts, rpts]):
                kdt = cKDTree(hemi.pts)
                diff, idx = kdt.query(pts)
                ptmap.append(idx)
                inverse.append(idx.argsort())

            np.savez(ptmapname, left=ptmap[0], right=ptmap[1])
        else:
            inverse = np.arange(len(self.left.ctm)), np.arange(len(self.right.ctm))

        ##### Save the SVG with remapped indices
        roipack = get_roipack(self.subject)
        layer = roipack.setup_labels()
        with open(svgname, "w") as fp:
            for element in layer.findall(".//{http://www.w3.org/2000/svg}text"):
                idx = int(element.attrib["data-ptidx"])
                if idx < len(inverse[0]):
                    idx = inverse[0][idx]
                else:
                    idx -= len(inverse[0])
                    idx = inverse[1][idx] + len(inverse[0])
                element.attrib["data-ptidx"] = str(idx)
            fp.write(roipack.toxml())

        return ptmap

class Hemi(object):
    def __init__(self, fiducial):
        self.nsurfs = 0
        self.tf = tempfile.NamedTemporaryFile()
        self.ctm = CTMfile(self.tf.name, "w")
        self.ctm.setMesh(*fiducial)
        
        self.pts = fiducial[0]
        self.polys = fiducial[1]
        self.aux = np.zeros((len(self.ctm), 4))

    def addSurf(self, pts):
        '''Scales the in-between surfaces to be same scale as fiducial'''
        norm = (pts - pts.min(0)) / (pts.max(0) - pts.min(0))
        rnorm = norm * (self.pts.max(0) - self.pts.min(0)) + self.pts.min(0)
        attrib = np.hstack([rnorm, np.zeros((len(rnorm),1))])
        self.ctm.addAttrib(attrib, 'morphTarget%d'%self.nsurfs)
        self.nsurfs += 1

    def setFlat(self, pts):
        assert np.all(pts[:,2] == 0)
        self.ctm.addUV(pts[:,:2], 'uv')

    def save(self, **kwargs):
        self.ctm.addAttrib(self.aux, 'auxdat')
        self.ctm.save(**kwargs)

        ctm = CTMfile(self.tf.name)
        return ctm.getMesh(), self.tf.read()

def make_pack(outfile, subj, xfm, types=("inflated",), method='raw', level=0):
    ctm = BrainCTM(subj, xfm)
    ctm.addMap()
    ctm.addDropout()
    ctm.addCurvature()
    for name in types:
        ctm.addSurf(name)

    return ctm.save(os.path.splitext(outfile)[0], method=method, level=level)
