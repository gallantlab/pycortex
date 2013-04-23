'''Generates the OpenCTM file that holds the brain mesh for the webgl viewer. 
This ctm file contains the following information:

pts, polys
  Forms the base of the brain mesh, in the fiducial space

morphTarget%d
  Holds additional surfaces (inflated, etc) as morphTargets for three.js.
  Morphtargets are normalized to the same extent as the fiducial, for better
  morphing effect.

uv
  Actually stores the raw flatmap coordinates, unnormalized. Normalization is handled
  in javascript, in the load function
'''
import os
import json
import tempfile
import numpy as np
from scipy.spatial import cKDTree

from .db import surfs
from .utils import get_cortical_mask, get_mapper, get_roipack, get_dropout
from openctm import CTMfile

class BrainCTM(object):
    def __init__(self, subject, xfmname):
        self.subject = subject
        self.xfmname = xfmname
        self.files = surfs.getFiles(subject)
        self.types = []

        xfm = surfs.getXfm(subject, xfmname)
        self.shape = xfm.shape

        try:
            self.left, self.right = list(map(Hemi, surfs.getSurf(subject, "pia")))
            left, right = surfs.getSurf(subject, "wm", nudge=False, merge=False)
            self.left.addSurf(left[0], name="wm")
            self.right.addSurf(right[0], name="wm")
        except IOError:
            self.left, self.right = list(map(Hemi, surfs.getSurf(subject, "fiducial")))

        #Find the flatmap limits
        left, right = surfs.getSurf(subject, "flat", nudge=True, merge=False)
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
            mwall[np.array(list(fidpolys - flatpolys)).astype(int)] = 1
            hemi.aux[:,2] = mwall

    def addSurf(self, typename):
        left, right = surfs.getSurf(self.subject, typename, nudge=False, merge=False)
        self.left.addSurf(left[0])
        self.right.addSurf(right[0])
        self.types.append(typename)

    def addDropout(self, projection='trilinear', power=20):
        left, right = get_dropout(self.subject, self.xfmname, projection, power)
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
            ptmap = inverse = np.arange(len(self.left.ctm)), np.arange(len(self.right.ctm))

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
        self.tf = tempfile.NamedTemporaryFile()
        self.ctm = CTMfile(self.tf.name, "w")
        self.ctm.setMesh(*fiducial)
        
        self.pts = fiducial[0]
        self.polys = fiducial[1]
        self.surfs = {}
        self.aux = np.zeros((len(self.ctm), 4))

    def addSurf(self, pts, name=None):
        '''Scales the in-between surfaces to be same scale as fiducial'''
        if name is None:
            name = 'morphTarget%d'%len(self.surfs)
        norm = (pts - pts.min(0)) / (pts.max(0) - pts.min(0))
        rnorm = norm * (self.pts.max(0) - self.pts.min(0)) + self.pts.min(0)
        attrib = np.hstack([rnorm, np.zeros((len(rnorm),1))])
        self.surfs[name] = attrib
        self.ctm.addAttrib(attrib, name)

    def setFlat(self, pts):
        #assert np.all(pts[:,2] == 0)
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
