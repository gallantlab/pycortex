import os
import tempfile
import numpy as np

from db import surfs
from utils import get_cortical_mask, get_mapper, get_roipack
from openctm import CTMfile

class BrainCTM(object):
    def __init__(self, subject, xfmname, base='fiducial'):
        self.subject = subject
        self.xfmname = xfmname
        self.files = surfs.getFiles(subject)
        self.types = []

        self.left, self.right = map(Hemi, surfs.getVTK(subject, base))

        #Find the flatmap limits
        left, right = surfs.getVTK(subject, "flat", nudge=True, merge=False)
        flatmerge = np.vstack([left[0][:,:2], right[0][:,:2]])
        fmin, fmax = flatmerge.min(0), flatmerge.max(0)
        self.flatlims = -fmin, fmax-fmin
        self.left.setFlat(left[0])
        self.right.setFlat(right[0])

        #set medial wall
        for hemi, ptpoly in zip([self.left, self.right], [left, right]):
            mwall = np.ones(len(hemi.ctm))
            mwall[np.unique(ptpoly[1])] = 0
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

    def save(self, path, method='mg2', **kwargs):
        ctmname = path+".ctm"
        svgname = path+".svg"
        jsname = path+".json"
        ptmapname = path+".npz"

        (lpts, _, _), lbin = self.left.save(method=method, **kwargs)
        (rpts, _, _), rbin = self.right.save(method=method, **kwargs)

        offsets = [0]
        with open(path+'.ctm', 'w') as fp:
            fp.write(lbin)
            offsets.append(fp.tell())
            fp.write(rbin)

        json.dump(dict(rois=os.path.split(svgname)[1], data=ctmname, names=self.types, 
            materials=[], offsets=offsets, flatlims=self.flatlims), open(jsname, 'w'))

        if method != 'raw':
            ptmap = []
            for hemi, pts in zip([self.left, self.right], [lpts, rpts]):
                kdt = cKDTree(hemi.pts)
                diff, idx = kdt.query(pts)
                ptmap.append(idx)
        else:
            ptmap = np.arange(self.left.ctm), np.arange(self.right.ctm)

        np.savez(ptmapname, left=ptmap[0], right=ptmap[1])

        with open(svgname, "w") as fp:
            for element in layer.findall(".//{http://www.w3.org/2000/svg}text"):
                idx = int(element.attrib["data-ptidx"])
                if idx < len(ptidx[0]):
                    idx = ptidx[0][idx]
                else:
                    idx -= len(ptidx[0])
                    idx = ptidx[1][idx] + len(ptidx[0])
                element.attrib["data-ptidx"] = str(idx)
            fp.write(roipack.toxml())

        return ptmap

class Hemi(object):
    def __init__(self, fiducial):
        self.nsurfs = 0
        self.tf = tempfile.NamedTemporaryFile()
        self.ctm = CTMfile(self.tf.name, "w")
        self.ctm.setMesh(*fiducial)
        self.aux = np.zeros((len(self.ctm), 4))        
        self.pts = fiducial[0]

    def addSurf(self, pts):
        '''Scales the in-between surfaces to be same scale as fiducial'''
        norm = (pts - pts.min(0)) / (pts.max(0) - pts.min(0))
        rnorm = (norm - self.pts.min(0)) * self.pts.max(0)
        self.ctm.addAttrib(rnorm, 'morphTarget%d'%self.nsurfs)
        self.nsurfs += 1

    def setFlat(self, pts):
        self.ctm.addUV(pts, 'uv')

    def save(self, **kwargs):
        self.ctm.addAttrib(self.aux, 'auxdat')
        self.ctm.save(**kwargs)
        self.tf.seek(0)

        ctm = CTMfile(fname)
        return ctm.getMesh(), self.tf.read()

if __name__ == "__main__":
    ctm = BrainCTM("AH", "AH_huth")
    ctm.addDropout()
    ctm.addCurvature()
    ctm.addSurf("inflated")
    ctm.save("/tmp/test")