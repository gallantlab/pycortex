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

from .database import db
from .utils import get_cortical_mask, get_mapper, get_dropout
from . import polyutils
from openctm import CTMfile

class BrainCTM(object):
    def __init__(self, subject, decimate=False):
        self.subject = subject
        self.types = []

        left, right = db.get_surf(subject, "fiducial")
        try:
            fleft, fright = db.get_surf(subject, "flat", nudge=True, merge=False)
        except IOError:
            fleft = None

        if decimate:
            try:
                pleft, pright = db.get_surf(subject, "pia")
                self.left = DecimatedHemi(left[0], left[1], fleft[1], pia=pleft[0])
                self.right = DecimatedHemi(right[0], right[1], fright[1], pia=pright[0])
                self.addSurf("wm", name="wm", addtype=False, renorm=False)
            except IOError:
                self.left = DecimatedHemi(left[0], left[1], fleft[1])
                self.right = DecimatedHemi(right[0], right[1], fright[1])
        else:
            try:
                pleft, pright = db.get_surf(subject, "pia")
                wleft, wright = db.get_surf(subject, "wm")
                self.left = Hemi(pleft[0], left[1])
                self.right = Hemi(pright[0], right[1])
                self.addSurf("wm", name="wm", addtype=False, renorm=False)
            except IOError:
                self.left = Hemi(left[0], left[1])
                self.right = Hemi(right[0], right[1])

            if fleft is not None:
                #set medial wall
                for hemi, ptpoly in ([self.left, fleft], [self.right, fright]):
                    fidpolys = set(tuple(f) for f in polyutils.sort_polys(hemi.polys))
                    flatpolys = set(tuple(f) for f in polyutils.sort_polys(ptpoly[1]))
                    hemi.aux[np.array(list(fidpolys - flatpolys)).astype(int), 0] = 1

        #Find the flatmap limits
        if fleft is not None:
            flatmerge = np.vstack([fleft[0][:,:2], fright[0][:,:2]])
            fmin, fmax = flatmerge.min(0), flatmerge.max(0)
            self.flatlims = map(float, -fmin), map(float, fmax-fmin)

            self.left.setFlat(fleft[0])
            self.right.setFlat(fright[0])
        else:
            self.flatlims = None

    def addSurf(self, typename, addtype=True, **kwargs):
        left, right = db.get_surf(self.subject, typename, nudge=False, merge=False)
        self.left.addSurf(left[0], **kwargs)
        self.right.addSurf(right[0], **kwargs)
        if addtype:
            self.types.append(typename)

    def addCurvature(self, **kwargs):
        npz = db.get_surfinfo(self.subject, type='curvature', **kwargs)
        try:
            self.left.aux[:,1] = npz.left[self.left.mask]
            self.right.aux[:,1] = npz.right[self.right.mask]
        except AttributeError:
            self.left.aux[:,1] = npz.left
            self.right.aux[:,1] = npz.right

    def save(self, path, method='mg2', disp_layers=['rois'], extra_disp=None, **kwargs):
        """Save CTM file for static html display. 

        Parameters
        ----------
        path : string filepath
        method : idkwtf
        disp_layers : tuple|list
            list of strings; names of layers from rois.svg to show
        extra_disp : tuple
            (svgfile,[layers]) - tuple of (external display .svg filename, [list_of,layers_in_file,to_display])
        """
        ctmname = path+".ctm"
        svgname = path+".svg"
        jsname = path+".json"

        # Save CTM concatenation
        (lpts, _, _), lbin = self.left.save(method=method, **kwargs)
        (rpts, _, _), rbin = self.right.save(method=method, **kwargs)

        offsets = [0]
        with open(path+'.ctm', 'w') as fp:
            fp.write(lbin)
            offsets.append(fp.tell())
            fp.write(rbin)

        # Save the JSON descriptor | Need to add to this for extra_disp?
        jsdict = dict(rois=os.path.split(svgname)[1],
                      data=os.path.split(ctmname)[1],
                      names=self.types, 
                      materials=[],
                      offsets=offsets)
        if self.flatlims is not None:
            jsdict['flatlims'] = self.flatlims
        json.dump(jsdict, open(jsname, 'w'))

        # Compute and save the index map
        if method != 'raw':
            ptmap, inverse = [], []
            for hemi, pts in zip([self.left, self.right], [lpts, rpts]):
                kdt = cKDTree(hemi.pts)
                diff, idx = kdt.query(pts)
                ptmap.append(idx)
                inverse.append(idx.argsort())
        else:
            ptmap = inverse = np.arange(len(self.left.ctm)), np.arange(len(self.right.ctm))

        # Save the SVG with remapped indices (map 2D flatmap locations to vertices)
        if self.left.flat is not None:
            # add sulci & display layers
            flatpts = np.vstack([self.left.flat, self.right.flat])
            roipack = db.get_overlay(self.subject, pts=flatpts, otype=disp_layers)
            # optionally add extra display layers
            if not extra_disp is None:
                esvgfile,elayerlist = extra_disp
                eroipack = db.get_overlay(self.subject,pts=flatpts, otype='external',
                    svgfile=esvgfile,layer=elayerlist)
                roipack = roipack + eroipack
            layers = roipack.setup_labels()
            if not isinstance(layers, (tuple, list)):
                layers = (layers,)
            
            # assign coordinates in left hemisphere negative values
            with open(svgname, "w") as fp:
                for layer in layers:
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
    def __init__(self, pts, polys, norms=None):
        self.tf = tempfile.NamedTemporaryFile()
        self.ctm = CTMfile(self.tf.name, "w")

        self.ctm.setMesh(pts.astype(np.float32), polys.astype(np.uint32), norms=norms)

        self.pts = pts
        self.polys = polys
        self.flat = None
        self.surfs = {}
        self.aux = np.zeros((len(self.ctm), 4))

    def addSurf(self, pts, name=None, renorm=True):
        '''Scales the in-between surfaces to be same scale as fiducial'''
        if name is None:
            name = 'morphTarget%d'%len(self.surfs)

        if renorm:
            norm = (pts - pts.min(0)) / (pts.max(0) - pts.min(0))
            rnorm = norm * (self.pts.max(0) - self.pts.min(0)) + self.pts.min(0)
        else:
            rnorm = pts

        attrib = np.hstack([rnorm, np.zeros((len(rnorm),1))])
        self.surfs[name] = attrib
        self.ctm.addAttrib(attrib, name)

    def setFlat(self, pts):
        self.ctm.addUV(pts[:,:2].astype(float), 'uv')
        self.flat = pts[:,:2]

    def save(self, **kwargs):
        self.ctm.addAttrib(self.aux, 'auxdat')
        self.ctm.save(**kwargs)

        ctm = CTMfile(self.tf.name)
        return ctm.getMesh(), self.tf.read()

class DecimatedHemi(Hemi):
    def __init__(self, pts, polys, fpolys, pia=None):
        print("Decimating...")
        kdt = cKDTree(pts)
        mask = np.zeros((len(pts),), dtype=bool)

        fidset = set([tuple(p) for p in polyutils.sort_polys(polys)])
        flatset = set([tuple(p) for p in polyutils.sort_polys(fpolys)])
        mwall = np.array(list(fidset - flatset))

        dpts, dpolys = polyutils.decimate(pts, fpolys)
        dist, didx = kdt.query(dpts)
        mask[didx] = True

        mwpts, mwpolys = polyutils.decimate(pts, mwall)
        dist, mwidx = kdt.query(mwpts)
        mask[mwidx] = True

        allpolys = np.vstack([didx[dpolys], mwidx[mwpolys]])
        idxmap = np.zeros((len(pts),), dtype=np.uint32)
        idxmap[mask] = np.arange(mask.sum()).astype(np.uint32)
        #norms = polyutils.Surface(pts, polys).normals[mask]
        basepts = pts[mask] if pia is None else pia[mask]
        super(DecimatedHemi, self).__init__(basepts, idxmap[allpolys])
        self.aux[idxmap[mwidx], 0] = 1
        self.mask = mask
        self.idxmap = idxmap

    def setFlat(self, pts):
        super(DecimatedHemi, self).setFlat(pts[self.mask])

    def addSurf(self, pts, **kwargs):
        super(DecimatedHemi, self).addSurf(pts[self.mask], **kwargs)

def make_pack(outfile, subj, types=("inflated",), method='raw', level=0,
              decimate=False, disp_layers=['rois'],extra_disp=None):
    """Generates a cached CTM file"""

    ctm = BrainCTM(subj, decimate=decimate)
    ctm.addCurvature()
    for name in types:
        ctm.addSurf(name)

    if not os.path.exists(os.path.split(outfile)[0]):
        os.makedirs(os.path.split(outfile)[0])

    return ctm.save(os.path.splitext(outfile)[0],
                    method=method,
                    level=level,
                    disp_layers=disp_layers,
                    extra_disp=extra_disp)

def read_pack(ctmfile):
    fname = os.path.splitext(ctmfile)[0]
    jsfile = json.load(open(fname+".json"))
    offset = jsfile['offsets']

    meshes = []

    with open(ctmfile, 'r') as ctmfp:
        ctmfp.seek(0, 2)
        offset.append(ctmfp.tell())

        for start, end in zip(offset[:-1], offset[1:]):
            ctmfp.seek(start)
            tf = tempfile.NamedTemporaryFile()
            tf.write(ctmfp.read(end-start))
            tf.seek(0)
            ctm = CTMfile(tf.name, "r")
            pts, polys, norms = ctm.getMesh()
            meshes.append((pts, polys))

    return meshes
