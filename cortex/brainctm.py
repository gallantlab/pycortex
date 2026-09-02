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
import sys
import json
import tempfile
import numpy as np
from scipy.spatial import cKDTree

from .database import db
from .utils import get_cortical_mask, get_mapper, get_dropout
from . import polyutils
from .openctm import CTMfile

class BrainCTM:
    def __init__(self, subject, decimate=False):
        self.subject = subject
        self.types = []
        self.has_volume = False

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
                self.addSurf("wm", addtype=False, renorm=False)
                self.has_volume = True
            except IOError:
                self.left = DecimatedHemi(left[0], left[1], fleft[1])
                self.right = DecimatedHemi(right[0], right[1], fright[1])
        else:
            try:
                pleft, pright = db.get_surf(subject, "pia")
                wleft, wright = db.get_surf(subject, "wm")
                self.left = Hemi(pleft[0], left[1])
                self.right = Hemi(pright[0], right[1])
                self.addSurf("wm", addtype=False, renorm=False)
                self.has_volume = True
            except IOError:
                self.left = Hemi(left[0], left[1])
                self.right = Hemi(right[0], right[1])

            if fleft is not None:
                #set medial wall
                for hemi, ptpoly in ([self.left, fleft], [self.right, fright]):
                    # fidpolys = set(tuple(f) for f in polyutils.sort_polys(hemi.polys))
                    # flatpolys = set(tuple(f) for f in polyutils.sort_polys(ptpoly[1]))
                    # medial_verts = set(np.ravel(list(fidpolys - flatpolys)))
                    medial_verts = set(hemi.polys.ravel()) - set(ptpoly[1].ravel())
                    hemi.aux[list(medial_verts), 0] = 1

                    connected = [set() for _ in range(len(ptpoly[0]))]
                    for p1, p2, p3 in hemi.polys:
                        if p1 not in medial_verts:
                            connected[p2].add(p1)
                            connected[p3].add(p1)
                        if p2 not in medial_verts:
                            connected[p1].add(p2)
                            connected[p3].add(p2)
                        if p3 not in medial_verts:
                            connected[p1].add(p3)
                            connected[p2].add(p3)

                    #move the medial wall vertices out of the flatmap
                    for vert in medial_verts:
                        candidates = connected[vert]
                        if len(candidates) > 0:
                            ptpoly[0][vert] = ptpoly[0][candidates.pop()]
                        else:
                            ptpoly[0][vert] = 0

        #Find the flatmap limits
        if fleft is not None:
            flatmerge = np.vstack([fleft[0][:,:2], fright[0][:,:2]])
            fmin, fmax = flatmerge.min(0), flatmerge.max(0)
            self.flatlims = [float(x) for x in -fmin], [float(x) for x in fmax-fmin]

            self.left.setFlat(fleft[0])
            self.right.setFlat(fright[0])
        else:
            self.flatlims = None

    def addSurf(self, typename, addtype=True, **kwargs):
        left, right = db.get_surf(self.subject, typename, nudge=False, merge=False)
        self.left.addSurf(left[0], typename, **kwargs)
        self.right.addSurf(right[0], typename, **kwargs)
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

    def addEquivolumeAreas(self, **kwargs):
        """Load the vertex areas the viewer's equivolume depth sampling needs.

        They ride in the two spare components of ``auxdat`` -- ``x`` is the
        medial wall mask and ``y`` the curvature -- rather than taking vertex
        attribute slots of their own, which the surface shaders are already
        right up against.
        """
        if not self.has_volume:
            return
        npz = db.get_surfinfo(self.subject, type='equivolume_areas', **kwargs)
        try:
            self.left.aux[:,2] = npz['wm_left'][self.left.mask]
            self.left.aux[:,3] = npz['pia_left'][self.left.mask]
            self.right.aux[:,2] = npz['wm_right'][self.right.mask]
            self.right.aux[:,3] = npz['pia_right'][self.right.mask]
        except AttributeError:
            self.left.aux[:,2] = npz['wm_left']
            self.left.aux[:,3] = npz['pia_left']
            self.right.aux[:,2] = npz['wm_right']
            self.right.aux[:,3] = npz['pia_right']
        npz.close()

    def addBumpyFlat(self, **kwargs):
        """Load the relief that gives the flatmap its bumps.

        Each vertex gets a (0, 0, height) offset from its position on the flat
        white matter surface, in flatmap units.
        """
        if self.flatlims is None or not self.has_volume:
            return

        npz = db.get_surfinfo(self.subject, type='bumpy_flatmap', **kwargs)
        self.left.setBump(npz['bump_left'])
        self.right.setBump(npz['bump_right'])
        npz.close()

    def save(self, path, method='mg2', external_svg=None, 
             overlays_available=None, **kwargs):
        """Save CTM file for static html display. 

        Parameters
        ----------
        path : string
            File path for cached ctm file to save
        method : str
            string specifying method of how inverse transforms for
            labels are computed (determines how labels are displayed
            on 3D viewer) one of ['mg2','raw']
        overlays_available : str
            Which overlays in the svg file to include in the viewer. If
            None, all layers in the relevant svg file are included.

        """
        ctmname = path + ".ctm"
        svgname = path + ".svg"
        jsname = path + ".json"
        mapname = path + ".npz"

        # Save CTM concatenation
        (lpts, _, _), lbin = self.left.save(method=method, **kwargs)
        (rpts, _, _), rbin = self.right.save(method=method, **kwargs)

        offsets = [0]
        with open(path+'.ctm', 'wb') as fp:
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
        with open(jsname, 'w') as fp:
            json.dump(jsdict, fp)

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

        np.savez(mapname, 
            index=np.hstack([ptmap[0], ptmap[1]+len(ptmap[0])]), 
            inverse=np.hstack([inverse[0], inverse[1]+len(inverse[0])]))

        # Save the SVG with remapped indices (map 2D flatmap locations to vertices)
        if self.left.flat is not None:
            flatpts = np.vstack([self.left.flat, self.right.flat])
            if external_svg is None:
                svg = db.get_overlay(self.subject, pts=flatpts, 
                                     overlays_available=overlays_available) 
            else:
                from .svgoverlay import get_overlay
                _, polys = db.get_surf(self.subject, "flat", merge=True, nudge=True)
                svg = get_overlay(self.subject, external_svg, flatpts, polys, 
                                  overlays_available=overlays_available)
            
            # assign coordinates in left hemisphere negative values
            with open(svgname, "wb") as fp:
                for element in svg.svg.findall(".//{http://www.w3.org/2000/svg}text"):
                    if 'data-ptidx' in element.attrib:
                        idx = int(element.attrib["data-ptidx"])
                        if idx < len(inverse[0]):
                            idx = inverse[0][idx]
                        else:
                            idx -= len(inverse[0])
                            idx = inverse[1][idx] + len(inverse[0])
                        element.attrib["data-ptidx"] = str(idx)
                fp.write(svg.toxml())
        return ptmap

class Hemi:
    def __init__(self, pts, polys, norms=None):
        # On Windows a NamedTemporaryFile cannot be opened a second time while
        # the Python handle is open, so close it and let the CTM library open it
        self.tf = tempfile.NamedTemporaryFile(delete = False)
        self.tf.close()
        self.tfName = os.fsencode(self.tf.name)
        self.ctm = CTMfile(self.tfName, "w")

        self.ctm.setMesh(pts.astype(np.float32),
                         polys.astype(np.uint32),
                         norms=norms)

        self.pts = pts
        self.polys = polys
        self.flat = None
        self.bump = None
        self.surfs = {}
        self.aux = np.zeros((len(self.ctm), 4))

    def addSurf(self, pts, name, renorm=True):
        '''Scales the in-between surfaces to be same scale as fiducial'''
        if renorm:
            norm = (pts - pts.min(0)) / (pts.max(0) - pts.min(0))
            rnorm = norm * (self.pts.max(0) - self.pts.min(0)) + self.pts.min(0)
        else:
            rnorm = pts

        attrib = np.hstack([rnorm, np.zeros((len(rnorm),1))])
        self.surfs[name] = attrib
        self.ctm.addAttrib(attrib, name)
        print(name)

    def setFlat(self, pts):
        self.ctm.addUV(pts[:,:2].astype(float), 'uv')
        self.flat = pts[:,:2]

    def setBump(self, offsets):
        '''Bumpy flatmap offsets, padded to the four components a ctm attribute
        map always has.'''
        self.bump = np.hstack([offsets, np.zeros((len(offsets), 1))])

    def save(self, **kwargs):
        self.ctm.addAttrib(self.aux, 'auxdat')
        if self.bump is not None:
            self.ctm.addAttrib(self.bump, 'flatoffset')

        # OpenCTM only has eight attribute map slots. Two go to auxdat and the
        # bumpy flatmap offsets and one to the white matter surface, so a viewer
        # can carry at most five extra surfaces.
        if len(self.ctm.attribs) > 8:
            raise ValueError(
                "too many surfaces for one ctm file: %d attribute maps, and "
                "OpenCTM allows 8. Pass fewer entries in `types`. (%s)"
                % (len(self.ctm.attribs), ", ".join(self.ctm.attribs)))
        self.ctm.save(**kwargs)
        ctm = CTMfile(self.tfName)
        mesh = ctm.getMesh()
        with open(self.tf.name, 'rb') as fp:
            binary = fp.read()
        os.unlink(self.tf.name)
        return mesh, binary

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
        super().__init__(basepts, idxmap[allpolys])
        self.aux[idxmap[mwidx], 0] = 1
        self.mask = mask
        self.idxmap = idxmap

    def setFlat(self, pts):
        super().setFlat(pts[self.mask])

    def addSurf(self, pts, **kwargs):
        super().addSurf(pts[self.mask], **kwargs)

    def setBump(self, offsets):
        super().setBump(offsets[self.mask])

def make_pack(outfile, subj, types=("inflated",), method='raw', level=0,
              decimate=False, disp_layers=['rois'], 
              external_svg=None, overlays_available=None,):
    """Generates a cached CTM file

    Parameters
    ----------

    """

    ctm = BrainCTM(subj, decimate=decimate)
    ctm.addCurvature()
    ctm.addEquivolumeAreas()
    ctm.addBumpyFlat()
    for name in types:
        ctm.addSurf(name)

    if not os.path.exists(os.path.split(outfile)[0]):
        os.makedirs(os.path.split(outfile)[0])
    return ctm.save(os.path.splitext(outfile)[0],
                    method=method,
                    level=level,
                    external_svg=external_svg,
                    overlays_available=overlays_available)

def read_pack(ctmfile):
    fname = os.path.splitext(ctmfile)[0]
    with open(fname + ".json") as fp:
        jsfile = json.load(fp)
    offset = jsfile['offsets']

    meshes = []

    with open(ctmfile, 'rb') as ctmfp:
        ctmfp.seek(0, 2)
        offset.append(ctmfp.tell())

        for start, end in zip(offset[:-1], offset[1:]):
            ctmfp.seek(start)
            tf = tempfile.NamedTemporaryFile(delete = False)
            tf.write(ctmfp.read(end-start))
            tf.close()
            ctm = CTMfile(tf.name, "r")
            pts, polys, norms = ctm.getMesh()
            meshes.append((pts, polys))
            os.unlink(tf.name)

    return meshes
