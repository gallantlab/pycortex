import numpy as np
from scipy import sparse

from . import Mapper
from . import samplers

class VolumeMapper(Mapper):
    @classmethod
    def _cache(cls, filename, subject, xfmname, **kwargs):
        from .. import db
        masks = []
        xfm = db.get_xfm(subject, xfmname, xfmtype='coord')
        pia = db.get_surf(subject, "pia", merge=False, nudge=False)
        wm = db.get_surf(subject, "wm", merge=False, nudge=False)
        
        #iterate over hemispheres
        for (wpts, polys), (ppts, _) in zip(pia, wm):
            masks.append(cls._getmask(xfm(ppts), xfm(wpts), polys, xfm.shape, **kwargs))
            
        _savecache(filename, masks[0], masks[1], xfm.shape)
        return cls(masks[0], masks[1], xfm.shape)

    @classmethod
    def _getmask(cls, pia, wm, polys, shape, **kwargs):
        from .. import mp
        rand = np.random.rand(npts, 3)
        csrshape = len(wm), np.prod(shape)

        def func(pts):
            if len(pts) > 0:
                #generate points within the bounding box
                samples = rand * (pts.max(0) - pts.min(0)) + pts.min(0)
                #check which points are inside the polyhedron
                inside = polyutils.inside_convex_poly(pts)(samples)
                return cls._sample(samples[inside], shape, np.sum(inside))

        surf = polyutils.Surface(pia, polys)
        samples = mp.map(func, surf.polyconvex(wm))
        #samples = map(func, surf.polyconvex(wm)) ## For debugging
        ij, data = [], []
        for i, sample in enumerate(samples):
            if sample is not None:
                idx = np.zeros((2, len(sample[0])))
                idx[0], idx[1] = i, sample[0]
                ij.append(idx)
                data.append(sample[1])

        return sparse.csr_matrix((np.hstack(data), np.hstack(ij)), shape=csrshape)

class PolyConstMapper(VolumeMapper):
    patchsize = 0.5
    

class PolyLinMapper(VolumeMapper):
    patchsize = 1

class Polyhedral(VolumeMapper):
    '''Uses an actual (likely concave) polyhedra betwen the pial and white surfaces
    to estimate the thickness'''
    @staticmethod
    def _getmask(pia, wm, polys, shape):
        from .. import polyutils
        mask = sparse.csr_matrix((len(wm), np.prod(shape)))

        from tvtk.api import tvtk
        measure = tvtk.MassProperties()
        planes = tvtk.PlaneCollection()
        for norm in np.vstack([-np.eye(3), np.eye(3)]):
            planes.append(tvtk.Plane(normal=norm))
        ccs = tvtk.ClipClosedSurface(clipping_planes=planes)
        feats = tvtk.FeatureEdges(boundary_edges=1, non_manifold_edges=0, manifold_edges=0, feature_edges=0)
        feats.set_input(ccs.output)

        surf = polyutils.Surface(pia, polys)
        for i, (pts, faces) in enumerate(surf.polyhedra(wm)):
            if len(pts) > 0:
                poly = tvtk.PolyData(points=pts, polys=faces)
                measure.set_input(poly)
                measure.update()
                totalvol = measure.volume
                ccs.set_input(poly)
                measure.set_input(ccs.output)

                bmin = pts.min(0).round().astype(int)
                bmax = (pts.max(0).round() + 1).astype(int)
                vidx = np.mgrid[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]]
                for vox in vidx.reshape(3, -1).T:
                    try:
                        idx = np.ravel_multi_index(vox[::-1], shape)
                        for plane, m in zip(planes, [.5, .5, .5, -.5, -.5, -.5]):
                            plane.origin = vox+m

                        ccs.update()
                        if ccs.output.number_of_cells > 2:
                            measure.update()
                            mask[i, idx] = measure.volume
    
                    except ValueError:
                        print('Voxel not in volume: (%d, %d, %d)'%tuple(vox))

                mask.data[mask.indptr[i]:mask.indptr[i+1]] /= mask[i].sum()

        return mask

class ConvexPolyhedra(VolumeMapper):
    @classmethod
    def _getmask(cls, pia, wm, polys, shape, npts=1024):
        from .. import mp
        from .. import polyutils
        rand = np.random.rand(npts, 3)
        csrshape = len(wm), np.prod(shape)

        def func(pts):
            if len(pts) > 0:
                #generate points within the bounding box
                samples = rand * (pts.max(0) - pts.min(0)) + pts.min(0)
                #check which points are inside the polyhedron
                inside = polyutils.inside_convex_poly(pts)(samples)
                return cls._sample(samples[inside], shape, np.sum(inside))

        surf = polyutils.Surface(pia, polys)
        samples = mp.map(func, surf.polyconvex(wm))
        #samples = map(func, surf.polyconvex(wm)) ## For debugging
        ij, data = [], []
        for i, sample in enumerate(samples):
            if sample is not None:
                idx = np.zeros((2, len(sample[0])))
                idx[0], idx[1] = i, sample[0]
                ij.append(idx)
                data.append(sample[1])

        return sparse.csr_matrix((np.hstack(data), np.hstack(ij)), shape=csrshape)

class ConvexNN(VolumeMapper):
    @staticmethod
    def _sample(pts, shape, norm):
        coords = pts.round().astype(int)[:,::-1]
        d1 = np.logical_and(0 <= coords[:,0], coords[:,0] < shape[0])
        d2 = np.logical_and(0 <= coords[:,1], coords[:,1] < shape[1])
        d3 = np.logical_and(0 <= coords[:,2], coords[:,2] < shape[2])
        valid = np.logical_and(d1, np.logical_and(d2, d3))
        if valid.any():
            idx = np.ravel_multi_index(coords[valid].T, shape)
            j, data = np.array(Counter(idx).items()).T
            return j, data / float(norm)

class ConvexTrilin(VolumeMapper):
    @staticmethod
    def _sample(pts, shape, norm):
        (x, y, z), floor = np.modf(pts.T)
        floor = floor.astype(int)
        ceil = floor + 1
        x[x < 0] = 0
        y[y < 0] = 0
        z[z < 0] = 0

        i000 = np.ravel_multi_index((floor[2], floor[1], floor[0]), shape, mode='clip')
        i100 = np.ravel_multi_index((floor[2], floor[1],  ceil[0]), shape, mode='clip')
        i010 = np.ravel_multi_index((floor[2],  ceil[1], floor[0]), shape, mode='clip')
        i001 = np.ravel_multi_index(( ceil[2], floor[1], floor[0]), shape, mode='clip')
        i101 = np.ravel_multi_index(( ceil[2], floor[1],  ceil[0]), shape, mode='clip')
        i011 = np.ravel_multi_index(( ceil[2],  ceil[1], floor[0]), shape, mode='clip')
        i110 = np.ravel_multi_index((floor[2],  ceil[1],  ceil[0]), shape, mode='clip')
        i111 = np.ravel_multi_index(( ceil[2],  ceil[1],  ceil[0]), shape, mode='clip')

        v000 = (1-x)*(1-y)*(1-z)
        v100 = x*(1-y)*(1-z)
        v010 = (1-x)*y*(1-z)
        v110 = x*y*(1-z)
        v001 = (1-x)*(1-y)*z
        v101 = x*(1-y)*z
        v011 = (1-x)*y*z
        v111 = x*y*z

        allj = np.vstack([i000, i100, i010, i001, i101, i011, i110, i111]).T.ravel()
        data = np.vstack([v000, v100, v010, v001, v101, v011, v110, v111]).T.ravel()

        uniquej = np.unique(allj)
        uniquejdata = np.array([data[allj==j].sum() for j in uniquej])
        
        return uniquej, uniquejdata / float(norm)


class ConvexLanczos(VolumeMapper):
    def _sample(self, pts):
        raise NotImplementedError
