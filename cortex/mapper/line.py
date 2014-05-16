import numpy as np
from scipy import sparse

from . import Mapper, _savecache
from . import samplers

class LineMapper(Mapper):
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
    def _getmask(cls, pia, wm, polys, shape, npts=64, mp=True, **kwargs):
        valid = np.unique(polys)
        #vidx = np.nonzero(valid)[0]
        mapper = sparse.csr_matrix((len(pia), np.prod(shape)))
        for t in np.linspace(0, 1, npts+2)[1:-1]:
            i, j, data = cls.sampler(pia*t + wm*(1-t), shape)
            mapper = mapper + sparse.csr_matrix((data / npts, (i, j)), shape=mapper.shape)
        return mapper

class LineNN(LineMapper):
    sampler = staticmethod(samplers.nearest)

class LineTrilin(LineMapper):
    sampler = staticmethod(samplers.trilinear)

class LineGauss(LineMapper):
    sampler = staticmethod(samplers.gaussian)

class LineLanczos(LineMapper):
    sampler = staticmethod(samplers.lanczos)
