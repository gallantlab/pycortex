import abc
from typing import Union, cast, overload

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .. import dataset

import sys
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import warnings

warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)

class Mapper(abc.ABC):
    '''Maps data from epi volume onto surface using various projections'''
    def __init__(self, left: sparse.csr_matrix, right: sparse.csr_matrix, shape: npt.NDArray[np.integer], subject: str, xfmname: str):
        self.idxmap = None
        self.masks = [left, right]
        self.nverts = left.shape[0] + right.shape[0]
        self.shape = shape
        self.subject = subject
        self.xfmname = xfmname

    @classmethod
    def from_cache(cls, cachefile: str, subject: str, xfmname: str) -> Self:
        npz = np.load(cachefile)
        left = (npz['left_data'], npz['left_indices'], npz['left_indptr'])
        right = (npz['right_data'], npz['right_indices'], npz['right_indptr'])
        lsparse = sparse.csr_matrix(left, shape=npz['left_shape'])
        rsparse = sparse.csr_matrix(right, shape=npz['right_shape'])
        return cls(lsparse, rsparse, npz['shape'], subject, xfmname)

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        mask = np.array(self.masks[0].sum(0) + self.masks[1].sum(0))
        return (mask.squeeze() != 0).reshape(self.shape)

    @property
    def hemimasks(self) -> list[npt.NDArray[np.bool_]]:
        func = lambda m: (np.array(m.sum(0)).squeeze() != 0).reshape(self.shape)
        return [func(x) for x in self.masks]

    def __repr__(self):
        ptype = self.__class__.__name__
        return '<%s mapper with %d vertices>'%(ptype, self.nverts)

    @overload
    def __call__(self, data: dataset.Volume) -> dataset.Vertex: ...

    @overload
    def __call__(self, data: dataset.Vertex) -> tuple[npt.NDArray, npt.NDArray]: ...

    def __call__(
        self, data: Union[dataset.Vertex, dataset.Volume, tuple]
    ) -> Union[tuple[npt.NDArray, npt.NDArray], dataset.Vertex]:
        """Project a Volume onto the surface, or split a Vertex per hemisphere.

        The overloads above are the typed contract: this maps between the two
        view classes it knows, ``Volume`` in and ``Vertex`` out. A 3-tuple of
        ``(data, subject, xfmname)`` is still accepted at runtime as a shorthand
        for the ``Volume`` it constructs, but is not part of that contract --
        build the ``Volume`` at the call site instead.
        """
        if isinstance(data, tuple):
            data = dataset.Volume(*data)

        if isinstance(data, dataset.Vertex):
            llen = self.masks[0].shape[0]
            if data.raw:
                left, right = data.data[..., :llen,:], data.data[..., llen:,:]
                if self.idxmap is not None:
                    left = left[..., self.idxmap[0], :]
                    right = right[..., self.idxmap[1], :]
            else:
                left, right = data[..., :llen], data[..., llen:]
                if self.idxmap is not None:
                    left = left[..., self.idxmap[0]]
                    right = right[..., self.idxmap[1]]
            # `Vertex.raw` is a property that builds a VertexRGB, so it is always
            # truthy and only the first branch above runs -- which is the one that
            # yields arrays. The second indexes the Vertex itself, so it yields
            # Vertex objects and contradicts the overload. Cast rather than
            # silently widen the contract; see KNOWN_ISSUES.md.
            return cast(tuple[npt.NDArray, npt.NDArray], (left, right))

        volume = np.ascontiguousarray(data.volume)
        volume.shape = len(volume), -1
        volume = volume.T

        mapped: list[npt.NDArray] = []
        for mask in self.masks:
            mapped.append(np.array(mask * volume).T) # change to @ matmul

        if self.idxmap is not None:
            mapped[0] = mapped[0][:, self.idxmap[0]]
            mapped[1] = mapped[1][:, self.idxmap[1]]

        return dataset.Vertex(np.hstack(mapped).squeeze(), data.subject)

    @overload
    def backwards(self, vertexdata: dataset.Vertex) -> dataset.Volume: ...

    @overload
    def backwards(self, vertexdata: npt.NDArray) -> npt.NDArray: ...

    def backwards(
        self, vertexdata: Union[dataset.Vertex, npt.NDArray]
    ) -> Union[dataset.Volume, npt.NDArray]:
        '''Projects vertex data back into volume space.

        The view direction is ``Vertex`` in, ``Volume`` out -- the inverse of
        :meth:`__call__`. A bare array is also accepted and returns a bare array,
        for callers that have per-vertex values without a subject to attach them
        to; ``cortex.utils`` projects ROI masks that way.

        Parameters
        ----------
        vertexdata : Vertex or ndarray
            Per-vertex values to project back into voxel space.
        '''
        # `isinstance` on the parameter rather than a bool flag, so that the two
        # returns below are each checked against the overload they satisfy.
        if isinstance(vertexdata, dataset.Vertex):
            to_map: npt.NDArray = vertexdata.data
        else:
            to_map = vertexdata
        # stack the two mappers together
        bothmappers = sparse.vstack(self.masks)
        # dot the vertex data with the stacked mappers
        partial_vertex = bothmappers.T.dot(to_map)
        # solve the inverse mapping problem
        voxeldata: npt.NDArray = self._get_backmapper().solve(partial_vertex).reshape(self.shape)
        if isinstance(vertexdata, dataset.Vertex):
            return dataset.Volume(voxeldata, self.subject, self.xfmname)
        return voxeldata

    def _get_backmapper(self):
        if not hasattr(self, '_backmapper'):
            # stack the two mappers together to get one voxel -> vertex mapper
            bothmappers = sparse.vstack(self.masks)
            # take inner product to get symmetric matrix
            symmappers = bothmappers.T.dot(bothmappers)
            # add (very) small diagonal to make sure it's full rank
            symmappers_reg = symmappers + 1e-9 * sparse.eye(symmappers.shape[0])
            # factorize it using splu so that inversion is fast
            self._backmapper = sparse.linalg.splu(symmappers_reg)

        return self._backmapper

    @classmethod
    def _cache(cls, filename: str, subject: str, xfmname: str, **kwargs) -> Self:
        print('Caching mapper...')
        from ..database import db
        masks: list[sparse.csr_matrix] = []
        xfm = db.get_xfm(subject, xfmname, xfmtype='coord')
        fid = db.get_surf(subject, 'fiducial', merge=False, nudge=False)

        try:
            flat = db.get_surf(subject, 'flat', merge=False, nudge=False)
        except IOError:
            flat = fid

        for (pts, _), (_, polys) in zip(fid, flat):
            masks.append(cls._getmask(xfm(pts), polys, xfm.shape, **kwargs))

        _savecache(filename, masks[0], masks[1], xfm.shape)
        return cls(masks[0], masks[1], xfm.shape, subject, xfmname)

    @classmethod
    @abc.abstractmethod
    def _getmask(cls, coords: npt.NDArray[np.floating], polys: npt.NDArray[np.intp], shape: tuple[int, int, int], **kwargs) -> sparse.csr_matrix:
        '''Generates a sparse matrix mapping from volume to surface vertices'''
        pass

    @staticmethod
    @abc.abstractmethod
    def sampler(coords: npt.NDArray[np.floating], shape: tuple[int, int, int], **kwargs) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.floating]]:
        '''Generates a sparse matrix mapping from volume to surface vertices'''
        pass

def _savecache(filename: str, left: sparse.csr_matrix, right: sparse.csr_matrix, shape: npt.NDArray[np.integer]) -> None:
    np.savez(filename,
             left_data=left.data,
             left_indices=left.indices,
             left_indptr=left.indptr,
             left_shape=left.shape,
             right_data=right.data,
             right_indices=right.indices,
             right_indptr=right.indptr,
             right_shape=right.shape,
             shape=shape)
