"""Streamline (tractography) dataviews.

This module defines :class:`Tractogram`, a :class:`~cortex.dataset.views.Dataview`
subclass that holds a bundle of 3-D polylines (streamlines) plus optional
per-vertex/per-streamline scalar data and named groups, for rendering
alongside cortical surfaces in the pycortex WebGL viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from .braindata import _hash
from .views import Dataview

if TYPE_CHECKING:
    from trx.trx_file_memmap import TrxFile

#: Accepted spellings for the ``color`` display attribute.
ColorSpec = Union[str, tuple, list, npt.NDArray]


class Tractogram(Dataview):
    """Encapsulates a bundle of streamlines (a tractogram) for display.

    Unlike :class:`~cortex.dataset.views.Volume` and
    :class:`~cortex.dataset.views.Vertex`, a ``Tractogram`` is not backed by
    :class:`~cortex.dataset.braindata.BrainData`: its geometry is a flat
    array of 3-D points plus an offset table delimiting individual
    streamlines (the TRX convention), rather than a volume- or
    surface-sampled array.

    Parameters
    ----------
    points : (N, 3) array_like
        Streamline vertex coordinates, in the subject's fiducial-surface mm
        space (see the Notes below), concatenated across all streamlines.
        Cast to ``float32``.
    offsets : (M,) or (M+1,) array_like
        Index into `points` where each streamline starts. Following the TRX
        convention, this should have ``M + 1`` entries (`M` = number of
        streamlines) with a trailing sentinel equal to `N` (the number of
        points); if exactly `M` entries are given (no sentinel), the
        sentinel is appended automatically. Must be non-decreasing and start
        at 0.
    subject : str
        Subject identifier. Must exist in the pycortex database.
    dpv : dict[str, array_like], optional
        Data-per-vertex: for each name, an array with one entry (or row) per
        point, aligned with `points`.
    dps : dict[str, array_like], optional
        Data-per-streamline: for each name, an array with one entry (or row)
        per streamline.
    groups : dict[str, array_like of int], optional
        Named subsets of streamlines, given as arrays of streamline indices.
        Groups may overlap, and a streamline may belong to no group. Dict
        order is preserved and is significant: it is the order in which
        `groups_wire` concatenates them into the wire-format buffer.
    color : str or tuple, optional
        How to color the streamlines. One of:

        - ``"orientation"`` (default): color by the absolute value of the
          local (unit) tangent direction, the standard "directionally
          encoded color" tractography convention.
        - A ``(r, g, b)`` tuple (0-1 or 0-255 range): a constant color.
        - ``"dpv:<name>"`` or ``"dps:<name>"``: color by the named scalar
          data-per-vertex or data-per-streamline array, mapped through
          `cmap`/`vmin`/`vmax`.
    cmap : str, optional
        Colormap name used when `color` selects a scalar ``dpv``/``dps``
        array. Defaults to the pycortex default colormap.
    vmin : float, optional
        Minimum value of the colormap, when `color` selects a scalar array.
        Defaults to the ``nanmin`` of that array.
    vmax : float, optional
        Maximum value of the colormap, when `color` selects a scalar array.
        Defaults to the ``nanmax`` of that array.
    alpha : float, optional
        Opacity of the streamlines, in [0, 1]. Defaults to 1.0.
    linewidth : float, optional
        Line width in pixels (the WebGL viewer clamps this to 1px on most
        platforms). Defaults to 1.0.
    description : str, optional
        String describing this dataset. Displayed in the webgl viewer.
    **kwargs
        Additional arguments passed to :class:`~cortex.dataset.views.Dataview`
        (e.g. ``state``, ``priority``).

    Notes
    -----
    `points` must be expressed in the same mm space as the subject's
    fiducial surfaces (FreeSurfer scanner RAS mm; see `cortex.freesurfer`).
    Diffusion tractography output is often in a DWI/T1 RASmm space that
    already matches this when the DWI was registered to the T1 -- in that
    case no transform is needed. Use the `xfm` argument of :meth:`from_trx`
    to align data expressed in a different space (e.g. MNI).
    """

    def __init__(
        self,
        points: npt.ArrayLike,
        offsets: npt.ArrayLike,
        subject: str,
        *,
        # Mapping, not Dict: dict is invariant in its value type, so a plain
        # dict[str, ndarray] -- what every caller actually has -- would not
        # satisfy dict[str, ArrayLike].
        dpv: Optional[Mapping[str, npt.ArrayLike]] = None,
        dps: Optional[Mapping[str, npt.ArrayLike]] = None,
        groups: Optional[Mapping[str, npt.ArrayLike]] = None,
        color: ColorSpec = "orientation",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        alpha: float = 1.0,
        linewidth: float = 1.0,
        description: str = "",
        **kwargs: Any,
    ) -> None:
        points_arr = np.asarray(points, dtype=np.float32)
        if points_arr.ndim != 2 or points_arr.shape[1] != 3:
            raise ValueError(
                "points must have shape (N, 3), got %r" % (points_arr.shape,)
            )
        n_points = points_arr.shape[0]

        offsets_arr = np.asarray(offsets)
        if offsets_arr.ndim != 1:
            raise ValueError(
                "offsets must be 1-D, got shape %r" % (offsets_arr.shape,)
            )
        offsets_arr = offsets_arr.astype(np.int64)
        if offsets_arr.size == 0 or offsets_arr[-1] != n_points:
            # M offsets given without the trailing sentinel: append it.
            offsets_arr = np.concatenate([offsets_arr, [n_points]]).astype(np.int64)
        if offsets_arr.size == 0 or offsets_arr[0] != 0:
            raise ValueError("offsets must start at 0")
        if np.any(np.diff(offsets_arr) < 0):
            raise ValueError("offsets must be monotonically non-decreasing")

        self.points = points_arr
        self.offsets = offsets_arr
        n_streamlines = offsets_arr.shape[0] - 1

        self.subject = subject if isinstance(subject, str) else subject.decode("utf-8")

        self.dpv: Dict[str, npt.NDArray] = {}
        for name, arr in (dpv or {}).items():
            arr = np.asarray(arr)
            if arr.shape[0] != n_points:
                raise ValueError(
                    "dpv[%r] must have %d entries (one per point), got %d"
                    % (name, n_points, arr.shape[0])
                )
            self.dpv[name] = arr

        self.dps: Dict[str, npt.NDArray] = {}
        for name, arr in (dps or {}).items():
            arr = np.asarray(arr)
            if arr.shape[0] != n_streamlines:
                raise ValueError(
                    "dps[%r] must have %d entries (one per streamline), got %d"
                    % (name, n_streamlines, arr.shape[0])
                )
            self.dps[name] = arr

        self.groups: Dict[str, npt.NDArray] = {}
        for name, idx in (groups or {}).items():
            idx_arr = np.asarray(idx, dtype=np.int64)
            if idx_arr.size and (idx_arr.min() < 0 or idx_arr.max() >= n_streamlines):
                raise ValueError(
                    "groups[%r] contains streamline indices out of range "
                    "[0, %d)" % (name, n_streamlines)
                )
            self.groups[name] = idx_arr

        if isinstance(color, str):
            if color != "orientation":
                if color.startswith("dpv:"):
                    key = color[len("dpv:") :]
                    if key not in self.dpv:
                        raise ValueError("Unknown dpv field %r for color" % key)
                elif color.startswith("dps:"):
                    key = color[len("dps:") :]
                    if key not in self.dps:
                        raise ValueError("Unknown dps field %r for color" % key)
                else:
                    raise ValueError(
                        "Unrecognized color spec %r: must be 'orientation', "
                        "'dpv:<name>', 'dps:<name>', or an (r, g, b) tuple" % (color,)
                    )
        else:
            color = tuple(color)
            if len(color) != 3:
                raise ValueError(
                    "color tuple must have exactly 3 entries (r, g, b), got %d"
                    % len(color)
                )
        self.color: ColorSpec = color

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1], got %r" % (alpha,))
        if linewidth <= 0:
            raise ValueError("linewidth must be positive, got %r" % (linewidth,))
        self.alpha = float(alpha)
        self.linewidth = float(linewidth)

        super().__init__(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            **kwargs,
        )

    # -- basic geometry -----------------------------------------------

    @property
    def n_points(self) -> int:
        """Total number of points across all streamlines."""
        return self.points.shape[0]

    @property
    def n_streamlines(self) -> int:
        """Number of streamlines."""
        return self.offsets.shape[0] - 1

    def __len__(self) -> int:
        return self.n_streamlines

    @property
    def lengths(self) -> npt.NDArray[np.int64]:
        """Number of points in each streamline, shape ``(n_streamlines,)``."""
        return np.diff(self.offsets)

    @property
    def streamlines(self) -> list:
        """List of ``(L_i, 3)`` views into `points`, one per streamline."""
        return [
            self.points[start:stop]
            for start, stop in zip(self.offsets[:-1], self.offsets[1:])
        ]

    @property
    def name(self) -> str:
        """Name of this Tractogram, computed from a hash of `points`.

        Mirrors :attr:`cortex.dataset.braindata.BrainData.name`, so that
        code keying on that convention (e.g. `cortex.webgl.data.Package`)
        works unchanged for tractograms.
        """
        return "__%s" % _hash(self.points)[:16]

    # -- selection / decimation -----------------------------------------

    def select(self, indices: npt.ArrayLike) -> "Tractogram":
        """Return a new `Tractogram` containing only the given streamlines.

        Parameters
        ----------
        indices : array_like of int or bool
            Streamline indices to keep (or a boolean mask of length
            `n_streamlines`). Order is preserved; repeats are allowed.

        Returns
        -------
        Tractogram
            A new tractogram with `points`/`offsets`/`dpv`/`dps` sliced and
            `groups` remapped to the new streamline indexing. Display
            attributes (`color`, `cmap`, `vmin`, `vmax`, `alpha`,
            `linewidth`, `description`) are copied.
        """
        idx = np.asarray(indices)
        if idx.dtype == np.bool_:
            idx = np.nonzero(idx)[0]
        idx = idx.astype(np.int64)

        starts = self.offsets[:-1]
        stops = self.offsets[1:]

        point_chunks = []
        dpv_chunks: Dict[str, list] = {name: [] for name in self.dpv}
        new_lengths = []
        for i in idx:
            s, e = int(starts[i]), int(stops[i])
            point_chunks.append(self.points[s:e])
            for name in self.dpv:
                dpv_chunks[name].append(self.dpv[name][s:e])
            new_lengths.append(e - s)

        if point_chunks:
            new_points = np.concatenate(point_chunks, axis=0)
        else:
            new_points = np.zeros((0, 3), dtype=np.float32)
        new_offsets = np.concatenate([[0], np.cumsum(new_lengths)]).astype(np.int64)

        new_dpv = {}
        for name, chunks in dpv_chunks.items():
            if chunks:
                new_dpv[name] = np.concatenate(chunks, axis=0)
            else:
                new_dpv[name] = np.zeros(
                    (0,) + self.dpv[name].shape[1:], dtype=self.dpv[name].dtype
                )

        new_dps = {name: self.dps[name][idx] for name in self.dps}

        # A group keeps every new streamline whose old index was a member
        # (repeated indices in `idx` therefore stay in the group).
        new_groups = {}
        for name, members in self.groups.items():
            new_groups[name] = np.nonzero(np.isin(idx, members))[0].astype(np.int64)

        return Tractogram(
            new_points,
            new_offsets,
            self.subject,
            dpv=new_dpv,
            dps=new_dps,
            groups=new_groups,
            color=self.color,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            alpha=self.alpha,
            linewidth=self.linewidth,
            description=self.description,
            state=self.state,
            **self.attrs,
        )

    def get_group(self, name: str) -> "Tractogram":
        """Return a new `Tractogram` restricted to the named group.

        Parameters
        ----------
        name : str
            A key of `groups`.
        """
        if name not in self.groups:
            raise KeyError("Unknown group %r" % name)
        return self.select(self.groups[name])

    def subsample(
        self,
        max_streamlines: Optional[int] = None,
        step: Optional[int] = None,
        seed: int = 0,
    ) -> "Tractogram":
        """Return a decimated `Tractogram`, for faster interactive display.

        Parameters
        ----------
        max_streamlines : int, optional
            If given (and `step` is not), and there are more than this many
            streamlines, a random subset of this size is kept (order
            preserved), chosen with the given `seed`.
        step : int, optional
            If given, keep every `step`-th streamline (``indices[::step]``);
            takes precedence over `max_streamlines`.
        seed : int, optional
            Seed for the random subset chosen when `max_streamlines` is used.

        Returns
        -------
        Tractogram
        """
        n = self.n_streamlines
        if step is not None:
            indices = np.arange(0, n, step)
        elif max_streamlines is not None and n > max_streamlines:
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(n, size=max_streamlines, replace=False))
        else:
            indices = np.arange(n)
        return self.select(indices)

    # -- coloring ---------------------------------------------------------

    def vertex_colors(self) -> npt.NDArray[np.uint8]:
        """Compute per-point RGB colors according to `color`.

        Returns
        -------
        (N, 3) ndarray of uint8
            One RGB color per point in `points`.
        """
        color = self.color
        if isinstance(color, str):
            if color == "orientation":
                return self._orientation_colors()
            if color.startswith("dpv:"):
                name = color[len("dpv:") :]
                scalar = np.asarray(self.dpv[name], dtype=np.float64).reshape(
                    self.n_points, -1
                )[:, 0]
                return self._scalar_colors(scalar)
            if color.startswith("dps:"):
                name = color[len("dps:") :]
                per_streamline = np.asarray(self.dps[name], dtype=np.float64).reshape(
                    self.n_streamlines, -1
                )[:, 0]
                return self._scalar_colors(self._broadcast_dps(per_streamline))
            raise ValueError("Unrecognized color spec %r" % (color,))
        return self._constant_colors(color)

    def _orientation_colors(self) -> npt.NDArray[np.uint8]:
        # Vectorized over all points: central differences everywhere, then
        # the first/last point of every streamline is fixed up with a
        # one-sided difference (single-point streamlines get a zero tangent).
        pts = self.points.astype(np.float64)
        tangent = np.zeros_like(pts)
        if pts.shape[0] >= 3:
            tangent[1:-1] = pts[2:] - pts[:-2]
        lengths = self.lengths
        nonempty = lengths > 0
        starts = self.offsets[:-1][nonempty]
        ends = self.offsets[1:][nonempty] - 1
        multi = lengths[nonempty] > 1
        first, last = starts[multi], ends[multi]
        tangent[first] = pts[first + 1] - pts[first]
        tangent[last] = pts[last] - pts[last - 1]
        tangent[starts[~multi]] = 0.0
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        unit = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 0)
        return np.clip(np.abs(unit) * 255.0, 0, 255).astype(np.uint8)

    def _constant_colors(self, color: ColorSpec) -> npt.NDArray[np.uint8]:
        rgb = np.asarray(color, dtype=np.float64)
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.tile(rgb, (self.n_points, 1))

    def _broadcast_dps(self, values: npt.NDArray) -> npt.NDArray[np.float64]:
        return np.repeat(np.asarray(values, dtype=np.float64), self.lengths)

    def _scalar_colors(self, scalar: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
        from matplotlib import cm, colors as mcolors

        nan_mask = np.isnan(scalar)
        finite = scalar[~nan_mask]
        vmin = self.vmin if self.vmin is not None else (
            float(np.nanmin(finite)) if finite.size else 0.0
        )
        vmax = self.vmax if self.vmax is not None else (
            float(np.nanmax(finite)) if finite.size else 1.0
        )
        cmap = self.get_cmapdict()["cmap"]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgba = mappable.to_rgba(np.nan_to_num(scalar))
        rgb = (np.clip(rgba[:, :3], 0, 1) * 255.0).astype(np.uint8)
        rgb[nan_mask] = 128
        return rgb

    # -- serialization ------------------------------------------------

    def groups_wire(self) -> "tuple[npt.NDArray[np.uint32], Dict[str, tuple]]":
        """Concatenate `groups` into one buffer, for the wire format.

        Returns
        -------
        indices : (K,) ndarray of uint32
            Concatenation of every entry of `groups`, in `groups` dict
            order, as streamline indices. Empty when `groups` is empty.
        slices : dict[str, tuple[int, int]]
            For each group name (in the same order as `indices`), the
            ``(start, stop)`` slice bounds into `indices` holding that
            group's streamline indices (``stop - start`` equals
            ``len(groups[name])``).
        """
        chunks = []
        slices: Dict[str, tuple] = {}
        pos = 0
        for name, idx in self.groups.items():
            idx_arr = np.asarray(idx, dtype=np.uint32)
            chunks.append(idx_arr)
            slices[name] = (pos, pos + idx_arr.shape[0])
            pos += idx_arr.shape[0]
        if chunks:
            indices = np.concatenate(chunks).astype(np.uint32)
        else:
            indices = np.zeros((0,), dtype=np.uint32)
        return indices, slices

    # The base class narrows this to DataviewJSON, a TypedDict describing a
    # colormapped brain-data payload. A tractogram's metadata shares almost
    # none of those keys (see below) and TypedDicts admit no extra ones, so
    # the override is deliberate rather than an oversight.
    def to_json(self, simple: bool = False) -> dict:  # type: ignore[override]
        """Return the wire-format metadata dict for this tractogram.

        Parameters
        ----------
        simple : bool, optional
            If True, omit the colormap-related keys (`cmap`/`vmin`/`vmax`)
            even when `color` selects a scalar `dpv`/`dps` array.

        Notes
        -----
        ``result["groups"]`` maps each group name to a ``[start, stop]``
        slice (not a count): these are bounds into the fourth per-tractogram
        wire buffer, ``groups`` (little-endian uint32 streamline indices,
        the concatenation of every group in dict order -- see
        :meth:`groups_wire`, which `cortex.webgl.data.Package` uses to build
        that buffer so the two stay in sync).
        """
        if isinstance(self.color, str):
            color_repr = self.color
        else:
            color_repr = str(tuple(float(c) for c in self.color))

        _, group_slices = self.groups_wire()
        result: dict = {
            "subject": self.subject,
            "n_points": int(self.n_points),
            "n_streamlines": int(self.n_streamlines),
            "alpha": self.alpha,
            "linewidth": self.linewidth,
            "visible": True,
            "color": color_repr,
            "groups": {
                name: [int(start), int(stop)]
                for name, (start, stop) in group_slices.items()
            },
            "description": self.description,
        }

        is_scalar_mode = isinstance(self.color, str) and (
            self.color.startswith("dpv:") or self.color.startswith("dps:")
        )
        if not simple and is_scalar_mode:
            result["cmap"] = self.cmap
            result["vmin"] = self.vmin
            result["vmax"] = self.vmax
        return result

    def _write_hdf(self, h5, name: str = "data", data=None, xfmname=None):
        raise NotImplementedError(
            "HDF5 persistence of Tractogram is not implemented yet"
        )

    # -- construction from TRX / raw streamlines --------------------------

    @classmethod
    def from_trx(
        cls,
        path_or_trxfile: Union[str, Path, "TrxFile"],
        subject: str,
        xfm: Optional[npt.ArrayLike] = None,
        **display: Any,
    ) -> "Tractogram":
        """Build a `Tractogram` from a TRX file (via `trx-python`).

        Parameters
        ----------
        path_or_trxfile : str, Path, or trx.trx_file_memmap.TrxFile
            Path to a ``.trx`` file, or an already-loaded `TrxFile`. When a
            path is given, `trx.io.load` is used to load it, and the file is
            closed before returning.
        subject : str
            Subject identifier. Must exist in the pycortex database.
        xfm : (4, 4) array_like, optional
            Affine applied to the streamline points (as homogeneous
            coordinates) after loading, to align data expressed in a
            different RASmm space onto the subject's fiducial surfaces.
        **display
            Additional display keyword arguments forwarded to the
            `Tractogram` constructor (`color`, `cmap`, `vmin`, `vmax`,
            `alpha`, `linewidth`, `description`, ...).

        Returns
        -------
        Tractogram
        """
        close_after = False
        if isinstance(path_or_trxfile, (str, Path)):
            from trx.io import load as trx_load

            trx_file = trx_load(str(path_or_trxfile))
            close_after = True
        else:
            trx_file = path_or_trxfile

        try:
            streamlines = trx_file.streamlines
            points = np.array(streamlines._data, dtype=np.float32, copy=True)
            n_points = points.shape[0]
            raw_offsets = np.array(streamlines._offsets, copy=True).astype(np.int64)
            if raw_offsets.size == 0 or raw_offsets[-1] != n_points:
                offsets = np.concatenate([raw_offsets, [n_points]]).astype(np.int64)
            else:
                offsets = raw_offsets

            def _extract(value: Any) -> npt.NDArray:
                # trx-python memmaps dpv/dps as (N, k); a single scalar per
                # entry comes back as (N, 1), which we flatten to (N,).
                data = np.array(getattr(value, "_data", value), copy=True)
                if data.ndim == 2 and data.shape[1] == 1:
                    data = data[:, 0]
                return data

            dpv = {
                name: _extract(arr) for name, arr in trx_file.data_per_vertex.items()
            }
            dps = {
                name: _extract(arr)
                for name, arr in trx_file.data_per_streamline.items()
            }
            groups = {
                name: np.array(idx, dtype=np.int64, copy=True)
                for name, idx in trx_file.groups.items()
            }

            if xfm is not None:
                xfm_arr = np.asarray(xfm, dtype=np.float64)
                homogeneous = np.concatenate(
                    [points.astype(np.float64), np.ones((n_points, 1))], axis=1
                )
                points = (homogeneous @ xfm_arr.T)[:, :3].astype(np.float32)

            return cls(
                points, offsets, subject, dpv=dpv, dps=dps, groups=groups, **display
            )
        finally:
            if close_after:
                trx_file.close()

    @classmethod
    def from_streamlines(
        cls,
        streamlines: Sequence[npt.ArrayLike],
        subject: str,
        **kwargs: Any,
    ) -> "Tractogram":
        """Build a `Tractogram` from a list of ``(L_i, 3)`` point arrays.

        Parameters
        ----------
        streamlines : sequence of array_like
            Each element is an ``(L_i, 3)`` array of points for one
            streamline.
        subject : str
            Subject identifier. Must exist in the pycortex database.
        **kwargs
            Additional keyword arguments forwarded to the `Tractogram`
            constructor.
        """
        arrays = [np.asarray(s, dtype=np.float32) for s in streamlines]
        if not arrays:
            points = np.zeros((0, 3), dtype=np.float32)
            offsets = np.array([0], dtype=np.int64)
        else:
            lengths = [a.shape[0] for a in arrays]
            points = np.concatenate(arrays, axis=0).astype(np.float32)
            offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        return cls(points, offsets, subject, **kwargs)
