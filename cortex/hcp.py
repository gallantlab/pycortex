"""Interfacing with HCP fs_LR 32k data.

This module lets you visualize HCP data with pycortex, both natively on the
fs_LR 32k mesh (via the ``32k_fs_LR`` pycortex subject) and projected to
``fsaverage``.

It provides three functions:

1. :func:`download_fs_lr` -- fetch the prebuilt ``32k_fs_LR`` pycortex subject
   into the filestore (a thin wrapper around :func:`cortex.download_subject`),
   so HCP data can be rendered with the usual
   ``cortex.Vertex(data, "32k_fs_LR")`` + ``cortex.quickshow`` path.
2. :func:`cifti_to_surface` -- expand CIFTI grayordinate data (59412 cortical
   vertices, no medial wall) to the full 64984-vertex fs_LR 32k surface, filling
   the medial wall with NaN.
3. :func:`to_fsaverage` -- resample fs_LR 32k data to ``fsaverage`` (the
   lower-level :func:`project_fslr_to_fsaverage` does the surface-to-surface
   step). The projection matrix is built directly from the HCP standard-mesh
   spheres using spherical barycentric interpolation -- the same interpolation
   ``wb_command -metric-resample ... BARYCENTRIC`` performs -- so no Connectome
   Workbench installation is required at runtime.

Notes
-----
The ``fs_LR-deformed_to-fsaverage`` source sphere and the ``fsaverageN``
standard spheres live in the same spherical-registration space, so a target
vertex can be located inside a source-mesh triangle directly (see
:func:`_barycentric_resample_matrix`). This mirrors, and reuses the caching
conventions of, :func:`cortex.freesurfer.get_mri_surf2surf_matrix`.
"""

import logging
from pathlib import Path
from typing import Literal, Optional, Union, cast
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import nibabel as nib
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix, load_npz, save_npz
from scipy.spatial import KDTree

from cortex import appdirs
from cortex.freesurfer import upsample_to_fsaverage
from cortex.utils import download_subject

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Name of the pycortex subject shipping the HCP fs_LR 32k surfaces.
SUBJECT = "32k_fs_LR"

#: Vertices per hemisphere on the fs_LR 32k surface.
N_VERTICES_FS_LR_32K_HEM = 32492
#: Vertices on both fs_LR 32k hemispheres (full surface, includes medial wall).
N_VERTICES_FS_LR_32K = 2 * N_VERTICES_FS_LR_32K_HEM  # 64984
#: Cortical grayordinates in a standard HCP CIFTI file (no medial wall).
N_VERTICES_CIFTI_CORTEX = 59412

#: Vertices per hemisphere on the target fsaverage meshes.
N_VERTICES_TARGET_HEM = {
    "fsaverage5": 10242,
    "fsaverage6": 40962,
}

# ---------------------------------------------------------------------------
# Sphere file download helpers
# ---------------------------------------------------------------------------

_SPHERE_FILES_URL = (
    "https://raw.githubusercontent.com/Washington-University/HCPpipelines/"
    "master/global/templates/standard_mesh_atlases/resample_fsaverage/"
)

#: Standard-mesh sphere GIFTIs, keyed by space then hemisphere ("L"/"R").
_SPHERE_FILES = {
    "fs_LR_32k": {
        "L": "fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii",
        "R": "fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii",
    },
    "fsaverage5": {
        "L": "fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii",
        "R": "fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii",
    },
    "fsaverage6": {
        "L": "fsaverage6_std_sphere.L.41k_fsavg_L.surf.gii",
        "R": "fsaverage6_std_sphere.R.41k_fsavg_R.surf.gii",
    },
}


_HEMI_ALIASES: dict[str, Literal["L", "R"]] = {
    "L": "L",
    "lh": "L",
    "R": "R",
    "rh": "R",
}


def _normalize_hemi(hemi: str) -> Literal["L", "R"]:
    """Normalize a hemisphere name to ``"L"``/``"R"`` (accepts ``lh``/``rh``)."""
    try:
        return _HEMI_ALIASES[hemi]
    except (KeyError, TypeError):
        raise ValueError(
            f"hemi must be 'L'/'lh' or 'R'/'rh', got {hemi!r}"
        ) from None


def _default_cache_dir() -> Path:
    """Directory for downloaded spheres and cached projection matrices."""
    return Path(appdirs.user_cache_dir("pycortex")) / "hcp"


def ensure_sphere_files(
    space: str,
    hemi: str,
    cache_dir: Optional[Union[str, Path]] = None,
    download: bool = True,
) -> Path:
    """Return the path to a standard-mesh sphere GIFTI, downloading if needed.

    Parameters
    ----------
    space : {"fs_LR_32k", "fsaverage5", "fsaverage6"}
        Surface space of the sphere.
    hemi : {"L", "R"}
        Hemisphere.
    cache_dir : str or Path or None
        Directory to store sphere files. Defaults to a pycortex user cache dir.
    download : bool
        If True (default), download the file from the HCPpipelines repository
        when it is missing.

    Returns
    -------
    path : pathlib.Path
        Path to the sphere ``.surf.gii`` file.
    """
    if space not in _SPHERE_FILES:
        raise ValueError(
            f"Unknown space {space!r}; choose from {sorted(_SPHERE_FILES)}"
        )
    hemi = _normalize_hemi(hemi)

    if cache_dir is None:
        cache_dir = _default_cache_dir()
    atlas_dir = Path(cache_dir) / "standard_mesh_atlases"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    fname = _SPHERE_FILES[space][hemi]
    fpath = atlas_dir / fname
    if fpath.exists():
        return fpath
    if not download:
        raise FileNotFoundError(
            f"Sphere file not found: {fpath}. Set download=True to fetch it."
        )

    url = _SPHERE_FILES_URL + fname
    logger.info("Downloading %s -> %s", url, fpath)
    tmp_path = fpath.with_suffix(".tmp")
    try:
        urlretrieve(url, tmp_path)
    except (URLError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download sphere file from {url}. Check your internet "
            f"connection or download it manually to {fpath}."
        ) from exc
    if tmp_path.stat().st_size == 0:
        tmp_path.unlink()
        raise RuntimeError(f"Downloaded file is empty: {url}")
    tmp_path.rename(fpath)
    return fpath


# ---------------------------------------------------------------------------
# Subject download
# ---------------------------------------------------------------------------


def download_fs_lr(
    pycortex_store: Optional[str] = None, download_again: bool = False
) -> None:
    """Download the ``32k_fs_LR`` pycortex subject into the filestore.

    Thin wrapper around :func:`cortex.download_subject` for the HCP fs_LR 32k
    surfaces. Once present, HCP data can be rendered natively with
    ``cortex.Vertex(data, "32k_fs_LR")`` and ``cortex.quickshow``.

    Parameters
    ----------
    pycortex_store : str or None
        Directory to place the subject folder. If None, uses the current
        filestore (``cortex.db.filestore``).
    download_again : bool
        Re-download even if the subject is already present.

    Notes
    -----
    The ``32k_fs_LR`` subject is derived from HCP S1200 group-average Open
    Access surfaces and is redistributed under the WU-Minn HCP Consortium Open
    Access Data Use Terms; use of it requires the standard HCP acknowledgment.
    """
    download_subject(
        subject_id=SUBJECT, pycortex_store=pycortex_store, download_again=download_again
    )


# ---------------------------------------------------------------------------
# CIFTI -> fs_LR 32k surface expansion
# ---------------------------------------------------------------------------


def get_cifti_vertex_indices(
    cifti: Union[str, Path, nib.Cifti2Image],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """Left/right surface-vertex indices for a CIFTI file's cortical models.

    HCP CIFTI files store only non-medial-wall cortical grayordinates. This
    reads, from the CIFTI brain-model axis, which fs_LR 32k surface vertices
    each hemisphere's grayordinates correspond to.

    Parameters
    ----------
    cifti : str or Path or nibabel.Cifti2Image
        A CIFTI-2 file (``.dscalar.nii``/``.dtseries.nii``/...) or loaded image.

    Returns
    -------
    left_indices : ndarray, shape (n_left,)
        Surface vertex indices (into a 32492-vertex hemisphere) for the left
        cortex grayordinates.
    right_indices : ndarray, shape (n_right,)
        Same for the right cortex.
    """
    if isinstance(cifti, (str, Path)):
        cifti = nib.load(str(cifti))
    if not isinstance(cifti, nib.Cifti2Image):
        raise TypeError("cifti must be a Cifti2Image or a path to a CIFTI-2 file.")

    # Find the brain-model axis (the one carrying the grayordinate structures).
    bm_axis = None
    for i in range(cifti.ndim):
        axis = cifti.header.get_axis(i)
        if isinstance(axis, nib.cifti2.BrainModelAxis):
            bm_axis = axis
            break
    if bm_axis is None:
        raise ValueError("CIFTI file has no BrainModelAxis (grayordinate axis).")

    indices: dict[str, npt.NDArray[np.integer]] = {}
    for name, data_slice, model in bm_axis.iter_structures():
        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            indices["L"] = np.asarray(model.vertex)
        elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            indices["R"] = np.asarray(model.vertex)
    if "L" not in indices and "R" not in indices:
        raise ValueError("CIFTI file is missing both left and right cortex structures.")
    # Tolerate hemispherically split files: a missing hemisphere is an empty
    # index set, which cifti_to_surface / project_fslr_to_fsaverage handle by
    # leaving that hemisphere as NaN.
    empty = np.array([], dtype=np.int64)
    return indices.get("L", empty), indices.get("R", empty)


def cifti_to_surface(
    data: npt.ArrayLike,
    left_indices: npt.ArrayLike,
    right_indices: npt.ArrayLike,
) -> npt.NDArray:
    """Expand CIFTI cortical grayordinates onto the full fs_LR 32k surface.

    Parameters
    ----------
    data : ndarray, shape (n_grayordinates,) or (n_samples, n_grayordinates)
        Cortical CIFTI data (59412 vertices for standard HCP files). Only the
        cortical grayordinates are used; the left hemisphere's values come
        first, followed by the right hemisphere's.
    left_indices, right_indices : ndarray
        Surface vertex indices per hemisphere, e.g. from
        :func:`get_cifti_vertex_indices`.

    Returns
    -------
    full_data : ndarray, shape (64984,) or (n_samples, 64984)
        Data on the full fs_LR 32k surface, with NaN at medial-wall vertices.
    """
    data = np.asarray(data)
    left_indices = np.asarray(left_indices)
    right_indices = np.asarray(right_indices)
    n_left = len(left_indices)
    n_right = len(right_indices)
    right_offset = N_VERTICES_FS_LR_32K_HEM

    n_grayordinates = n_left + n_right
    if data.shape[-1] != n_grayordinates:
        raise ValueError(
            f"data has {data.shape[-1]} grayordinates on its last axis, but the "
            f"indices specify {n_grayordinates} ({n_left} left + {n_right} right)."
        )

    # Ellipsis indexing supports 1-D, 2-D, and higher-dimensional inputs
    # (e.g. subjects x time x grayordinates).
    out_dtype = np.result_type(data.dtype, np.float32)
    full = np.full(data.shape[:-1] + (N_VERTICES_FS_LR_32K,), np.nan, dtype=out_dtype)
    full[..., left_indices] = data[..., :n_left]
    full[..., right_offset + right_indices] = data[..., n_left : n_left + n_right]
    return full


# ---------------------------------------------------------------------------
# Spherical barycentric resampling matrix
# ---------------------------------------------------------------------------


def _read_sphere(
    path: Union[str, Path],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return (points, triangles) from a sphere ``.surf.gii`` file."""
    gii = cast(nib.GiftiImage, nib.load(str(path)))
    pts = gii.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data
    tris = gii.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")[0].data
    return np.asarray(pts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def _vertex_to_triangles(
    triangles: npt.NDArray[np.integer], n_vertices: int
) -> list[list[int]]:
    """List of incident triangle indices for each vertex."""
    incident: list[list[int]] = [[] for _ in range(n_vertices)]
    for ti, tri in enumerate(triangles):
        for v in tri:
            incident[v].append(ti)
    return incident


def _ray_triangle_bary(
    direction: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    tol: float = 1e-6,
) -> Optional[tuple[float, float, float]]:
    """Barycentric weights where the ray ``t*direction`` (from 0) hits triangle abc.

    Uses the Moller-Trumbore intersection. Returns ``(wa, wb, wc)`` summing to 1
    if the central projection of ``direction`` onto the plane of ``abc`` lies
    inside the triangle (within ``tol``), else ``None``. Because candidate
    triangles are restricted to those incident to the target's nearest source
    vertices, only the correct near-side triangle is ever tested.
    """
    edge1 = b - a
    edge2 = c - a
    pvec = np.cross(direction, edge2)
    det = np.dot(edge1, pvec)
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    tvec = -a
    u = np.dot(tvec, pvec) * inv_det
    if u < -tol or u > 1.0 + tol:
        return None
    qvec = np.cross(tvec, edge1)
    v = np.dot(direction, qvec) * inv_det
    if v < -tol or u + v > 1.0 + tol:
        return None
    w = 1.0 - u - v
    # Clamp tiny negative weights from edge/vertex hits, then renormalize.
    weights = np.array([w, u, v], dtype=np.float64)
    weights[weights < 0] = 0.0
    total = weights.sum()
    if total <= 0:
        return None
    return tuple(weights / total)


def _barycentric_resample_matrix(
    src_sphere: Union[str, Path], tgt_sphere: Union[str, Path], k: int = 15
) -> csr_matrix:
    """Build a sparse barycentric resampling matrix between two spheres.

    For every target-sphere vertex, the containing triangle on the *source*
    mesh is located and the target row is set to that triangle's three
    barycentric weights -- exactly the interpolation
    ``wb_command -metric-resample ... BARYCENTRIC`` performs.

    Parameters
    ----------
    src_sphere, tgt_sphere : str or Path
        Paths to the source and target sphere ``.surf.gii`` files. Both must be
        in the same spherical-registration space.
    k : int
        Number of nearest source vertices whose incident triangles are searched
        for the one containing each target vertex.

    Returns
    -------
    matrix : scipy.sparse.csr_matrix, shape (n_target, n_source)
        Apply with ``target_data = matrix.dot(source_data)``. Rows sum to 1.
    """
    src_pts, src_tris = _read_sphere(src_sphere)
    tgt_pts, _ = _read_sphere(tgt_sphere)
    n_src = len(src_pts)
    n_tgt = len(tgt_pts)

    # Project onto the unit sphere so the ray directions are well-conditioned.
    src_unit = src_pts / np.linalg.norm(src_pts, axis=1, keepdims=True)
    tgt_unit = tgt_pts / np.linalg.norm(tgt_pts, axis=1, keepdims=True)

    incident = _vertex_to_triangles(src_tris, n_src)
    tree = KDTree(src_unit)
    k = min(k, n_src)
    _, nn = tree.query(tgt_unit, k=k)
    nn = cast(npt.NDArray[np.integer], nn)
    if nn.ndim == 1:
        nn = nn[:, np.newaxis]

    rows = np.empty(3 * n_tgt, dtype=np.int64)
    cols = np.empty(3 * n_tgt, dtype=np.int64)
    vals = np.empty(3 * n_tgt, dtype=np.float64)
    n_entries = 0
    n_fallback = 0

    for t in range(n_tgt):
        direction = tgt_unit[t]
        found: Optional[tuple[int, int, int, tuple[float, float, float]]] = None
        seen: set[int] = set()
        for v in nn[t]:
            for ti in incident[v]:
                if ti in seen:
                    continue
                seen.add(ti)
                ia, ib, ic = src_tris[ti]
                w = _ray_triangle_bary(
                    direction, src_unit[ia], src_unit[ib], src_unit[ic]
                )
                if w is not None:
                    found = (ia, ib, ic, w)
                    break
            if found is not None:
                break

        if found is None:
            # No containing triangle among candidates: assign nearest vertex.
            n_fallback += 1
            va = int(nn[t][0])
            rows[n_entries] = t
            cols[n_entries] = va
            vals[n_entries] = 1.0
            n_entries += 1
            continue

        ia, ib, ic, (wa, wb, wc) = found
        rows[n_entries : n_entries + 3] = t
        cols[n_entries : n_entries + 3] = (ia, ib, ic)
        vals[n_entries : n_entries + 3] = (wa, wb, wc)
        n_entries += 3

    if n_fallback:
        logger.warning(
            "barycentric resample: %d/%d target vertices had no containing "
            "source triangle within k=%d neighbours; used nearest-vertex "
            "fallback.",
            n_fallback,
            n_tgt,
            k,
        )

    matrix = coo_matrix(
        (vals[:n_entries], (rows[:n_entries], cols[:n_entries])), shape=(n_tgt, n_src)
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def get_fslr_to_fsaverage_matrix(
    hemi: str,
    target: str = "fsaverage6",
    cache_dir: Optional[Union[str, Path]] = None,
    cache: bool = True,
) -> csr_matrix:
    """Sparse fs_LR 32k -> fsaverage matrix for one hemisphere.

    Parameters
    ----------
    hemi : {"L", "R"}
        Hemisphere.
    target : {"fsaverage5", "fsaverage6"}
        Target fsaverage density. Use :func:`project_fslr_to_fsaverage` with
        ``target="fsaverage"`` to reach the full-resolution surface.
    cache_dir : str or Path or None
        Directory for the sphere files and matrix cache. Defaults to a pycortex
        user cache dir.
    cache : bool
        If True (default), load/save the computed matrix as an ``.npz`` file.

    Returns
    -------
    matrix : scipy.sparse.csr_matrix, shape (n_target_hemi, 32492)
    """
    hemi = _normalize_hemi(hemi)
    if target not in N_VERTICES_TARGET_HEM:
        raise ValueError(
            f"target must be one of {sorted(N_VERTICES_TARGET_HEM)}, got {target!r}"
        )

    if cache_dir is None:
        cache_dir = _default_cache_dir()
    cache_dir = Path(cache_dir)

    cache_path = cache_dir / "mappers" / f"{hemi}_fs_LR_32k_to_{target}.npz"
    if cache and cache_path.exists():
        logger.info("Loading cached matrix from %s", cache_path)
        return cast(csr_matrix, load_npz(str(cache_path)))

    src_sphere = ensure_sphere_files("fs_LR_32k", hemi, cache_dir=cache_dir)
    tgt_sphere = ensure_sphere_files(target, hemi, cache_dir=cache_dir)
    matrix = _barycentric_resample_matrix(src_sphere, tgt_sphere)

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(str(cache_path), matrix)
        logger.info("Saved matrix to %s", cache_path)
    return matrix


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _project_hemi(
    matrix: csr_matrix, hemi_data: npt.NDArray, nanmean: bool
) -> npt.NDArray:
    """Apply one hemisphere's resampling matrix with NaN-aware weighting."""
    nan_mask = np.isnan(hemi_data)
    hemi_clean = np.where(nan_mask, 0.0, hemi_data)
    projected = (matrix @ hemi_clean.T).T

    # Weight actually contributed by valid (non-NaN) source vertices. Matrix
    # weights are non-negative (barycentric weights / unit fallback), so no abs.
    valid_weight = (matrix @ (~nan_mask).astype(np.float64).T).T
    all_nan = valid_weight < 1e-10
    if nanmean:
        with np.errstate(invalid="ignore", divide="ignore"):
            projected = np.where(all_nan, np.nan, projected / valid_weight)
    else:
        projected[all_nan] = np.nan
    return projected


def project_fslr_to_fsaverage(
    data: npt.ArrayLike,
    target: str = "fsaverage",
    cache_dir: Optional[Union[str, Path]] = None,
    freesurfer_subjects_dir: Optional[str] = None,
    nanmean: bool = True,
) -> npt.NDArray:
    """Project fs_LR 32k data to an fsaverage surface.

    Parameters
    ----------
    data : ndarray, shape (64984,) or (n_samples, 64984)
        Concatenated left+right data on the full fs_LR 32k surface (medial wall
        included; NaN there is fine). Use :func:`cifti_to_surface` to expand
        CIFTI grayordinate data to this shape.
    target : {"fsaverage", "fsaverage6", "fsaverage5"}
        Destination surface. ``"fsaverage"`` (full, 327684 vertices) is reached
        by projecting to fsaverage6 and then upsampling with
        :func:`cortex.freesurfer.upsample_to_fsaverage`.
    cache_dir : str or Path or None
        Directory for sphere files and matrix caches.
    freesurfer_subjects_dir : str or None
        FreeSurfer ``SUBJECTS_DIR`` containing ``fsaverage``/``fsaverage6``.
        Only required for ``target="fsaverage"``. If None, uses ``$SUBJECTS_DIR``.
    nanmean : bool
        If True (default), renormalize weights to exclude NaN sources so medial
        wall / missing data do not dilute neighbouring valid values.

    Returns
    -------
    projected : ndarray
        Data on the target surface. Medial-wall and all-NaN-source vertices are
        NaN; apply ``numpy.nan_to_num`` before handing to pycortex if needed.
    """
    data = np.asarray(data)
    if data.shape[-1] != N_VERTICES_FS_LR_32K:
        raise ValueError(
            f"Expected {N_VERTICES_FS_LR_32K} fs_LR 32k vertices on the last "
            f"axis, got {data.shape[-1]}."
        )
    # Preserve the input's floating precision (float32 timeseries are common and
    # large); only integer inputs are promoted, to float32. Flatten any leading
    # dimensions so 1-D, 2-D, and N-D inputs are all handled uniformly.
    out_dtype = data.dtype if np.issubdtype(data.dtype, np.floating) else np.float32
    leading_shape = data.shape[:-1]
    data_2d = data.reshape(-1, N_VERTICES_FS_LR_32K).astype(out_dtype, copy=False)

    matrix_target = "fsaverage6" if target == "fsaverage" else target
    if matrix_target not in N_VERTICES_TARGET_HEM:
        raise ValueError(
            f"target must be 'fsaverage', 'fsaverage6', or 'fsaverage5', got {target!r}"
        )

    n_tgt_hemi = N_VERTICES_TARGET_HEM[matrix_target]
    result = np.full((data_2d.shape[0], 2 * n_tgt_hemi), np.nan, dtype=out_dtype)
    for ih, hemi in enumerate(("L", "R")):
        src_sl = slice(
            ih * N_VERTICES_FS_LR_32K_HEM, (ih + 1) * N_VERTICES_FS_LR_32K_HEM
        )
        tgt_sl = slice(ih * n_tgt_hemi, (ih + 1) * n_tgt_hemi)
        matrix = get_fslr_to_fsaverage_matrix(
            hemi, target=matrix_target, cache_dir=cache_dir
        )
        result[:, tgt_sl] = _project_hemi(matrix, data_2d[:, src_sl], nanmean)

    if target == "fsaverage":
        result = upsample_to_fsaverage(
            result, "fsaverage6", freesurfer_subjects_dir=freesurfer_subjects_dir
        )

    return result.reshape(leading_shape + (result.shape[-1],))


def to_fsaverage(
    cifti: Union[str, Path, nib.Cifti2Image],
    target: str = "fsaverage",
    cache_dir: Optional[Union[str, Path]] = None,
    freesurfer_subjects_dir: Optional[str] = None,
    nanmean: bool = True,
) -> npt.NDArray:
    """Project HCP CIFTI cortical data straight to an fsaverage surface.

    Convenience wrapper chaining :func:`get_cifti_vertex_indices`,
    :func:`cifti_to_surface`, and :func:`project_fslr_to_fsaverage`.

    Parameters
    ----------
    cifti : str or Path or nibabel.Cifti2Image
        A CIFTI-2 file (or loaded image) whose data will be projected. The
        cortical grayordinates are read and expanded to the fs_LR 32k surface
        before resampling.
    target : {"fsaverage", "fsaverage6", "fsaverage5"}
        Destination surface.
    cache_dir, freesurfer_subjects_dir, nanmean
        See :func:`project_fslr_to_fsaverage`.

    Returns
    -------
    projected : ndarray
        CIFTI data on the target fsaverage surface (NaN medial wall).
    """
    if isinstance(cifti, (str, Path)):
        cifti = nib.load(str(cifti))
    if not isinstance(cifti, nib.Cifti2Image):
        raise TypeError("cifti must be a Cifti2Image or a path to a CIFTI-2 file.")
    left_indices, right_indices = get_cifti_vertex_indices(cifti)
    # CIFTI grayordinate axis is the last axis. A single-map file (e.g. a
    # one-column dscalar) is squeezed to a single surface map for convenience;
    # genuine multi-map/timeseries files keep their leading sample axis.
    data = np.asarray(cifti.get_fdata())
    if data.ndim == 2 and data.shape[0] == 1:
        data = data[0]
    full = cifti_to_surface(data, left_indices, right_indices)
    return project_fslr_to_fsaverage(
        full,
        target=target,
        cache_dir=cache_dir,
        freesurfer_subjects_dir=freesurfer_subjects_dir,
        nanmean=nanmean,
    )
