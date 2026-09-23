import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest
from scipy.spatial import ConvexHull

import cortex
from cortex import hcp
from cortex.hcp import (
    _barycentric_resample_matrix,
    _normalize_hemi,
    cifti_to_surface,
    get_cifti_vertex_indices,
)


def test_normalize_hemi_accepts_aliases():
    assert _normalize_hemi("L") == "L" and _normalize_hemi("lh") == "L"
    assert _normalize_hemi("R") == "R" and _normalize_hemi("rh") == "R"
    with pytest.raises(ValueError):
        _normalize_hemi("left")


def test_cifti_to_surface_ndim_and_validation():
    left_idx = [0, 2, 5]
    right_idx = [1, 3]
    # 3-D input (subjects x time x grayordinates), 5 grayordinates on last axis.
    data = np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5)
    full = cifti_to_surface(data, left_idx, right_idx)
    assert full.shape == (2, 4, hcp.N_VERTICES_FS_LR_32K)
    np.testing.assert_array_equal(full[..., left_idx], data[..., :3])
    np.testing.assert_array_equal(
        full[..., hcp.N_VERTICES_FS_LR_32K_HEM + np.array(right_idx)], data[..., 3:5]
    )
    # Wrong number of grayordinates on the last axis is rejected.
    with pytest.raises(ValueError):
        cifti_to_surface(np.zeros((2, 7)), left_idx, right_idx)


# ---------------------------------------------------------------------------
# Barycentric resampling matrix (pure geometry, no network / wb_command)
# ---------------------------------------------------------------------------


def _triangulated_sphere(n, seed):
    """n points on the unit sphere plus a covering triangulation (convex hull)."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    tris = ConvexHull(pts).simplices
    return pts, tris


def _write_sphere_gii(path, pts, tris):
    """Write points + triangles to a ``.surf.gii`` file like the HCP spheres."""
    img = nib.gifti.GiftiImage(
        darrays=[
            nib.gifti.GiftiDataArray(
                np.asarray(pts, dtype=np.float32), intent="NIFTI_INTENT_POINTSET"
            ),
            nib.gifti.GiftiDataArray(
                np.asarray(tris, dtype=np.int32), intent="NIFTI_INTENT_TRIANGLE"
            ),
        ]
    )
    nib.save(img, str(path))


def test_barycentric_identity_when_source_equals_target(tmp_path):
    # Same source and target sphere: every target vertex sits on a source
    # vertex, so the matrix is the identity.
    pts, tris = _triangulated_sphere(60, seed=0)
    src = tmp_path / "src.surf.gii"
    _write_sphere_gii(src, pts, tris)
    m = _barycentric_resample_matrix(src, src)
    assert m.shape == (60, 60)
    np.testing.assert_allclose(m.toarray(), np.eye(60), atol=1e-9)


def test_barycentric_rows_sum_to_one(tmp_path):
    src_pts, src_tris = _triangulated_sphere(200, seed=1)
    tgt_pts, tgt_tris = _triangulated_sphere(80, seed=2)
    src = tmp_path / "src.surf.gii"
    tgt = tmp_path / "tgt.surf.gii"
    _write_sphere_gii(src, src_pts, src_tris)
    _write_sphere_gii(tgt, tgt_pts, tgt_tris)
    m = _barycentric_resample_matrix(src, tgt)
    assert m.shape == (80, 200)
    row_sums = np.asarray(m.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.ones(80), atol=1e-9)


def test_barycentric_preserves_constants(tmp_path):
    src_pts, src_tris = _triangulated_sphere(200, seed=3)
    tgt_pts, tgt_tris = _triangulated_sphere(70, seed=4)
    src = tmp_path / "src.surf.gii"
    tgt = tmp_path / "tgt.surf.gii"
    _write_sphere_gii(src, src_pts, src_tris)
    _write_sphere_gii(tgt, tgt_pts, tgt_tris)
    m = _barycentric_resample_matrix(src, tgt)
    const = np.full(200, 3.5)
    np.testing.assert_allclose(m.dot(const), np.full(70, 3.5), atol=1e-9)


def test_barycentric_weights_are_nonnegative(tmp_path):
    src_pts, src_tris = _triangulated_sphere(150, seed=5)
    tgt_pts, tgt_tris = _triangulated_sphere(60, seed=6)
    src = tmp_path / "src.surf.gii"
    tgt = tmp_path / "tgt.surf.gii"
    _write_sphere_gii(src, src_pts, src_tris)
    _write_sphere_gii(tgt, tgt_pts, tgt_tris)
    m = _barycentric_resample_matrix(src, tgt)
    assert m.data.min() >= 0.0
    # Barycentric interpolation uses at most three source vertices per target.
    assert m.getnnz(axis=1).max() <= 3


# ---------------------------------------------------------------------------
# CIFTI -> fs_LR 32k surface expansion
# ---------------------------------------------------------------------------


def test_cifti_to_surface_1d_places_values_and_nans_medial_wall():
    left_idx = np.array([0, 2, 5])
    right_idx = np.array([1, 3])
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # 3 left then 2 right
    full = cifti_to_surface(data, left_idx, right_idx)

    assert full.shape == (hcp.N_VERTICES_FS_LR_32K,)
    off = hcp.N_VERTICES_FS_LR_32K_HEM
    np.testing.assert_array_equal(full[left_idx], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(full[off + right_idx], [40.0, 50.0])
    # Everything else is medial wall -> NaN.
    n_valid = np.sum(~np.isnan(full))
    assert n_valid == 5


def test_cifti_to_surface_2d_matches_1d_per_row():
    left_idx = np.array([4, 1])
    right_idx = np.array([7])
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 left, 1 right per row
    full = cifti_to_surface(data, left_idx, right_idx)
    assert full.shape == (2, hcp.N_VERTICES_FS_LR_32K)
    off = hcp.N_VERTICES_FS_LR_32K_HEM
    np.testing.assert_array_equal(full[:, left_idx], [[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_array_equal(full[:, off + right_idx], [[3.0], [6.0]])


def test_get_cifti_vertex_indices_reads_brain_model_axis():
    left_idx = [0, 2, 5, 9]
    right_idx = [1, 3, 8]
    bm = nib.cifti2.BrainModelAxis.from_surface(
        left_idx, hcp.N_VERTICES_FS_LR_32K_HEM, "CIFTI_STRUCTURE_CORTEX_LEFT"
    ) + nib.cifti2.BrainModelAxis.from_surface(
        right_idx, hcp.N_VERTICES_FS_LR_32K_HEM, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    scalar = nib.cifti2.ScalarAxis(["map"])
    img = nib.cifti2.Cifti2Image(
        np.zeros((1, len(left_idx) + len(right_idx))), header=(scalar, bm)
    )

    got_left, got_right = get_cifti_vertex_indices(img)
    np.testing.assert_array_equal(got_left, left_idx)
    np.testing.assert_array_equal(got_right, right_idx)


# ---------------------------------------------------------------------------
# Validation against wb_command (requires Connectome Workbench + network)
# ---------------------------------------------------------------------------


def _wb_metric_resample(data, src_sphere, tgt_sphere):
    """Reference resample via ``wb_command -metric-resample ... BARYCENTRIC``."""
    with tempfile.TemporaryDirectory() as td:
        in_p = Path(td) / "in.func.gii"
        out_p = Path(td) / "out.func.gii"
        img = nib.gifti.GiftiImage(
            darrays=[
                nib.gifti.GiftiDataArray(
                    np.asarray(data, dtype=np.float32),
                    intent="NIFTI_INTENT_NORMAL",
                    datatype="NIFTI_TYPE_FLOAT32",
                )
            ]
        )
        nib.save(img, str(in_p))
        subprocess.run(
            [
                "wb_command",
                "-metric-resample",
                str(in_p),
                str(src_sphere),
                str(tgt_sphere),
                "BARYCENTRIC",
                str(out_p),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return np.asarray(nib.load(str(out_p)).darrays[0].data)


@pytest.mark.skipif(
    shutil.which("wb_command") is None,
    reason="Connectome Workbench (wb_command) not installed",
)
def test_barycentric_matches_wb_command():
    # Uses the real HCP fs_LR 32k -> fsaverage5 spheres (downloaded on demand),
    # so it also exercises ensure_sphere_files. Skipped if the spheres cannot
    # be fetched.
    cache_dir = Path(tempfile.gettempdir()) / "pycortex_hcp_test_spheres"
    try:
        src = hcp.ensure_sphere_files("fs_LR_32k", "L", cache_dir=cache_dir)
        tgt = hcp.ensure_sphere_files("fsaverage5", "L", cache_dir=cache_dir)
    except RuntimeError as exc:  # network unavailable
        pytest.skip(f"could not download sphere files: {exc}")

    m = _barycentric_resample_matrix(src, tgt)
    rng = np.random.RandomState(0)
    data = rng.randn(hcp.N_VERTICES_FS_LR_32K_HEM).astype(np.float32)

    ours = m.dot(data)
    reference = _wb_metric_resample(data, src, tgt)
    assert ours.shape == reference.shape
    corr = np.corrcoef(ours, reference)[0, 1]
    assert corr > 0.999
    # Spread of the reference is O(1); the discrepancy should be tiny.
    assert np.abs(ours - reference).max() < 0.05 * reference.std() + 1e-4


# ---------------------------------------------------------------------------
# Native-space rendering integration (requires the 32k_fs_LR subject)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "32k_fs_LR" not in cortex.db.subjects, reason="32k_fs_LR subject not in filestore"
)
def test_native_vertex_matches_subject():
    # The subject carries the full 64984-vertex fs_LR 32k surface, so a Vertex
    # built from cifti_to_surface output must line up with it.
    data = np.zeros(hcp.N_VERTICES_FS_LR_32K)
    v = cortex.Vertex(data, "32k_fs_LR")
    assert v.data.shape[-1] == hcp.N_VERTICES_FS_LR_32K
