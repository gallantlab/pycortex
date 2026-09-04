import os
import shutil

import numpy as np
import pytest

import cortex._autoflatten as af
import cortex.freesurfer as fs
from cortex.freesurfer import (
    _remove_disconnected_polys,
    _surf2surf_nnfr_matrix,
    get_mri_surf2surf_matrix,
    upsample_to_fsaverage,
)


def test_remove_disconnected_polys_examples():
    polys = np.array([[0, 1, 2],
                      [0, 1, 3],
                      [1, 2, 4],
                      [5, 6, 7]])
    expected_result = np.array([[0, 1, 2],
                                [0, 1, 3],
                                [1, 2, 4]])
    result = _remove_disconnected_polys(polys)
    np.testing.assert_array_equal(result, expected_result)


def test_remove_disconnected_polys_idempotence():
    rng = np.random.RandomState(0)
    for n_polys in [10, 20, 30, 40]:
        polys_0 =rng.randint(0, 100, size=3 * n_polys).reshape(-1, 3)
        
        # make sure this example filters something
        polys_1 = _remove_disconnected_polys(polys_0)
        assert len(polys_0) != len(polys_1)
        
        # make sure calling the function does not change anything
        polys_2 = _remove_disconnected_polys(polys_1)
        np.testing.assert_array_equal(polys_1, polys_2)


# ---------------------------------------------------------------------------
# surf2surf (nnfr) matrix construction
# ---------------------------------------------------------------------------

def _random_sphere(n, seed):
    """n points on the unit sphere (so KDTree distances behave like the
    real ?h.sphere.reg geometry)."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def test_surf2surf_identity_when_source_equals_target():
    # If source and target spheres are identical, every target maps to the
    # co-located source vertex and there are no orphans -> identity matrix.
    sphere = _random_sphere(60, seed=0)
    m = _surf2surf_nnfr_matrix(sphere, sphere)
    assert m.shape == (60, 60)
    np.testing.assert_allclose(m.toarray(), np.eye(60))


def test_surf2surf_rows_sum_to_one():
    src = _random_sphere(80, seed=1)
    trg = _random_sphere(50, seed=2)
    m = _surf2surf_nnfr_matrix(src, trg)
    assert m.shape == (50, 80)
    row_sums = np.asarray(m.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.ones(50))


def test_surf2surf_preserves_constants():
    # A row-normalized averaging matrix maps a constant map to itself.
    src = _random_sphere(80, seed=3)
    trg = _random_sphere(50, seed=4)
    m = _surf2surf_nnfr_matrix(src, trg)
    const = np.full(80, 7.0)
    np.testing.assert_allclose(m.dot(const), np.full(50, 7.0))


def test_surf2surf_no_source_vertex_is_dropped():
    # The reverse pass guarantees every source vertex contributes to at least
    # one target vertex (this is the whole point of "forward and reverse").
    src = _random_sphere(120, seed=5)
    trg = _random_sphere(40, seed=6)  # heavy downsampling
    m = _surf2surf_nnfr_matrix(src, trg)
    col_nnz = m.getnnz(axis=0)
    assert (col_nnz > 0).all()


def test_surf2surf_known_averaging_example():
    # Four collinear source vertices, two target vertices placed so that each
    # target's nearest source is an endpoint, leaving the two middle source
    # vertices as orphans that get folded into their nearest target.
    src = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    trg = np.array([[0.4, 0, 0], [2.6, 0, 0]])
    m = _surf2surf_nnfr_matrix(src, trg)

    expected = np.array([[0.5, 0.5, 0.0, 0.0],   # mean of src 0 and 1
                         [0.0, 0.0, 0.5, 0.5]])   # mean of src 2 and 3
    np.testing.assert_allclose(m.toarray(), expected)

    data = np.array([10., 20., 30., 40.])
    np.testing.assert_allclose(m.dot(data), [15., 35.])


def test_get_mri_surf2surf_matrix_ignores_legacy_kwargs(monkeypatch):
    # Legacy regression-based kwargs are accepted but ignored, with a warning.
    src = _random_sphere(20, seed=7)
    trg = _random_sphere(15, seed=8)
    monkeypatch.setattr(
        fs, "_read_sphere_reg",
        lambda subj, hemi, subjects_dir=None: src if subj == "A" else trg)

    with pytest.warns(DeprecationWarning):
        m = get_mri_surf2surf_matrix("A", "lh", target_subj="B",
                                     n_neighbors=20, n_test_images=40)
    assert m.shape == (15, 20)


def test_get_mri_surf2surf_matrix_rejects_unknown_kwargs():
    with pytest.raises(TypeError):
        get_mri_surf2surf_matrix("A", "lh", target_subj="B", bogus_kwarg=1)


# ---------------------------------------------------------------------------
# upsample_to_fsaverage (bundled neighbor tables, no freesurfer needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data_space,n_src", [("fsaverage6", 81924),
                                              ("fsaverage5", 20484)])
def test_upsample_to_fsaverage_bundled_no_freesurfer(data_space, n_src,
                                                     monkeypatch):
    # fsaverage5/6 upsampling tables ship with pycortex, so this must work with
    # no $SUBJECTS_DIR and no freesurfer install.
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    rng = np.random.RandomState(0)
    data = rng.randn(3, n_src)
    out = upsample_to_fsaverage(data, data_space)
    assert out.shape == (3, 327684)
    # The low-resolution vertices are carried over unchanged.
    np.testing.assert_array_equal(out[:, :n_src // 2], data[:, :n_src // 2])
    # A 1-D input yields a 1-D result.
    assert upsample_to_fsaverage(data[0], data_space).ndim == 1


def test_upsample_to_fsaverage_constant_preserved(monkeypatch):
    # Nearest-neighbor fill of a constant map stays constant everywhere.
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    out = upsample_to_fsaverage(np.full(81924, 2.5), "fsaverage6")
    np.testing.assert_array_equal(out, np.full(327684, 2.5))


def _have_template(subjects_dir, name, hemi="lh"):
    return bool(subjects_dir) and os.path.exists(
        os.path.join(subjects_dir, name, "surf", hemi + ".sphere.reg"))


@pytest.mark.skipif(shutil.which("mri_surf2surf") is None,
                    reason="freesurfer mri_surf2surf not available")
def test_surf2surf_matches_freesurfer_identity():
    """fsaverage -> fsaverage must be the identity, matching mri_surf2surf.

    Uses only the standard fsaverage template (no individual subjects), so it
    is reproducible anywhere freesurfer is installed and skipped otherwise.
    """
    subjects_dir = os.environ.get("SUBJECTS_DIR")
    hemi = "lh"
    if not _have_template(subjects_dir, "fsaverage", hemi):
        pytest.skip("fsaverage template with sphere.reg not found in SUBJECTS_DIR")

    m = get_mri_surf2surf_matrix("fsaverage", hemi, target_subj="fsaverage",
                                 subjects_dir=subjects_dir)
    rng = np.random.RandomState(0)
    data = rng.randn(4, m.shape[1]).astype(np.float32)
    reference = fs.mri_surf2surf(data, "fsaverage", "fsaverage", hemi,
                                 subjects_dir=subjects_dir)
    got = np.stack([m.dot(data[i]) for i in range(data.shape[0])])

    np.testing.assert_allclose(got, data, atol=1e-4)        # identity
    np.testing.assert_allclose(got, reference, atol=1e-4)   # matches freesurfer


@pytest.mark.skipif(shutil.which("mri_surf2surf") is None,
                    reason="freesurfer mri_surf2surf not available")
def test_surf2surf_matches_freesurfer_downsample():
    """fsaverage -> fsaverage6 (icosahedral downsample) against mri_surf2surf.

    Uses only standard fsaverage templates. This is the documented inexact
    case: freesurfer's tie-breaking on exactly-equidistant vertices of the
    regular mesh differs, so we only require a high correlation rather than a
    bit-exact match (see _surf2surf_nnfr_matrix notes).
    """
    subjects_dir = os.environ.get("SUBJECTS_DIR")
    hemi = "lh"
    if not (_have_template(subjects_dir, "fsaverage", hemi)
            and _have_template(subjects_dir, "fsaverage6", hemi)):
        pytest.skip("fsaverage/fsaverage6 templates not found in SUBJECTS_DIR")

    m = get_mri_surf2surf_matrix("fsaverage", hemi, target_subj="fsaverage6",
                                 subjects_dir=subjects_dir)
    rng = np.random.RandomState(0)
    data = rng.randn(4, m.shape[1]).astype(np.float32)
    reference = fs.mri_surf2surf(data, "fsaverage", "fsaverage6", hemi,
                                 subjects_dir=subjects_dir)
    got = np.stack([m.dot(data[i]) for i in range(data.shape[0])])

    corr = np.corrcoef(reference.ravel(), got.ravel())[0, 1]
    assert corr > 0.99


# ---------------------------------------------------------------------------
# freesurfer subjects directory
# ---------------------------------------------------------------------------

def test_get_freesurfer_subject_dir_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv("SUBJECTS_DIR", "/from/env")
    assert fs._get_freesurfer_subject_dir("/explicit") == "/explicit"


def test_get_freesurfer_subject_dir_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("SUBJECTS_DIR", "/from/env")
    assert fs._get_freesurfer_subject_dir() == "/from/env"


def test_get_freesurfer_subject_dir_raises_without_env(monkeypatch):
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    with pytest.raises(ValueError):
        fs._get_freesurfer_subject_dir()



# ---------------------------------------------------------------------------
# autoflatten step of import_subj (see test_autoflatten.py for the step itself)
# ---------------------------------------------------------------------------

def _stub_import_subj(tmp_path, monkeypatch, subject="S1"):
    """Stub out everything `import_subj` shells out to, and the filestore."""
    filestore = tmp_path / "filestore"
    for folder in ("anatomicals", "surfaces", "surface-info"):
        (filestore / subject / folder).mkdir(parents=True)

    class _FakeDatabase(object):
        def make_subj(self, subject):
            pass

    monkeypatch.setattr(fs.database, "default_filestore", str(filestore))
    monkeypatch.setattr(fs.database, "db", _FakeDatabase(), raising=False)
    monkeypatch.setattr(fs.database, "Database", _FakeDatabase)
    monkeypatch.setattr(fs.sp, "check_output", lambda *args, **kwargs: b"")
    monkeypatch.setattr(fs, "make_fiducial", lambda *args, **kwargs: None)
    monkeypatch.setattr(fs, "parse_curv", lambda path: np.zeros(3))

    calls = []
    # import_subj calls the name re-exported into cortex.freesurfer, so that is
    # what has to be patched
    monkeypatch.setattr(
        fs, "autoflatten_subject",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    return calls


def test_import_subj_runs_autoflatten_by_default(tmp_path, monkeypatch):
    calls = _stub_import_subj(tmp_path, monkeypatch)
    monkeypatch.setattr(af, "is_available", lambda: True)

    fs.import_subj("S1", freesurfer_subject_dir=str(tmp_path / "fs_subjects"))

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("S1",)
    assert kwargs["pycortex_subject"] == "S1"
    assert kwargs["freesurfer_subject_dir"] == str(tmp_path / "fs_subjects")
    assert kwargs["autoflatten_args"] is None


def test_import_subj_passes_autoflatten_args(tmp_path, monkeypatch):
    calls = _stub_import_subj(tmp_path, monkeypatch)
    monkeypatch.setattr(af, "is_available", lambda: True)

    fs.import_subj("S1", freesurfer_subject_dir=str(tmp_path / "fs_subjects"),
                   autoflatten_args=["--n-cores", "4"])

    _, kwargs = calls[0]
    assert kwargs["autoflatten_args"] == ["--n-cores", "4"]


def test_import_subj_can_disable_autoflatten(tmp_path, monkeypatch):
    calls = _stub_import_subj(tmp_path, monkeypatch)
    # even when autoflatten is available, it must not run when disabled
    monkeypatch.setattr(af, "is_available", lambda: True)

    fs.import_subj("S1", freesurfer_subject_dir=str(tmp_path / "fs_subjects"),
                   autoflatten=False)

    assert calls == []


def test_import_subj_warns_and_skips_when_autoflatten_missing(tmp_path, monkeypatch):
    calls = _stub_import_subj(tmp_path, monkeypatch)
    monkeypatch.setattr(af, "is_available", lambda: False)

    with pytest.warns(UserWarning, match="not installed"):
        fs.import_subj("S1", freesurfer_subject_dir=str(tmp_path / "fs_subjects"))

    assert calls == []
