import os
import shutil
import sys
import warnings

import numpy as np
import pytest

import cortex._autoflatten as af
import cortex.freesurfer as fs


def _make_freesurfer_subject(tmp_path, subject="S1"):
    """Create a minimal freesurfer subject directory with an empty surf/ folder."""
    surf_dir = tmp_path / "fs_subjects" / subject / "surf"
    surf_dir.mkdir(parents=True)
    return str(tmp_path / "fs_subjects"), surf_dir


def _patch_autoflatten_run(monkeypatch, calls, surf_dir=None):
    """Pretend autoflatten is installed, and record the command it is invoked with.

    If `surf_dir` is given, the flat patch files that autoflatten would create are
    written there when the (fake) command runs.
    """
    monkeypatch.setattr(af, "get_command", lambda: ["fake-autoflatten"])

    def fake_check_call(cmd):
        calls["cmd"] = cmd
        if surf_dir is not None:
            for hemi in ("lh", "rh"):
                (surf_dir / (hemi + ".autoflatten.flat.patch.3d")).write_bytes(b"")

    monkeypatch.setattr(af.sp, "check_call", fake_check_call)

    def fake_import_flat(*args, **kwargs):
        calls["import_flat"] = (args, kwargs)

    monkeypatch.setattr(fs, "import_flat", fake_import_flat)


def test_get_command_uses_the_current_interpreter():
    cmd = af.get_command()
    if cmd is None:  # autoflatten is an optional dependency
        assert af.is_available() is False
    else:
        assert cmd == [sys.executable, "-m", "autoflatten.cli"]
        assert af.is_available() is True


def test_get_command_is_none_when_autoflatten_is_not_installed(monkeypatch):
    find_spec = af.importlib.util.find_spec
    monkeypatch.setattr(
        af.importlib.util, "find_spec",
        lambda name, *args, **kwargs: (
            None if name == "autoflatten" else find_spec(name, *args, **kwargs)
        ),
    )
    assert af.get_command() is None
    assert af.is_available() is False


def test_autoflatten_subject_raises_when_not_installed(tmp_path, monkeypatch):
    subjects_dir, _ = _make_freesurfer_subject(tmp_path)
    monkeypatch.setattr(af, "get_command", lambda: None)
    with pytest.raises(ImportError, match="autoflatten"):
        af.autoflatten_subject("S1", freesurfer_subject_dir=subjects_dir)


def test_autoflatten_subject_raises_on_missing_subject_dir(tmp_path, monkeypatch):
    subjects_dir, _ = _make_freesurfer_subject(tmp_path)
    monkeypatch.setattr(af, "get_command", lambda: ["fake-autoflatten"])
    with pytest.raises(IOError, match="not found"):
        af.autoflatten_subject("nosuchsubject", freesurfer_subject_dir=subjects_dir)


def test_autoflatten_subject_runs_cli_and_imports_flatmaps(tmp_path, monkeypatch):
    subjects_dir, surf_dir = _make_freesurfer_subject(tmp_path)
    calls = {}
    _patch_autoflatten_run(monkeypatch, calls, surf_dir=surf_dir)

    with pytest.warns(UserWarning, match="15-30 minutes"):
        flat_files = af.autoflatten_subject(
            "S1",
            pycortex_subject="cx_S1",
            freesurfer_subject_dir=subjects_dir,
            autoflatten_args=["--backend", "freesurfer"],
        )

    # the extra arguments are appended to `autoflatten run <subject_dir>`
    assert calls["cmd"] == ["fake-autoflatten", "run",
                            os.path.join(subjects_dir, "S1"),
                            "--backend", "freesurfer"]

    # the flat patches are expected in the freesurfer subject's surf/ directory
    for hemi in ("lh", "rh"):
        assert flat_files[hemi] == str(
            surf_dir / (hemi + ".autoflatten.flat.patch.3d")
        )

    args, kwargs = calls["import_flat"]
    assert args == ("S1", "autoflatten")
    assert kwargs["cx_subject"] == "cx_S1"
    assert kwargs["flat_type"] == "freesurfer"
    assert kwargs["auto_overwrite"] is True
    assert kwargs["freesurfer_subject_dir"] == subjects_dir


def test_autoflatten_subject_no_runtime_warning_if_already_flattened(
        tmp_path, monkeypatch):
    subjects_dir, surf_dir = _make_freesurfer_subject(tmp_path)
    for hemi in ("lh", "rh"):
        (surf_dir / (hemi + ".autoflatten.flat.patch.3d")).write_bytes(b"")
    calls = {}
    _patch_autoflatten_run(monkeypatch, calls)

    # autoflatten skips existing patches, so there is nothing slow to warn about
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        af.autoflatten_subject("S1", freesurfer_subject_dir=subjects_dir)

    # the existing flatmaps are still imported into the pycortex database
    assert "import_flat" in calls


def test_autoflatten_subject_can_skip_import(tmp_path, monkeypatch):
    subjects_dir, surf_dir = _make_freesurfer_subject(tmp_path)
    calls = {}
    _patch_autoflatten_run(monkeypatch, calls, surf_dir=surf_dir)

    with pytest.warns(UserWarning):
        af.autoflatten_subject("S1", freesurfer_subject_dir=subjects_dir,
                               import_flatmaps=False)

    assert "import_flat" not in calls


def test_autoflatten_subject_raises_when_no_flatmap_produced(tmp_path, monkeypatch):
    subjects_dir, _ = _make_freesurfer_subject(tmp_path)
    calls = {}
    # surf_dir=None: the fake command does not write any flat patch file
    _patch_autoflatten_run(monkeypatch, calls, surf_dir=None)

    with pytest.warns(UserWarning):
        with pytest.raises(IOError, match="did not produce"):
            af.autoflatten_subject("S1", freesurfer_subject_dir=subjects_dir)

    assert "import_flat" not in calls


# ---------------------------------------------------------------------------
# End-to-end run against a real freesurfer subject
#
# The bundled S1 subject is a pycortex filestore entry (gifti surfaces), not a
# freesurfer subject directory, so it cannot be flattened: autoflatten needs
# `?h.sphere.reg` and freesurfer's `mri_label2label` to map the fsaverage
# template cuts onto the subject. Point $PYCORTEX_TEST_FS_SUBJECT at a real
# recon-all'd subject in $SUBJECTS_DIR to run this for real, e.g.
#
#     PYCORTEX_TEST_FS_SUBJECT=sub-01 pytest -m slow cortex/tests/test_autoflatten.py
#
# It is marked `slow` (deselected by default) because a full run takes 15-30
# minutes for both hemispheres.
# ---------------------------------------------------------------------------

FS_TEST_SUBJECT = os.environ.get("PYCORTEX_TEST_FS_SUBJECT")


def _freesurfer_is_sourced():
    return "SUBJECTS_DIR" in os.environ and all(
        shutil.which(binary) is not None
        for binary in ("mri_label2label", "mri_info")
    )


def _link_subject(src_dir, dst_dir):
    """Mirror a freesurfer subject into `dst_dir` without writing to the original.

    Everything is symlinked, except that ``surf/`` is a real directory holding
    symlinks to the individual surface files, so that the patches autoflatten
    writes there land in `dst_dir` rather than in the real subject.
    """
    os.makedirs(dst_dir)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if name == "surf":
            os.makedirs(dst)
            for surf_file in os.listdir(src):
                os.symlink(os.path.join(src, surf_file),
                           os.path.join(dst, surf_file))
        else:
            os.symlink(src, dst)


@pytest.mark.slow
@pytest.mark.timeout(5400)
@pytest.mark.skipif(FS_TEST_SUBJECT is None,
                    reason="set $PYCORTEX_TEST_FS_SUBJECT to a freesurfer subject")
@pytest.mark.skipif(not af.is_available(), reason="autoflatten is not installed")
@pytest.mark.skipif(not _freesurfer_is_sourced(), reason="freesurfer is not sourced")
def test_autoflatten_subject_end_to_end(tmp_path, monkeypatch):
    """Really run autoflatten, and check that it produces usable flat patches."""
    from cortex import freesurfer

    subjects_dir = os.environ["SUBJECTS_DIR"]
    assert os.path.isdir(os.path.join(subjects_dir, "fsaverage")), (
        "autoflatten maps the template cuts from fsaverage, which must be in "
        "$SUBJECTS_DIR"
    )

    # run against a throwaway $SUBJECTS_DIR so the real subject is not modified
    test_subjects_dir = tmp_path / "subjects"
    test_subjects_dir.mkdir()
    _link_subject(os.path.join(subjects_dir, FS_TEST_SUBJECT),
                  str(test_subjects_dir / FS_TEST_SUBJECT))
    os.symlink(os.path.join(subjects_dir, "fsaverage"),
               str(test_subjects_dir / "fsaverage"))
    # mri_label2label resolves subjects through $SUBJECTS_DIR in the subprocess
    monkeypatch.setenv("SUBJECTS_DIR", str(test_subjects_dir))

    # import_flatmaps=False: the flatmaps are checked here directly rather than
    # written into the user's filestore, which this test must not touch
    flat_files = af.autoflatten_subject(
        FS_TEST_SUBJECT,
        freesurfer_subject_dir=str(test_subjects_dir),
        import_flatmaps=False,
    )

    assert sorted(flat_files) == ["lh", "rh"]
    for hemi, flat_file in flat_files.items():
        assert os.path.exists(flat_file), hemi
        # get_surf returns the whole smoothwm surface, with the patch vertices
        # replaced by their flattened positions; idx marks which those are
        pts, polys, idx = freesurfer.get_surf(
            FS_TEST_SUBJECT, hemi, "patch", "autoflatten.flat",
            freesurfer_subject_dir=str(test_subjects_dir),
        )
        in_patch = idx != 0
        # a cut hemisphere keeps most of its vertices, and every triangle left
        # in the patch is made of patch vertices
        assert in_patch.sum() > 10000, hemi
        assert len(polys) > 10000, hemi
        assert in_patch[polys].all(), hemi
        # the patch really is flat: it is spread out over two axes, and has no
        # thickness at all along the third
        spread = np.ptp(pts[in_patch], axis=0)
        assert (np.sort(spread)[-2:] > 1.0).all(), (hemi, spread)
        assert np.sort(spread)[0] < 1e-3, (hemi, spread)
