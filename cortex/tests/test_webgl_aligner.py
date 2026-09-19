"""Tests for the browser-based manual aligner (cortex.webgl.aligner).

The pure-python tests cover the world frame the aligner works in and the
tornado endpoints of its server. The browser test drives the aligner in
headless Chromium and is skipped without playwright.
"""

import base64
import io
import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pytest

import cortex
from cortex import align, database
from cortex.webgl import aligner
from cortex.tests.testing_utils import has_playwright

subj = "S1"
xfmname = "fullhead"


def _reference():
    return database.db.get_xfm(subj, xfmname).reference_nifti


def _open(url, data=None, timeout=30):
    """Fetch `url`; posting `data` (a dict) as a form when given."""
    body = None if data is None else urllib.parse.urlencode(data).encode()
    with urllib.request.urlopen(url, data=body, timeout=timeout) as resp:
        return resp.read()


class _SaveRecorder:
    """Stands in for db.save_xfm, so tests never write into the filestore."""

    def __init__(self):
        self.calls = []

    def __call__(self, subject, name, xfm, xfmtype="magnet", reference=None):
        self.calls.append(dict(subject=subject, name=name, xfm=np.asarray(xfm, dtype=float),
                               xfmtype=xfmtype, reference=reference))

    def wait(self, count=1, timeout=20):
        deadline = time.monotonic() + timeout
        while len(self.calls) < count and time.monotonic() < deadline:
            time.sleep(0.1)
        return len(self.calls) >= count


@pytest.fixture
def recorder(monkeypatch):
    rec = _SaveRecorder()
    monkeypatch.setattr(database.db, "save_xfm", rec)
    return rec


@pytest.fixture
def server(request):
    """A running aligner server for the bundled transform, stopped at teardown."""
    kwargs = dict(open_browser=False, display_url=False)
    kwargs.update(getattr(request, "param", {}))
    srv = aligner.show(subj, xfmname, **kwargs)
    yield srv
    srv.stop()


# ---------------------------------------------------------------------------
# World frame and reference loading
# ---------------------------------------------------------------------------


def test_reference_frame_is_scaled_signed_permutation():
    """Each voxel axis maps to one world axis, scaled by its voxel size."""
    import nibabel

    nii = _reference()
    world = aligner.reference_frame(nii)
    zooms = np.asarray(nii.header.get_zooms()[:3])

    assert world.shape == (4, 4)
    assert np.allclose(world[3], [0, 0, 0, 1])
    assert np.allclose(world[:3, 3], 0)
    linear = world[:3, :3]
    # one nonzero entry per row and per column, of the voxel size
    assert np.array_equal((linear != 0).sum(axis=0), [1, 1, 1])
    assert np.array_equal((linear != 0).sum(axis=1), [1, 1, 1])
    assert np.allclose(np.abs(linear).sum(axis=0), zooms)

    # the bundled reference is stored L, P, S: x and y flip, z does not
    assert nibabel.aff2axcodes(nii.affine) == ("L", "P", "S")
    assert world[0, 0] < 0 and world[1, 1] < 0 and world[2, 2] > 0


def test_reference_frame_follows_axis_permutation():
    """A reference stored in a different axis order gets its axes permuted
    so that world x, y, z point right, anterior and superior."""
    import nibabel

    # voxel axes are (superior, right, anterior) with 2, 3 and 4 mm voxels
    affine = np.array([[0, 3.0, 0, 0],
                       [0, 0, 4.0, 0],
                       [2.0, 0, 0, 0],
                       [0, 0, 0, 1]])
    nii = nibabel.Nifti1Image(np.zeros((5, 6, 7), dtype=np.float32), affine)
    world = aligner.reference_frame(nii)
    expected = np.array([[0, 3.0, 0, 0],
                         [0, 0, 4.0, 0],
                         [2.0, 0, 0, 0],
                         [0, 0, 0, 1]])
    assert np.allclose(world, expected)


def test_load_reference_takes_first_volume_and_drops_nans():
    import nibabel

    data = np.random.RandomState(0).rand(4, 5, 6, 3)
    data[0, 0, 0, 0] = np.nan
    nii = nibabel.Nifti1Image(data, np.eye(4))
    epi = aligner.load_reference(nii)
    assert epi.shape == (4, 5, 6)
    assert epi.dtype == np.float32
    assert epi[0, 0, 0] == 0
    assert np.allclose(epi[1:], data[1:, :, :, 0])


# ---------------------------------------------------------------------------
# Argument checks
# ---------------------------------------------------------------------------


def test_new_transform_requires_reference():
    with pytest.raises(ValueError, match="does not exist"):
        aligner.show(subj, "aligner_test_missing_xfm", open_browser=False, display_url=False)


def test_existing_transform_refuses_new_reference():
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        aligner.show(subj, xfmname, reference=_reference().get_filename(),
                     open_browser=False, display_url=False)


def test_transform_with_masks_requires_view_only(monkeypatch):
    import types

    monkeypatch.setattr(aligner, "glob", types.SimpleNamespace(glob=lambda pattern: ["mask_thick.nii.gz"]))
    with pytest.raises(ValueError, match="cached masks"):
        aligner.show(subj, xfmname, open_browser=False, display_url=False)


def test_align_entry_point_forwards(monkeypatch):
    seen = {}

    def fake_show(subject, name, reference=None, **kwargs):
        seen.update(subject=subject, name=name, reference=reference, kwargs=kwargs)
        return "handle"

    monkeypatch.setattr(aligner, "show", fake_show)
    assert align.webgl_manual(subj, xfmname, view_only=True) == "handle"
    assert seen == dict(subject=subj, name=xfmname, reference=None, kwargs=dict(view_only=True))


# ---------------------------------------------------------------------------
# Server endpoints (no browser)
# ---------------------------------------------------------------------------


def test_page_carries_config(server):
    from PIL import Image

    base = "http://localhost:%d" % server.port
    html = _open(base + "/aligner.html").decode()
    assert "aligner.Aligner" in html
    start = html.index('viewer = figure.add(aligner.Aligner, "main", true, ') + len(
        'viewer = figure.add(aligner.Aligner, "main", true, ')
    end = html.index(");", start)
    config = json.loads(html[start:end])

    xfm = database.db.get_xfm(subj, xfmname)
    nii = xfm.reference_nifti
    assert config["subject"] == subj
    assert config["xfmname"] == xfmname
    assert config["view_only"] is False
    assert np.allclose(config["xfm"], xfm.xfm)
    assert np.allclose(config["world"], aligner.reference_frame(nii))
    assert config["volume"]["shape"] == list(nii.shape[::-1])
    assert config["cmap"] == "gray"
    assert config["mesh_color"] == "#ffffff"
    assert config["vmin"] < config["vmax"]

    # the reference is served as the float mosaic the viewer expects
    png = _open(base + "/data/reference.png")
    image = Image.open(io.BytesIO(png))
    nwide, ntall = config["volume"]["mosaic"]
    assert image.size == (nwide * (nii.shape[0] + 1) + 1, ntall * (nii.shape[1] + 1) + 1)

    # the surfaces come from the viewer's CTM pack
    ctm = json.loads(_open(base + "/ctm/%s/" % subj).decode())
    assert len(ctm["offsets"]) == 2
    assert len(_open(base + "/ctm/%s/%s" % (subj, ctm["data"]))) > 0


def test_save_endpoint_stores_coord_transform(server, recorder):
    base = "http://localhost:%d" % server.port
    xfm = np.arange(16, dtype=float).reshape(4, 4)
    resp = json.loads(_open(base + "/save", dict(xfm=json.dumps(xfm.tolist()))).decode())
    assert resp["status"] == "ok"
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["subject"] == subj
    assert call["name"] == xfmname
    assert call["xfmtype"] == "coord"
    assert np.allclose(call["xfm"], xfm)

    resp = json.loads(_open(base + "/save", dict(xfm=json.dumps([1, 2, 3]))).decode())
    assert resp["status"] == "error"
    assert len(recorder.calls) == 1


@pytest.mark.parametrize("server", [dict(view_only=True)], indirect=True)
def test_view_only_never_saves(server, recorder):
    base = "http://localhost:%d" % server.port
    resp = json.loads(_open(base + "/save", dict(xfm=json.dumps(np.eye(4).tolist()))).decode())
    assert resp["status"] == "error"
    assert "view only" in resp["message"]
    assert recorder.calls == []
    assert '"view_only": true' in _open(base + "/").decode()


# ---------------------------------------------------------------------------
# Headless browser
# ---------------------------------------------------------------------------


def _quadrants(png):
    from PIL import Image

    rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(np.uint32)
    h, w = rgb.shape[:2]
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    return [packed[:h // 2, :w // 2], packed[h // 2:, :w // 2],
            packed[:h // 2, w // 2:], packed[h // 2:, w // 2:]]


def _translation(vector):
    mat = np.eye(4)
    mat[:3, 3] = vector
    return mat


@pytest.mark.skipif(not has_playwright, reason="playwright and chromium are required")
@pytest.mark.timeout(400)
def test_aligner_in_headless_browser(recorder):
    """The aligner loads, edits the transform in world millimeters and saves it."""
    from cortex.export.headless import _PlaywrightThread, _wait_for_viewer_loaded, filter_webgl_failures

    server = aligner.show(subj, xfmname, open_browser=False, display_url=False)
    server.disconnect_on_close = False
    pw = _PlaywrightThread()
    handle = None
    try:
        pw.start("http://localhost:%d/aligner.html" % server.port, timeout=120)
        handle = server.get_client()
        object.__setattr__(handle, "server", server)
        _wait_for_viewer_loaded(handle, timeout=240)
        # software rendering is slow: the first frame follows the load
        assert handle.wait_for_frame(timeout=120) >= 1

        nii = _reference()
        world = aligner.reference_frame(nii)
        coord0 = np.asarray(database.db.get_xfm(subj, xfmname).xfm)
        assert np.allclose(handle.get_xfm(), coord0, atol=1e-3)

        # a translation in world millimeters is applied ahead of the transform
        handle.translate([2.0, -3.0, 1.5])
        coord1 = handle.get_xfm()
        expected = np.linalg.inv(world) @ _translation([2.0, -3.0, 1.5]) @ world @ coord0
        assert np.allclose(coord1, expected, atol=1e-3)

        # a rotation about the cursor keeps the cursor fixed
        cursor_voxel = np.asarray(handle._call("getCursor"), dtype=float)
        cursor_world = (world @ np.append(cursor_voxel, 1))[:3]
        handle.rotate([0, 0, 1], 10)
        coord2 = handle.get_xfm()
        anat = np.linalg.inv(world @ coord2) @ np.append(cursor_world, 1)
        assert np.allclose((world @ coord1 @ anat)[:3], cursor_world, atol=1e-2)
        assert not np.allclose(coord2, coord1, atol=1e-4)

        handle.undo()
        assert np.allclose(handle.get_xfm(), coord1, atol=1e-3)

        # every view drew something, and the two view modes differ
        # (snapshot waits for the frame that shows the last change)
        outline = handle.snapshot()
        for quadrant in _quadrants(outline):
            assert len(np.unique(quadrant)) > 10
        handle.set_control("view", aligner.MODES["projected"])
        projected = handle.snapshot()
        for quadrant in _quadrants(projected):
            assert len(np.unique(quadrant)) > 10
        assert outline != projected

        # a colormap change and a new mesh color reach the shaders
        handle.set_control("view", aligner.MODES["outline"])
        handle.set_control("image.colormap", "hot")
        handle.set_control("mesh.color", "#ff0000")
        recolored = handle.snapshot()
        assert recolored != outline
        assert handle.get_control("mesh.color") == "#ff0000"
        assert handle.get_control("image.colormap") == "hot"
        assert handle.get_control("view") == aligner.MODES["outline"]

        # saving stores the current transform as a coord transform
        handle.save()
        assert recorder.wait(1), "the save request never reached the server"
        call = recorder.calls[0]
        assert call["subject"] == subj and call["name"] == xfmname
        assert call["xfmtype"] == "coord"
        assert np.allclose(call["xfm"], coord1, atol=1e-3)

        errors = pw.browser_errors
        assert not [e for e in errors if "[pageerror]" in e], errors
        assert not filter_webgl_failures(errors), errors
    finally:
        pw.shutdown()
        server.stop()
