import shutil
import tarfile
from unittest import mock

import pytest

import cortex


@pytest.fixture
def fake_fsaverage_tarball(tmp_path):
    """Build a minimal fsaverage.tar.gz that pycortex will recognize as a subject."""
    subj_src = tmp_path / "src" / "fsaverage"
    subj_src.mkdir(parents=True)
    (subj_src / "marker").write_text("ok")
    tarball = tmp_path / "fsaverage.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(subj_src, arcname="fsaverage")
    return tarball


@pytest.fixture
def isolated_filestore(tmp_path, monkeypatch):
    """Redirect cortex.db.filestore to an empty tmp dir and reset the subject cache."""
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(cortex.db, "filestore", str(store))
    original_subjects = cortex.db._subjects
    cortex.db._subjects = None
    yield store
    cortex.db._subjects = original_subjects


def test_download_subject(isolated_filestore, fake_fsaverage_tarball, monkeypatch):
    # Newly downloaded subjects are added to the current database.
    def fake_retrieve(url, dest):
        shutil.copy(fake_fsaverage_tarball, dest)
        return dest, None

    mock_retrieve = mock.Mock(side_effect=fake_retrieve)
    monkeypatch.setattr(cortex.utils.urllib.request, "urlretrieve", mock_retrieve)

    assert "fsaverage" not in cortex.db.subjects
    cortex.utils.download_subject(subject_id='fsaverage')
    assert "fsaverage" in cortex.db.subjects
    assert mock_retrieve.call_count == 1


def test_download_subject_skips_when_present(isolated_filestore, monkeypatch):
    # If the subject is already in the database and download_again is False,
    # download_subject warns and returns without touching the network.
    cortex.db._subjects = {"fsaverage": mock.MagicMock()}

    mock_retrieve = mock.Mock()
    monkeypatch.setattr(cortex.utils.urllib.request, "urlretrieve", mock_retrieve)

    with pytest.warns(UserWarning, match="already present"):
        cortex.utils.download_subject(subject_id='fsaverage')
    mock_retrieve.assert_not_called()


def test_download_subject_download_again(
    isolated_filestore, fake_fsaverage_tarball, monkeypatch
):
    # download_again=True re-downloads even when the subject is already present.
    def fake_retrieve(url, dest):
        shutil.copy(fake_fsaverage_tarball, dest)
        return dest, None

    mock_retrieve = mock.Mock(side_effect=fake_retrieve)
    monkeypatch.setattr(cortex.utils.urllib.request, "urlretrieve", mock_retrieve)

    cortex.utils.download_subject(subject_id='fsaverage')
    assert "fsaverage" in cortex.db.subjects
    cortex.utils.download_subject(subject_id='fsaverage', download_again=True)
    assert mock_retrieve.call_count == 2


def test_get_roi_masks_missing_roi_does_not_fail_when_not_required():
    # When fail_for_missing_rois=False and a requested ROI isn't in
    # overlays.svg, get_roi_masks falls back to computing the missing-ROI
    # list from all available ROIs. That fallback used to crash with
    # `TypeError: unsupported operand type(s) for +: 'dict_keys' and 'list'`
    # because dict.keys() (a dict_keys view) doesn't support `+` with a list.
    result = cortex.utils.get_roi_masks(
        "S1", "fullhead", roi_list=["V1", "NotARealROI"],
        fail_for_missing_rois=False,
    )
    assert "V1" in result
    assert "NotARealROI" not in result


def test_decimate_faces_draws_from_the_vertices_it_was_given():
    """The coarse faces are written over a subset of the original vertices,
    which is what lets a viewer draw them from the arrays it already holds."""
    import numpy as np

    from cortex import polyutils

    pts, polys = cortex.db.get_surf("S1", "pia", merge=False)[0]
    verts, faces = polyutils.decimate_faces(pts, polys, 8)

    assert len(faces) < len(polys) / 10
    assert faces.max() < len(pts), "a face points past the end of the surface"
    assert set(np.unique(faces)) <= set(verts.tolist()), (
        "a face is drawn from a vertex that is not one of the ones kept")
    #a face with a vertex twice over has no area left to draw
    assert (faces[:, 0] != faces[:, 1]).all()
    assert (faces[:, 1] != faces[:, 2]).all()
    assert (faces[:, 0] != faces[:, 2]).all()
    #and the same triangle is in there once
    assert len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces)

    #a wider group leaves fewer faces
    assert len(polyutils.decimate_faces(pts, polys, 16)[1]) < len(faces)
    #and no group at all leaves the surface as it is
    assert len(polyutils.decimate_faces(pts, polys, 0)[1]) == len(polys)


def test_decimate_faces_keeps_the_labels_apart():
    """A label a group may not cross, the medial wall for one, puts the
    vertices on either side of it in groups of their own."""
    import numpy as np

    from cortex import polyutils

    #a strip of a surface, as one row of squares cut into triangles, with its
    #two halves under different labels and narrow enough to be one group
    x = np.arange(9.)
    pts = np.vstack([np.repeat(x, 2), np.tile([0., 1.], len(x)),
                     np.zeros(2 * len(x))]).T
    polys = np.array([[i, i + 1, i + 2] for i in range(2 * len(x) - 2)])
    split = pts[:, 0] >= 4

    whole = polyutils.decimate_faces(pts, polys, 100)[0]
    halves = polyutils.decimate_faces(pts, polys, 100, split=split)[0]
    assert len(whole) == 1, "the strip is one group without a label"
    assert len(halves) == 2, "the label did not divide it"
    assert split[halves[0]] != split[halves[1]], (
        "the two groups stand for the same side of the label")


def test_get_lod_is_built_once_and_read_back(tmp_path):
    """The coarse faces are cut when they are first asked for and kept."""
    import os

    import numpy as np

    from cortex import brainctm, utils

    ctmargs = dict(method="mg2", level=9)
    path = utils.get_lod("S1", 16, **ctmargs)
    assert os.path.exists(path)
    written = os.path.getmtime(path)
    assert utils.get_lod("S1", 16, **ctmargs) == path
    assert os.path.getmtime(path) == written, "the file was cut a second time"

    raw = np.fromfile(path, dtype="<u4")
    counts = raw[:2]
    faces = raw[2:].reshape(-1, 3)
    assert len(faces) == counts.sum()

    base = os.path.splitext(utils.get_ctmpack("S1", **ctmargs))[0]
    hemis = brainctm.read_pack(base + ".ctm")
    at = 0
    for (pts, polys), count in zip(hemis, counts):
        hemi = faces[at:at + count]
        at += count
        assert hemi.max() < len(pts), "a face points past the end of the surface"
        assert len(hemi) < len(polys) / 50


def _patch(corner, du, dv, n):
    """A square patch of `n` by `n` cells from `corner` along `du` and `dv`,
    wound so that its normals point along cross(du, dv)."""
    import numpy as np

    corner, du, dv = (np.asarray(a, dtype=float) for a in (corner, du, dv))
    u, v = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    pts = corner + u.reshape(-1, 1) * du / n + v.reshape(-1, 1) * dv / n
    ids = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = (ids[:-1, :-1].ravel(), ids[1:, :-1].ravel(),
                  ids[1:, 1:].ravel(), ids[:-1, 1:].ravel())
    polys = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return pts, polys


def test_vertex_occlusion_is_none_on_a_plane_and_most_in_a_pit():
    """A plane blocks none of its own sky; the floor of a pit misses most
    of it and the rim of the pit about half."""
    import numpy as np

    from cortex import polyutils

    pts, polys = _patch([0, 0, 0], [40, 0, 0], [0, 40, 0], 20)
    assert polyutils.vertex_occlusion(pts, polys).max() == 0

    #an open shaft 10 mm wide and 20 mm deep, its walls facing in: from its
    #floor the rays that are followed reach every wall
    size, depth, n = 10., 20., 10
    parts = [
        _patch([0, 0, 0], [size, 0, 0], [0, size, 0], n),
        _patch([0, 0, 0], [0, 0, depth], [size, 0, 0], n),
        _patch([0, size, 0], [size, 0, 0], [0, 0, depth], n),
        _patch([0, 0, 0], [0, size, 0], [0, 0, depth], n),
        _patch([size, 0, 0], [0, 0, depth], [0, size, 0], n),
    ]
    pts, polys, at = [], [], 0
    for p, f in parts:
        pts.append(p)
        polys.append(f + at)
        at += len(p)
    occlusion = polyutils.vertex_occlusion(np.vstack(pts), np.vstack(polys))
    floor = occlusion[:(n + 1) ** 2].reshape(n + 1, n + 1)
    walls = occlusion[(n + 1) ** 2:].reshape(4, n + 1, n + 1)
    assert floor[n // 2, n // 2] > 0.6, "the floor of the shaft sees too much sky"
    #the top edge of a wall looks half into the shaft and half out of it
    assert 0.3 < walls[:, :, -1].mean() < 0.7
    assert floor.mean() > walls[:, :, -1].mean(), "the floor is no darker than the rim"
    assert len(occlusion) == len(np.vstack(pts))
    assert occlusion.dtype == np.float32


def test_get_occlusion_is_built_once_and_read_back():
    """The occlusion is worked out when it is first asked for and kept."""
    import json
    import os

    import numpy as np

    from cortex import brainctm, utils

    ctmargs = dict(method="mg2", level=9)
    path = utils.get_occlusion("S1", **ctmargs)
    assert os.path.exists(path)
    written = os.path.getmtime(path)
    assert utils.get_occlusion("S1", **ctmargs) == path
    assert os.path.getmtime(path) == written, "the file was built a second time"

    base = os.path.splitext(utils.get_ctmpack("S1", **ctmargs))[0]
    with open(base + ".json") as fp:
        pack = json.load(fp)
    counts = [len(pts) for pts, _ in brainctm.read_pack(base + ".ctm")]
    head = np.fromfile(path, dtype="<u4", count=3)
    values = np.fromfile(path, dtype=np.uint8, offset=12)
    #the folded surface at both depths, each of the pack's names, and the
    #flat surface
    assert head[0] == 3 + len(pack["names"])
    assert list(head[1:]) == counts
    assert len(values) == head[0] * sum(counts)
    values = values.reshape(head[0], sum(counts)) / 255.
    pial, wm, inflated, flat = values[0], values[1], values[2], values[-1]
    assert flat.max() == 0, "a flat surface blocks none of its own sky"
    assert 0.3 < pial.mean() < 0.9, "the pial surface is mostly sulcal wall"
    assert 0.3 < wm.mean() < 0.95, "the white matter surface is mostly sulcal wall"
    assert abs(pial - wm).mean() > 0.05, "the two depths came out the same"
    assert inflated.mean() < pial.mean() / 5, "inflation did not open the sulci"
