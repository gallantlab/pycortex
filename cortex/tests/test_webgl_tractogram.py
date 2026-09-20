"""Tests for shipping a `cortex.Tractogram` to the WebGL viewer.

Covers the python transport (`cortex.webgl.data.Package`, `make_static`) and,
when playwright + Chromium are available, the javascript side
(`resources/js/tractogram.js`).
"""

import os
import re

import numpy as np
import pytest

import cortex
from cortex.webgl.data import Package, _tract_id

from .testing_utils import has_playwright
from .test_tractogram import _make_tractogram, _overlapping_groups

subj = "S1"


def _vertex():
    """A small Vertex dataview for S1, to accompany the tractogram."""
    pts, _ = cortex.db.get_surf(subj, "fiducial", merge=True)
    return cortex.Vertex(np.zeros(pts.shape[0], dtype=np.float32), subj)


def _dataset():
    # Overlapping groups + an ungrouped streamline, to exercise the viewer's
    # per-group visibility toggles (a streamline is visible iff it is a
    # member of >=1 visible group; ungrouped streamlines follow a pseudo
    # "(ungrouped)" group).
    tract = _make_tractogram(
        n_streamlines=8, n_points=15, groups=_overlapping_groups(8)
    )
    return cortex.Dataset(overlay=_vertex(), af=tract), tract


# ---------------------------------------------------------------------------
# Package: wire format
# ---------------------------------------------------------------------------


def test_package_tract_metadata_and_buffers():
    ds, tract = _dataset()
    pkg = Package(ds)

    meta = pkg.metadata()
    assert "af" in meta["tracts"]
    tmeta = meta["tracts"]["af"]

    assert tmeta["subject"] == subj
    assert tmeta["n_points"] == tract.n_points
    assert tmeta["n_streamlines"] == tract.n_streamlines
    assert tmeta["alpha"] == tract.alpha
    assert tmeta["linewidth"] == tract.linewidth
    assert tmeta["visible"] is True
    assert set(tmeta["urls"]) == {"points", "offsets", "colors", "groups"}
    assert tmeta["urls"]["points"] == "/tract/af/points/"

    bufs = pkg.tracts["af"]
    n, m = tract.n_points, tract.n_streamlines
    n_group_entries = sum(len(idx) for idx in tract.groups.values())
    assert len(bufs["points"]) == 12 * n
    assert len(bufs["offsets"]) == 4 * (m + 1)
    assert len(bufs["colors"]) == 3 * n
    assert len(bufs["groups"]) == 4 * n_group_entries

    # The buffers must round-trip as little-endian arrays of the right dtype.
    points = np.frombuffer(bufs["points"], dtype="<f4").reshape(-1, 3)
    assert np.allclose(points, tract.points)
    offsets = np.frombuffer(bufs["offsets"], dtype="<u4")
    assert np.array_equal(offsets, tract.offsets)
    assert offsets[-1] == n

    # Metadata "groups" is {name: [start, stop]} slicing into the buffer,
    # contiguous and in the same order as `tract.groups`.
    group_buf = np.frombuffer(bufs["groups"], dtype="<u4")
    assert tmeta["groups"].keys() == tract.groups.keys()
    pos = 0
    for name, idx in tract.groups.items():
        start, stop = tmeta["groups"][name]
        assert (start, stop) == (pos, pos + len(idx))
        assert stop - start == len(idx)
        np.testing.assert_array_equal(group_buf[start:stop], idx)
        pos += len(idx)


def test_tract_id_is_url_and_path_safe():
    # Ordinary names travel unchanged, so urls and file names stay readable.
    assert _tract_id("af") == "af"
    assert _tract_id("AF_left-1") == "AF_left-1"

    # Anything else is slugified: a dataset key is an arbitrary string, and
    # both the "/tract/{id}/{buf}/" urls and the "tracts/{id}_{buf}.bin"
    # files would otherwise take a "/" or a ".." literally.
    for name in ["../escape", "a/b", "tract name", "#frag?q", ".."]:
        ident = _tract_id(name)
        assert re.match(r"^[A-Za-z0-9_-]+$", ident), ident
        assert ident == _tract_id(name), "ids must be deterministic"

    # Names that slugify alike stay distinct.
    assert _tract_id("a/b") != _tract_id("a b")
    # ... and a very long key cannot produce an unopenable file name.
    assert len(_tract_id("x" * 500)) <= 64


def test_package_tract_ids_used_in_urls():
    tract = _make_tractogram(n_streamlines=4, n_points=10)
    name = "../escape"
    pkg = Package(cortex.Dataset(**{name: tract}), require_brains=False)

    ident = pkg.tract_ids[name]
    assert ".." not in ident and "/" not in ident
    # The metadata is still keyed by the name the user gave; only the
    # transport urls use the id.
    urls = pkg.metadata()["tracts"][name]["urls"]
    assert urls["points"] == "/tract/%s/points/" % ident


def test_make_static_sanitizes_tract_names(tmp_path):
    # A dataset key is an arbitrary string, and it used to be spliced
    # straight into the buffer file names: "../../escape" would have written
    # two directories up from `outpath`.
    tract = _make_tractogram(n_streamlines=4, n_points=10)
    ds = cortex.Dataset(**{"overlay": _vertex(), "../../escape": tract})
    outpath = tmp_path / "static"
    cortex.webgl.make_static(str(outpath), ds, recache=False)

    written = sorted(os.listdir(str(outpath / "tracts")))
    assert len(written) == 4
    for fname in written:
        assert ".." not in fname
    # Nothing landed next to (rather than inside) the output directory.
    assert sorted(os.listdir(str(tmp_path))) == ["static"]


def test_package_empty_groups_gives_empty_buffer():
    tract = _make_tractogram(n_streamlines=4, n_points=10, groups={})
    pkg = Package(cortex.Dataset(af=tract), require_brains=False)

    assert pkg.tracts["af"]["groups"] == b""
    meta = pkg.metadata()
    assert meta["tracts"]["af"]["groups"] == {}


def test_package_keeps_tracts_out_of_views_and_data():
    """Every existing javascript path must see exactly what it saw before."""
    ds, tract = _dataset()
    pkg = Package(ds)
    meta = pkg.metadata()

    assert [view["name"] for view in meta["views"]] == ["overlay"]
    assert "af" not in meta["data"]
    assert "af" not in meta["images"]
    assert tract.name not in meta["data"]
    # ... and the tractogram is not among the BrainData that get reordered.
    assert all(not isinstance(u, cortex.Tractogram) for u in pkg.uniques)


def test_package_uniques_are_deduplicated_by_name():
    # Two dataviews holding the same array are one brain on the wire (they
    # share a name, which is a hash of the data), so they must appear once in
    # `uniques`: reorder() rewrites self.images[name] in place, and a second
    # pass would run over the first pass's output.
    zeros = np.zeros(cortex.db.get_surf(subj, "fiducial", merge=True)[0].shape[0],
                     dtype=np.float32)
    ds = cortex.Dataset(
        a=cortex.Vertex(zeros.copy(), subj),
        b=cortex.Vertex(zeros.copy(), subj),
    )
    pkg = Package(ds)
    assert len(pkg.uniques) == 1
    assert len(pkg.brains) == 1
    # ... while both dataviews still reach the viewer.
    assert [view["name"] for view in pkg.views] == ["a", "b"]


def test_package_reorder_ignores_tracts():
    tract = _make_tractogram(n_streamlines=8, n_points=15)
    pkg = Package(cortex.Dataset(af=tract), require_brains=False)
    before = pkg.tracts["af"]["points"]
    # reorder() needs a per-subject CTM index for every BrainData it holds;
    # with only a tractogram there is nothing to reorder, so no index is
    # needed and the buffers must come out untouched.
    pkg.reorder({})
    assert pkg.tracts["af"]["points"] == before


def test_package_tractogram_only_raises():
    tract = _make_tractogram(n_streamlines=4, n_points=10)
    with pytest.raises(ValueError, match="cannot be displayed on its own"):
        Package(cortex.Dataset(af=tract))
    with pytest.raises(ValueError, match="cannot be displayed on its own"):
        Package(tract)

    # A running viewer can still be handed a tractogram alone.
    pkg = Package(cortex.Dataset(af=tract), require_brains=False)
    assert "af" in pkg.tracts


def test_package_tract_without_a_dataview_of_its_subject_raises():
    # A tractogram is only drawn while a dataview of its own subject is
    # active, so one whose subject has no dataview at all would never show.
    tract = _make_tractogram(n_streamlines=4, n_points=10)
    tract.subject = "someone_else"
    with pytest.raises(ValueError, match="someone_else"):
        Package(cortex.Dataset(overlay=_vertex(), af=tract))

    # Not the boot path's problem when the viewer is already up, though: the
    # caller there gets the "unknown subject" check in JSMixer.addData.
    pkg = Package(
        cortex.Dataset(overlay=_vertex(), af=tract), require_brains=False
    )
    assert pkg.tract_meta["af"]["subject"] == "someone_else"


# ---------------------------------------------------------------------------
# make_static
# ---------------------------------------------------------------------------


def test_make_static_writes_tract_buffers(tmp_path):
    ds, tract = _dataset()
    outpath = str(tmp_path / "static")
    cortex.webgl.make_static(outpath, ds, recache=False)

    n, m = tract.n_points, tract.n_streamlines
    n_group_entries = sum(len(idx) for idx in tract.groups.values())
    expected = {
        "af_points.bin": 12 * n,
        "af_offsets.bin": 4 * (m + 1),
        "af_colors.bin": 3 * n,
        "af_groups.bin": 4 * n_group_entries,
    }
    for fname, size in expected.items():
        path = os.path.join(outpath, "tracts", fname)
        assert os.path.exists(path), "%s was not written" % fname
        assert os.path.getsize(path) == size

    with open(os.path.join(outpath, "index.html")) as fp:
        html = fp.read()
    assert "tracts/af_points.bin" in html


# ---------------------------------------------------------------------------
# save_3d_views subject resolution
# ---------------------------------------------------------------------------


def test_save_3d_views_resolves_subject_from_dataset():
    """A tractogram can only be rendered inside a Dataset, so the screenshot
    helpers have to find the subject there rather than on a lone dataview."""
    from cortex.export.save_views import _view_subject

    ds, _ = _dataset()
    assert _view_subject(ds) == subj
    assert _view_subject(_vertex()) == subj


def test_save_3d_views_subject_survives_a_view_named_subject():
    """`Dataset.__getattr__` falls through to the views it holds, so a view
    named "subject" would answer `getattr(ds, "subject")` with a dataview.
    Resolution has to go by type, not by duck-typing on the attribute."""
    from cortex.export.save_views import _view_subject

    _, tract = _dataset()
    ds = cortex.Dataset(subject=_vertex(), bundles=tract)
    assert _view_subject(ds) == subj


def test_save_3d_views_rejects_ambiguous_subject():
    """Two subjects in one Dataset is not renderable: the viewer's surface
    controls are addressed by subject name, so there is nothing to format
    "surface.{subject}.unfold" with. Stand-ins rather than real dataviews,
    since the bundled filestore only has one subject."""
    from types import SimpleNamespace

    from cortex.export.save_views import _view_subject

    two_subjects = cortex.Dataset(a=_vertex())
    two_subjects.views["b"] = SimpleNamespace(subject="S2")
    with pytest.raises(ValueError, match="exactly one subject"):
        _view_subject(two_subjects)

    with pytest.raises(ValueError, match="Cannot determine the subject"):
        _view_subject(object())


# ---------------------------------------------------------------------------
# Headless browser
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not has_playwright, reason="playwright + Chromium not available"
)
def test_tractogram_renders_in_headless_viewer():
    ds, tract = _dataset()

    with cortex.export.headless_viewer(ds, viewer_params={}) as handle:
        # `handle` is a JSProxy rooted at window.viewer: attribute access
        # queries the live javascript object graph over the websocket.
        # The tracts load asynchronously and deliberately do not block
        # viewer.loaded, but headless_viewer waits for viewer.tractsState()
        # on top of it -- so the geometry is already built here, with no
        # polling of our own (Tractogram.n_points stays 0 until then).
        assert handle.send(
            method="run", params=["window.viewer.tractsState", []]
        ) == ["resolved"]

        n, m = tract.n_points, tract.n_streamlines
        assert handle.tracts.af.n_points == n
        assert handle.tracts.af.n_streamlines == m
        assert handle.tracts.af.object.visible is True
        # Controls now live in a dedicated DOM panel (#tracts) instead of a
        # dat.gui folder: just confirm the per-tractogram element was built.
        assert "element" in dir(handle.tracts.af)
        # Either the indexed geometry (one vertex per point) or the
        # duplicated-vertex fallback (two per segment).
        full_n_vertices = handle.tracts.af.n_vertices
        assert full_n_vertices in (n, 2 * (n - m))

        # Per-group visibility: hiding "first_half" must shrink the drawn
        # geometry (fewer visible streamlines -> fewer segments; on the
        # indexed path only the index shrinks, so count segments rather
        # than vertices), and showAllGroups() must restore it. JSProxy calls
        # return the per-client response list, hence the [0].
        full_n_segments = handle.tracts.af.n_segments
        assert full_n_segments == n - m
        # Alphabetical, with the synthetic "(ungrouped)" entry last -- not
        # the order the groups arrived in, which the fixture deliberately
        # makes the reverse of alphabetical.
        assert handle.tracts.af.groupNames()[0] == [
            "first_half",
            "second_half",
            "(ungrouped)",
        ]
        handle.tracts.af.setGroupVisible("first_half", False)
        shrunk_n_segments = handle.tracts.af.n_segments
        assert 0 < shrunk_n_segments < full_n_segments

        handle.tracts.af.showAllGroups()
        assert handle.tracts.af.n_segments == full_n_segments

        # Translucent streamlines must keep writing depth. Without it they
        # have nothing to depth-test against each other and blend in buffer
        # order, so whichever bundle sits last in the geometry paints over
        # the ones in front of it and the picture reorders itself the moment
        # opacity leaves 1.
        assert handle.tracts.af.material.depthWrite is True
        handle.tracts.af.setOpacity(0.5)
        assert handle.tracts.af.material.transparent is True
        assert handle.tracts.af.material.depthWrite is True
        # Called with no argument setOpacity is the getter; as with
        # groupNames() above, a JSProxy call answers with the per-client
        # response list.
        assert handle.tracts.af.setOpacity()[0] == 0.5
        # Out-of-range and unparsable input is clamped / ignored rather than
        # reaching the material (the panel's number box accepts typing).
        handle.tracts.af.setOpacity(5)
        assert handle.tracts.af.setOpacity()[0] == 1
        handle.tracts.af.setOpacity("")
        assert handle.tracts.af.setOpacity()[0] == 1

        # A tractogram belongs to one subject's scanner space, so it steps
        # aside (lines and panel entry both) while another subject's dataview
        # is the active one -- there is only one subject in the test
        # filestore, so move the tractogram to a fictitious one instead of
        # building a two-subject dataset.
        handle.send(
            method="set",
            params=["window.viewer.tracts.af.meta.subject", "not_" + subj],
        )
        handle.send(method="run", params=["window.viewer._updateTractSubjects", []])
        assert handle.tracts.af.object.visible is False
        assert handle.tracts.af.setSubjectShown()[0] is False
        handle.send(
            method="set", params=["window.viewer.tracts.af.meta.subject", subj]
        )
        handle.send(method="run", params=["window.viewer._updateTractSubjects", []])
        assert handle.tracts.af.object.visible is True


@pytest.mark.skipif(
    not has_playwright, reason="playwright + Chromium not available"
)
def test_tractogram_index_modes_in_headless_viewer():
    """The three element-index strategies, and the fallback's geometry.

    Which one a geometry gets depends on its size and on whether the GL
    context has OES_element_index_uint, so a small tractogram in one browser
    only ever exercises one of them. `mriview.indexModeFor` is the choice on
    its own (checkable with plain arguments), and `forceIndexMode` pins it,
    so the duplicated-vertex fallback can be built and compared against the
    indexed geometry it replaces without uploading 65536+ points.
    """
    ds, tract = _dataset()
    n, m = tract.n_points, tract.n_streamlines

    def run(func, *args):
        return handle.send(method="run", params=[func, list(args)])[0]

    with cortex.export.headless_viewer(ds, viewer_params={}) as handle:
        # Small geometries stay on uint16 whatever the extension says; above
        # 65535 vertices the extension decides between a 32-bit index and
        # duplicating the endpoints of every segment.
        assert run("window.mriview.indexModeFor", 100, True) == "uint16"
        assert run("window.mriview.indexModeFor", 100, False) == "uint16"
        assert run("window.mriview.indexModeFor", 65535, False) == "uint16"
        assert run("window.mriview.indexModeFor", 65536, True) == "uint32"
        assert run("window.mriview.indexModeFor", 65536, False) == "duplicate"

        af = handle.tracts.af
        assert af.indexType == "uint16"
        indexed_segments = af.n_segments
        assert indexed_segments == n - m
        assert af.n_vertices == n
        # Endpoints and colors of a segment, as they reach the GPU.
        indexed = run("window.viewer.tracts.af.segment", 3)

        # Same geometry without an element index: the endpoints of every
        # segment are duplicated instead, so the vertex count doubles per
        # segment while the drawn segments -- and their positions and
        # colors -- must come out identical.
        handle.send(
            method="set",
            params=["window.viewer.tracts.af.forceIndexMode", "duplicate"],
        )
        handle.tracts.af._rebuildGeometry()
        assert handle.tracts.af.indexType == "duplicate"
        assert handle.tracts.af.n_segments == indexed_segments
        assert handle.tracts.af.n_vertices == 2 * indexed_segments
        assert run("window.viewer.tracts.af.segment", 3) == indexed

        # ... and back, so the fallback is not left pinned on the viewer.
        handle.send(
            method="set",
            params=["window.viewer.tracts.af.forceIndexMode", None],
        )
        handle.tracts.af._rebuildGeometry()
        assert handle.tracts.af.indexType == "uint16"
        assert handle.tracts.af.n_vertices == n


@pytest.mark.skipif(
    not has_playwright, reason="playwright + Chromium not available"
)
def test_tractogram_named_like_an_object_prototype_member():
    """A dataset key is an arbitrary string, including "constructor".

    The viewer keys its tractograms by name in a plain object, so a name that
    collides with an inherited Object.prototype member used to look like a
    tractogram that was already loaded -- addTracts would then "replace" it
    and rmTracts would trip over the inherited value.
    """
    tract = _make_tractogram(n_streamlines=4, n_points=10)
    ds = cortex.Dataset(overlay=_vertex(), constructor=tract)

    with cortex.export.headless_viewer(ds, viewer_params={}) as handle:
        assert handle.send(
            method="run", params=["window.viewer.hasTract", ["constructor"]]
        ) == [True]
        assert handle.send(
            method="run", params=["window.viewer.hasTract", ["toString"]]
        ) == [False]
        assert handle.tracts.constructor.n_points == tract.n_points
