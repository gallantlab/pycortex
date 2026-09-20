"""Tests for shipping a `cortex.Tractogram` to the WebGL viewer.

Covers the python transport (`cortex.webgl.data.Package`, `make_static`) and,
when playwright + Chromium are available, the javascript side
(`resources/js/tractogram.js`).
"""

import os

import numpy as np
import pytest

import cortex
from cortex.webgl.data import Package

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
        # viewer.loaded, so poll until the geometry has been built
        # (Tractogram.n_points stays 0 until then).
        import time

        deadline = time.monotonic() + 30
        n_points = 0
        while time.monotonic() < deadline:
            n_points = handle.tracts.af.n_points
            if n_points:
                break
            time.sleep(0.2)

        n, m = tract.n_points, tract.n_streamlines
        assert n_points == n
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
        assert set(handle.tracts.af.groupNames()[0]) == {
            "first_half",
            "second_half",
            "(ungrouped)",
        }
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
