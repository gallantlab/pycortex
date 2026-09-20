"""Tests for cortex.dataset.tractogram.Tractogram."""

import zipfile
import json

import numpy as np
import pytest

import cortex
from cortex import db, dataset
from cortex.dataset import Tractogram

subj = "S1"


def _make_streamlines(n_streamlines=20, n_points=30, seed=0):
    """Build synthetic streamlines derived from S1's fiducial surface.

    Each streamline is a straight line between two random fiducial
    vertices, with a sinusoidal bulge added orthogonal to the line, so
    that the local tangent direction varies along the streamline (needed
    to exercise the "orientation" coloring mode).
    """
    rng = np.random.default_rng(seed)
    pts, _ = db.get_surf(subj, "fiducial", merge=True)
    n_verts = pts.shape[0]

    streamlines = []
    for _ in range(n_streamlines):
        i0, i1 = rng.choice(n_verts, size=2, replace=False)
        p0, p1 = pts[i0], pts[i1]
        t = np.linspace(0, 1, n_points)[:, None]
        line = p0[None, :] * (1 - t) + p1[None, :] * t

        # orthogonal bulge so tangents aren't all parallel
        direction = p1 - p0
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            ortho = np.cross(direction, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(ortho) < 1e-6:
                ortho = np.cross(direction, np.array([0.0, 1.0, 0.0]))
            ortho = ortho / (np.linalg.norm(ortho) + 1e-12)
            bulge = 5.0 * np.sin(np.linspace(0, np.pi, n_points))
            line = line + bulge[:, None] * ortho[None, :]

        streamlines.append(line.astype(np.float32))
    return streamlines


def _make_tractogram(n_streamlines=20, n_points=30, seed=0, groups=None, **kwargs):
    streamlines = _make_streamlines(n_streamlines, n_points, seed=seed)
    n_total = sum(s.shape[0] for s in streamlines)

    rng = np.random.default_rng(seed + 1)
    dpv = {"scalar": rng.standard_normal(n_total).astype(np.float32)}
    dps = {"length": np.array([s.shape[0] for s in streamlines], dtype=np.float32)}
    if groups is None:
        groups = {
            "even": np.arange(0, n_streamlines, 2),
            "odd": np.arange(1, n_streamlines, 2),
        }
    return Tractogram.from_streamlines(
        streamlines, subj, dpv=dpv, dps=dps, groups=groups, **kwargs
    )


def _overlapping_groups(n_streamlines):
    """Two overlapping groups that leave the last streamline ungrouped.

    Used by tests (and by `test_webgl_tractogram.py`) that need to exercise
    the "streamline belongs to >=1 group" / "streamline belongs to no
    group" cases together, which the disjoint even/odd split above does not.
    """
    half = n_streamlines // 2
    return {
        "first_half": np.arange(0, half + 1),
        "second_half": np.arange(half - 1, n_streamlines - 1),
    }


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_construction_basic():
    tract = _make_tractogram()
    assert tract.n_streamlines == 20
    assert len(tract) == 20
    assert tract.n_points == tract.points.shape[0]
    assert tract.offsets.shape == (21,)
    assert tract.offsets[0] == 0
    assert tract.offsets[-1] == tract.n_points
    assert tract.subject == subj


def test_offsets_sentinel_appended():
    points = np.random.randn(10, 3).astype(np.float32)
    offsets_no_sentinel = np.array([0, 5])  # M=2 offsets, no trailing N
    tract = Tractogram(points, offsets_no_sentinel, subj)
    assert tract.offsets.shape == (3,)
    assert tract.offsets[-1] == 10
    assert tract.n_streamlines == 2


def test_points_wrong_shape_raises():
    with pytest.raises(ValueError):
        Tractogram(np.zeros((10, 2)), np.array([0, 10]), subj)


def test_offsets_not_monotonic_raises():
    points = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        Tractogram(points, np.array([0, 8, 5, 10]), subj)


def test_offsets_must_start_at_zero():
    points = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        Tractogram(points, np.array([1, 10]), subj)


def test_dpv_wrong_size_raises():
    points = np.zeros((10, 3), dtype=np.float32)
    offsets = np.array([0, 10])
    with pytest.raises(ValueError):
        Tractogram(points, offsets, subj, dpv={"bad": np.zeros(5)})


def test_dps_wrong_size_raises():
    points = np.zeros((10, 3), dtype=np.float32)
    offsets = np.array([0, 5, 10])
    with pytest.raises(ValueError):
        Tractogram(points, offsets, subj, dps={"bad": np.zeros(1)})


def test_group_out_of_range_raises():
    points = np.zeros((10, 3), dtype=np.float32)
    offsets = np.array([0, 5, 10])
    with pytest.raises(ValueError):
        Tractogram(points, offsets, subj, groups={"bad": np.array([5])})


def test_unrecognized_color_raises():
    points = np.zeros((10, 3), dtype=np.float32)
    offsets = np.array([0, 10])
    with pytest.raises(ValueError):
        Tractogram(points, offsets, subj, color="not-a-valid-spec")


def test_unknown_dpv_dps_color_raises():
    points = np.zeros((10, 3), dtype=np.float32)
    offsets = np.array([0, 10])
    with pytest.raises(ValueError):
        Tractogram(points, offsets, subj, color="dpv:missing")
    with pytest.raises(ValueError):
        Tractogram(points, offsets, subj, color="dps:missing")


# ---------------------------------------------------------------------------
# lengths / offsets / streamlines invariants
# ---------------------------------------------------------------------------


def test_lengths_and_streamlines():
    tract = _make_tractogram(n_streamlines=5, n_points=7)
    assert np.array_equal(tract.lengths, np.diff(tract.offsets))
    assert tract.lengths.sum() == tract.n_points
    streamlines = tract.streamlines
    assert len(streamlines) == tract.n_streamlines
    for sl, length in zip(streamlines, tract.lengths):
        assert sl.shape == (length, 3)


def test_name_is_deterministic_hash():
    tract1 = _make_tractogram(seed=1)
    tract2 = Tractogram(tract1.points.copy(), tract1.offsets.copy(), subj)
    assert tract1.name == tract2.name
    tract3 = _make_tractogram(seed=2)
    assert tract1.name != tract3.name


# ---------------------------------------------------------------------------
# select / get_group / subsample
# ---------------------------------------------------------------------------


def _assert_offsets_valid(tract):
    assert tract.offsets[0] == 0
    assert tract.offsets[-1] == tract.n_points
    assert np.all(np.diff(tract.offsets) >= 0)


def test_select_basic():
    tract = _make_tractogram(n_streamlines=10, n_points=5)
    sub = tract.select([0, 2, 4])
    assert sub.n_streamlines == 3
    _assert_offsets_valid(sub)
    assert sub.dpv["scalar"].shape[0] == sub.n_points
    assert sub.dps["length"].shape[0] == sub.n_streamlines
    np.testing.assert_array_equal(sub.streamlines[0], tract.streamlines[0])
    np.testing.assert_array_equal(sub.streamlines[1], tract.streamlines[2])


def test_select_remaps_groups():
    tract = _make_tractogram(n_streamlines=10, n_points=5)
    sub = tract.select([1, 3, 5, 7, 9])  # exactly the "odd" group
    assert set(sub.groups["odd"].tolist()) == {0, 1, 2, 3, 4}
    assert sub.groups["even"].size == 0


def test_get_group():
    tract = _make_tractogram(n_streamlines=10, n_points=5)
    even = tract.get_group("even")
    assert even.n_streamlines == 5
    _assert_offsets_valid(even)
    with pytest.raises(KeyError):
        tract.get_group("nope")


def test_subsample_max_streamlines():
    tract = _make_tractogram(n_streamlines=20, n_points=5)
    sub = tract.subsample(max_streamlines=5, seed=42)
    assert sub.n_streamlines == 5
    _assert_offsets_valid(sub)
    # deterministic given seed
    sub2 = tract.subsample(max_streamlines=5, seed=42)
    np.testing.assert_array_equal(sub.points, sub2.points)


def test_subsample_step():
    tract = _make_tractogram(n_streamlines=20, n_points=5)
    sub = tract.subsample(step=4)
    assert sub.n_streamlines == 5
    _assert_offsets_valid(sub)


def test_subsample_noop_when_small():
    tract = _make_tractogram(n_streamlines=5, n_points=5)
    sub = tract.subsample(max_streamlines=100)
    assert sub.n_streamlines == tract.n_streamlines


# ---------------------------------------------------------------------------
# vertex_colors
# ---------------------------------------------------------------------------


def test_vertex_colors_orientation():
    tract = _make_tractogram(color="orientation")
    colors = tract.vertex_colors()
    assert colors.shape == (tract.n_points, 3)
    assert colors.dtype == np.uint8
    assert colors.min() >= 0 and colors.max() <= 255

    # unit-tangent property: recompute manually for the first streamline
    seg = tract.streamlines[0].astype(np.float64)
    tangent = np.empty_like(seg)
    tangent[1:-1] = seg[2:] - seg[:-2]
    tangent[0] = seg[1] - seg[0]
    tangent[-1] = seg[-1] - seg[-2]
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    expected = np.clip(np.abs(tangent / norm) * 255, 0, 255).astype(np.uint8)
    start, stop = tract.offsets[0], tract.offsets[1]
    np.testing.assert_allclose(colors[start:stop], expected, atol=1)


def test_vertex_colors_constant():
    tract = _make_tractogram(color=(1.0, 0.0, 0.5))
    colors = tract.vertex_colors()
    assert colors.shape == (tract.n_points, 3)
    expected = np.array([255, 0, 127], dtype=np.uint8)
    assert np.all(colors[0] == expected)
    assert np.all(colors == colors[0])  # constant everywhere

    tract255 = _make_tractogram(color=(255, 0, 128))
    colors255 = tract255.vertex_colors()
    assert np.all(colors255[0] == np.array([255, 0, 128], dtype=np.uint8))


def test_vertex_colors_dpv():
    tract = _make_tractogram(color="dpv:scalar", cmap="gray")
    colors = tract.vertex_colors()
    assert colors.shape == (tract.n_points, 3)
    assert colors.dtype == np.uint8
    scalar = tract.dpv["scalar"]
    order = np.argsort(scalar)
    # a gray colormap should be monotonic with the scalar (all channels equal)
    gray_channel = colors[:, 0].astype(np.float64)
    assert np.all(colors[:, 0] == colors[:, 1])
    assert np.all(colors[:, 1] == colors[:, 2])
    sorted_gray = gray_channel[order]
    assert np.all(np.diff(sorted_gray) >= -1)  # monotone non-decreasing (allow tie noise)


def test_vertex_colors_dpv_nan_is_gray():
    tract = _make_tractogram(color="dpv:scalar", cmap="gray")
    tract.dpv["scalar"][0] = np.nan
    colors = tract.vertex_colors()
    assert tuple(colors[0]) == (128, 128, 128)


def test_vertex_colors_dps():
    tract = _make_tractogram(color="dps:length", cmap="gray")
    colors = tract.vertex_colors()
    assert colors.shape == (tract.n_points, 3)
    # all points within a streamline share the same dps-derived color
    start, stop = tract.offsets[0], tract.offsets[1]
    assert np.all(colors[start:stop] == colors[start])


# ---------------------------------------------------------------------------
# to_json / _write_hdf / normalize
# ---------------------------------------------------------------------------


def test_to_json_keys():
    tract = _make_tractogram(color="dpv:scalar")
    j = tract.to_json()
    expected_keys = {
        "subject",
        "n_points",
        "n_streamlines",
        "alpha",
        "linewidth",
        "visible",
        "color",
        "groups",
        "description",
        "cmap",
        "vmin",
        "vmax",
    }
    assert expected_keys <= set(j.keys())
    assert j["subject"] == subj
    assert j["n_points"] == tract.n_points
    assert j["n_streamlines"] == tract.n_streamlines
    assert j["visible"] is True
    assert j["groups"] == {"even": [0, 10], "odd": [10, 20]}
    # must be JSON serializable
    json.dumps(j)


def test_groups_wire_disjoint():
    tract = _make_tractogram(n_streamlines=10, n_points=5)
    indices, slices = tract.groups_wire()
    assert indices.dtype == np.uint32
    assert indices.shape == (10,)
    assert slices == {"even": (0, 5), "odd": (5, 10)}
    for name, (start, stop) in slices.items():
        assert stop - start == len(tract.groups[name])
        np.testing.assert_array_equal(indices[start:stop], tract.groups[name])
    # to_json must report the very same slice bounds.
    j = tract.to_json()
    assert j["groups"] == {name: list(s) for name, s in slices.items()}


def test_groups_wire_overlapping_and_ungrouped():
    n_streamlines = 8
    groups = _overlapping_groups(n_streamlines)
    tract = _make_tractogram(
        n_streamlines=n_streamlines, n_points=5, groups=groups
    )
    indices, slices = tract.groups_wire()
    assert indices.shape[0] == sum(len(g) for g in groups.values())
    # slices are contiguous and in the same order as `groups`
    assert list(slices.keys()) == list(groups.keys())
    pos = 0
    for name, idx in groups.items():
        start, stop = slices[name]
        assert (start, stop) == (pos, pos + len(idx))
        np.testing.assert_array_equal(indices[start:stop], idx)
        pos += len(idx)
    # groups overlap: streamline `half - 1` .. `half` appear in both
    first, second = slices["first_half"], slices["second_half"]
    first_members = set(indices[first[0] : first[1]].tolist())
    second_members = set(indices[second[0] : second[1]].tolist())
    assert first_members & second_members
    # the last streamline belongs to no group
    all_members = first_members | second_members
    assert (n_streamlines - 1) not in all_members


def test_groups_wire_empty():
    tract = _make_tractogram(n_streamlines=5, n_points=5, groups={})
    indices, slices = tract.groups_wire()
    assert indices.shape == (0,)
    assert indices.dtype == np.uint32
    assert slices == {}
    assert tract.to_json()["groups"] == {}


def test_to_json_simple_omits_cmap():
    tract = _make_tractogram(color="dpv:scalar")
    j = tract.to_json(simple=True)
    assert "cmap" not in j
    assert "vmin" not in j


def test_to_json_non_scalar_color_no_cmap():
    tract = _make_tractogram(color="orientation")
    j = tract.to_json()
    assert "cmap" not in j


def test_write_hdf_raises():
    tract = _make_tractogram()
    with pytest.raises(NotImplementedError):
        tract._write_hdf(None)


def test_normalize_passthrough():
    tract = _make_tractogram()
    assert dataset.normalize(tract) is tract
    assert cortex.dataset.normalize(tract) is tract


# ---------------------------------------------------------------------------
# from_trx (requires trx-python)
# ---------------------------------------------------------------------------


def _write_trx_fixture(path, points, offsets, dpv=None, dps=None, groups=None):
    """Write a minimal TRX zip archive by hand (no trx-python dependency)."""
    n_points = points.shape[0]
    n_streamlines = offsets.shape[0] - 1
    header = {
        "VOXEL_TO_RASMM": np.eye(4).tolist(),
        "DIMENSIONS": [10, 10, 10],
        "NB_VERTICES": int(n_points),
        "NB_STREAMLINES": int(n_streamlines),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("header.json", json.dumps(header))
        zf.writestr(
            "positions.3.float32", points.astype("<f4").tobytes()
        )
        zf.writestr("offsets.uint64", offsets.astype("<u8").tobytes())
        for name, arr in (dpv or {}).items():
            zf.writestr(f"dpv/{name}.float32", arr.astype("<f4").tobytes())
        for name, arr in (dps or {}).items():
            zf.writestr(f"dps/{name}.float32", arr.astype("<f4").tobytes())
        for name, idx in (groups or {}).items():
            zf.writestr(f"groups/{name}.uint32", idx.astype("<u4").tobytes())


def _build_trx_fixture(tmp_path):
    streamlines = _make_streamlines(n_streamlines=6, n_points=8, seed=3)
    lengths = [s.shape[0] for s in streamlines]
    points = np.concatenate(streamlines, axis=0).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint64)
    n_points = points.shape[0]
    dpv = {"scalar": np.arange(n_points, dtype=np.float32)}
    dps = {"length": np.array(lengths, dtype=np.float32)}
    groups = {"first_half": np.array([0, 1, 2], dtype=np.uint32)}

    path = tmp_path / "fixture.trx"
    _write_trx_fixture(path, points, offsets, dpv=dpv, dps=dps, groups=groups)
    return path, points, offsets, dpv, dps, groups


def test_from_trx_roundtrip(tmp_path):
    pytest.importorskip("trx")
    path, points, offsets, dpv, dps, groups = _build_trx_fixture(tmp_path)

    tract = Tractogram.from_trx(path, subj)
    np.testing.assert_allclose(tract.points, points, atol=1e-4)
    assert tract.n_streamlines == offsets.shape[0] - 1
    np.testing.assert_array_equal(tract.offsets, offsets.astype(np.int64))
    np.testing.assert_allclose(tract.dpv["scalar"], dpv["scalar"], atol=1e-4)
    np.testing.assert_allclose(tract.dps["length"], dps["length"], atol=1e-4)
    np.testing.assert_array_equal(
        np.sort(tract.groups["first_half"]), np.sort(groups["first_half"])
    )


def test_from_trx_xfm_shifts_points(tmp_path):
    pytest.importorskip("trx")
    path, points, offsets, dpv, dps, groups = _build_trx_fixture(tmp_path)

    shift = np.array([1.0, 2.0, 3.0])
    xfm = np.eye(4)
    xfm[:3, 3] = shift

    tract = Tractogram.from_trx(path, subj, xfm=xfm)
    np.testing.assert_allclose(tract.points, points + shift, atol=1e-3)
