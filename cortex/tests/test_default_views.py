"""Tests for the views every subject gets, and the quickflat size hint.

The camera geometry is asserted rather than described: `default_subject_views`
names its four orientations anatomically ("dorsal", "lateral_left", ...), and
nothing else in the codebase checks that those names match what the viewer
actually shows. The helpers below reproduce the viewer's own camera arithmetic
so the claim is testable without a browser.
"""

import json
import math

import pytest

from cortex.export.save_views import (
    DEFAULT_VIEW_ANGLES,
    FLAT_INERT_PROPS,
    FLAT_VIEW_NAME,
    INFLATED_SUFFIX,
    angle_view_params,
    camera_basis,
    default_subject_views,
)

# Surface RAS, which is what pycortex surfaces are in.
RIGHT = (1.0, 0.0, 0.0)
ANTERIOR = (0.0, 1.0, 0.0)
SUPERIOR = (0.0, 0.0, 1.0)

def points_along(vector, axis, tol=1e-3):
    """Whether `vector` points the same way as the unit `axis`."""
    dot = sum(a * b for a, b in zip(vector, axis))
    return dot > 1 - tol


def basis_of(view):
    return camera_basis(view["camera.azimuth"], view["camera.altitude"])


# ---------------------------------------------------------------------------
# The set itself
# ---------------------------------------------------------------------------


def test_every_subject_gets_nine_views():
    views = default_subject_views(has_flatmap=True)
    expected = set(DEFAULT_VIEW_ANGLES)
    expected |= {name + INFLATED_SUFFIX for name in DEFAULT_VIEW_ANGLES}
    expected.add(FLAT_VIEW_NAME)
    assert set(views) == expected
    assert len(views) == 9


def test_the_named_orientations_are_the_ones_asked_for():
    assert set(DEFAULT_VIEW_ANGLES) == {
        "dorsal", "ventral", "lateral_left", "lateral_right"}


@pytest.mark.parametrize("name", list(DEFAULT_VIEW_ANGLES))
def test_the_plain_views_are_on_the_fiducial_surface(name):
    """"...and inflated" is a separate view, so these must not be inflated."""
    views = default_subject_views(has_flatmap=True)
    assert views[name]["surface.{subject}.unfold"] == 0


@pytest.mark.parametrize("name", list(DEFAULT_VIEW_ANGLES))
def test_each_view_has_an_inflated_twin_at_the_same_angle(name):
    views = default_subject_views(has_flatmap=True)
    plain, inflated = views[name], views[name + INFLATED_SUFFIX]
    assert inflated["surface.{subject}.unfold"] == 0.5
    assert inflated["camera.azimuth"] == plain["camera.azimuth"]
    assert inflated["camera.altitude"] == plain["camera.altitude"]


def test_without_a_flatmap_there_is_no_flat_view():
    views = default_subject_views(has_flatmap=False)
    assert FLAT_VIEW_NAME not in views
    assert len(views) == 8


def test_without_a_flatmap_inflated_is_fully_unfolded():
    """Without a flat surface the inflated surface sits at unfold 1, not 0.5.

    The same correction save_3d_views makes.
    """
    views = default_subject_views(has_flatmap=False)
    for name in DEFAULT_VIEW_ANGLES:
        assert views[name + INFLATED_SUFFIX]["surface.{subject}.unfold"] == 1
        assert views[name]["surface.{subject}.unfold"] == 0


def test_the_flat_view_is_fully_unfolded_and_pivoted():
    view = default_subject_views(has_flatmap=True)[FLAT_VIEW_NAME]
    assert view["surface.{subject}.unfold"] == 1
    assert view == {**view, **angle_view_params["flatmap"]}


def test_the_flat_view_carries_no_camera_angle():
    """A flattened surface pins the camera square-on and ignores the angle.

    Naming one would do nothing to the view itself while giving an animation
    something spurious to interpolate towards on the way in -- spinning the
    brain as it flattens, and overwriting the folded angle it would return to.
    """
    view = default_subject_views(has_flatmap=True)[FLAT_VIEW_NAME]
    assert not set(FLAT_INERT_PROPS) & set(view), view
    # Not in the table it is built from either, so save_3d_views does not ask
    # for one when it renders a flatmap.
    assert not set(FLAT_INERT_PROPS) & set(angle_view_params["flatmap"])
    # Every other view still names its angle.
    for name in DEFAULT_VIEW_ANGLES:
        assert set(FLAT_INERT_PROPS) <= set(default_subject_views(True)[name])


def test_the_flat_view_names_no_target():
    """It says nothing about the folded pose, and the viewer centres the flatmap.

    default_view_params' origin is a folded target; carried by a flat view with
    no camera.flat_target it would be read as the flat target, and put the
    flatmap back where it used to sit, sixty units low.
    """
    view = default_subject_views(has_flatmap=True)[FLAT_VIEW_NAME]
    assert "camera.target" not in view
    assert "camera.flat_target" not in view


def test_views_keep_the_subject_placeholder():
    """So one view still applies in a viewer showing several subjects."""
    for view in default_subject_views(has_flatmap=True).values():
        assert any("{subject}" in key for key in view)
        for key in view:
            assert "S1" not in key


def test_views_survive_the_trip_to_the_browser():
    """They are shipped inside viewopts, so they have to be JSON."""
    views = default_subject_views(has_flatmap=True)
    assert json.loads(json.dumps(views)) == views


def test_callers_cannot_corrupt_the_shared_tables():
    """default_subject_views must hand out copies, not the module's own dicts."""
    first = default_subject_views(has_flatmap=True)
    first["dorsal"]["camera.azimuth"] = 12345
    assert default_subject_views(has_flatmap=True)["dorsal"]["camera.azimuth"] != 12345
    # ... and the tables it is built from are untouched.
    assert angle_view_params["top"]["camera.azimuth"] == 180


# ---------------------------------------------------------------------------
# Do the anatomical names describe what the camera actually shows?
# ---------------------------------------------------------------------------


def test_dorsal_looks_down_with_the_frontal_lobe_up():
    right, up, back = basis_of(default_subject_views()["dorsal"])
    assert points_along(back, SUPERIOR)      # camera above the brain
    assert points_along(up, ANTERIOR)        # frontal lobe at the top
    assert points_along(right, RIGHT)        # subject's right on the right


def test_ventral_looks_up_with_the_frontal_lobe_up():
    right, up, back = basis_of(default_subject_views()["ventral"])
    assert points_along(back, tuple(-c for c in SUPERIOR))   # camera below
    assert points_along(up, ANTERIOR)                        # frontal lobe up
    # Seen from underneath the subject's left falls on the image right, which
    # is what looking at the underside of something does.
    assert points_along(right, tuple(-c for c in RIGHT))


def test_lateral_left_looks_from_the_left_with_the_brain_upright():
    right, up, back = basis_of(default_subject_views()["lateral_left"])
    assert points_along(back, tuple(-c for c in RIGHT))   # camera on the left
    assert points_along(up, SUPERIOR)                     # upright
    # Facing the left side of a head, the nose points to the image left.
    assert points_along(right, tuple(-c for c in ANTERIOR))


def test_lateral_right_looks_from_the_right_with_the_brain_upright():
    right, up, back = basis_of(default_subject_views()["lateral_right"])
    assert points_along(back, RIGHT)                      # camera on the right
    assert points_along(up, SUPERIOR)                     # upright
    assert points_along(right, ANTERIOR)                  # nose to the right


def test_the_two_lateral_views_are_opposite_each_other():
    _, _, left_back = basis_of(default_subject_views()["lateral_left"])
    _, _, right_back = basis_of(default_subject_views()["lateral_right"])
    dot = sum(a * b for a, b in zip(left_back, right_back))
    assert dot == pytest.approx(-1, abs=1e-6)


@pytest.mark.parametrize("name", list(DEFAULT_VIEW_ANGLES))
def test_inflating_does_not_move_the_camera(name):
    views = default_subject_views()
    assert basis_of(views[name]) == basis_of(views[name + INFLATED_SUFFIX])


# ---------------------------------------------------------------------------
# Serving them: defaults, and the filestore overriding them
# ---------------------------------------------------------------------------


def test_quickflat_size_matches_the_flatmask_formula(monkeypatch):
    """The reported size must be the one quickflat.make_png actually writes.

    make_png resizes the figure to the flatmap image and saves at `dpi`, so the
    png comes out exactly that image's pixel size: 1024 tall, and as wide as the
    flat surface's bounding box makes it.
    """
    import numpy as np

    from cortex.webgl import view as webgl_view

    # A bounding box 3 units wide and 2 tall, so the image is 1.5x as wide as
    # it is tall. The offset is there to catch an implementation using max()
    # where it should use the span.
    pts = np.array([[10.0, 5.0, 0.0], [13.0, 7.0, 1.0]])
    monkeypatch.setattr(webgl_view, "_has_flatmap", lambda subject: True)
    monkeypatch.setattr(webgl_view.db, "get_surf",
                        lambda *a, **k: (pts, None))

    assert webgl_view._quickflat_size("S1", height=1024) == [1536, 1024]
    assert webgl_view._quickflat_size("S1", height=512) == [768, 512]


def test_quickflat_size_is_none_without_a_flat_surface(monkeypatch):
    """Asked before reading anything, so no warning and no wasted surface load."""
    from cortex.webgl import view as webgl_view

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("get_surf must not be called without a flatmap")

    monkeypatch.setattr(webgl_view, "_has_flatmap", lambda subject: False)
    monkeypatch.setattr(webgl_view.db, "get_surf", should_not_be_called)
    assert webgl_view._quickflat_size("S1") is None


def test_quickflat_size_warns_if_the_surface_cannot_be_read(monkeypatch):
    from cortex.webgl import view as webgl_view

    def no_surface(*args, **kwargs):
        raise ValueError("unreadable")

    monkeypatch.setattr(webgl_view, "_has_flatmap", lambda subject: True)
    monkeypatch.setattr(webgl_view.db, "get_surf", no_surface)
    with pytest.warns(UserWarning, match="quickflat size"):
        assert webgl_view._quickflat_size("S1") is None


def test_a_subject_with_no_views_directory_still_gets_the_defaults(monkeypatch,
                                                                   tmp_path):
    from cortex.webgl import view as webgl_view

    monkeypatch.setattr(webgl_view.db, "filestore", str(tmp_path))
    monkeypatch.setattr(webgl_view, "_has_flatmap", lambda subject: True)

    loaded = webgl_view._load_saved_views(["S1"])
    assert set(loaded) == {"S1"}
    assert loaded["S1"] == default_subject_views(has_flatmap=True)


def test_a_saved_view_replaces_the_default_of_the_same_name(monkeypatch,
                                                            tmp_path):
    """The override the whole arrangement exists for."""
    from cortex.webgl import view as webgl_view

    viewdir = tmp_path / "S1" / "views"
    viewdir.mkdir(parents=True)
    mine = {"camera.azimuth": 33.0, "camera.altitude": 44.0,
            "surface.{subject}.unfold": 0.25}
    (viewdir / "dorsal.json").write_text(json.dumps(mine))
    (viewdir / "my_own_view.json").write_text(json.dumps(mine))

    monkeypatch.setattr(webgl_view.db, "filestore", str(tmp_path))
    monkeypatch.setattr(webgl_view, "_has_flatmap", lambda subject: True)
    loaded = webgl_view._load_saved_views(["S1"])["S1"]

    # The saved one wins outright; it is not merged with the default.
    assert loaded["dorsal"] == mine
    # Every other default survives, and unrelated saved views are still added.
    defaults = default_subject_views(has_flatmap=True)
    for name in defaults:
        if name != "dorsal":
            assert loaded[name] == defaults[name]
    assert loaded["my_own_view"] == mine
    assert set(loaded) == set(defaults) | {"my_own_view"}


def test_defaults_are_per_subject(monkeypatch, tmp_path):
    """Overriding a view for one subject must not touch another's."""
    from cortex.webgl import view as webgl_view

    viewdir = tmp_path / "S1" / "views"
    viewdir.mkdir(parents=True)
    mine = {"camera.azimuth": 33.0}
    (viewdir / "ventral.json").write_text(json.dumps(mine))

    monkeypatch.setattr(webgl_view.db, "filestore", str(tmp_path))
    monkeypatch.setattr(webgl_view, "_has_flatmap", lambda subject: True)
    loaded = webgl_view._load_saved_views(["S1", "S2"])

    assert loaded["S1"]["ventral"] == mine
    assert loaded["S2"]["ventral"] == default_subject_views()["ventral"]


# ---------------------------------------------------------------------------
# Framing the default views for a subject
# ---------------------------------------------------------------------------


def _surface_points(kind):
    """The surface `kind` as the viewer draws it."""
    from cortex.export.save_views import _viewer_points

    return _viewer_points("S1", kind)


def test_framing_sees_the_surfaces_the_viewer_draws():
    """The surfaces are fitted as the surface packs lay them out.

    Checked against brainctm itself, which builds the surface packs the
    viewer loads: framing the raw inflated file instead leaves the inflated
    views at about half the size they should be.
    """
    import numpy as np

    from cortex.brainctm import BrainCTM

    pack = BrainCTM("S1")
    pack.addSurf("inflated")
    hemis = (pack.left, pack.right)
    drawn = np.vstack([hemi.surfs["inflated"][:, :3] for hemi in hemis])
    assert _surface_points("inflated") == pytest.approx(drawn, abs=1e-3)
    # S1 is packed on its pial surface with the white matter alongside; the
    # folded brain is drawn at their midpoint (the depth slider's default).
    folded = np.vstack([(hemi.pts + hemi.surfs["wm"][:, :3]) / 2 for hemi in hemis])
    assert _surface_points("fiducial") == pytest.approx(folded, abs=1e-3)


def _ndc(points, view, target, radius):
    """Where each point lands in the frame, in units of the half-frame.

    Through the viewer's perspective camera, for a frame FRAMING_ASPECT wide:
    +-1 is the frame's edge on each axis.
    """
    import numpy as np

    from cortex.export import save_views

    right, up, back = (np.array(v) for v in basis_of(view))
    rel = points - np.asarray(target)
    depth = radius - rel @ back
    tan = math.tan(math.radians(save_views.VIEWER_FOV / 2))
    return (rel @ right) / (depth * tan * save_views.FRAMING_ASPECT), \
        (rel @ up) / (depth * tan)


@pytest.fixture
def framing_cache(tmp_path, monkeypatch):
    """Keep each test's framing cache to itself, out of the S1 filestore."""
    import cortex

    monkeypatch.setattr(cortex.db, "get_cache", lambda subject: str(tmp_path))
    return tmp_path


def _counting_get_surf(monkeypatch):
    """Wrap db.get_surf so a test can see whether surfaces were read."""
    import cortex

    calls = []
    real = cortex.db.get_surf

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(cortex.db, "get_surf", counted)
    return calls


def test_framing_covers_every_view_but_flat(framing_cache):
    from cortex.export.save_views import RADIUS_LIMITS, default_view_framing

    framing = default_view_framing("S1")
    views = default_subject_views(has_flatmap=True)
    assert set(framing) == set(views) - {FLAT_VIEW_NAME}
    for frame in framing.values():
        assert set(frame) == {"camera.target", "camera.radius"}
        assert RADIUS_LIMITS[0] <= frame["camera.radius"] <= RADIUS_LIMITS[1]


def test_framing_aims_at_the_middle_of_the_surface_shown(framing_cache):
    import numpy as np

    from cortex.export.save_views import default_view_framing

    framing = default_view_framing("S1")
    for kind, names in (("fiducial", DEFAULT_VIEW_ANGLES),
                        ("inflated", [n + INFLATED_SUFFIX for n in DEFAULT_VIEW_ANGLES])):
        points = _surface_points(kind)
        centre = (points.min(0) + points.max(0)) / 2
        for name in names:
            assert framing[name]["camera.target"] == pytest.approx(centre.tolist())


def test_framing_fills_the_frame_and_clips_nothing(framing_cache):
    """The brain reaches exactly FRAMING_FILL of the half-frame, and no further."""
    import numpy as np

    from cortex.export.save_views import FRAMING_FILL, default_view_framing

    framing = default_view_framing("S1")
    views = default_subject_views(has_flatmap=True)
    for name, frame in framing.items():
        view = views[name]
        points = _surface_points(
            "fiducial" if view["surface.{subject}.unfold"] == 0 else "inflated")
        x, y = _ndc(points, view, frame["camera.target"], frame["camera.radius"])
        reach = max(np.abs(x).max(), np.abs(y).max())
        assert reach == pytest.approx(FRAMING_FILL, abs=1e-6), name


def test_framing_is_read_from_the_cache(framing_cache, monkeypatch):
    from cortex.export.save_views import default_view_framing

    first = default_view_framing("S1")
    assert (framing_cache / "default_view_framing.json").exists()

    calls = _counting_get_surf(monkeypatch)
    assert default_view_framing("S1") == first
    assert calls == []


def test_framing_is_refitted_when_a_surface_changes(framing_cache, monkeypatch):
    import os

    from cortex.export.save_views import default_view_framing

    first = default_view_framing("S1")
    real = os.path.getmtime
    monkeypatch.setattr(os.path, "getmtime", lambda path: real(path) + 1)
    calls = _counting_get_surf(monkeypatch)
    assert default_view_framing("S1") == first
    assert calls != []


def test_framing_is_refitted_when_the_framing_changes(framing_cache, monkeypatch):
    from cortex.export import save_views

    first = save_views.default_view_framing("S1")
    monkeypatch.setattr(save_views, "FRAMING_FILL", save_views.FRAMING_FILL / 2)
    second = save_views.default_view_framing("S1")
    for name in first:
        assert second[name]["camera.radius"] > first[name]["camera.radius"]


def test_framing_without_a_writable_cache(tmp_path, monkeypatch):
    """A read-only filestore just means no caching."""
    import cortex

    from cortex.export.save_views import default_view_framing

    monkeypatch.setattr(cortex.db, "get_cache",
                        lambda subject: str(tmp_path / "does" / "not" / "exist"))
    assert len(default_view_framing("S1")) == 8

    def refuse(subject):
        raise PermissionError("read-only")

    monkeypatch.setattr(cortex.db, "get_cache", refuse)
    assert len(default_view_framing("S1")) == 8


def test_default_views_are_framed_only_for_a_subject(framing_cache):
    from cortex.export.save_views import default_view_framing

    plain = default_subject_views(has_flatmap=True)
    assert not any("camera.radius" in view for view in plain.values())

    framed = default_subject_views(has_flatmap=True, subject="S1")
    framing = default_view_framing("S1")
    for name, view in framed.items():
        if name == FLAT_VIEW_NAME:
            assert "camera.radius" not in view and "camera.target" not in view
        else:
            assert view == {**plain[name], **framing[name]}
