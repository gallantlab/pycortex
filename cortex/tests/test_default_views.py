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
    default_subject_views,
)

# Surface RAS, which is what pycortex surfaces are in.
RIGHT = (1.0, 0.0, 0.0)
ANTERIOR = (0.0, 1.0, 0.0)
SUPERIOR = (0.0, 0.0, 1.0)

# LandscapeControls clamps altitude into (0.0001, 179.9999) before building the
# camera position, so a view asking for 0 or 180 is rendered a hair off the
# pole. That matters: exactly at the pole the view direction is parallel to the
# up vector and the image orientation is undefined.
POLE_EPSILON = 1e-4


def camera_basis(azimuth, altitude):
    """The camera's axes in world space, as the viewer computes them.

    Reproduces the eye position from ``LandscapeControls.update`` in
    resources/js/LandscapeControls.js, with ``camera.up`` fixed at +z by
    ``axes3d.js``, then three.js's ``Matrix4.lookAt``.

    Returns
    -------
    tuple
        ``(right, up, back)`` unit vectors: where the image's right edge, top
        edge, and the direction from the target towards the camera point in
        world space.
    """
    altitude = min(max(altitude, POLE_EPSILON), 180 - POLE_EPSILON)
    altrad = math.radians(altitude)
    azirad = math.radians(azimuth + 90)
    eye = (math.sin(altrad) * math.cos(azirad),
           math.sin(altrad) * math.sin(azirad),
           math.cos(altrad))

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def unit(v):
        length = math.sqrt(sum(c * c for c in v))
        return tuple(c / length for c in v)

    back = unit(eye)                       # the camera looks along -back
    right = unit(cross(SUPERIOR, back))
    up = cross(back, right)
    return right, up, back


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
