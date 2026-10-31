"""Tests for the WebGL viewer's views: saved and default views, and the camera.

These cover capturing and restoring views, saving them to the filestore, the
views menu, the views every subject gets by default, framing the flat surface,
the camera target and angle across folding and flattening, how static
exports name views by subject, and that an older custom template still opens.

Related tests live elsewhere:

- test_webgl_headless.py: what the viewer draws.
- test_webgl_animation.py: the animation panel, including the camera while
  animating into the flat view.
- test_webgl_view.py: the parts of cortex.webgl.view that need no browser.

All tests are skipped if playwright is not installed.
"""

import json
import os
import time

import numpy as np
import pytest

import cortex
import cortex.export
from cortex.export.save_views import default_view_params
from cortex.tests.testing_utils import (
    alpha_mask,
    flatmap_centre,
    js_attrs,
    js_run,
    js_value,
    mask_bbox,
    requires_playwright,
    subj,
    volshape,
    wait_for_file,
    xfmname,
)

pytestmark = requires_playwright

# ---------------------------------------------------------------------------
# _capture_view roundtrip
# ---------------------------------------------------------------------------


def test_capture_view_roundtrip():
    """Setting view parameters and capturing them back should match."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        target_params = {
            "camera.azimuth": 90,
            "camera.altitude": 90,
        }
        handle._set_view(**target_params)
        time.sleep(2)
        captured = handle._capture_view()
        for key, expected in target_params.items():
            assert captured[key] == pytest.approx(
                expected, abs=1.0
            ), f"{key}: expected {expected}, got {captured[key]}"


# ---------------------------------------------------------------------------
# Saved views
# ---------------------------------------------------------------------------


def test_retrieve_new_views_roundtrip():
    """Views saved through the GUI come back to python via the handle."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        assert handle.retrieve_new_views() == {}

        target = {"camera.azimuth": 90, "camera.altitude": 90}
        handle._set_view(**target)
        time.sleep(2)

        # What the "save view" button calls.
        handle.send(method="run",
                    params=["window.viewer.saveNewView", ["from_gui"]])

        views = handle.retrieve_new_views()
        assert set(views) == {"from_gui"}
        saved = views["from_gui"]
        for key, expected in target.items():
            assert saved[key] == pytest.approx(expected, abs=1.0)

        # Keys keep the {subject} placeholder, so the view stays interchangeable
        # with what _capture_view writes and with saved views/*.json files.
        assert "surface.{subject}.unfold" in saved

        # The javascript capture must be a subset of the python one; otherwise
        # _set_view would reject keys coming back out of the browser.
        captured = handle._capture_view()
        assert set(saved) <= set(captured), set(saved) - set(captured)

        # And it must round-trip back in without complaint.
        handle._set_view(**saved)

        pageerrors = [e for e in handle._pw_thread.browser_errors if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"


def test_save_new_views_writes_to_the_filestore():
    """save_new_views stores GUI views and promotes them out of new_views."""
    viewdir = os.path.join(cortex.db.filestore, subj, "views")
    names = ["_pytest_saved_a", "_pytest saved b"]
    written = []

    try:
        vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
            handle._set_view(**{"camera.azimuth": 90})
            time.sleep(2)
            for name in names:
                handle.send(method="run",
                            params=["window.viewer.saveNewView", [name]])
            assert set(handle.retrieve_new_views()) == set(names)

            paths = handle.save_new_views()
            written = list(paths.values())
            assert set(paths) == set(names)

            for name in names:
                path = os.path.join(viewdir, name + ".json")
                assert paths[name] == path
                assert os.path.isfile(path)
                with open(path) as fp:
                    stored = json.load(fp)
                # Stored in _capture_view's format, so it feeds straight back in.
                assert "surface.{subject}.unfold" in stored
                assert stored["camera.azimuth"] == pytest.approx(90, abs=1.0)
                handle._set_view(**stored)

            # Saved views are no longer "new": they move into the views menu.
            assert handle.retrieve_new_views() == {}
            buttons = js_attrs(
                handle, "window.viewer.ui._desc.camera._desc.views._desc")
            for name in names:
                assert name in buttons

            # A second save is a no-op rather than a rewrite, since there is
            # nothing left to promote.
            assert handle.save_new_views() == {}

            # Re-saving under an existing name needs is_overwrite.
            handle.send(method="run",
                        params=["window.viewer.saveNewView", [names[0]]])
            with pytest.raises(IOError):
                handle.save_new_views()
            assert handle.save_new_views(is_overwrite=True) == {
                names[0]: os.path.join(viewdir, names[0] + ".json")}

            # A name that would escape the views directory is refused outright.
            handle.send(method="run",
                        params=["window.viewer.saveNewView", ["../_pytest_evil"]])
            with pytest.raises(ValueError):
                handle.save_new_views()
            assert not os.path.exists(
                os.path.join(cortex.db.filestore, subj, "_pytest_evil.json"))

            with pytest.raises(KeyError):
                handle.save_new_views(names=["_pytest_no_such_view"])

            pageerrors = [e for e in handle._pw_thread.browser_errors
                          if "[pageerror]" in e]
            assert len(pageerrors) == 0, f"JS errors: {pageerrors}"
    finally:
        # The S1 filestore is checked in; leave nothing behind.
        for path in set(written) | {os.path.join(viewdir, n + ".json")
                                    for n in names}:
            if os.path.exists(path):
                os.remove(path)


def test_saved_views_are_loaded_into_the_viewer():
    """views/*.json for the displayed subject reach the browser and the menu."""
    from cortex.export.save_views import default_view_params

    viewdir = os.path.join(cortex.db.filestore, subj, "views")
    os.makedirs(viewdir, exist_ok=True)
    name = "_pytest_tmp_view"
    viewfile = os.path.join(viewdir, name + ".json")
    with open(viewfile, "w") as fp:
        json.dump(dict(default_view_params), fp)

    try:
        vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
            # Only the displayed subject's views are shipped to the browser.
            assert set(js_attrs(handle, "window.viewopts.saved_views")) == {subj}
            assert name in js_attrs(
                handle, "window.viewopts.saved_views.%s" % subj)

            # ... and each one becomes a button under camera > views.
            buttons = js_attrs(
                handle, "window.viewer.ui._desc.camera._desc.views._desc")
            assert name in buttons

            # Clicking it applies the view.
            handle._set_view(**{"camera.azimuth": 10})
            time.sleep(1)
            handle.send(method="run", params=[
                "window.viewer.ui._desc.camera._desc.views._desc"
                ".%s.action" % name, []])
            time.sleep(2)
            assert handle.ui.get("camera.azimuth")[0] == pytest.approx(
                default_view_params["camera.azimuth"], abs=1.0)
    finally:
        os.remove(viewfile)


# ---------------------------------------------------------------------------
# Default views every subject gets
# ---------------------------------------------------------------------------


def test_default_views_reach_the_browser_and_the_menu():
    """Every subject gets the standard orientations without saving anything."""
    from cortex.export.save_views import default_subject_views
    from cortex.webgl.view import _has_flatmap

    expected = set(default_subject_views(_has_flatmap(subj)))
    assert "dorsal" in expected and "lateral_left_inflated" in expected

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        shipped = js_attrs(handle, "window.viewopts.saved_views.%s" % subj)
        assert expected <= set(shipped), expected - set(shipped)

        buttons = js_attrs(
            handle, "window.viewer.ui._desc.camera._desc.views._desc")
        assert expected <= set(buttons), expected - set(buttons)


def test_clicking_a_default_view_applies_it():
    """The buttons are wired, not just present."""
    from cortex.export.save_views import default_subject_views
    from cortex.webgl.view import _has_flatmap

    views = default_subject_views(_has_flatmap(subj))
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        # Somewhere that is not the view we are about to ask for.
        handle._set_view(**{"camera.azimuth": 10, "camera.altitude": 45})
        time.sleep(1)

        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.views._desc"
            ".lateral_left.action", []])
        time.sleep(2)

        want = views["lateral_left"]
        assert handle.ui.get("camera.azimuth")[0] == pytest.approx(
            want["camera.azimuth"], abs=1.0)
        assert handle.ui.get("camera.altitude")[0] == pytest.approx(
            want["camera.altitude"], abs=1.0)


def test_a_saved_view_overrides_the_default_of_the_same_name():
    """A views/dorsal.json in the filestore wins over the built-in dorsal."""
    from cortex.export.save_views import default_subject_views

    viewdir = os.path.join(cortex.db.filestore, subj, "views")
    os.makedirs(viewdir, exist_ok=True)
    viewfile = os.path.join(viewdir, "dorsal.json")
    assert not os.path.exists(viewfile), (
        "%s already exists; this test would overwrite it" % viewfile)

    builtin = default_subject_views()["dorsal"]
    mine = dict(default_view_params)
    mine["camera.azimuth"] = 123.0
    mine["camera.altitude"] = 47.0
    assert mine["camera.azimuth"] != builtin["camera.azimuth"]

    with open(viewfile, "w") as fp:
        json.dump(mine, fp)
    try:
        vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
            shipped = js_attrs(
                handle, "window.viewopts.saved_views.%s.dorsal" % subj)
            assert "camera.azimuth" in shipped

            handle.send(method="run", params=[
                "window.viewer.ui._desc.camera._desc.views._desc"
                ".dorsal.action", []])
            time.sleep(2)
            assert handle.ui.get("camera.azimuth")[0] == pytest.approx(
                mine["camera.azimuth"], abs=1.0)

            # The other defaults are untouched by the override.
            buttons = js_attrs(
                handle, "window.viewer.ui._desc.camera._desc.views._desc")
            assert "ventral" in buttons and "lateral_right" in buttons
    finally:
        os.remove(viewfile)


def test_fit_flat_view_frames_the_flat_surface():
    """The flat view looks at the middle of the flatmap, from far enough back.

    Far enough back being the distance at which the camera's field of view
    spans the flatmap, which is what makes the render fill the frame the way
    quickflat's image does.
    """
    from cortex.webgl.view import _has_flatmap, _quickflat_size

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        # Clicking the view in the camera > views menu, the way a person does.
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.views._desc.flat.action", []])
        time.sleep(3)

        box = js_run(handle, "window.viewer.flatBBox", [])
        span = [box["max"][i] - box["min"][i] for i in range(3)]

        # The viewer's flatmap and quickflat's are the same surface, so the
        # frame quickflat writes has the same shape as the flatmap does here.
        width, height = _quickflat_size(subj)
        assert span[0] / span[1] == pytest.approx(width / height, rel=1e-3)

        fov = js_value(handle, "window.viewer.camera.fov")
        aspect = js_value(handle, "window.viewer.camera.aspect")
        view = handle._capture_view()
        assert view["camera.flat_target"] == pytest.approx(
            [(box["min"][0] + box["max"][0]) / 2,
             (box["min"][1] + box["max"][1]) / 2, 0], abs=1e-3)
        # As large as fits in the frame on screen: filling it top to bottom,
        # unless that window is narrower than the flatmap, which would crop it.
        assert view["camera.radius"] == pytest.approx(
            max(span[1] / 2, span[0] / 2 / aspect) / np.tan(np.radians(fov / 2)),
            rel=1e-3)
        assert js_run(handle, "window.viewer.isFlatFitted", []) is True

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"


def test_setting_the_flat_view_from_python_leaves_the_camera_alone():
    """The framing is asked for, never applied behind a caller's back.

    save_3d_views sets the flat view through _set_view and renders it at its
    own size; re-framing it there would silently change every flatmap that
    function has ever written.
    """
    from cortex.export.save_views import default_subject_views
    from cortex.webgl.view import _has_flatmap

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        before = handle._capture_view()
        handle._set_view(**default_subject_views(True)["flat"])
        time.sleep(2)

        after = handle._capture_view()
        assert after["surface.{subject}.unfold"] == 1
        assert after["camera.radius"] == pytest.approx(before["camera.radius"])
        assert js_run(handle, "window.viewer.isFlatFitted", []) is False

        # ... and asking for it does frame the flatmap.
        framing = handle.fit_flat_view()
        assert framing is not None
        time.sleep(1)
        assert handle._capture_view()["camera.radius"] == pytest.approx(
            framing["radius"], rel=1e-3)


def test_flat_view_renders_what_quickflat_draws(tmp_path):
    """The point of the framing: a flat frame is the png make_png writes.

    Rendered at the subject's quickflat size, a framed flat view has to put the
    flatmap where quickflat puts it -- same position, same scale -- or an
    animation that visits the flat surface does not line up with a flatmap.
    """
    from cortex.export.save_views import default_subject_views
    from cortex.webgl.view import _has_flatmap, _quickflat_size

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    size = _quickflat_size(subj)
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    quickflat = str(tmp_path / "quickflat.png")
    cortex.quickflat.make_png(quickflat, vol, with_rois=False,
                              with_labels=False, with_colorbar=False)
    drawn = alpha_mask(quickflat)
    assert list(drawn.shape[::-1]) == size

    rendered = str(tmp_path / "webgl.png")
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        handle._set_view(**default_subject_views(True)["flat"])
        time.sleep(2)
        handle.fit_flat_view()
        time.sleep(1)
        # The window is not the shape of the image, so this only lines up
        # because getImage re-frames for what it is about to write.
        handle.getImage(rendered, size=tuple(size))
        wait_for_file(rendered, timeout=60)
        time.sleep(1)

    shot = alpha_mask(rendered)
    assert shot.shape == drawn.shape

    # Both fill the frame: quickflat's fills it exactly, and the render is
    # within a pixel of that (the frame is a whole number of pixels, so its
    # aspect ratio is quickflat's rounded).
    x0, y0, x1, y1 = mask_bbox(shot)
    assert (x0, y0) == pytest.approx((0, 0), abs=2)
    assert (x1, y1) == pytest.approx(shot.shape[::-1], abs=2)

    # And they are the same flatmap in the same place, not merely two things
    # that happen to fill the frame. What is left is edge pixels: the two
    # rasterize the outline differently, and nothing else may differ.
    overlap = (shot & drawn).sum() / (shot | drawn).sum()
    assert overlap > 0.97, "masks overlap by only %.3f" % overlap


# ---------------------------------------------------------------------------
# A flat pose carries no camera angle
# ---------------------------------------------------------------------------


def test_a_flat_pose_records_no_camera_angle():
    """Both capture paths agree: azimuth and altitude are not part of it."""
    from cortex.webgl.view import _has_flatmap

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        folded = handle._capture_view()
        assert "camera.azimuth" in folded and "camera.altitude" in folded

        handle._set_view(**{"surface.{subject}.unfold": 1})
        time.sleep(2)

        flat = handle._capture_view()
        assert "camera.azimuth" not in flat, flat
        assert "camera.altitude" not in flat, flat

        # The browser's own capture (vt.captureView, behind "save view") drops
        # them too, or a view saved in the GUI would carry what python's does
        # not.
        handle.send(method="run",
                    params=["window.viewer.saveNewView", ["_pytest_flat"]])
        saved = handle.retrieve_new_views()["_pytest_flat"]
        assert "camera.azimuth" not in saved, saved
        assert "camera.altitude" not in saved, saved

        # Tilting the flat surface gives the angle back its meaning, so it is
        # recorded again.
        handle._set_view(**{"surface.{subject}.allow_tilt": True})
        time.sleep(2)
        tilted = handle._capture_view()
        assert "camera.azimuth" in tilted, tilted
        assert "camera.altitude" in tilted, tilted


# ---------------------------------------------------------------------------
# The folded and the flat camera target
# ---------------------------------------------------------------------------


def test_flat_target_starts_at_the_flatmap_centre():
    """Flattening lands centred on the flatmap without anything asking for it.

    The controls used to start the flat target at y = -60, a guess close to
    S1's flatmap and wrong for other subjects; setting the flat view from
    python then replaced even that with the origin, putting the flatmap
    sixty units low.
    """
    from cortex.export.save_views import default_subject_views
    from cortex.webgl.view import _has_flatmap

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        before = handle._capture_view()
        assert "camera.flat_target" in before

        handle._set_view(**default_subject_views(True)["flat"])
        time.sleep(2)
        centre = flatmap_centre(handle)

        # Worked out at load, before the surface was ever flat, and right.
        assert before["camera.flat_target"] == pytest.approx(centre, abs=1e-3)
        after = handle._capture_view()
        assert after["camera.flat_target"] == pytest.approx(centre, abs=1e-3)
        assert js_run(handle, "window.viewer.controls.setTarget", []) == \
            pytest.approx(centre, abs=1e-3)
        # ... and the folded target is where it was.
        assert after["camera.target"] == pytest.approx(before["camera.target"])


def test_a_flat_view_saved_before_flat_target_still_loads():
    """A flat view that stores only camera.target means a flat target.

    That is where camera.target went once the surface was flat, so every flat
    view saved before camera.flat_target existed stores it that way -- and it
    is what save_3d_views passes with its flatmap, which keeps that function's
    output exactly as it was.
    """
    from cortex.webgl.view import _has_flatmap

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    legacy = {"surface.{subject}.unfold": 1, "camera.target": [5.0, -40.0, 0.0],
              "camera.radius": 300.0}
    viewdir = os.path.join(cortex.db.filestore, subj, "views")
    os.makedirs(viewdir, exist_ok=True)
    name = "_pytest_legacy_flat"
    viewfile = os.path.join(viewdir, name + ".json")
    with open(viewfile, "w") as fp:
        json.dump(legacy, fp)

    try:
        vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
        with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
            folded = handle._capture_view()["camera.target"]

            # From python ...
            handle._set_view(**legacy)
            time.sleep(2)
            view = handle._capture_view()
            assert view["camera.flat_target"] == pytest.approx([5, -40, 0])
            assert view["camera.target"] == pytest.approx(folded)

            # ... and clicked in the views menu, after moving it elsewhere.
            handle._set_view(**{"camera.flat_target": [0.0, 0.0, 0.0]})
            time.sleep(1)
            handle.send(method="run", params=[
                "window.viewer.ui._desc.camera._desc.views._desc"
                ".%s.action" % name, []])
            time.sleep(2)
            view = handle._capture_view()
            assert view["camera.flat_target"] == pytest.approx([5, -40, 0])
            assert view["camera.target"] == pytest.approx(folded)
    finally:
        os.remove(viewfile)


# ---------------------------------------------------------------------------
# Default views carry the whole camera
# ---------------------------------------------------------------------------


def test_default_views_set_the_fitted_target_and_radius():
    """Clicking a default view returns the same scene, zoom included."""
    from cortex.export.save_views import default_view_framing

    framing = default_view_framing(subj)
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        for name in ("dorsal", "lateral_left_inflated"):
            # Zoomed and aimed somewhere else first.
            handle._set_view(**{"camera.radius": 520, "camera.target": [30, -20, 5]})
            time.sleep(1)
            handle.send(method="run", params=[
                "window.viewer.ui._desc.camera._desc.views._desc.%s.action" % name,
                []])
            time.sleep(2)
            view = handle._capture_view()
            assert view["camera.radius"] == pytest.approx(
                framing[name]["camera.radius"], rel=1e-6), name
            assert view["camera.target"] == pytest.approx(
                framing[name]["camera.target"], abs=1e-6), name


@pytest.mark.parametrize("name", ["dorsal", "lateral_left_inflated"])
def test_a_default_view_fills_a_4_by_3_frame(tmp_path, name):
    """The framing holds in the real viewer, not just in python's model of it.

    Fitted so the brain's farthest point from the middle reaches
    FRAMING_FILL of the half-frame: rendered 4:3, the tightest margin is
    (1 - FRAMING_FILL) / 2 of the frame, and nothing is clipped. Dorsal is
    limited by its height, the lateral view by its width, and the inflated
    view checks that the fit sees the inflated surface the way the viewer's
    surface packs lay it out.
    """
    from cortex.export.save_views import FRAMING_FILL

    width, height = 800, 600
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    shot = str(tmp_path / (name + ".png"))
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.views._desc.%s.action" % name, []])
        time.sleep(3)
        handle.getImage(shot, size=(width, height))
        wait_for_file(shot, timeout=60)
        time.sleep(1)

    x0, y0, x1, y1 = mask_bbox(alpha_mask(shot))
    margins = [x0 / width, (width - x1) / width, y0 / height, (height - y1) / height]
    assert min(margins) > 0, margins                     # nothing clipped
    assert min(margins) == pytest.approx((1 - FRAMING_FILL) / 2, abs=0.015), margins


# ---------------------------------------------------------------------------
# Static exports and legacy templates
# ---------------------------------------------------------------------------


def _static_page_json(html, name):
    """The JSON static.html assigns to `name` (``name = {...};`` on one line)."""
    import re

    match = re.search(r"^\s*%s = (\{.*\});\s*$" % name, html, re.M)
    assert match, "no %s in the page" % name
    return json.loads(match.group(1))


@pytest.mark.parametrize("anonymize", [False, True])
def test_static_export_names_subjects_consistently(tmp_path, anonymize):
    """Views and the quickflat hint are keyed by the name the page knows.

    In an anonymized export that is the anonymized name: keyed by the real one,
    the page would carry the real subject ID the export exists to hide, and the
    viewer could not find either entry.
    """
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    outpath = str(tmp_path / "static")
    # An anonymized export ships its own renamed surface files (make_static
    # rewrites the names inside them), so it needs them copied.
    cortex.webgl.make_static(outpath, vol, html_embed=False,
                             copy_ctmfiles=anonymize, anonymize=anonymize)
    with open(os.path.join(outpath, "index.html")) as fp:
        html = fp.read()

    surfaces = set(_static_page_json(html, "subjects"))
    viewopts = _static_page_json(html, "viewopts")
    assert set(viewopts["saved_views"]) == surfaces
    assert set(viewopts["quickflat_size"]) == surfaces
    if anonymize:
        assert surfaces == {"S0"}
        assert subj not in viewopts["saved_views"]
        assert subj not in viewopts["quickflat_size"]
    else:
        assert surfaces == {subj}


def test_a_template_without_the_new_scripts_still_opens(tmp_path):
    """A custom template.html from before viewtools.js leaves the viewer usable.

    Template directories can shadow template.html, and one written before this
    viewer gained its views menu loads none of viewtools.js, interpolation.js,
    zipstore.js or mp4mux.js. The viewer must open without them -- just without
    the views menu and the animation panel.
    """
    webgl = os.path.dirname(cortex.webgl.view.__file__)
    with open(os.path.join(webgl, "template.html")) as fp:
        legacy = [line for line in fp if not any(
            script in line for script in ("viewtools.js", "interpolation.js",
                                          "zipstore.js", "mp4mux.js"))]
    (tmp_path / "template.html").write_text("".join(legacy))
    with open(os.path.join(webgl, "mixer.html")) as fp:
        (tmp_path / "mixer.html").write_text(fp.read())

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(
            vol, viewer_params=dict(template=str(tmp_path / "mixer.html"))) as handle:
        assert js_attrs(handle, "window.jsplot").get("viewtools") is None
        camera = js_attrs(handle, "window.viewer.ui._desc.camera._desc")
        assert "views" not in camera and "create animation" not in camera

        handle._set_view(**{"camera.azimuth": 123})
        time.sleep(1)
        assert handle._capture_view()["camera.azimuth"] == pytest.approx(123, abs=1)

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"
