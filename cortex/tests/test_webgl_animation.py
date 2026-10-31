"""Tests for the WebGL animation panel: keyframes, smoothing and movies.

Everything here drives the animation panel or renders what it produces:
keyframe interpolation in the browser and in python, flat keyframes and the
quickflat render size, the keyframe dots under the slider, and movies
downloaded as a zip of PNGs or an MP4. The camera while animating into the
flat view is tested here because it needs the panel.

Related tests live elsewhere:

- test_webgl_headless.py: what the viewer draws.
- test_webgl_camera_views.py: saved and default views, and the camera outside
  the animation panel.

All tests are skipped if playwright is not installed.
"""

import os
import struct
import time
import zipfile

import numpy as np
import pytest

import cortex
import cortex.export
from cortex.tests.testing_utils import (
    alpha_mask,
    flatmap_centre,
    js_attrs,
    js_query,
    js_run,
    js_value,
    requires_playwright,
    subj,
    volshape,
    xfmname,
)


pytestmark = requires_playwright


# ---------------------------------------------------------------------------
# Group 1: Movies render in the browser, not on the server
# ---------------------------------------------------------------------------


def _post(url, **fields):
    """POST form fields, returning the HTTP status (including error statuses)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    data = urllib.parse.urlencode(fields).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code


def test_the_server_accepts_no_frames():
    """Rendered movies are downloaded by the browser; the server writes nothing.

    The animation panel used to POST each frame to a /movie endpoint that wrote
    it on the serving machine. That endpoint must stay gone.
    """
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        url = f"http://localhost:{handle.server.port}/movie"
        # Refused either way: nothing routes /movie any more, and the server's
        # catch-all file handler (serve.py) only answers GET, hence 405.
        assert _post(url, name="f", frame=0, png="x") in (404, 405)
        assert "movie_post" not in js_attrs(handle, "window.viewopts")


def test_static_viewer_can_render_movies(tmp_path):
    """A static export carries saved views, and renders without a server.

    Movies are built in the page and downloaded, so the viewer only needs its
    own scripts -- the zip and mp4 writers ship with it -- and nothing that
    points back at a python server.
    """
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    outpath = str(tmp_path / "static")
    cortex.webgl.make_static(outpath, vol, html_embed=False, copy_ctmfiles=False)

    with open(os.path.join(outpath, "index.html")) as fp:
        html = fp.read()
    assert "viewtools.js" in html
    assert "saved_views" in html
    # Only the script tags can be checked here: make_static does not copy the
    # resources tree (see test_static_viewer_ships_the_interpolation_module).
    # They must load before viewtools.js, which uses them when rendering.
    for script in ("zipstore.js", "mp4mux.js"):
        assert script in html
        assert html.index(script) < html.index("viewtools.js")
    assert "movie_post" not in html


# ---------------------------------------------------------------------------
# Group 2: Smoothed animation trajectories
# ---------------------------------------------------------------------------

# Keyframes exercising every kind of channel at once: an angle that crosses the
# 0/360 wrap, a plain scalar, a vector, a discrete property, a boolean and a
# string -- and a different interpolation mode on each keyframe.
SMOOTHING_KEYFRAMES = [
    {"frame": 0, "interpolation": "Bezier",
     "camera.azimuth": 300.0, "camera.altitude": 10.0,
     "camera.target": [0.0, 0.0, 0.0], "surface.S1.layers": 1,
     "surface.S1.dither": False, "surface.S1.sampler": "nearest"},
    {"frame": 10, "interpolation": "CubicHermite",
     "camera.azimuth": 40.0, "camera.altitude": 90.0,
     "camera.target": [10.0, 20.0, 30.0], "surface.S1.layers": 4,
     "surface.S1.dither": True, "surface.S1.sampler": "trilinear"},
    {"frame": 20, "interpolation": "BezierInHoldOut",
     "camera.azimuth": 140.0, "camera.altitude": 30.0,
     "camera.target": [5.0, 5.0, 5.0], "surface.S1.layers": 2,
     "surface.S1.dither": False, "surface.S1.sampler": "nearest"},
    {"frame": 30, "interpolation": "Linear",
     "camera.azimuth": 200.0, "camera.altitude": 55.0,
     "camera.target": [1.0, 2.0, 3.0], "surface.S1.layers": 3,
     "surface.S1.dither": True, "surface.S1.sampler": "trilinear"},
]


def _assert_views_match(expected, actual, tol=1e-6):
    """Compare two view dicts property by property."""
    assert set(expected) == set(actual), set(expected) ^ set(actual)
    for prop, want in expected.items():
        got = actual[prop]
        if isinstance(want, list):
            assert got == pytest.approx(want, abs=tol), prop
        elif isinstance(want, bool) or not isinstance(want, (int, float)):
            assert got == want, prop
        else:
            assert got == pytest.approx(want, abs=tol), prop


def test_interpolation_js_is_loaded_with_all_eight_modes():
    """The browser knows the same modes python does."""
    from cortex.webgl.interpolation import Interpolation

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        modes = js_attrs(handle, "window.jsplot.interpolation.Interpolation")
        assert set(modes) == {mode.value for mode in Interpolation}
        assert js_value(handle, "window.jsplot.interpolation.DEFAULT_MODE") == \
            Interpolation.Bezier.value


def test_browser_and_python_interpolate_identically():
    """The whole point of keeping two implementations: they must agree.

    An animation built in the panel is played back in javascript but rendered
    to disk through _get_anim_seq in python, so any divergence would show up as
    a movie that does not match its preview.
    """
    from cortex.webgl.interpolation import build_channels, evaluate

    frames = [i * 0.5 for i in range(61)]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        from_js = js_run(handle, "window.jsplot.viewtools.viewsAt",
                          [SMOOTHING_KEYFRAMES, frames])
        assert isinstance(from_js, list) and len(from_js) == len(frames), from_js

        channels = build_channels(SMOOTHING_KEYFRAMES, time_key="frame")
        for frame, js_view in zip(frames, from_js):
            _assert_views_match(evaluate(channels, frame), js_view)

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"


def test_animation_panel_defaults_to_bezier():
    """Opening the panel sets up per-keyframe smoothing state."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        # What the "create animation" button calls. The key has a space in it,
        # which the dotted-path walker in python_interface.js handles fine.
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.create animation.action", []])
        time.sleep(1)

        state = js_attrs(handle, "window.viewer._anim")
        assert "mode" in state, state
        assert js_value(handle, "window.viewer._anim.mode") == "Bezier"


def test_get_anim_seq_linear_path_is_unchanged():
    """The pairwise easings still produce exactly what they always did."""
    keyframes = [
        {"time": 0.0, "camera.azimuth": 10.0, "camera.altitude": 20.0},
        {"time": 1.0, "camera.azimuth": 90.0, "camera.altitude": 60.0},
        {"time": 2.0, "camera.azimuth": 170.0, "camera.altitude": 40.0},
    ]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        seq = handle._get_anim_seq([dict(k) for k in keyframes], fps=30,
                                   interpolation="linear")
        # 30 frames per second over two seconds, plus the closing frame.
        assert len(seq) == 61
        assert "time" not in seq[0]
        assert seq[0]["camera.azimuth"] == pytest.approx(10.0)
        assert seq[30]["camera.azimuth"] == pytest.approx(90.0)
        assert seq[-1]["camera.azimuth"] == pytest.approx(170.0)
        # Straight lines between the keyframes: the quarter point is halfway
        # from the first keyframe to the second.
        assert seq[15]["camera.azimuth"] == pytest.approx(50.0)
        assert seq[15]["camera.altitude"] == pytest.approx(40.0)


def test_get_anim_seq_all_linear_modes_reproduce_the_legacy_path():
    """'linear' and a list of Linear keyframes are the same curve.

    This ties the two code paths together: whatever the smoothed path does to
    frame times and property handling, it has to land on the old answer when
    every keyframe is linear.
    """
    keyframes = [
        {"time": 0.0, "camera.altitude": 20.0, "camera.target": [0.0, 0.0, 0.0]},
        {"time": 0.7, "camera.altitude": 60.0, "camera.target": [3.0, 6.0, 9.0]},
        {"time": 2.0, "camera.altitude": 40.0, "camera.target": [1.0, 1.0, 1.0]},
    ]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        legacy = handle._get_anim_seq([dict(k) for k in keyframes], fps=30,
                                      interpolation="linear")
        smoothed = handle._get_anim_seq([dict(k) for k in keyframes], fps=30,
                                        interpolation="Linear")
        assert len(legacy) == len(smoothed)
        for want, got in zip(legacy, smoothed):
            _assert_views_match(want, got)


def test_get_anim_seq_honours_per_keyframe_modes():
    """A keyframe's own mode takes over, and selects the smoothed path."""
    keyframes = [
        {"time": 0.0, "camera.altitude": 0.0,
         "interpolation": "BezierInHoldOut"},
        {"time": 1.0, "camera.altitude": 40.0, "interpolation": "Linear"},
        {"time": 2.0, "camera.altitude": 80.0, "interpolation": "Linear"},
    ]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        # interpolation defaults to 'linear', but the keyframes override it.
        seq = handle._get_anim_seq([dict(k) for k in keyframes], fps=30)
        assert len(seq) == 61
        # Held across the first second...
        assert seq[15]["camera.altitude"] == pytest.approx(0.0)
        assert seq[29]["camera.altitude"] == pytest.approx(0.0)
        # ... then linear to the end.
        assert seq[30]["camera.altitude"] == pytest.approx(40.0)
        assert seq[45]["camera.altitude"] == pytest.approx(60.0)
        assert seq[-1]["camera.altitude"] == pytest.approx(80.0)
        assert "interpolation" not in seq[0]


def test_get_anim_seq_rejects_mixing_an_easing_with_keyframe_modes():
    """smoothstep eases a segment and has no per-keyframe equivalent."""
    keyframes = [
        {"time": 0.0, "camera.altitude": 0.0, "interpolation": "Bezier"},
        {"time": 1.0, "camera.altitude": 40.0, "interpolation": "Bezier"},
    ]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        with pytest.raises(ValueError, match="whole segment"):
            handle._get_anim_seq([dict(k) for k in keyframes], fps=30,
                                 interpolation="smoothstep")
        with pytest.raises(ValueError, match="Unknown interpolation"):
            handle._get_anim_seq([dict(k) for k in keyframes], fps=30,
                                 interpolation="wobble")


def test_static_viewer_ships_the_interpolation_module(tmp_path):
    """Smoothing is pure browser-side, so static exports get it too.

    Only the script tag is checked here: make_static does not copy the
    resources tree, it is either inlined by htmlembed or served alongside.
    """
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    outpath = str(tmp_path / "static")
    cortex.webgl.make_static(outpath, vol, html_embed=False, copy_ctmfiles=False)

    with open(os.path.join(outpath, "index.html")) as fp:
        html = fp.read()
    assert "interpolation.js" in html
    # It has to come before viewtools.js, which uses it at panel-open time.
    assert html.index("interpolation.js") < html.index("viewtools.js")


def test_quickflat_size_reaches_the_browser():
    """The animation panel's flat-render hint is computed in python."""
    from cortex.webgl.view import _quickflat_size

    expected = _quickflat_size(subj)
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        assert subj in js_attrs(handle, "window.viewopts.quickflat_size")
        # .slice() hands back a plain array, which survives the JSON round trip
        # that a bare property read does not.
        shipped = js_run(
            handle, "window.viewopts.quickflat_size.%s.slice" % subj, [])
        assert shipped == expected

    if expected is not None:
        width, height = expected
        assert height == 1024
        assert width > 0


def test_quickflat_size_matches_a_real_quickflat_png(tmp_path):
    """The hint has to be the size make_png actually writes, not near it."""
    from PIL import Image

    from cortex.webgl.view import _quickflat_size

    expected = _quickflat_size(subj)
    if expected is None:
        pytest.skip("%s has no flat surface" % subj)

    out = str(tmp_path / "flat.png")
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    cortex.quickflat.make_png(out, vol, with_rois=False, with_labels=False,
                              with_colorbar=False)
    assert list(Image.open(out).size) == expected


def test_flattening_leaves_the_camera_angle_alone():
    """An animation into the flat view must not spin the brain on the way.

    The controls discard the angle once flat and write the *folded* one on the
    way there, so a keyframe carrying an angle both fights the controls' own
    blend and leaves the folded view rotated once it unfolds again.
    """
    from cortex.webgl.view import _has_flatmap

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        handle._set_view(**{"camera.azimuth": 45, "camera.altitude": 70,
                            "surface.{subject}.unfold": 0})
        time.sleep(2)
        folded = handle._capture_view()

        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.create animation.action", []])
        time.sleep(1)
        handle.send(method="run",
                    params=["window.viewer._animPanel.addKeyframe", []])

        handle.send(method="run",
                    params=["window.viewer._animPanel.setFrame", [30]])
        time.sleep(1)
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.views._desc.flat.action", []])
        time.sleep(3)
        handle.send(method="run",
                    params=["window.viewer._animPanel.addKeyframe", []])
        time.sleep(1)

        keyframes = js_run(handle, "window.viewer._anim.keyframes.slice", [])
        assert len(keyframes) == 2
        assert "camera.azimuth" not in keyframes[1], keyframes[1]

        # Nothing along the way asks for an angle either: the one channel the
        # animation has for it is fed by the single keyframe that carries one.
        frames = [f * 1.0 for f in range(0, 31, 5)]
        views = js_run(handle, "window.jsplot.viewtools.viewsAt",
                        [keyframes, frames])
        azimuths = [view["camera.azimuth"] for view in views]
        assert azimuths == pytest.approx([folded["camera.azimuth"]] * len(frames))

        # Step the animation through the transition, which is where the angle
        # used to be rewritten, and then unfold by hand: the folded camera has
        # to be the one we set, not wherever the flat keyframe dragged it.
        for f in (10, 20, 30):
            handle.send(method="run",
                        params=["window.viewer._animPanel.setFrame", [f]])
            time.sleep(0.5)
        handle._set_view(**{"surface.{subject}.unfold": 0})
        time.sleep(2)

        back = handle._capture_view()
        assert back["camera.azimuth"] == pytest.approx(
            folded["camera.azimuth"], abs=1.0)
        assert back["camera.altitude"] == pytest.approx(
            folded["camera.altitude"], abs=1.0)

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"


def test_flattening_leaves_the_folded_target_alone():
    """An animation into the flat view must not drag the folded brain with it.

    One camera.target used to stand for two targets, written to whichever
    matched the unfold state at the moment: every partly unfolded frame of a
    transition wrote the flat target's values into the folded one, sliding the
    brain down ahead of the flattening and leaving it displaced afterwards.
    """
    from cortex.webgl.view import _has_flatmap

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.create animation.action", []])
        time.sleep(1)
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.views._desc.lateral_left.action",
            []])
        time.sleep(2)
        folded = handle._capture_view()["camera.target"]
        js_run(handle, "window.viewer._animPanel.addKeyframe", [])

        js_run(handle, "window.viewer._animPanel.setFrame", [30])
        time.sleep(1)
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.views._desc.flat.action", []])
        time.sleep(3)
        js_run(handle, "window.viewer._animPanel.addKeyframe", [])
        centre = flatmap_centre(handle)

        for frame in (5, 10, 15, 20, 25, 30):
            js_run(handle, "window.viewer._animPanel.setFrame", [frame])
            time.sleep(0.5)
            view = handle._capture_view()
            assert view["camera.target"] == pytest.approx(folded, abs=1e-6), frame
            assert view["camera.flat_target"] == pytest.approx(centre, abs=1e-3), frame

        # Unfolding by hand returns the brain to exactly where it started.
        handle._set_view(**{"surface.{subject}.unfold": 0})
        time.sleep(2)
        assert js_run(handle, "window.viewer.controls.setTarget", []) == \
            pytest.approx(folded, abs=1e-6)

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"


def test_browser_and_python_agree_with_a_flat_keyframe_in_the_middle():
    """The two interpolators must treat a missing property the same way."""
    from cortex.webgl.interpolation import build_channels, evaluate

    keyframes = [
        {"frame": 0, "camera.azimuth": 45.0, "camera.altitude": 70.0,
         "surface.{subject}.unfold": 0.0, "interpolation": "Bezier"},
        {"frame": 15, "surface.{subject}.unfold": 1.0,
         "interpolation": "Bezier"},
        {"frame": 30, "camera.azimuth": 270.0, "camera.altitude": 40.0,
         "surface.{subject}.unfold": 0.0, "interpolation": "Bezier"},
    ]
    frames = [i * 1.5 for i in range(21)]

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        from_js = js_run(handle, "window.jsplot.viewtools.viewsAt",
                          [keyframes, frames])
        assert len(from_js) == len(frames)

        channels = build_channels(keyframes, time_key="frame")
        for frame, js_view in zip(frames, from_js):
            _assert_views_match(evaluate(channels, frame), js_view)

        # The flat keyframe is not a knot on the angle's curve: it sweeps
        # across the whole animation rather than pausing in the middle.
        mid = from_js[len(from_js) // 2]["camera.azimuth"]
        assert mid != pytest.approx(45.0) and mid != pytest.approx(270.0)


# ---------------------------------------------------------------------------
# Group 3: The animation panel's "match quickflat size" option
# ---------------------------------------------------------------------------


def test_the_panel_leaves_the_render_size_alone_until_asked():
    """The quickflat size is offered, not imposed."""
    from cortex.webgl.view import _has_flatmap, _quickflat_size

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    size = _quickflat_size(subj)
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.create animation.action", []])
        time.sleep(1)

        # Lay down a flat keyframe, the case the option is about.
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.views._desc.flat.action", []])
        time.sleep(3)
        handle.send(method="run",
                    params=["window.viewer._animPanel.addKeyframe", []])
        time.sleep(1)

        assert js_run(handle, "window.viewer._animPanel.matchesFlat", []) is False
        default_size = js_run(handle, "window.viewer._animPanel.renderSize", [])
        assert default_size != size

        # Ticking the box takes over the size and re-frames the keyframe that
        # is already down, which was framed for the untouched default.
        before = js_run(handle, "window.viewer._anim.keyframes.slice", [])[0]
        assert js_run(handle, "window.viewer._animPanel.setMatchFlat",
                       [True]) is True
        time.sleep(2)

        assert js_run(handle, "window.viewer._animPanel.renderSize", []) == size
        after = js_run(handle, "window.viewer._anim.keyframes.slice", [])[0]
        assert after["camera.radius"] != pytest.approx(before["camera.radius"])

        framing = js_run(handle, "window.viewer.flatFraming",
                          [size[0] / size[1]])
        assert after["camera.radius"] == pytest.approx(framing["radius"],
                                                       rel=1e-3)

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"


def test_rendering_an_animation_reproduces_the_quickflat_png(tmp_path):
    """End to end: with the box ticked, a flat keyframe renders as the flatmap.

    The panel picks the render size, the framing follows it, and the frame the
    browser downloads has to be the png make_png writes -- that is what lets a
    flat frame of an animation be cut against a flatmap made in python.
    """
    from cortex.webgl.view import _has_flatmap, _quickflat_size

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    size = _quickflat_size(subj)
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)

    quickflat = str(tmp_path / "quickflat.png")
    cortex.quickflat.make_png(quickflat, vol, with_rois=False,
                              with_labels=False, with_colorbar=False)
    drawn = alpha_mask(quickflat)

    with cortex.export.headless_viewer(
            vol, viewer_params={}, download_dir=str(tmp_path)) as handle:
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.create animation.action", []])
        time.sleep(1)

        # Tick "match quickflat size" before laying the keyframe down.
        handle.send(method="run",
                    params=["window.viewer._animPanel.setMatchFlat", [True]])
        time.sleep(1)

        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.views._desc.flat.action", []])
        time.sleep(3)
        handle.send(method="run",
                    params=["window.viewer._animPanel.addKeyframe", []])
        time.sleep(1)

        assert js_run(handle, "window.viewer._animPanel.renderSize", []) == size

        # One frame is enough, and 31 of them at this size are not free.
        handle.send(method="set", params=["window.viewer._anim.last", 0])
        handle.send(method="run",
                    params=["window.viewer._animPanel.render", []])

        movie = handle._pw_thread.wait_for_download(timeout=120)

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"

    frame = str(tmp_path / "flat_frame.png")
    with zipfile.ZipFile(movie) as archive:
        with open(frame, "wb") as fp:
            fp.write(archive.read("brainmovie/brainmovie_00000.png"))
    shot = alpha_mask(frame)
    assert list(shot.shape[::-1]) == size
    overlap = (shot & drawn).sum() / (shot | drawn).sum()
    assert overlap > 0.97, "masks overlap by only %.3f" % overlap


# ---------------------------------------------------------------------------
# Group 4: Rendering movies to a download
# ---------------------------------------------------------------------------


def _render_setup(handle, last=2, size=(320, 240)):
    """An animation of `last` + 1 frames, turning the brain, at a small size."""
    handle.send(method="run", params=[
        "window.viewer.ui._desc.camera._desc.create animation.action", []])
    time.sleep(1)
    handle._set_view(**{"camera.azimuth": 45})
    time.sleep(1)
    js_run(handle, "window.viewer._animPanel.addKeyframe", [])
    js_run(handle, "window.viewer._animPanel.setFrame", [last])
    handle._set_view(**{"camera.azimuth": 135})
    time.sleep(1)
    js_run(handle, "window.viewer._animPanel.addKeyframe", [])
    handle.send(method="set", params=["window.viewer._anim.last", last])
    assert js_run(handle, "window.viewer._animPanel.setRenderSize",
                   list(size)) == list(size)


def _panel_status(handle):
    """What the animation panel's status line says."""
    return js_run(handle, "window.viewer._animPanel._statusText", [])


def _mp4_boxes(data, start=0, end=None):
    """The box tree of an MP4, as a list of (type, payload) with containers expanded."""
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"dinf"}
    end = len(data) if end is None else end
    boxes = []
    while start < end:
        size, kind = struct.unpack(">I4s", data[start:start + 8])
        assert size >= 8, "bad box size %d at %d" % (size, start)
        payload = data[start + 8:start + size]
        if kind in containers:
            boxes.append((kind, _mp4_boxes(data, start + 8, start + size)))
        else:
            boxes.append((kind, payload))
        start += size
    assert start == end, "boxes overrun their parent"
    return boxes


def _find_box(boxes, *path):
    for kind, payload in boxes:
        if kind == path[0]:
            return payload if len(path) == 1 else _find_box(payload, *path[1:])
    raise KeyError(b"/".join(path))


_PLAYBACK_PAGE = """<html><body><video id=v muted playsinline></video><canvas id=c></canvas>
<script>
window.result = new Promise(function(resolve) {
  var v = document.getElementById('v');
  v.onerror = function() { resolve({error: v.error ? v.error.message : 'error'}); };
  v.onloadeddata = function() {
    var c = document.getElementById('c');
    c.width = v.videoWidth; c.height = v.videoHeight;
    var ctx = c.getContext('2d');
    ctx.drawImage(v, 0, 0);
    var px = ctx.getImageData(0, 0, c.width, c.height).data, lit = 0;
    for (var i = 0; i < px.length; i += 4)
      if (px[i] + px[i + 1] + px[i + 2] > 30) lit++;
    resolve({width: v.videoWidth, height: v.videoHeight, duration: v.duration,
             lit: lit / (px.length / 4)});
  };
  v.src = '/movie.mp4';
});
</script></body></html>"""


def _play_mp4(path):
    """Load an MP4 into a <video> in a fresh browser and report what it decodes.

    Run once the viewer's own browser has shut down: sync Playwright objects
    belong to the thread that made them, and that browser lives on the viewer's
    worker thread. Served from a routed http://localhost page, the same kind of
    secure context the viewer runs in.
    """
    from playwright.sync_api import sync_playwright

    with open(path, "rb") as fp:
        movie = fp.read()

    def serve(route):
        if route.request.url.endswith("/movie.mp4"):
            route.fulfill(status=200, content_type="video/mp4", body=movie)
        else:
            route.fulfill(status=200, content_type="text/html",
                          body=_PLAYBACK_PAGE)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--use-gl=swiftshader", "--no-sandbox"])
        try:
            page = browser.new_page()
            page.route("http://localhost:9/**", serve)
            page.goto("http://localhost:9/")
            return page.evaluate("window.result")
        finally:
            browser.close()


def test_rendering_downloads_a_zip_of_pngs(tmp_path):
    """PNG frames arrive as one zip download, one lossless frame per entry."""
    from PIL import Image

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(
            vol, viewer_params={}, download_dir=str(tmp_path)) as handle:
        _render_setup(handle, last=2, size=(320, 240))
        assert js_run(handle, "window.viewer._animPanel.setRenderFormat",
                       ["png"]) is True
        js_run(handle, "window.viewer._animPanel.render", [])
        movie = handle._pw_thread.wait_for_download(timeout=120)

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"

    assert os.path.basename(movie) == "brainmovie.zip"
    with zipfile.ZipFile(movie) as archive:
        # testzip reads every entry back and checks its CRC-32.
        assert archive.testzip() is None
        names = archive.namelist()
        assert names == ["brainmovie/brainmovie_%05d.png" % f for f in range(3)]
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED

        frames = []
        for name in names:
            with archive.open(name) as fp:
                image = Image.open(fp)
                image.load()
            assert image.size == (320, 240)
            assert image.mode == "RGBA"
            # Transparent outside the brain, as Save image is.
            alpha = np.array(image)[..., 3]
            assert alpha.min() == 0 and alpha.max() == 255
            frames.append(np.array(image))

    # The brain turned between frames, so they are not the same picture.
    assert not np.array_equal(frames[0], frames[-1])


def test_rendering_downloads_an_mp4(tmp_path):
    """MP4 arrives as one well-formed H.264 file that a browser plays."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(
            vol, viewer_params={}, download_dir=str(tmp_path)) as handle:
        if not js_run(handle, "window.jsplot.viewtools.canEncodeVideo", []):
            pytest.skip("this browser cannot encode video")

        _render_setup(handle, last=5, size=(320, 240))
        assert js_run(handle, "window.viewer._animPanel.setRenderFormat",
                       ["mp4"]) is True
        js_run(handle, "window.viewer._animPanel.render", [])
        movie = handle._pw_thread.wait_for_download(timeout=120)
        fps = js_value(handle, "window.viewer._anim.fps")

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"

    assert os.path.basename(movie) == "brainmovie.mp4"
    with open(movie, "rb") as fp:
        data = fp.read()
    boxes = _mp4_boxes(data)
    assert [kind for kind, _ in boxes] == [b"ftyp", b"moov", b"mdat"]

    stbl = (b"moov", b"trak", b"mdia", b"minf", b"stbl")
    stsd = _find_box(boxes, *stbl, b"stsd")
    assert stsd[12:16] == b"avc1"          # after version/flags and entry count
    assert b"avcC" in stsd

    stsz = _find_box(boxes, *stbl, b"stsz")
    count = struct.unpack(">I", stsz[8:12])[0]
    sizes = struct.unpack(">%dI" % count, stsz[12:12 + 4 * count])
    assert count == 6
    assert sum(sizes) == len(_find_box(boxes, b"mdat"))

    stts = _find_box(boxes, *stbl, b"stts")
    assert struct.unpack(">III", stts[4:16]) == (1, 6, 1000)

    stss = _find_box(boxes, *stbl, b"stss")
    assert struct.unpack(">II", stss[4:12]) == (1, 1)   # first sample is a keyframe

    played = _play_mp4(movie)
    assert "error" not in played, played
    assert (played["width"], played["height"]) == (320, 240)
    assert played["duration"] == pytest.approx(6 / fps, abs=1e-3)
    # It decodes to a picture of the brain, on black: neither blank nor noise.
    assert 0.02 < played["lit"] < 0.9, played


def test_mp4_refuses_a_size_the_encoder_cannot_do(tmp_path):
    """Too large for H.264 here: said up front, and nothing is downloaded."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(
            vol, viewer_params={}, download_dir=str(tmp_path)) as handle:
        if not js_run(handle, "window.jsplot.viewtools.canEncodeVideo", []):
            pytest.skip("this browser cannot encode video")

        _render_setup(handle, last=1, size=(320, 240))
        js_run(handle, "window.viewer._animPanel.setRenderSize", [8192, 8192])
        js_run(handle, "window.viewer._animPanel.setRenderFormat", ["mp4"])
        js_run(handle, "window.viewer._animPanel.render", [])

        deadline = time.monotonic() + 30
        while js_value(handle, "window.viewer._animPanel.rendering"):
            assert time.monotonic() < deadline, "the render never gave up"
            time.sleep(0.5)
        assert "cannot make" in _panel_status(handle)
        time.sleep(2)
        assert handle._pw_thread.downloads == []


# ---------------------------------------------------------------------------
# Group 5: The keyframe dots under the animation panel's slider
# ---------------------------------------------------------------------------

_THUMB_RGB = (0x2f, 0xa1, 0xd6)      # .keyframe-track's slider thumb (mriview.css)
_DOT_RGB = (0xff, 0xd7, 0x00)        # .keyframe-dot


def _colour_centres(png, rgb, rows=None, tol=60):
    """The x centres of each run of columns where `rgb` appears in a png."""
    import io

    from PIL import Image

    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    if rows is not None:
        pixels = pixels[rows]
    columns = np.where((np.abs(pixels - np.array(rgb)).sum(-1) < tol).any(0))[0]
    runs, run = [], []
    for x in columns:
        if run and x - run[-1] > 1:
            runs.append(run)
            run = []
        run.append(x)
    if run:
        runs.append(run)
    return [(r[0] + r[-1]) / 2 for r in runs]


def _panel_with_keyframes(handle, frames, azimuths=None):
    """Open the animation panel and lay down a keyframe at each of `frames`."""
    handle.send(method="run", params=[
        "window.viewer.ui._desc.camera._desc.create animation.action", []])
    time.sleep(1)
    for i, frame in enumerate(frames):
        js_run(handle, "window.viewer._animPanel.goToFrame", [frame])
        if azimuths is not None:
            handle._set_view(**{"camera.azimuth": azimuths[i]})
            time.sleep(0.5)
        js_run(handle, "window.viewer._animPanel.addKeyframe", [])
    time.sleep(0.5)


def test_keyframe_dots_line_up_with_the_slider():
    """A keyframe's dot sits under the slider's thumb when the playhead is on it.

    It used to drift by several pixels towards either end: the dots were inset
    by a guess at half the browser's own thumb, whose size varies by browser.
    """
    frames = [0, 7, 15, 23, 30]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        _panel_with_keyframes(handle, frames)

        def track(page):
            return page.locator(".keyframe-track").screenshot()

        dots = _colour_centres(handle._pw_thread.run_on_page(track), _DOT_RGB)
        assert len(dots) == len(frames), dots

        for frame, dot in zip(frames, dots):
            js_run(handle, "window.viewer._animPanel.goToFrame", [frame])
            time.sleep(0.3)
            png = handle._pw_thread.run_on_page(track)
            thumbs = _colour_centres(png, _THUMB_RGB)
            assert len(thumbs) == 1, thumbs
            assert abs(thumbs[0] - dot) <= 1, (frame, thumbs[0], dot)


def test_clicking_a_keyframe_dot_goes_to_its_frame():
    """Clicking a dot puts the playhead, the slider and the view on that keyframe."""
    frames, azimuths = [0, 12, 30], [45, 120, 200]
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        _panel_with_keyframes(handle, frames, azimuths)
        js_run(handle, "window.viewer._animPanel.goToFrame", [5])
        time.sleep(0.5)

        for frame, azimuth in zip(frames, azimuths):
            def click(page, frame=frame):
                page.locator('.keyframe-dot[data-frame="%d"]' % frame).click()
                return (page.locator(".anim-slider").input_value(),
                        page.locator(".anim-frame").input_value())

            slider, field = handle._pw_thread.run_on_page(click)
            time.sleep(0.5)
            assert js_value(handle, "window.viewer._anim.frame") == frame
            assert (int(float(slider)), int(float(field))) == (frame, frame)
            assert handle._capture_view()["camera.azimuth"] == pytest.approx(
                azimuth, abs=0.5)

        pageerrors = [e for e in handle._pw_thread.browser_errors
                      if "[pageerror]" in e]
        assert len(pageerrors) == 0, f"JS errors: {pageerrors}"


def test_the_slider_still_drags_over_the_dots():
    """The dots take clicks without covering any part of the slider."""
    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        _panel_with_keyframes(handle, [0, 10, 20, 30])
        js_run(handle, "window.viewer._animPanel.goToFrame", [0])
        time.sleep(0.5)

        def drag(page):
            slider = page.locator(".anim-slider")
            slider.scroll_into_view_if_needed()
            box = slider.bounding_box()
            y = box["y"] + box["height"] / 2
            page.mouse.move(box["x"] + 6, y)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] / 2, y, steps=10)
            page.mouse.up()

        handle._pw_thread.run_on_page(drag)
        time.sleep(0.5)
        assert js_value(handle, "window.viewer._anim.frame") == pytest.approx(15, abs=1)


def test_changing_the_render_size_reframes_flat_keyframes():
    """With "match quickflat size" ticked, flat keyframes follow the size fields.

    Typed into the fields, set from code, or changed without either -- a flat
    keyframe is framed for the size the animation renders at.
    """
    from cortex.webgl.view import _has_flatmap

    if not _has_flatmap(subj):
        pytest.skip("%s has no flat surface" % subj)

    vol = cortex.Volume(np.random.randn(*volshape), subj, xfmname)
    with cortex.export.headless_viewer(vol, viewer_params={}) as handle:
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.create animation.action", []])
        time.sleep(1)
        js_run(handle, "window.viewer._animPanel.setMatchFlat", [True])
        handle.send(method="run", params=[
            "window.viewer.ui._desc.camera._desc.views._desc.flat.action", []])
        time.sleep(3)
        js_run(handle, "window.viewer._animPanel.addKeyframe", [])
        time.sleep(0.5)

        def flat_radius():
            return js_query(handle, "window.viewer._anim.keyframes.slice",
                             [])[0]["camera.radius"]

        def fitted(width, height):
            return js_query(handle, "window.viewer.flatFraming",
                             [width / height])["radius"]

        # Set from code.
        js_run(handle, "window.viewer._animPanel.setRenderSize", [600, 1000])
        assert flat_radius() == pytest.approx(fitted(600, 1000), rel=1e-6)

        # Typed into the fields.
        def type_size(page):
            page.locator(".anim-render").click()      # show the render form
            for field, value in ((".anim-width", "1600"), (".anim-height", "400")):
                page.locator(field).fill(value)
                page.locator(field).press("Tab")      # a change event, as typing gives

        handle._pw_thread.run_on_page(type_size)
        time.sleep(0.5)
        assert flat_radius() == pytest.approx(fitted(1600, 400), rel=1e-6)

        # Changed with no event at all: the render itself re-frames first.
        handle._pw_thread.run_on_page(lambda page: page.evaluate(
            "() => { document.querySelector('.anim-width').value = 700;"
            "        document.querySelector('.anim-height').value = 700; }"))
        handle.send(method="set", params=["window.viewer._anim.last", 0])
        js_run(handle, "window.viewer._animPanel.render", [])
        # Read back after the render: while it draws, a slow page can't answer in time.
        handle._pw_thread.wait_for_download(timeout=120)
        assert flat_radius() == pytest.approx(fitted(700, 700), rel=1e-6)
