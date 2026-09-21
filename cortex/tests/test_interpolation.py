"""Tests for the keyframe interpolation shared by the viewer and the movie path.

The arithmetic here is duplicated in ``cortex/webgl/resources/js/interpolation.js``
so that the browser's animation panel and ``JSMixer._get_anim_seq`` produce the
same trajectories. These tests pin the python half; the two halves are checked
against each other in ``test_webgl_headless.py``.
"""

import math

import pytest

from cortex.webgl.interpolation import (
    Interpolation,
    Interpolator,
    Keyframe,
    build_channels,
    evaluate,
    from_values,
    shortest_step,
    unwrap_angles,
    wrap_angle,
)

# A run with a rise, a fall and a rise, so that every keyframe but the ends is
# an interior one with neighbours on both sides.
TIMES = [0.0, 1.0, 2.0, 3.0]
VALUES = [0.0, 10.0, 4.0, 7.0]

HOLD_OUT = [
    Interpolation.LinearInHoldOut,
    Interpolation.BezierInHoldOut,
    Interpolation.CubicHermiteInHoldOut,
]


def sample(interp, start, stop, count=101):
    """``count`` evenly spaced samples of ``interp`` over ``[start, stop]``."""
    step = (stop - start) / (count - 1)
    return [interp(start + i * step) for i in range(count)]


# ---------------------------------------------------------------------------
# Every mode, at the keyframes themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(Interpolation))
def test_keyframes_are_reproduced_exactly(mode):
    """Whatever happens in between, a keyframe's own value is returned at its time."""
    interp = from_values(TIMES, VALUES, mode)
    for time, value in zip(TIMES, VALUES):
        assert interp(time) == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize("mode", list(Interpolation))
def test_values_are_held_outside_the_keyframe_range(mode):
    interp = from_values(TIMES, VALUES, mode)
    assert interp(-10.0) == VALUES[0]
    assert interp(TIMES[-1] + 10.0) == VALUES[-1]


@pytest.mark.parametrize("mode", HOLD_OUT)
def test_hold_out_modes_are_constant_until_the_next_keyframe(mode):
    """A hold keeps its keyframe's value right up to the following one."""
    interp = from_values(TIMES, VALUES, mode)
    for index in range(len(TIMES) - 1):
        # Stop just short of the next keyframe, which belongs to the next segment.
        end = TIMES[index + 1] - 1e-9
        held = sample(interp, TIMES[index], end)
        assert held == pytest.approx([VALUES[index]] * len(held), abs=1e-9)


# ---------------------------------------------------------------------------
# Over/undershoot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [Interpolation.CubicHermite, Interpolation.Bezier])
@pytest.mark.parametrize("values", [
    [0.0, 10.0, 4.0, 7.0],      # a peak and a trough
    [0.0, 1.0, 2.0, 3.0],       # monotonically increasing
    [3.0, 2.0, 1.0, 0.0],       # monotonically decreasing
    [5.0, 5.0, 1.0, 1.0],       # flat stretches
])
def test_smooth_modes_never_leave_the_bracketing_values(mode, values):
    """The whole point of the automatic tangents: no overshoot anywhere."""
    interp = from_values(TIMES, values, mode)
    for index in range(len(TIMES) - 1):
        low = min(values[index], values[index + 1])
        high = max(values[index], values[index + 1])
        seen = sample(interp, TIMES[index], TIMES[index + 1])
        assert min(seen) >= low - 1e-9
        assert max(seen) <= high + 1e-9


@pytest.mark.parametrize("mode", [Interpolation.LinearInCubicHermiteOut,
                                  Interpolation.LinearInBezierOut])
def test_linear_in_modes_are_allowed_to_overshoot(mode):
    """These modes carry the entry slope through, so they can and do overshoot.

    Pinned as a test because it is the only thing separating them from the
    plain smooth modes.
    """
    # A steep climb into a long, nearly flat run. Leaving the middle keyframe
    # along the entry slope has to carry the curve above the final value.
    # The second segment has to be long: a Bezier stays inside the hull of its
    # control points, and a short one would keep the handle under the endpoint.
    interp = from_values([0.0, 1.0, 11.0], [0.0, 10.0, 10.1], mode)
    assert max(sample(interp, 1.0, 11.0)) > 10.1 + 1e-6


# ---------------------------------------------------------------------------
# Deviations from the reference implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("values", [
    [0.0, 10.0, 4.0, 7.0],
    [10.0, 2.0, -5.0, -1.0],     # descending, where an unsigned offset breaks
])
def test_control_points_lie_on_the_keyframe_tangent(values):
    """Both handles sit on the tangent line through the keyframe.

    The reference implementation takes an unsigned square root for the value
    offset, which puts the handles of a descending keyframe on the wrong side
    of it and breaks the C1 continuity the scheme exists to provide.
    """
    interp = Interpolator([Keyframe(t, v, Interpolation.Bezier)
                           for t, v in zip(TIMES, values)])
    for frame in interp.keyframes:
        for handle in (frame.control_point_1, frame.control_point_2):
            if handle is None:
                continue
            dtime = handle[0] - frame.time
            if abs(dtime) < 1e-12:      # a vertical handle carries no slope
                continue
            assert (handle[1] - frame.value) / dtime == pytest.approx(
                frame.derivative, abs=1e-6)


def test_bezier_solves_for_its_parameter():
    """The parameter is found from the time, not assumed equal to it."""
    interp = from_values(TIMES, VALUES, Interpolation.Bezier)
    segment = interp.segments[0]
    for i in range(101):
        time = TIMES[0] + (TIMES[1] - TIMES[0]) * i / 100
        u = segment.parameter_at(time)
        x = ((1 - u) ** 3 * segment.p1[0] + 3 * (1 - u) ** 2 * u * segment.p2[0]
             + 3 * (1 - u) * u ** 2 * segment.p3[0] + u ** 3 * segment.p4[0])
        assert x == pytest.approx(time, abs=1e-9)


def test_linear_into_a_smooth_keyframe_enters_along_the_chord():
    """The combination the reference implementation builds no segment for.

    It must produce a value at all -- the reference desynchronizes its segment
    and end-time lists here -- and it must leave the first keyframe along the
    straight line to the second.
    """
    interp = Interpolator([
        Keyframe(0.0, 0.0, Interpolation.Linear),
        Keyframe(1.0, 10.0, Interpolation.CubicHermite),
        Keyframe(2.0, 4.0, Interpolation.Bezier),
    ])
    chord = (10.0 - 0.0) / (1.0 - 0.0)
    measured = (interp(1e-6) - interp(0.0)) / 1e-6
    assert measured == pytest.approx(chord, rel=1e-3)
    assert interp(0.5) == pytest.approx(6.25, abs=1e-9)


def test_a_single_keyframe_is_constant():
    interp = from_values([5.0], [3.0])
    assert interp(-1.0) == 3.0
    assert interp(5.0) == 3.0
    assert interp(99.0) == 3.0


# ---------------------------------------------------------------------------
# Building the interpolator
# ---------------------------------------------------------------------------


def test_keyframe_order_does_not_matter():
    forward = from_values(TIMES, VALUES, Interpolation.Bezier)
    shuffled = from_values([TIMES[i] for i in (2, 0, 3, 1)],
                           [VALUES[i] for i in (2, 0, 3, 1)],
                           Interpolation.Bezier)
    assert sample(forward, 0.0, 3.0) == pytest.approx(sample(shuffled, 0.0, 3.0))


def test_keyframes_sharing_a_time_keep_the_last():
    """Matching the way the animation panel replaces a keyframe in place."""
    interp = Interpolator([Keyframe(0.0, 0.0), Keyframe(1.0, 5.0),
                           Keyframe(1.0, 9.0)])
    assert len(interp) == 2
    assert interp(1.0) == 9.0


def test_an_empty_keyframe_list_is_rejected():
    with pytest.raises(ValueError):
        Interpolator([])


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------


def anim_interp(start, end, t):
    """``Viewer._animInterp`` from resources/js/mriview.js, for camera.azimuth."""
    if abs(end - start) >= 180:
        if start > end:
            return (start * (1 - t) + (end + 360) * t + 360) % 360
        return (start * (1 - t) + (end - 360) * t + 360) % 360
    return start * (1 - t) + end * t


@pytest.mark.parametrize("start,end", [
    (350, 10), (10, 350), (0, 180), (180, 0), (20, 200), (200, 20),
    (90, 100), (0, 0), (45, 225), (225, 45),
])
def test_angle_unwrapping_matches_the_viewer(start, end):
    """Two keyframes must spin the way the viewer already spins them.

    Half a turn is equally short either way and ``_animInterp`` breaks the tie
    by travelling against the sign of the raw difference; a symmetric
    shortest-angle formula would reverse one of those two cases.
    """
    unwrapped = unwrap_angles([start, end])
    for i in range(11):
        t = i / 10
        mine = wrap_angle(unwrapped[0] * (1 - t) + unwrapped[1] * t)
        theirs = anim_interp(start, end, t)
        # Compare as angles, so 0 and 360 count as equal.
        assert abs(wrap_angle(mine - theirs + 180) - 180) < 1e-9


def test_shortest_step_never_exceeds_half_a_turn():
    for start in range(0, 360, 7):
        for end in range(0, 360, 11):
            assert abs(shortest_step(start, end)) <= 180 + 1e-12


def test_azimuth_channel_stays_in_range_across_the_wrap():
    keyframes = [
        {"time": 0.0, "camera.azimuth": 300.0},
        {"time": 1.0, "camera.azimuth": 40.0},
        {"time": 2.0, "camera.azimuth": 140.0},
    ]
    channels = build_channels(keyframes)
    for i in range(101):
        azimuth = evaluate(channels, 2.0 * i / 100)["camera.azimuth"]
        assert 0.0 <= azimuth < 360.0


# ---------------------------------------------------------------------------
# Splitting view dicts into channels
# ---------------------------------------------------------------------------


def channel_keyframes():
    return [
        {"time": 0.0, "camera.altitude": 0.0, "camera.target": [0.0, 0.0, 0.0],
         "surface.S1.layers": 1, "surface.S1.dither": False, "name": "a"},
        {"time": 1.0, "camera.altitude": 90.0, "camera.target": [10.0, 20.0, 30.0],
         "surface.S1.layers": 4, "surface.S1.dither": True, "name": "b"},
        {"time": 2.0, "camera.altitude": 30.0, "camera.target": [5.0, 5.0, 5.0],
         "surface.S1.layers": 2, "surface.S1.dither": False, "name": "c"},
    ]


def test_bookkeeping_keys_do_not_become_channels():
    keyframes = channel_keyframes()
    keyframes[0]["interpolation"] = Interpolation.Linear
    view = evaluate(build_channels(keyframes), 0.5)
    assert "time" not in view
    assert "interpolation" not in view
    assert "frame" not in view


def test_discrete_and_non_numeric_properties_step():
    """`layers` recompiles shaders, and booleans and strings cannot be blended."""
    channels = build_channels(channel_keyframes())
    half = evaluate(channels, 0.5)
    assert half["surface.S1.layers"] == 1          # not 2.5
    assert half["surface.S1.dither"] is False
    assert half["name"] == "a"
    # ... and they step at the keyframe, not before or after it.
    assert evaluate(channels, 0.999)["surface.S1.layers"] == 1
    assert evaluate(channels, 1.0)["surface.S1.layers"] == 4


def test_arrays_interpolate_component_by_component():
    channels = build_channels(channel_keyframes())
    assert evaluate(channels, 0.5)["camera.target"] == pytest.approx([5.0, 10.0, 15.0])
    assert evaluate(channels, 0.0)["camera.target"] == pytest.approx([0.0, 0.0, 0.0])
    assert evaluate(channels, 1.0)["camera.target"] == pytest.approx([10.0, 20.0, 30.0])


def test_arrays_of_inconsistent_length_step_instead():
    keyframes = [
        {"time": 0.0, "camera.target": [0.0, 0.0, 0.0]},
        {"time": 1.0, "camera.target": [1.0, 1.0]},
    ]
    assert evaluate(build_channels(keyframes), 0.5)["camera.target"] == [0.0, 0.0, 0.0]


def test_per_keyframe_modes_are_honoured():
    """A hold on the first keyframe freezes only its own segment."""
    keyframes = [
        {"time": 0.0, "v": 0.0, "interpolation": Interpolation.BezierInHoldOut},
        {"time": 1.0, "v": 10.0, "interpolation": Interpolation.Linear},
        {"time": 2.0, "v": 20.0, "interpolation": Interpolation.Linear},
    ]
    channels = build_channels(keyframes)
    assert evaluate(channels, 0.5)["v"] == 0.0        # held
    assert evaluate(channels, 1.5)["v"] == pytest.approx(15.0)   # linear


def test_modes_may_be_plain_strings():
    """Keyframes arriving from the browser carry the mode as JSON text."""
    keyframes = [{"time": 0.0, "v": 0.0, "interpolation": "LinearInHoldOut"},
                 {"time": 1.0, "v": 10.0, "interpolation": "Linear"}]
    assert evaluate(build_channels(keyframes), 0.5)["v"] == 0.0


def test_an_unknown_mode_falls_back_to_the_default():
    keyframes = [{"time": 0.0, "v": 0.0, "interpolation": "NoSuchMode"},
                 {"time": 1.0, "v": 10.0}]
    channels = build_channels(keyframes, default_mode=Interpolation.Linear)
    assert evaluate(channels, 0.5)["v"] == pytest.approx(5.0)


def test_properties_missing_from_the_first_keyframe_are_skipped():
    keyframes = [{"time": 0.0, "a": 0.0}, {"time": 1.0, "a": 1.0, "b": 2.0}]
    assert set(build_channels(keyframes)) == {"a"}


def test_channels_are_indexed_by_a_chosen_time_key():
    """The browser stores frame numbers under "frame" rather than "time"."""
    keyframes = [{"frame": 0, "v": 0.0}, {"frame": 10, "v": 10.0}]
    channels = build_channels(keyframes, time_key="frame",
                              default_mode=Interpolation.Linear)
    assert evaluate(channels, 5)["v"] == pytest.approx(5.0)


def test_nan_and_infinity_step_rather_than_propagating():
    keyframes = [{"time": 0.0, "v": 1.0}, {"time": 1.0, "v": float("nan")}]
    value = evaluate(build_channels(keyframes), 0.5)["v"]
    assert value == 1.0 and not math.isnan(value)


# ---------------------------------------------------------------------------
# The javascript twin
# ---------------------------------------------------------------------------


def test_the_javascript_twin_is_packaged():
    """setup.py's resources/js/*.js pattern has to actually pick the file up.

    The browser half of this module is a plain file in the webgl resources; if
    it stops shipping, the animation panel silently falls back to the old
    linear interpolation rather than failing loudly.
    """
    import os

    import cortex.webgl

    path = os.path.join(os.path.dirname(cortex.webgl.__file__),
                        "resources", "js", "interpolation.js")
    assert os.path.exists(path), path


def test_every_mode_is_offered_by_the_javascript_panel():
    """The dropdown must list all eight, or a mode becomes unreachable.

    Read out of the javascript source rather than a browser, so this runs
    without playwright; the browser-side check lives in test_webgl_headless.py.
    """
    import os
    import re

    import cortex.webgl

    path = os.path.join(os.path.dirname(cortex.webgl.__file__),
                        "resources", "js", "interpolation.js")
    with open(path) as handle:
        source = handle.read()

    order = re.search(r"ip\.MODE_ORDER\s*=\s*\[(.*?)\]", source, re.S)
    assert order is not None, "MODE_ORDER not found in interpolation.js"
    listed = set(re.findall(r'"([A-Za-z]+)"', order.group(1)))
    assert listed == {mode.value for mode in Interpolation}

    labels = re.search(r"ip\.MODE_LABELS\s*=\s*\{(.*?)\n    \};", source, re.S)
    assert labels is not None, "MODE_LABELS not found in interpolation.js"
    labelled = set(re.findall(r"(\w+):", labels.group(1)))
    assert labelled == {mode.value for mode in Interpolation}
