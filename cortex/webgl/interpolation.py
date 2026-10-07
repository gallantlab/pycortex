"""Keyframe interpolation for viewer animations.

A one-dimensional piecewise interpolator built from a list of keyframes, each
carrying its own :class:`Interpolation` mode. The modes follow the way Adobe
products describe keyframe interpolation -- a mode names how a keyframe is
entered and how it is left, so the shape of the curve between two keyframes is
decided by the pair of modes at its ends -- with cubic Hermite added.

``resources/js/interpolation.js`` is the twin of this module and must stay in
step with it: the browser's animation panel interpolates in javascript while
:meth:`~cortex.webgl.view.show.<locals>.JSMixer._get_anim_seq` interpolates here,
and an animation should look the same either way. Both use the same closed forms
and both carry the mode as one of the strings in :class:`Interpolation`, so a
keyframe crosses between them unchanged.

Nothing here needs numpy or scipy, which keeps the arithmetic identical to the
javascript side rather than deferring to a library the browser does not have.

Deviations from the reference implementation this was ported from
-----------------------------------------------------------------

1. Control-point offsets are signed (see :func:`_tangent_step`). The reference
   takes a square root for the value offset, which is always positive, so the
   handles of a keyframe with a negative derivative end up off its own tangent
   line -- breaking the C1 continuity the whole scheme exists to provide.
2. The Bezier parameter is solved for (see :meth:`_BezierSegment.__call__`)
   rather than assumed equal to normalized time. ``x`` is cubic in the
   parameter, so the two agree only when the control points happen to be evenly
   spaced in time, which is exactly when the smoothing is doing nothing.
3. A ``Linear`` keyframe followed by one entered smoothly produces a Hermite
   segment with a linear entry slope (see :func:`_make_segment`). The reference
   enumerates no segment for that combination at all.
4. A lone keyframe yields a constant interpolator (see :meth:`Interpolator.build`).

Values outside the keyframe range are always held at the nearest keyframe, which
is what the animation panel's frame slider does at either end of its range.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = ["Interpolation", "Keyframe", "Interpolator", "from_values",
           "build_channels", "evaluate", "shortest_step", "unwrap_angles",
           "wrap_angle"]


class Interpolation(str, Enum):
    """How a keyframe is entered and left.

    A ``str`` enum so that a mode survives ``json`` serialization as its own
    name, which is how keyframes travel between python and the browser.
    """

    Linear = "Linear"
    """Linear in, linear out."""

    LinearInHoldOut = "LinearInHoldOut"
    """Linear in, then hold this value until the next keyframe."""

    CubicHermite = "CubicHermite"
    """Cubic Hermite with automatic derivatives; will not over/undershoot."""

    LinearInCubicHermiteOut = "LinearInCubicHermiteOut"
    """Linear in, cubic Hermite out; may over/undershoot."""

    Bezier = "Bezier"
    """Cubic Bezier with automatic control points; will not over/undershoot."""

    BezierInHoldOut = "BezierInHoldOut"
    """Bezier in, then hold this value until the next keyframe."""

    CubicHermiteInHoldOut = "CubicHermiteInHoldOut"
    """Cubic Hermite in, then hold this value until the next keyframe."""

    LinearInBezierOut = "LinearInBezierOut"
    """Linear in, Bezier out; may over/undershoot."""


#: Modes that hold their value from this keyframe until the next one.
HOLD_OUT_MODES = frozenset({
    Interpolation.LinearInHoldOut,
    Interpolation.BezierInHoldOut,
    Interpolation.CubicHermiteInHoldOut,
})

#: Modes entered linearly.
LINEAR_IN_MODES = frozenset({
    Interpolation.Linear,
    Interpolation.LinearInHoldOut,
    Interpolation.LinearInCubicHermiteOut,
    Interpolation.LinearInBezierOut,
})

#: Modes left along a Bezier.
BEZIER_OUT_MODES = frozenset({
    Interpolation.Bezier,
    Interpolation.LinearInBezierOut,
})

#: Modes left along a cubic Hermite.
HERMITE_OUT_MODES = frozenset({
    Interpolation.CubicHermite,
    Interpolation.LinearInCubicHermiteOut,
})

#: Modes whose incoming derivative is the slope of the chord from the previous
#: keyframe, which is what lets them over/undershoot.
LINEAR_SLOPE_MODES = frozenset({
    Interpolation.LinearInCubicHermiteOut,
    Interpolation.LinearInBezierOut,
})

#: How far Bezier handles reach towards the neighbouring keyframe, as a fraction
#: of the distance to it. A third is the usual choice: it is the spacing at which
#: a cubic Bezier reproduces the cubic Hermite through the same tangents.
DEFAULT_CONTROL_EXTENT = 3.0


def _tangent_step(slope: float, distance: float) -> Tuple[float, float]:
    """Walk ``distance`` along a line of gradient ``slope``.

    Returns the ``(time, value)`` components of the step, both signed so that
    adding them moves forward along the tangent and subtracting them moves
    back. The reference implementation takes a square root for the value
    component and so always returns it positive, which puts the handles of a
    descending keyframe on the wrong side of it.
    """
    dtime = distance / math.sqrt(1.0 + slope * slope)
    if slope == 0.0:
        return dtime, 0.0
    # distance / sqrt(1 + 1/slope**2) == abs(slope) * dtime, written this way to
    # match the reference's algebra.
    return dtime, math.copysign(distance / math.sqrt(1.0 + 1.0 / (slope * slope)),
                                slope)


class Keyframe:
    """A value at a time, and how the curve enters and leaves it.

    The neighbour links, derivative and control points are filled in by
    :meth:`Interpolator.build`; a keyframe on its own does not know enough to
    compute them.
    """

    def __init__(self, time: float, value: float,
                 interpolation: Interpolation = Interpolation.Bezier,
                 control_extent: float = DEFAULT_CONTROL_EXTENT) -> None:
        self.time = float(time)
        self.value = float(value)
        self.interpolation = Interpolation(interpolation)
        self.control_extent = float(control_extent)

        self.previous: Optional["Keyframe"] = None
        self.next: Optional["Keyframe"] = None

        #: Tangent used by the Hermite and Bezier segments touching this frame.
        self.derivative: float = 0.0
        #: Bezier handle on the incoming side, ``None`` at the first keyframe.
        self.control_point_1: Optional[Tuple[float, float]] = None
        #: Bezier handle on the outgoing side, ``None`` at the last keyframe.
        self.control_point_2: Optional[Tuple[float, float]] = None

    def __repr__(self) -> str:
        return "Keyframe(time=%g, value=%g, interpolation=%s)" % (
            self.time, self.value, self.interpolation.value)

    def compute_derivative(self) -> float:
        """The tangent to use at this keyframe.

        Returns zero wherever a non-zero tangent could push the curve outside
        the values being interpolated: at the ends, at a local extremum, and
        where this keyframe repeats a neighbour's value. That, rather than any
        clamping after the fact, is what stops the smooth modes from
        over/undershooting.
        """
        previous, following = self.previous, self.next
        if previous is None or following is None:
            return 0.0

        value, before, after = self.value, previous.value, following.value
        if value > before and value > after:      # local maximum
            return 0.0
        if value < before and value < after:      # local minimum
            return 0.0
        if value == before or value == after:
            return 0.0
        if self.interpolation in HOLD_OUT_MODES:
            return 0.0
        if self.interpolation in LINEAR_SLOPE_MODES:
            # Enter along the chord and keep going: this is the overshoot the
            # LinearIn* modes exist to allow.
            return (value - before) / (self.time - previous.time)

        # Otherwise take the tangent of the smooth curve that would pass
        # through the neighbours if this keyframe were not there -- a Hermite
        # over (previous, next) flat at both ends -- and then stretch it by how
        # far this keyframe sits from that curve. Sitting below the smooth path
        # (in the direction of travel) steepens the tangent, above flattens it.
        span = following.time - previous.time
        t = (self.time - previous.time) / span
        smooth_value = before + (after - before) * (3.0 * t ** 2 - 2.0 * t ** 3)
        smooth_slope = (after - before) * (6.0 * t - 6.0 * t ** 2) / span

        # abs(after - before) cannot be zero here: a keyframe between two equal
        # neighbours is either a local extremum or equal to them, and both cases
        # have already returned.
        scale = 2.0 * (smooth_value - value) / abs(after - before)
        scale *= math.copysign(1.0, after - before)
        return smooth_slope * math.exp(scale)

    def compute_control_points(self) -> None:
        """Place the Bezier handles along this keyframe's tangent.

        Each handle reaches ``1 / control_extent`` of the way to the
        neighbour it faces, measured as a straight-line distance in the
        (time, value) plane. Its time component is clamped inside the interval
        so that time stays monotonic along the curve, which is what lets
        :meth:`_BezierSegment.__call__` solve for the parameter.
        """
        slope = self.derivative

        if self.previous is None:
            self.control_point_1 = None
        else:
            distance = math.hypot(self.time - self.previous.time,
                                  self.value - self.previous.value)
            dtime, dvalue = _tangent_step(slope, distance / self.control_extent)
            self.control_point_1 = (max(self.time - dtime, self.previous.time),
                                    self.value - dvalue)

        if self.next is None:
            self.control_point_2 = None
        else:
            distance = math.hypot(self.time - self.next.time,
                                  self.value - self.next.value)
            dtime, dvalue = _tangent_step(slope, distance / self.control_extent)
            self.control_point_2 = (min(self.time + dtime, self.next.time),
                                    self.value + dvalue)


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


class _Segment:
    """A piece of the curve, valid between two keyframes."""

    def __call__(self, time: float) -> float:
        raise NotImplementedError


class _ConstantSegment(_Segment):
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self, time: float) -> float:
        return self.value


class _LinearSegment(_Segment):
    def __init__(self, t1: float, v1: float, t2: float, v2: float) -> None:
        self.t1, self.v1, self.t2, self.v2 = t1, v1, t2, v2

    def __call__(self, time: float) -> float:
        span = self.t2 - self.t1
        if span == 0:
            return self.v2
        t = min(1.0, max(0.0, (time - self.t1) / span))
        return self.v1 * (1.0 - t) + self.v2 * t


class _HermiteSegment(_Segment):
    """Cubic Hermite between two keyframes with prescribed end tangents."""

    def __init__(self, t1: float, v1: float, d1: float,
                 t2: float, v2: float, d2: float) -> None:
        self.t1, self.v1, self.d1 = t1, v1, d1
        self.t2, self.v2, self.d2 = t2, v2, d2

    def __call__(self, time: float) -> float:
        span = self.t2 - self.t1
        if span == 0:
            return self.v2
        t = min(1.0, max(0.0, (time - self.t1) / span))
        t2 = t * t
        t3 = t2 * t
        return ((2.0 * t3 - 3.0 * t2 + 1.0) * self.v1
                + (t3 - 2.0 * t2 + t) * span * self.d1
                + (-2.0 * t3 + 3.0 * t2) * self.v2
                + (t3 - t2) * span * self.d2)


def _bezier(a: float, b: float, c: float, d: float, u: float) -> float:
    m = 1.0 - u
    return (m * m * m * a + 3.0 * m * m * u * b
            + 3.0 * m * u * u * c + u * u * u * d)


def _bezier_slope(a: float, b: float, c: float, d: float, u: float) -> float:
    m = 1.0 - u
    return 3.0 * m * m * (b - a) + 6.0 * m * u * (c - b) + 3.0 * u * u * (d - c)


class _BezierSegment(_Segment):
    """Cubic Bezier through two keyframes and the handles facing each other."""

    #: Newton is seeded with normalized time and converges in a couple of steps
    #: for handles this well behaved; the cap is a guard, not a budget.
    _NEWTON_STEPS = 8
    _BISECTION_STEPS = 60
    _TOLERANCE = 1e-12

    def __init__(self, first: Keyframe, second: Keyframe) -> None:
        self.p1 = (first.time, first.value)
        self.p2 = first.control_point_2 or (first.time, first.value)
        self.p3 = second.control_point_1 or (second.time, second.value)
        self.p4 = (second.time, second.value)

    def parameter_at(self, time: float) -> float:
        """The curve parameter whose time component is ``time``.

        The parameter is not normalized time: both coordinates of a cubic
        Bezier are cubic in it. Time is monotonic along the curve because
        :meth:`Keyframe.compute_control_points` clamps the handles into the
        interval, so Newton from normalized time converges, and bisection is a
        safe fallback when a flat spot makes the Newton step useless.
        """
        x1, x2, x3, x4 = self.p1[0], self.p2[0], self.p3[0], self.p4[0]
        span = x4 - x1
        if span <= 0:
            return 0.0

        u = min(1.0, max(0.0, (time - x1) / span))
        for _ in range(self._NEWTON_STEPS):
            error = _bezier(x1, x2, x3, x4, u) - time
            if abs(error) < self._TOLERANCE:
                return u
            slope = _bezier_slope(x1, x2, x3, x4, u)
            if abs(slope) < 1e-9:
                break
            stepped = u - error / slope
            if not (0.0 <= stepped <= 1.0):
                break
            u = stepped

        low, high = 0.0, 1.0
        u = 0.5
        for _ in range(self._BISECTION_STEPS):
            x = _bezier(x1, x2, x3, x4, u)
            if abs(x - time) < self._TOLERANCE:
                break
            if x < time:
                low = u
            else:
                high = u
            u = 0.5 * (low + high)
        return u

    def __call__(self, time: float) -> float:
        u = self.parameter_at(time)
        return _bezier(self.p1[1], self.p2[1], self.p3[1], self.p4[1], u)


def _make_segment(first: Keyframe, second: Keyframe) -> _Segment:
    """The piece of curve running from ``first`` to ``second``.

    ``first``'s mode decides how the segment leaves and ``second``'s decides how
    it arrives, so both are consulted. A hold overrides everything else.
    """
    if first.interpolation in HOLD_OUT_MODES:
        return _ConstantSegment(first.value)

    if first.interpolation == Interpolation.Linear:
        if second.interpolation in LINEAR_IN_MODES:
            return _LinearSegment(first.time, first.value,
                                  second.time, second.value)
        # Leaves linearly but arrives smoothly. The reference implementation
        # produces no segment at all here, which desynchronizes its segment and
        # end-time lists; a Hermite entered along the chord is the reading its
        # mode names imply.
        span = second.time - first.time
        chord = 0.0 if span == 0 else (second.value - first.value) / span
        return _HermiteSegment(first.time, first.value, chord,
                               second.time, second.value, second.derivative)

    if first.interpolation in BEZIER_OUT_MODES:
        return _BezierSegment(first, second)

    if first.interpolation in HERMITE_OUT_MODES:
        return _HermiteSegment(first.time, first.value, first.derivative,
                               second.time, second.value, second.derivative)

    raise ValueError("Unhandled interpolation mode: %r" % (first.interpolation,))


# ---------------------------------------------------------------------------
# The interpolator
# ---------------------------------------------------------------------------


class Interpolator:
    """Piecewise interpolation of one value over a list of keyframes.

    >>> interp = Interpolator([Keyframe(0, 0.0), Keyframe(1, 10.0)])
    >>> interp(0.0), interp(1.0)
    (0.0, 10.0)

    Times before the first keyframe and after the last hold that keyframe's
    value.
    """

    def __init__(self, keyframes: Iterable[Keyframe]) -> None:
        self.keyframes: List[Keyframe] = list(keyframes)
        if not self.keyframes:
            raise ValueError("An interpolator needs at least one keyframe")
        self.segments: List[_Segment] = []
        #: ``segments[i]`` applies up to ``segment_end_times[i]``.
        self.segment_end_times: List[float] = []
        self.build()

    def build(self) -> None:
        """Sort and link the keyframes, then assemble the segments."""
        # Keep the last of any keyframes sharing a time, matching the way the
        # animation panel replaces a keyframe when one is added over another.
        unique: "dict[float, Keyframe]" = {}
        for frame in sorted(self.keyframes, key=lambda k: k.time):
            unique[frame.time] = frame
        frames = list(unique.values())
        self.keyframes = frames

        for index, frame in enumerate(frames):
            frame.previous = frames[index - 1] if index > 0 else None
            frame.next = frames[index + 1] if index + 1 < len(frames) else None

        # Derivatives first: the control points are placed along them.
        for frame in frames:
            frame.derivative = frame.compute_derivative()
        for frame in frames:
            frame.compute_control_points()

        if len(frames) == 1:
            self.segments = [_ConstantSegment(frames[0].value)]
            self.segment_end_times = [frames[0].time]
            return

        self.segments = []
        self.segment_end_times = []
        for first, second in zip(frames[:-1], frames[1:]):
            self.segments.append(_make_segment(first, second))
            self.segment_end_times.append(second.time)

    @property
    def start(self) -> float:
        return self.keyframes[0].time

    @property
    def end(self) -> float:
        return self.keyframes[-1].time

    def at(self, time: float) -> float:
        """The interpolated value at ``time``, held outside the keyframe range."""
        if time <= self.keyframes[0].time:
            return self.keyframes[0].value
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value
        # Each segment covers [start, end), so a time landing exactly on an
        # interior keyframe belongs to the segment starting there. Without the
        # strict comparison a hold-out segment would swallow the keyframe that
        # ends it and report the held value instead of the new one.
        for end_time, segment in zip(self.segment_end_times, self.segments):
            if time < end_time:
                return segment(time)
        return self.keyframes[-1].value

    def __call__(self, time: float) -> float:
        return self.at(time)

    def __len__(self) -> int:
        return len(self.keyframes)

    def __getitem__(self, index: int) -> Keyframe:
        return self.keyframes[index]


def from_values(times: Sequence[float], values: Sequence[float],
                interpolation: Interpolation = Interpolation.Bezier,
                modes: Optional[Sequence[Interpolation]] = None) -> Interpolator:
    """Build an :class:`Interpolator` from parallel time and value sequences.

    Parameters
    ----------
    times : sequence of float
        Keyframe times.
    values : sequence of float
        Keyframe values, the same length as ``times``.
    interpolation : Interpolation, optional
        Mode for keyframes with no entry in ``modes``. Default
        ``Interpolation.Bezier``.
    modes : sequence of Interpolation or None, optional
        Per-keyframe modes, the same length as ``times``. An entry of ``None``
        falls back to ``interpolation``.

    Returns
    -------
    Interpolator
    """
    if len(times) != len(values):
        raise ValueError("times and values must be the same length")
    if modes is not None and len(modes) != len(times):
        raise ValueError("modes must be the same length as times")

    frames = []
    for index, (time, value) in enumerate(zip(times, values)):
        mode = interpolation if modes is None or modes[index] is None \
            else modes[index]
        frames.append(Keyframe(time, value, mode))
    return Interpolator(frames)


# ---------------------------------------------------------------------------
# Whole-view interpolation
# ---------------------------------------------------------------------------
#
# The animation keyframes the viewer produces are whole view dicts, not scalars,
# so they are taken apart into independent one-dimensional "channels" -- one per
# property, or one per component for the array-valued ones -- each with its own
# Interpolator. This is the python side of vt.buildInterpolators/vt.evaluate in
# resources/js/viewtools.js and must agree with it.

#: Leaf property names that are numeric but discrete. Interpolating them yields
#: values their setter cannot use, and ``layers`` recompiles the shaders on every
#: assignment, so a fractional layer count is both wrong and very slow.
STEP_PROP_LEAVES = frozenset({"layers"})

#: Properties measured in degrees around a circle, which have to be unwrapped
#: before a spline can be fitted to them.
ANGLE_PROPS = frozenset({"camera.azimuth"})

#: Keys that are animation bookkeeping rather than view properties.
RESERVED_KEYS = frozenset({"time", "frame", "interpolation"})


def shortest_step(start: float, end: float) -> float:
    """The signed change in degrees from ``start`` to ``end``, at most half a turn.

    Reproduces ``Viewer._animInterp`` in ``resources/js/mriview.js`` exactly,
    tie-break included: half a turn is equally short either way, and
    ``_animInterp`` resolves it by travelling *against* the sign of the raw
    difference. ``math.fmod`` rather than ``%`` because the sign of the
    remainder has to match javascript's.
    """
    step = math.fmod(end - start, 360.0)
    if step >= 180.0:
        return step - 360.0
    if step <= -180.0:
        return step + 360.0
    return step


def unwrap_angles(values: Sequence[float]) -> List[float]:
    """Re-express an angle sequence so it never jumps by a full turn."""
    out = [float(values[0])]
    for index in range(1, len(values)):
        out.append(out[index - 1] + shortest_step(values[index - 1], values[index]))
    return out


def wrap_angle(value: float) -> float:
    """Bring an angle back into ``[0, 360)``."""
    return value % 360.0


def _is_sequence(value: object) -> bool:
    if isinstance(value, (str, bytes, dict)):
        return False
    return isinstance(value, (list, tuple)) or hasattr(value, "__len__")


def _is_blendable(value: object) -> bool:
    """Whether a value is a finite number. ``bool`` is excluded deliberately."""
    if isinstance(value, (bool, str, bytes)) or value is None:
        return False
    try:
        return math.isfinite(float(value))          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


class _StepChannel:
    """Holds the value of the latest keyframe at or before the requested time."""

    def __init__(self, frames: Sequence[dict], prop: str, time_key: str) -> None:
        self._points = [(frame[time_key], frame[prop])
                        for frame in frames if prop in frame]
        self._fallback = frames[0].get(prop)

    def at(self, time: float) -> object:
        value = self._fallback
        for point_time, point_value in self._points:
            if point_time > time:
                break
            value = point_value
        return value


class _SmoothChannel:
    def __init__(self, interpolator: Interpolator, is_angle: bool) -> None:
        self._interpolator = interpolator
        self._is_angle = is_angle

    def at(self, time: float) -> object:
        value = self._interpolator.at(time)
        return wrap_angle(value) if self._is_angle else value


class _ArrayChannel:
    def __init__(self, components: Sequence[_SmoothChannel]) -> None:
        self._components = list(components)

    def at(self, time: float) -> object:
        return [component.at(time) for component in self._components]


def _modes_of(frames: Sequence[dict],
              default_mode: Interpolation) -> List[Interpolation]:
    modes = []
    for frame in frames:
        mode = frame.get("interpolation", default_mode)
        try:
            modes.append(Interpolation(mode))
        except ValueError:
            modes.append(default_mode)
    return modes


def _smooth_channel(frames: Sequence[dict], times: Sequence[float],
                    values: Sequence[float], is_angle: bool,
                    default_mode: Interpolation) -> _SmoothChannel:
    if is_angle:
        values = unwrap_angles(values)
    keyframes = [Keyframe(time, value, mode) for time, value, mode
                 in zip(times, values, _modes_of(frames, default_mode))]
    return _SmoothChannel(Interpolator(keyframes), is_angle)


def _make_channel(frames: Sequence[dict], prop: str, time_key: str,
                  default_mode: Interpolation) -> object:
    times = [frame[time_key] for frame in frames]
    first = frames[0].get(prop)
    leaf = prop.split(".")[-1]

    if first is None or leaf in STEP_PROP_LEAVES or isinstance(first, (bool, str)):
        return _StepChannel(frames, prop, time_key)

    if _is_sequence(first):
        length = len(first)                          # type: ignore[arg-type]
        for frame in frames:
            value = frame.get(prop)
            if not _is_sequence(value) or len(value) != length:   # type: ignore[arg-type]
                return _StepChannel(frames, prop, time_key)
            if not all(_is_blendable(item) for item in value):    # type: ignore[union-attr]
                return _StepChannel(frames, prop, time_key)
        components = [
            _smooth_channel(frames, times,
                            [float(frame[prop][index]) for frame in frames],
                            False, default_mode)
            for index in range(length)
        ]
        return _ArrayChannel(components)

    for frame in frames:
        if not _is_blendable(frame.get(prop)):
            return _StepChannel(frames, prop, time_key)

    return _smooth_channel(frames, times,
                           [float(frame[prop]) for frame in frames],
                           prop in ANGLE_PROPS, default_mode)


def build_channels(keyframes: Sequence[dict], time_key: str = "time",
                   default_mode: Interpolation = Interpolation.Bezier
                   ) -> "dict[str, object]":
    """Split a list of keyframe view dicts into per-property interpolators.

    Parameters
    ----------
    keyframes : sequence of dict
        Each holds ``time_key`` plus view properties, and optionally an
        ``interpolation`` key naming that keyframe's :class:`Interpolation`.
    time_key : str, optional
        The key holding each keyframe's position on the time axis. ``"time"``
        in python, ``"frame"`` in the browser. Default ``"time"``.
    default_mode : Interpolation, optional
        Mode for keyframes with no ``interpolation`` key. Default
        ``Interpolation.Bezier``.

    Returns
    -------
    dict
        Property name to an object with an ``at(time)`` method. Each channel is
        built from the keyframes that carry its property: a property every
        keyframe carries behaves as it always did, one carried by a single
        keyframe is constant, and the keyframes that leave it out are simply
        not on its curve. That is how a flat keyframe, which records no camera
        angle, stays out of the ``camera.azimuth`` channel.
    """
    frames = sorted(keyframes, key=lambda frame: frame[time_key])
    if not frames:
        raise ValueError("Need at least one keyframe")

    channels: "dict[str, object]" = {}
    for frame in frames:
        for prop in frame:
            if prop in RESERVED_KEYS or prop in channels:
                continue
            carriers = [carrier for carrier in frames if prop in carrier]
            channels[prop] = _make_channel(carriers, prop, time_key, default_mode)
    return channels


def evaluate(channels: "dict[str, object]", time: float) -> "dict[str, object]":
    """Reassemble a view dict from the channels built by :func:`build_channels`."""
    return {prop: channel.at(time)          # type: ignore[attr-defined]
            for prop, channel in channels.items()}
