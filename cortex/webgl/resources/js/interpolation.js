// Keyframe interpolation for viewer animations.
//
// A one-dimensional piecewise interpolator built from a list of keyframes, each
// carrying its own interpolation mode. The modes follow the way Adobe products
// describe keyframe interpolation -- a mode names how a keyframe is entered and
// how it is left, so the shape of the curve between two keyframes is decided by
// the pair of modes at its ends -- with cubic hermite added.
//
// cortex/webgl/interpolation.py is the twin of this file and must stay in step
// with it: the animation panel interpolates here while JSMixer._get_anim_seq
// interpolates in python, and an animation should look the same either way.
// Both use the same closed forms and both carry the mode as one of the strings
// in `Interpolation`, so a keyframe crosses between them unchanged. There is a
// test asserting the two agree numerically.
//
// Deviations from the reference implementation this was ported from are marked
// with "Deviation:" comments at the four sites where they occur.

var jsplot = (function (module) {
    module.interpolation = (function (ip) {

    // How a keyframe is entered and left. Strings rather than numbers so a
    // keyframe survives JSON round-trips to python as its own name.
    ip.Interpolation = {
        Linear: "Linear",                                       // linear in, linear out
        LinearInHoldOut: "LinearInHoldOut",                     // linear in, then hold
        CubicHermite: "CubicHermite",                           // auto derivatives, no over/undershoot
        LinearInCubicHermiteOut: "LinearInCubicHermiteOut",     // linear in, hermite out, may overshoot
        Bezier: "Bezier",                                       // auto control points, no over/undershoot
        BezierInHoldOut: "BezierInHoldOut",                     // bezier in, then hold
        CubicHermiteInHoldOut: "CubicHermiteInHoldOut",         // hermite in, then hold
        LinearInBezierOut: "LinearInBezierOut"                  // linear in, bezier out, may overshoot
    };

    // The order the animation panel lists them in: the two that cannot
    // over/undershoot first, then the holds, then the ones that can.
    ip.MODE_ORDER = [
        "Bezier",
        "CubicHermite",
        "Linear",
        "BezierInHoldOut",
        "CubicHermiteInHoldOut",
        "LinearInHoldOut",
        "LinearInBezierOut",
        "LinearInCubicHermiteOut"
    ];

    // Short labels for the panel's dropdown.
    ip.MODE_LABELS = {
        Bezier: "bezier (smooth)",
        CubicHermite: "cubic hermite (smooth)",
        Linear: "linear",
        BezierInHoldOut: "bezier in, hold",
        CubicHermiteInHoldOut: "hermite in, hold",
        LinearInHoldOut: "linear in, hold",
        LinearInBezierOut: "linear in, bezier out",
        LinearInCubicHermiteOut: "linear in, hermite out"
    };

    ip.DEFAULT_MODE = ip.Interpolation.Bezier;

    function has(set, mode) {
        return set.indexOf(mode) >= 0;
    }

    // Modes that hold their value from this keyframe until the next one.
    var HOLD_OUT_MODES = ["LinearInHoldOut", "BezierInHoldOut",
                          "CubicHermiteInHoldOut"];
    // Modes entered linearly.
    var LINEAR_IN_MODES = ["Linear", "LinearInHoldOut",
                           "LinearInCubicHermiteOut", "LinearInBezierOut"];
    // Modes left along a bezier.
    var BEZIER_OUT_MODES = ["Bezier", "LinearInBezierOut"];
    // Modes left along a cubic hermite.
    var HERMITE_OUT_MODES = ["CubicHermite", "LinearInCubicHermiteOut"];
    // Modes whose incoming derivative is the chord from the previous keyframe,
    // which is what lets them over/undershoot.
    var LINEAR_SLOPE_MODES = ["LinearInCubicHermiteOut", "LinearInBezierOut"];

    ip.isValidMode = function(mode) {
        return ip.Interpolation.hasOwnProperty(mode);
    };

    // How far bezier handles reach towards the neighbouring keyframe, as a
    // fraction of the distance to it. A third is the usual choice: it is the
    // spacing at which a cubic bezier reproduces the cubic hermite through the
    // same tangents.
    ip.DEFAULT_CONTROL_EXTENT = 3.0;

    function sign(x) {
        return x < 0 ? -1 : 1;
    }

    // Walk `distance` along a line of gradient `slope`, returning the signed
    // (time, value) components of the step.
    //
    // Deviation: the reference takes a square root for the value component,
    // which is always positive, so the handles of a keyframe with a negative
    // derivative land off its own tangent line and break the C1 continuity the
    // whole scheme exists to provide. The sign puts them back on it.
    function tangentStep(slope, distance) {
        var dtime = distance / Math.sqrt(1.0 + slope * slope);
        if (slope === 0)
            return [dtime, 0.0];
        return [dtime,
                sign(slope) * distance / Math.sqrt(1.0 + 1.0 / (slope * slope))];
    }

    // ------------------------------------------------------------------
    // Keyframes
    // ------------------------------------------------------------------

    // A value at a time, and how the curve enters and leaves it. The neighbour
    // links, derivative and control points are filled in by Interpolator1D --
    // a keyframe on its own does not know enough to compute them.
    ip.Keyframe1D = function(time, value, mode, controlExtent) {
        this.time = time;
        this.value = value;
        this.mode = ip.isValidMode(mode) ? mode : ip.DEFAULT_MODE;
        this.controlExtent = controlExtent || ip.DEFAULT_CONTROL_EXTENT;

        this.previous = null;
        this.next = null;
        this.derivative = 0.0;
        this.controlPoint1 = null;   // handle on the incoming side
        this.controlPoint2 = null;   // handle on the outgoing side
    };

    // The tangent to use at this keyframe.
    //
    // Returns zero wherever a non-zero tangent could push the curve outside the
    // values being interpolated: at the ends, at a local extremum, and where
    // this keyframe repeats a neighbour's value. That, rather than any clamping
    // after the fact, is what stops the smooth modes from over/undershooting.
    ip.Keyframe1D.prototype.computeDerivative = function() {
        var p = this.previous, n = this.next;
        if (p === null || n === null)
            return 0.0;

        var value = this.value, before = p.value, after = n.value;
        if (value > before && value > after)    // local maximum
            return 0.0;
        if (value < before && value < after)    // local minimum
            return 0.0;
        if (value === before || value === after)
            return 0.0;
        if (has(HOLD_OUT_MODES, this.mode))
            return 0.0;
        if (has(LINEAR_SLOPE_MODES, this.mode))
            // Enter along the chord and keep going: this is the overshoot the
            // LinearIn* modes exist to allow.
            return (value - before) / (this.time - p.time);

        // Otherwise take the tangent of the smooth curve that would pass
        // through the neighbours if this keyframe were not there -- a hermite
        // over (previous, next) flat at both ends -- and then stretch it by how
        // far this keyframe sits from that curve. Sitting below the smooth path
        // (in the direction of travel) steepens the tangent, above flattens it.
        var span = n.time - p.time;
        var t = (this.time - p.time) / span;
        var smoothValue = before + (after - before) * (3.0 * t * t - 2.0 * t * t * t);
        var smoothSlope = (after - before) * (6.0 * t - 6.0 * t * t) / span;

        // abs(after - before) cannot be zero here: a keyframe between two equal
        // neighbours is either a local extremum or equal to them, and both
        // cases have already returned.
        var scale = 2.0 * (smoothValue - value) / Math.abs(after - before);
        scale *= sign(after - before);
        return smoothSlope * Math.exp(scale);
    };

    // Place the bezier handles along this keyframe's tangent. Each reaches
    // 1/controlExtent of the way to the neighbour it faces, measured as a
    // straight-line distance in the (time, value) plane. The time component is
    // clamped inside the interval so that time stays monotonic along the curve,
    // which is what lets BezierSegment solve for its parameter.
    ip.Keyframe1D.prototype.computeControlPoints = function() {
        var slope = this.derivative, step;

        if (this.previous === null) {
            this.controlPoint1 = null;
        } else {
            step = tangentStep(slope, Math.sqrt(
                Math.pow(this.time - this.previous.time, 2) +
                Math.pow(this.value - this.previous.value, 2)) / this.controlExtent);
            this.controlPoint1 = [Math.max(this.time - step[0], this.previous.time),
                                  this.value - step[1]];
        }

        if (this.next === null) {
            this.controlPoint2 = null;
        } else {
            step = tangentStep(slope, Math.sqrt(
                Math.pow(this.time - this.next.time, 2) +
                Math.pow(this.value - this.next.value, 2)) / this.controlExtent);
            this.controlPoint2 = [Math.min(this.time + step[0], this.next.time),
                                  this.value + step[1]];
        }
    };

    // ------------------------------------------------------------------
    // Segments
    // ------------------------------------------------------------------

    function ConstantSegment(value) {
        this.value = value;
    }
    ConstantSegment.prototype.at = function(time) {
        return this.value;
    };

    function LinearSegment(t1, v1, t2, v2) {
        this.t1 = t1; this.v1 = v1; this.t2 = t2; this.v2 = v2;
    }
    LinearSegment.prototype.at = function(time) {
        var span = this.t2 - this.t1;
        if (span === 0)
            return this.v2;
        var t = Math.min(1.0, Math.max(0.0, (time - this.t1) / span));
        return this.v1 * (1.0 - t) + this.v2 * t;
    };

    // Cubic hermite between two keyframes with prescribed end tangents.
    function HermiteSegment(t1, v1, d1, t2, v2, d2) {
        this.t1 = t1; this.v1 = v1; this.d1 = d1;
        this.t2 = t2; this.v2 = v2; this.d2 = d2;
    }
    HermiteSegment.prototype.at = function(time) {
        var span = this.t2 - this.t1;
        if (span === 0)
            return this.v2;
        var t = Math.min(1.0, Math.max(0.0, (time - this.t1) / span));
        var t2 = t * t, t3 = t2 * t;
        return ((2.0 * t3 - 3.0 * t2 + 1.0) * this.v1 +
                (t3 - 2.0 * t2 + t) * span * this.d1 +
                (-2.0 * t3 + 3.0 * t2) * this.v2 +
                (t3 - t2) * span * this.d2);
    };

    function bezier(a, b, c, d, u) {
        var m = 1.0 - u;
        return (m * m * m * a + 3.0 * m * m * u * b +
                3.0 * m * u * u * c + u * u * u * d);
    }

    function bezierSlope(a, b, c, d, u) {
        var m = 1.0 - u;
        return 3.0 * m * m * (b - a) + 6.0 * m * u * (c - b) + 3.0 * u * u * (d - c);
    }

    var NEWTON_STEPS = 8, BISECTION_STEPS = 60, TOLERANCE = 1e-12;

    // Cubic bezier through two keyframes and the handles facing each other.
    function BezierSegment(first, second) {
        this.p1 = [first.time, first.value];
        this.p2 = first.controlPoint2 || [first.time, first.value];
        this.p3 = second.controlPoint1 || [second.time, second.value];
        this.p4 = [second.time, second.value];
    }

    // The curve parameter whose time component is `time`.
    //
    // Deviation: the reference returns early using normalized time as the
    // parameter, leaving its own root-finding loop unreachable. Both
    // coordinates of a cubic bezier are cubic in the parameter, so the two
    // agree only when the handles happen to be evenly spaced in time -- which
    // is exactly when the smoothing is doing nothing. Time is monotonic along
    // the curve because computeControlPoints clamps the handles into the
    // interval, so newton from normalized time converges, and bisection is a
    // safe fallback when a flat spot makes the newton step useless.
    BezierSegment.prototype.parameterAt = function(time) {
        var x1 = this.p1[0], x2 = this.p2[0], x3 = this.p3[0], x4 = this.p4[0];
        var span = x4 - x1;
        if (span <= 0)
            return 0.0;

        var u = Math.min(1.0, Math.max(0.0, (time - x1) / span));
        var i, error, slope, stepped;
        for (i = 0; i < NEWTON_STEPS; i++) {
            error = bezier(x1, x2, x3, x4, u) - time;
            if (Math.abs(error) < TOLERANCE)
                return u;
            slope = bezierSlope(x1, x2, x3, x4, u);
            if (Math.abs(slope) < 1e-9)
                break;
            stepped = u - error / slope;
            if (!(stepped >= 0.0 && stepped <= 1.0))
                break;
            u = stepped;
        }

        var low = 0.0, high = 1.0, x;
        u = 0.5;
        for (i = 0; i < BISECTION_STEPS; i++) {
            x = bezier(x1, x2, x3, x4, u);
            if (Math.abs(x - time) < TOLERANCE)
                break;
            if (x < time)
                low = u;
            else
                high = u;
            u = 0.5 * (low + high);
        }
        return u;
    };

    BezierSegment.prototype.at = function(time) {
        return bezier(this.p1[1], this.p2[1], this.p3[1], this.p4[1],
                      this.parameterAt(time));
    };

    // The piece of curve running from `first` to `second`. `first`'s mode
    // decides how the segment leaves and `second`'s decides how it arrives, so
    // both are consulted. A hold overrides everything else.
    function makeSegment(first, second) {
        if (has(HOLD_OUT_MODES, first.mode))
            return new ConstantSegment(first.value);

        if (first.mode === ip.Interpolation.Linear) {
            if (has(LINEAR_IN_MODES, second.mode))
                return new LinearSegment(first.time, first.value,
                                         second.time, second.value);
            // Deviation: leaves linearly but arrives smoothly. The reference
            // produces no segment at all for this combination, desynchronizing
            // its segment and end-time lists; a hermite entered along the chord
            // is the reading its mode names imply.
            var span = second.time - first.time;
            var chord = span === 0 ? 0 : (second.value - first.value) / span;
            return new HermiteSegment(first.time, first.value, chord,
                                      second.time, second.value, second.derivative);
        }

        if (has(BEZIER_OUT_MODES, first.mode))
            return new BezierSegment(first, second);

        if (has(HERMITE_OUT_MODES, first.mode))
            return new HermiteSegment(first.time, first.value, first.derivative,
                                      second.time, second.value, second.derivative);

        throw new Error("Unhandled interpolation mode: " + first.mode);
    }

    // ------------------------------------------------------------------
    // The interpolator
    // ------------------------------------------------------------------

    // Piecewise interpolation of one value over a list of Keyframe1D. Times
    // before the first keyframe and after the last hold that keyframe's value,
    // which is what the panel's frame slider does at either end of its range.
    ip.Interpolator1D = function(keyframes) {
        this.keyframes = keyframes.slice();
        if (this.keyframes.length === 0)
            throw new Error("An interpolator needs at least one keyframe");
        this.segments = [];
        this.segmentEndTimes = [];   // segments[i] applies up to segmentEndTimes[i]
        this.build();
    };

    ip.Interpolator1D.prototype.build = function() {
        var i, frames = this.keyframes.slice().sort(function(a, b) {
            return a.time - b.time;
        });

        // Keep the last of any keyframes sharing a time, matching the way the
        // animation panel replaces a keyframe when one is added over another.
        var unique = [];
        for (i = 0; i < frames.length; i++) {
            if (unique.length > 0 &&
                    unique[unique.length - 1].time === frames[i].time)
                unique[unique.length - 1] = frames[i];
            else
                unique.push(frames[i]);
        }
        frames = unique;
        this.keyframes = frames;

        for (i = 0; i < frames.length; i++) {
            frames[i].previous = i > 0 ? frames[i - 1] : null;
            frames[i].next = i + 1 < frames.length ? frames[i + 1] : null;
        }

        // Derivatives first: the control points are placed along them.
        for (i = 0; i < frames.length; i++)
            frames[i].derivative = frames[i].computeDerivative();
        for (i = 0; i < frames.length; i++)
            frames[i].computeControlPoints();

        this.segments = [];
        this.segmentEndTimes = [];

        // Deviation: the reference builds a one-element segment list for a lone
        // keyframe and then keeps going, immediately discarding it.
        if (frames.length === 1) {
            this.segments.push(new ConstantSegment(frames[0].value));
            this.segmentEndTimes.push(frames[0].time);
            return;
        }

        for (i = 0; i < frames.length - 1; i++) {
            this.segments.push(makeSegment(frames[i], frames[i + 1]));
            this.segmentEndTimes.push(frames[i + 1].time);
        }
    };

    ip.Interpolator1D.prototype.at = function(time) {
        var frames = this.keyframes;
        if (time <= frames[0].time)
            return frames[0].value;
        if (time >= frames[frames.length - 1].time)
            return frames[frames.length - 1].value;

        // Each segment covers [start, end), so a time landing exactly on an
        // interior keyframe belongs to the segment starting there. Without the
        // strict comparison a hold-out segment would swallow the keyframe that
        // ends it and report the held value instead of the new one.
        for (var i = 0; i < this.segments.length; i++)
            if (time < this.segmentEndTimes[i])
                return this.segments[i].at(time);
        return frames[frames.length - 1].value;
    };

    // Build an Interpolator1D from parallel time and value arrays. `modes` is
    // optional and may hold a null per keyframe to fall back to `mode`.
    ip.fromValues = function(times, values, mode, modes) {
        var frames = [];
        for (var i = 0; i < times.length; i++) {
            var m = (modes !== undefined && modes !== null && modes[i]) ?
                modes[i] : mode;
            frames.push(new ip.Keyframe1D(times[i], values[i], m));
        }
        return new ip.Interpolator1D(frames);
    };

    return ip;
    }(module.interpolation || {}));

    return module;
}(jsplot || {}));
