// Saved views and keyframe animation for the webgl viewer.
//
// Adds three things to the "camera" folder of the viewer controls:
//   * a "views" sub-folder holding one button per view saved in the subject's
//     filestore views/ directory (shipped to the browser in viewopts.saved_views)
//   * a "save view" button, which captures the current view into viewer._new_views.
//     Those stay in the browser until python asks for them with
//     JSMixer.retrieve_new_views(), so that saving a view here does not touch
//     the filestore.
//   * a "create animation" button, which opens a panel for laying down keyframes,
//     playing them back, and rendering one png per frame.
//
// The view dicts produced here use the same keys as JSMixer._capture_view in
// cortex/webgl/view.py -- including the literal "{subject}" placeholder -- so
// they interchange with views saved from python.

var jsplot = (function (module) {
    module.viewtools = (function (vt) {

    // Lighting used to sit directly in the surface menu. Mirrors
    // JSMixer._legacy_props, so view files written before the move still load.
    var LEGACY_PROPS = {
        'surface.{subject}.specularity':
            'surface.{subject}.lighting.specularity',
        'surface.{subject}.uniform_illumination':
            'surface.{subject}.lighting.uniform_illumination',
    };

    var CURVATURE_PROPS = ['brightness', 'contrast', 'smoothness'];
    var LIGHTING_PROPS = ['topleft_lighting', 'uniform_illumination', 'specularity'];

    // Properties that are numeric but discrete: interpolating them produces
    // values the setter cannot use. `layers` recompiles the shaders on every
    // assignment, so a fractional layer count is both wrong and very slow.
    var STEP_PROPS = {'layers': true};

    // Properties a flattened surface ignores: the controls pin the camera
    // square-on to the flatmap and discard whatever angle they are given
    // (setAzimuth/setAltitude in resources/js/movement.js). The python side
    // names the same pair in cortex/export/save_views.py.
    var FLAT_INERT_PROPS = ['camera.azimuth', 'camera.altitude'];

    var SUBJ = '{subject}';

    function subst(prop, subject) {
        return prop.indexOf(SUBJ) >= 0 ? prop.replace(SUBJ, subject) : prop;
    }

    // ------------------------------------------------------------------
    // Reading the menu tree
    // ------------------------------------------------------------------

    // Subjects present in this viewer, as named in the surface menu.
    vt.subjects = function(viewer) {
        var surface = viewer.ui._desc.surface;
        if (surface === undefined || surface._desc === undefined)
            return [];
        return Object.keys(surface._desc);
    };

    // The settable view properties, discovered from the live menu tree the same
    // way JSMixer.view_props does it.
    //
    // Python walks `_controls`, which also contains the plain buttons (fold,
    // reset, inflate, "pial surface", ...); _capture_view then throws on each of
    // them and swallows the error. Keeping only array-form actions -- the ones
    // backed by a property or a getter/setter -- gives exactly the subset python
    // captures successfully, so the two stay interchangeable.
    vt.viewProps = function(viewer) {
        var props = [], name;

        var camera = viewer.ui._desc.camera;
        if (camera !== undefined && camera._desc !== undefined) {
            for (name in camera._desc)
                if (camera._desc[name].action instanceof Array)
                    props.push('camera.' + name);
        }

        var subjects = vt.subjects(viewer);
        if (subjects.length > 0) {
            var smenu = viewer.ui._desc.surface._desc[subjects[0]];
            for (name in smenu._desc)
                if (smenu._desc[name].action instanceof Array)
                    props.push('surface.' + SUBJ + '.' + name);
            for (var i = 0; i < CURVATURE_PROPS.length; i++)
                props.push('surface.' + SUBJ + '.curvature.' + CURVATURE_PROPS[i]);
            for (var j = 0; j < LIGHTING_PROPS.length; j++)
                props.push('surface.' + SUBJ + '.lighting.' + LIGHTING_PROPS[j]);
        }
        return props;
    };

    // ------------------------------------------------------------------
    // Capturing and applying views
    // ------------------------------------------------------------------

    // The javascript counterpart of JSMixer._capture_view. With several
    // subjects only the first is read, matching python.
    vt.captureView = function(viewer) {
        var view = {};
        var subjects = vt.subjects(viewer);
        var subject = subjects.length > 0 ? subjects[0] : null;
        var props = vt.viewProps(viewer);

        for (var i = 0; i < props.length; i++) {
            var path = subst(props[i], subject);
            try {
                var value = viewer.ui.get(path);
                if (value !== undefined)
                    view[props[i]] = value;
            } catch (e) {
                console.warn("Could not capture " + path + ": " + e.message);
            }
        }

        // A flat pose records no camera angle, since the flat surface ignores
        // it: an animation would otherwise have something spurious to
        // interpolate towards on the way in, spinning the brain as it flattens
        // and leaving the folded angle overwritten on the way out. A tilted
        // flat surface does use the angle, so it keeps it.
        if (vt.isFlat(view) && !view['surface.' + SUBJ + '.allow_tilt'])
            for (var j = 0; j < FLAT_INERT_PROPS.length; j++)
                delete view[FLAT_INERT_PROPS[j]];

        return view;
    };

    function sameValue(a, b) {
        if (a instanceof Array && b instanceof Array) {
            if (a.length != b.length)
                return false;
            for (var i = 0; i < a.length; i++)
                if (a[i] !== b[i])
                    return false;
            return true;
        }
        return a === b;
    }

    // The javascript counterpart of JSMixer._set_view.
    //
    // `previous`, when given, is the view that is currently applied; properties
    // that have not changed are skipped. That matters while animating, because
    // some setters (layers, dither, sampler) rebuild the shaders on every call.
    vt.applyView = function(viewer, view, previous) {
        var prop, key;

        // Copy so the caller's dict is not mutated by the legacy renaming.
        var params = {};
        for (prop in view)
            params[prop] = view[prop];
        for (key in LEGACY_PROPS) {
            if (key in params && !(LEGACY_PROPS[key] in params)) {
                params[LEGACY_PROPS[key]] = params[key];
                delete params[key];
            }
        }
        // A flat view saved before camera.flat_target existed stores its flat
        // target as camera.target -- that is where setTarget put it once the
        // surface was flat -- so read it as one. save_3d_views relies on the
        // same reading for the target it passes along with its flatmap.
        // JSMixer._set_view applies the same rule.
        if (vt.isFlat(params) && ('camera.target' in params) &&
                !('camera.flat_target' in params)) {
            params['camera.flat_target'] = params['camera.target'];
            delete params['camera.target'];
        }
        delete params['frame'];         // animation bookkeeping, not a menu path
        delete params['interpolation']; // ditto: the keyframe's smoothing mode
        delete params['time'];          // written by _capture_view(frame_time=...)

        var subjects = vt.subjects(viewer);
        var unfold = 'surface.' + SUBJ + '.unfold';

        for (var i = 0; i < subjects.length; i++) {
            var subject = subjects[i];
            // Unfolding interacts with the other surface parameters, so it goes
            // first -- the same reason _set_view does it first.
            if (unfold in params &&
                    (previous === undefined || !sameValue(params[unfold], previous[unfold])))
                vt.setProp(viewer, subst(unfold, subject), params[unfold]);

            for (prop in params) {
                if (prop === unfold)
                    continue;
                if (previous !== undefined && prop in previous &&
                        sameValue(params[prop], previous[prop]))
                    continue;
                vt.setProp(viewer, subst(prop, subject), params[prop]);
            }
        }

        // Applying a flat view that says nothing about how far away the
        // camera is frames the flatmap the way quickflat frames the image it
        // writes (see Viewer.fitFlatView). That is how the built-in "flat"
        // view is defined: it names neither an angle, which a flat surface
        // ignores, nor a zoom. A view saved from the GUI, and every frame
        // interpolated between keyframes, carries a camera.radius, so both are
        // left exactly as they were captured.
        //
        // This is the interactive path -- the views menu. JSMixer._set_view
        // deliberately does not do it, so that setting a flat view from python
        // renders what it always rendered; fit_flat_view() asks for it there.
        if (vt.isFlat(params) && !('camera.radius' in params) &&
                viewer.fitFlatView !== undefined)
            viewer.fitFlatView();
    };

    // Whether a view has the surface flattened. The unfold value is what
    // decides it, rather than the name the view was saved under, because a
    // keyframe records the pose and not where it came from.
    vt.isFlat = function(view) {
        var unfold = view['surface.' + SUBJ + '.unfold'];
        return unfold !== undefined && unfold >= 0.999;
    };

    vt.setProp = function(viewer, path, value) {
        try {
            viewer.ui.set(path, value);
        } catch (e) {
            console.warn("Could not set " + path + ": " + e.message);
        }
    };

    // Interpolate between two views. `a` and `b` are view dicts (optionally
    // carrying a "frame" key); `t` runs from 0 at `a` to 1 at `b`.
    //
    // Values that cannot be blended -- null, booleans, strings, discrete
    // numbers, and anything missing from `b` -- hold their starting value, the
    // same rule JSMixer._get_anim_seq uses. A property only `b` carries holds
    // its value instead: it is the one view of the pair that constrains it.
    // Numbers go through the viewer's own _animInterp so that camera.azimuth
    // still takes the short way around.
    vt.interpolate = function(viewer, a, b, t) {
        var subjects = vt.subjects(viewer);
        var subject = subjects.length > 0 ? subjects[0] : null;
        var out = {};
        var props = {};
        for (var key in a) props[key] = true;
        for (key in b) props[key] = true;

        for (var prop in props) {
            if (prop === 'frame')
                continue;
            var av = a[prop], bv = b[prop];
            var leaf = prop.split('.').pop();

            // Only the later view carries it: it is the one keyframe of the
            // pair that constrains the property, so it holds throughout.
            if (av === undefined) {
                out[prop] = bv;
                continue;
            }

            if (bv === undefined || av === null || STEP_PROPS[leaf] ||
                    typeof av === 'boolean' || typeof av === 'string') {
                out[prop] = av;
                continue;
            }

            var state = subst(prop, subject);
            if (av instanceof Array) {
                if (!(bv instanceof Array) || bv.length !== av.length) {
                    out[prop] = av;
                    continue;
                }
                var vals = [];
                for (var i = 0; i < av.length; i++)
                    vals.push(viewer._animInterp(state, av[i], bv[i], t));
                out[prop] = vals;
            } else if (typeof av === 'number' && typeof bv === 'number') {
                out[prop] = viewer._animInterp(state, av, bv, t);
            } else {
                out[prop] = av;
            }
        }
        return out;
    };

    // ------------------------------------------------------------------
    // Smoothed interpolation across a whole keyframe list
    // ------------------------------------------------------------------
    //
    // vt.interpolate above blends one pair of views, which is all linear
    // interpolation ever needs. The smooth modes need more: a tangent at a
    // keyframe depends on the keyframes on *both* sides of it, so the curve
    // cannot be built from a bracketing pair alone.
    //
    // buildInterpolators therefore takes the keyframe list apart into
    // independent one-dimensional "channels" -- one per property, or one per
    // component for the array-valued properties -- and hands each to an
    // Interpolator1D from resources/js/interpolation.js. evaluate() puts a view
    // dict back together from them.

    // camera.azimuth is an angle, so a spline over it needs a sequence that
    // does not jump by 360 at the wrap. Re-express each value as the previous
    // one plus the shortest signed step to it; the caller wraps the result back
    // into [0, 360).
    //
    // The step reproduces Viewer._animInterp exactly, tie-break included: half
    // a turn is equally short either way, and _animInterp resolves it by going
    // *against* the sign of the raw difference (it adds 360 to the end value
    // when travelling backwards and subtracts it when travelling forwards). A
    // symmetric "shortest angle" formula picks the other direction for one of
    // the two cases, which would reverse a 180-degree spin that used to work.
    function shortestStep(from, to) {
        var step = (to - from) % 360;   // truncating remainder: keeps the sign
        if (step >= 180)
            return step - 360;
        if (step <= -180)
            return step + 360;
        return step;
    }

    function unwrapAngles(values) {
        var out = [values[0]];
        for (var i = 1; i < values.length; i++)
            out.push(out[i - 1] + shortestStep(values[i - 1], values[i]));
        return out;
    }

    function wrapAngle(value) {
        return ((value % 360) + 360) % 360;
    }

    // Properties that cannot be blended hold the value of the most recent
    // keyframe at or before the requested frame -- the same rule vt.interpolate
    // applies to the earlier of its pair.
    function stepChannel(frames, prop) {
        return {at: function(f) {
            var value = frames[0][prop];
            for (var i = 0; i < frames.length; i++) {
                if (frames[i].frame > f)
                    break;
                if (frames[i][prop] !== undefined)
                    value = frames[i][prop];
            }
            return value;
        }};
    }

    function modesOf(frames) {
        var modes = [];
        for (var i = 0; i < frames.length; i++)
            modes.push(frames[i].interpolation);
        return modes;
    }

    // One Interpolator1D over `values`, wrapping the result if it is an angle.
    function smoothChannel(frames, values, isAngle) {
        var ip = jsplot.interpolation;
        var times = [], i;
        for (i = 0; i < frames.length; i++)
            times.push(frames[i].frame);

        var interp = ip.fromValues(times, isAngle ? unwrapAngles(values) : values,
                                   ip.DEFAULT_MODE, modesOf(frames));
        return {at: function(f) {
            var value = interp.at(f);
            return isAngle ? wrapAngle(value) : value;
        }};
    }

    function isBlendable(value) {
        return typeof value === 'number' && isFinite(value);
    }

    // Decide how one property should be animated, and build the channel for it.
    function makeChannel(frames, prop) {
        var leaf = prop.split('.').pop();
        var first = frames[0][prop], i, j;

        // Discrete or non-numeric in the first keyframe: nothing to blend.
        if (first === null || first === undefined || STEP_PROPS[leaf] ||
                typeof first === 'boolean' || typeof first === 'string')
            return stepChannel(frames, prop);

        if (first instanceof Array) {
            // Every keyframe must agree on the length, and every element must
            // be a finite number, or the whole property steps.
            for (i = 0; i < frames.length; i++) {
                var value = frames[i][prop];
                if (!(value instanceof Array) || value.length !== first.length)
                    return stepChannel(frames, prop);
                for (j = 0; j < value.length; j++)
                    if (!isBlendable(value[j]))
                        return stepChannel(frames, prop);
            }

            var components = [];
            for (j = 0; j < first.length; j++) {
                var column = [];
                for (i = 0; i < frames.length; i++)
                    column.push(frames[i][prop][j]);
                components.push(smoothChannel(frames, column, false));
            }
            return {at: function(f) {
                var out = [];
                for (var k = 0; k < components.length; k++)
                    out.push(components[k].at(f));
                return out;
            }};
        }

        for (i = 0; i < frames.length; i++)
            if (!isBlendable(frames[i][prop]))
                return stepChannel(frames, prop);

        return smoothChannel(frames, (function() {
            var column = [];
            for (var k = 0; k < frames.length; k++)
                column.push(frames[k][prop]);
            return column;
        }()), prop === 'camera.azimuth');
    }

    // The keyframes that carry `prop`. A keyframe that leaves a property out
    // does not constrain it -- which is how a flat keyframe stays out of the
    // camera.azimuth channel, since a flat pose records no angle.
    function carriersOf(frames, prop) {
        var out = [];
        for (var i = 0; i < frames.length; i++)
            if (frames[i][prop] !== undefined)
                out.push(frames[i]);
        return out;
    }

    // Take a keyframe list apart into per-property channels. Each channel is
    // built from the keyframes that carry its property, so a property every
    // keyframe carries behaves as it always did, one carried by a single
    // keyframe is constant, and the keyframes in between are simply not on the
    // curve.
    vt.buildInterpolators = function(keyframes) {
        var frames = keyframes.slice().sort(function(a, b) {
            return a.frame - b.frame;
        });
        var channels = {};
        for (var i = 0; i < frames.length; i++) {
            for (var prop in frames[i]) {
                if (prop === 'frame' || prop === 'interpolation' ||
                        channels[prop] !== undefined)
                    continue;
                channels[prop] = makeChannel(carriersOf(frames, prop), prop);
            }
        }
        return {frames: frames, channels: channels};
    };

    // The view dict at (possibly fractional) frame `f`.
    vt.evaluate = function(built, f) {
        var view = {};
        for (var prop in built.channels)
            view[prop] = built.channels[prop].at(f);
        return view;
    };

    // A whole animation in one call: the counterpart of JSMixer._get_anim_seq,
    // useful for checking that the two implementations still agree.
    vt.viewsAt = function(keyframes, frames) {
        var built = vt.buildInterpolators(keyframes), views = [];
        for (var i = 0; i < frames.length; i++)
            views.push(vt.evaluate(built, frames[i]));
        return views;
    };

    // ------------------------------------------------------------------
    // Small floating panels
    // ------------------------------------------------------------------

    function makePanel(id, title, bodyHTML, onClose) {
        var panel = $("<div class='pycortex-panel' id='" + id + "'>" +
            "<div class='pycortex-panel-title'>" + title +
            "<span class='pycortex-panel-close'>&times;</span></div>" +
            "<div class='pycortex-panel-body'>" + bodyHTML + "</div>" +
            "</div>");
        panel.find(".pycortex-panel-close").click(function() {
            panel.hide();
            if (onClose !== undefined)
                onClose();
        });
        $("body").append(panel);
        if ($.fn.draggable !== undefined)
            panel.draggable({handle: ".pycortex-panel-title"});
        return panel;
    }

    // ------------------------------------------------------------------
    // The animation panel
    // ------------------------------------------------------------------

    // Every input carries an id as well as a class. The viewer's global keyboard
    // shortcuts (see jsplot.Menu._add in menu.js) only step aside for an INPUT
    // with a non-empty id -- without one, typing "brainmovie" into a field would
    // fold, inflate and re-layer the brain as it went.
    var ANIM_HTML = [
        "<div class='pycortex-row'>",
        "  <label>frame</label>",
        "  <input type='number' id='anim-frame' class='anim-frame' step='1'>",
        "  <label>fps</label>",
        "  <input type='number' id='anim-fps' class='anim-fps' min='1' step='1'>",
        "</div>",
        "<div class='keyframe-track'>",
        "  <input type='range' id='anim-slider' class='anim-slider' step='1'>",
        "  <div class='keyframe-ticks'></div>",
        "</div>",
        "<div class='pycortex-row'>",
        "  <label>first</label>",
        "  <input type='number' id='anim-first' class='anim-first' step='1'>",
        "  <label>last</label>",
        "  <input type='number' id='anim-last' class='anim-last' step='1'>",
        "</div>",
        "<div class='pycortex-row pycortex-buttons'>",
        "  <button class='anim-add'>add keyframe</button>",
        "  <button class='anim-clear'>clear keyframe</button>",
        "</div>",
        "<div class='pycortex-row anim-interp-row'>",
        "  <label>smoothing</label>",
        "  <select id='anim-interp' class='anim-interp'></select>",
        "</div>",
        "<div class='pycortex-row pycortex-buttons'>",
        "  <button class='anim-play'>play animation</button>",
        "  <button class='anim-render'>render animation</button>",
        "</div>",
        "<div class='anim-render-form'>",
        "  <div class='pycortex-row'><label>name</label>",
        "    <input type='text' id='anim-name' class='anim-name' value='brainmovie'></div>",
        "  <div class='pycortex-row'><label>format</label>",
        "    <select id='anim-format' class='anim-format'>",
        "      <option value='png'>PNG frames (.zip)</option>",
        "      <option value='mp4'>MP4 video</option>",
        "    </select></div>",
        "  <div class='pycortex-hint anim-format-hint'></div>",
        "  <div class='pycortex-row'><label>size</label>",
        "    <input type='number' id='anim-width' class='anim-width' step='1'>",
        "    <span>&times;</span>",
        "    <input type='number' id='anim-height' class='anim-height' step='1'></div>",
        "  <div class='pycortex-row anim-flatmatch-row'>",
        "    <label class='anim-flatmatch-label'>",
        "      <input type='checkbox' id='anim-flatmatch' class='anim-flatmatch'>",
        "      match quickflat size</label></div>",
        "  <div class='pycortex-hint anim-flatsize'></div>",
        "  <div class='pycortex-row pycortex-buttons'>",
        "    <button class='anim-render-ok'>OK</button>",
        "    <button class='anim-render-cancel'>cancel</button></div>",
        "</div>",
        "<div class='pycortex-status anim-status'></div>",
    ].join("\n");

    // resources/js/interpolation.js is loaded from template.html, but a user
    // template dir can shadow that template (see FallbackLoader), so an older
    // copy may not pull it in. Everywhere the smoothing needs it we fall back
    // to the previous behaviour -- linear between the bracketing pair -- rather
    // than throwing on every frame.
    function hasInterpolation() {
        return typeof jsplot !== "undefined" &&
               jsplot.interpolation !== undefined;
    }

    function defaultMode() {
        return hasInterpolation() ? jsplot.interpolation.DEFAULT_MODE : "Linear";
    }

    function modeLabel(mode) {
        // A keyframe built by hand may carry no mode; the interpolator treats
        // that as the default, so label it that way too.
        if (!mode)
            mode = defaultMode();
        if (!hasInterpolation())
            return mode;
        return jsplot.interpolation.MODE_LABELS[mode] || mode;
    }

    function AnimationPanel(viewer) {
        this.viewer = viewer;
        // `mode` is the smoothing applied to keyframes added from here on; each
        // keyframe carries its own copy in an "interpolation" key.
        viewer._anim = {frame: 0, first: 0, last: 30, fps: 30, keyframes: [],
                        mode: defaultMode()};
        this.state = viewer._anim;
        this.playing = false;
        this.rendering = false;
        this._applied = undefined;
        // Channel interpolators, rebuilt lazily whenever the keyframes change.
        this._interp = null;

        this.panel = makePanel("animpanel", "Animation", ANIM_HTML,
                               this.close.bind(this));
        this._bind();
        this.sync();
    }

    AnimationPanel.prototype._el = function(cls) {
        return this.panel.find("." + cls);
    };

    AnimationPanel.prototype._bind = function() {
        var self = this, st = this.state;

        this._el("anim-frame").on("change", function() {
            self.setFrame(parseFloat(this.value));
            self.sync();
        });
        this._el("anim-slider").on("input change", function() {
            self.setFrame(parseFloat(this.value));
            self._el("anim-frame").val(Math.round(st.frame));
        });
        this._el("anim-fps").on("change", function() {
            var v = parseInt(this.value, 10);
            st.fps = (isFinite(v) && v > 0) ? v : st.fps;
            self.sync();
        });
        this._el("anim-first").on("change", function() {
            var v = parseInt(this.value, 10);
            if (isFinite(v)) {
                st.first = v;
                if (st.last <= st.first)
                    st.last = st.first + 1;
            }
            self.sync();
        });
        this._el("anim-last").on("change", function() {
            var v = parseInt(this.value, 10);
            if (isFinite(v)) {
                st.last = v;
                if (st.last <= st.first)
                    st.first = st.last - 1;
            }
            self.sync();
        });

        this._el("anim-add").click(this.addKeyframe.bind(this));
        this._el("anim-clear").click(this.clearKeyframe.bind(this));
        this._el("anim-play").click(this.playPause.bind(this));

        var select = this._el("anim-interp");
        if (hasInterpolation()) {
            var order = jsplot.interpolation.MODE_ORDER;
            for (var i = 0; i < order.length; i++)
                $("<option></option>").attr("value", order[i])
                                      .text(modeLabel(order[i]))
                                      .appendTo(select);
            select.val(st.mode);
            select.on("change", function() { self.setMode(this.value); });
        } else {
            select.prop("disabled", true)
                  .attr("title", "resources/js/interpolation.js is not loaded");
        }
        // An id is enough to keep the viewer's single-letter shortcuts off an
        // INPUT, but the guard in jsplot.Menu._add tests for INPUT only, and a
        // focused SELECT still gets keypress events as the user types ahead.
        // Without this, typing "b" to reach "bezier" would fold the brain.
        select.on("keypress keydown keyup", function(event) {
            event.stopPropagation();
        });

        // Rendering builds the movie in the page and downloads it, so it needs
        // no server: it works the same in a static export.
        this._el("anim-render").click(function() {
            self._el("anim-render-form").toggle();
        });
        this._el("anim-render-cancel").click(function() {
            self._el("anim-render-form").hide();
        });
        this._el("anim-render-ok").click(this.render.bind(this));

        var format = this._el("anim-format");
        if (!vt.canEncodeVideo())
            format.find("option[value=mp4]").prop("disabled", true).attr(
                "title", "Needs a browser that can encode video (WebCodecs), " +
                         "on localhost, https or a local file");
        format.on("change", function() { self.updateFormatHint(); });
        // Same reason as the smoothing select above: keep typing in it from
        // reaching the viewer's keyboard shortcuts.
        format.on("keypress keydown keyup", function(event) {
            event.stopPropagation();
        });
        this.updateFormatHint();

        this._el("anim-render-form").hide();
        this._el("anim-width").val(this.viewer.imageWidth || 2400);
        this._el("anim-height").val(this.viewer.imageHeight || 1200);
        this._el("anim-flatmatch").prop("checked", false).on("change", function() {
            self.matchFlatChanged();
        });
        this._el("anim-flatmatch-row").hide();
    };

    AnimationPanel.prototype.show = function() {
        this.panel.show();
        this.sync();
    };

    AnimationPanel.prototype.status = function(msg) {
        this._el("anim-status").text(msg === undefined ? "" : msg);
    };

    // What the status line says -- how a script (or a test) learns why a
    // render stopped.
    AnimationPanel.prototype._statusText = function() {
        return this._el("anim-status").text();
    };

    // Push the internal state out to every widget, and redraw the keyframe dots.
    AnimationPanel.prototype.sync = function() {
        var st = this.state;
        if (st.frame < st.first) st.frame = st.first;
        if (st.frame > st.last) st.frame = st.last;

        this._el("anim-first").val(st.first);
        this._el("anim-last").val(st.last);
        this._el("anim-fps").val(st.fps);
        this._el("anim-frame").val(Math.round(st.frame)).attr({min: st.first, max: st.last});
        this._el("anim-slider").attr({min: st.first, max: st.last}).val(st.frame);

        // The dropdown shows the mode that "add keyframe" would apply here:
        // the existing keyframe's own mode when there is one, otherwise the
        // mode chosen for new keyframes.
        if (hasInterpolation()) {
            var here = this.keyframeAt(Math.round(st.frame));
            this._el("anim-interp").val(here ? here.interpolation : st.mode);
        }
        this.drawTicks();
        this.updateFlatHint();
    };

    // The keyframe laid down at exactly `frame`, or null.
    AnimationPanel.prototype.keyframeAt = function(frame) {
        var kfs = this.state.keyframes;
        for (var i = 0; i < kfs.length; i++)
            if (kfs[i].frame === frame)
                return kfs[i];
        return null;
    };

    // The keyframes changed, so the channel interpolators and the record of
    // what is currently applied are both out of date.
    AnimationPanel.prototype.invalidate = function() {
        this._interp = null;
        this._applied = undefined;
        this.drawTicks();
        this.updateFlatHint();
    };

    // Whether the animation passes through the flattened surface at any
    // keyframe. Tested on the unfold value rather than on a view name, because
    // a keyframe records the pose, not the view it was posed from.
    AnimationPanel.prototype.usesFlat = function() {
        var kfs = this.state.keyframes;
        for (var i = 0; i < kfs.length; i++)
            if (vt.isFlat(kfs[i]))
                return true;
        return false;
    };

    // The size quickflat.make_png would write for the subject on show, shipped
    // from python in viewopts.quickflat_size because it follows from the flat
    // surface rather than from anything the browser knows. Null if the subject
    // has no flatmap.
    AnimationPanel.prototype.flatSize = function() {
        var sizes = (typeof viewopts !== "undefined") ?
            viewopts.quickflat_size : undefined;
        var subjects = vt.subjects(this.viewer);
        if (sizes === undefined || subjects.length == 0)
            return null;
        var size = sizes[subjects[0]];
        return (size === undefined || size === null) ? null : size;
    };

    // Whether the render form's "match quickflat size" box is ticked. Ticking
    // it renders an animation that visits the flat surface at the size
    // quickflat uses, so its flat frames come out as the png
    // quickflat.make_png writes: a flat keyframe is framed to fill the frame
    // (Viewer.flatFraming), which reproduces make_png only at make_png's own
    // aspect ratio. It starts unticked, so nothing about an animation changes
    // unless it is asked for.
    AnimationPanel.prototype.matchesFlat = function() {
        return this._el("anim-flatmatch").prop("checked") === true;
    };

    // Put quickflat's size in the size fields, if the box is ticked and the
    // fields still hold a size the panel is free to overwrite -- the one it
    // wrote last time, or the untouched default it starts out with. Typing a
    // size is how a person says they want a different one, and flat keyframes
    // are then framed for that size instead. Returns whether the fields ended
    // up holding quickflat's size.
    AnimationPanel.prototype.useFlatSize = function() {
        var size = this.flatSize();
        if (size === null || !this.matchesFlat())
            return false;

        var width = this._el("anim-width"), height = this._el("anim-height");
        var have = [parseInt(width.val(), 10), parseInt(height.val(), 10)];
        if (have[0] === size[0] && have[1] === size[1])
            return true;

        var mine = this._flatsized ||
                   [this.viewer.imageWidth || 2400, this.viewer.imageHeight || 1200];
        if (have[0] !== mine[0] || have[1] !== mine[1])
            return false;

        width.val(size[0]);
        height.val(size[1]);
        this._flatsized = [size[0], size[1]];
        return true;
    };

    // Frame every flat keyframe for the size the animation renders at.
    //
    // A flat keyframe is framed when it is laid down, so one laid down before
    // the box was ticked -- or before the size was changed -- carries a framing
    // for the wrong frame. Written straight into the keyframes rather than by
    // posing the viewer, since these are keyframes the playhead is not on.
    AnimationPanel.prototype.reframeFlatKeyframes = function() {
        var size = this.renderSize();
        if (size === null || this.viewer.flatFraming === undefined)
            return 0;

        var framing = this.viewer.flatFraming(size[0] / size[1]);
        if (framing === null)
            return 0;

        var kfs = this.state.keyframes, reframed = 0;
        for (var i = 0; i < kfs.length; i++) {
            if (!vt.isFlat(kfs[i]))
                continue;
            kfs[i]['camera.target'] = framing.target.slice();
            kfs[i]['camera.radius'] = framing.radius;
            reframed++;
        }
        if (reframed > 0) {
            this.invalidate();
            this.setFrame(this.state.frame);
        }
        return reframed;
    };

    // Tick or untick the box from code, as the menu click would.
    AnimationPanel.prototype.setMatchFlat = function(on) {
        this._el("anim-flatmatch").prop("checked", on === true);
        this.matchFlatChanged();
        return this.matchesFlat();
    };

    // The box was ticked or unticked. Ticking takes over the size and re-frames
    // the flat keyframes for it; unticking leaves both alone, since the size in
    // the fields is the one the keyframes are now framed for.
    AnimationPanel.prototype.matchFlatChanged = function() {
        if (this.matchesFlat()) {
            this.useFlatSize();
            this.reframeFlatKeyframes();
        }
        this.updateFlatHint();
    };

    AnimationPanel.prototype.updateFlatHint = function() {
        var hint = this._el("anim-flatsize");
        var size = this.flatSize();

        if (!size || !this.usesFlat()) {
            this._el("anim-flatmatch-row").hide();
            hint.text("");
            return;
        }

        this._el("anim-flatmatch-row").show();
        hint.text((this.useFlatSize() ? "size set to " : "use ") +
                  size[0] + " \u00d7 " + size[1] +
                  " to match quickflat.make_png()");
    };

    // Set the smoothing mode: on the keyframe under the playhead if there is
    // one, and always as the mode new keyframes will be created with.
    AnimationPanel.prototype.setMode = function(mode) {
        if (!hasInterpolation() || !jsplot.interpolation.isValidMode(mode))
            return;
        var st = this.state;
        st.mode = mode;

        var here = this.keyframeAt(Math.round(st.frame));
        if (here !== null) {
            here.interpolation = mode;
            this.invalidate();
            this.setFrame(st.frame);
            this.status("Keyframe " + here.frame + ": " + modeLabel(mode));
        } else {
            this.status("New keyframes will use " + modeLabel(mode));
        }
    };

    // One yellow dot per keyframe, positioned along the slider. Redrawn from
    // scratch so that changing first/last simply repositions everything.
    AnimationPanel.prototype.drawTicks = function() {
        var st = this.state;
        var ticks = this._el("keyframe-ticks").empty();
        var span = st.last - st.first;
        if (span <= 0)
            return;
        for (var i = 0; i < st.keyframes.length; i++) {
            var frame = st.keyframes[i].frame;
            if (frame < st.first || frame > st.last)
                continue;
            var pct = 100 * (frame - st.first) / span;
            $("<div class='keyframe-dot'></div>")
                .css("left", pct + "%")
                .attr("title", "keyframe at frame " + frame + " (" +
                               modeLabel(st.keyframes[i].interpolation) + ")")
                .appendTo(ticks);
        }
    };

    AnimationPanel.prototype.sorted = function() {
        return this.state.keyframes.slice().sort(function(a, b) {
            return a.frame - b.frame;
        });
    };

    // The interpolated view at (possibly fractional) frame `f`.
    //
    // Every property is carried by its own interpolator spanning the whole
    // keyframe list, because the tangent at a keyframe depends on the ones on
    // either side of it -- a bracketing pair is not enough to smooth. The
    // channels are rebuilt only when the keyframes change; see invalidate().
    AnimationPanel.prototype.viewAt = function(f) {
        var kfs = this.state.keyframes;
        if (kfs.length === 0)
            return null;

        if (!hasInterpolation())
            return this._viewAtLinear(f);

        if (this._interp === null)
            this._interp = vt.buildInterpolators(kfs);
        return vt.evaluate(this._interp, f);
    };

    // The pre-smoothing path, kept for templates that do not load
    // resources/js/interpolation.js.
    AnimationPanel.prototype._viewAtLinear = function(f) {
        var kfs = this.sorted();
        if (f <= kfs[0].frame)
            return kfs[0];
        if (f >= kfs[kfs.length - 1].frame)
            return kfs[kfs.length - 1];
        for (var i = 0; i < kfs.length - 1; i++) {
            if (f >= kfs[i].frame && f <= kfs[i + 1].frame) {
                var span = kfs[i + 1].frame - kfs[i].frame;
                var t = span === 0 ? 0 : (f - kfs[i].frame) / span;
                return vt.interpolate(this.viewer, kfs[i], kfs[i + 1], t);
            }
        }
        return kfs[kfs.length - 1];
    };

    // Scrubbing, playback and rendering all move the viewer through here, so
    // they cannot drift apart.
    AnimationPanel.prototype.setFrame = function(f) {
        var st = this.state;
        if (!isFinite(f))
            return;
        st.frame = Math.min(Math.max(f, st.first), st.last);
        var view = this.viewAt(st.frame);
        if (view !== null) {
            vt.applyView(this.viewer, view, this._applied);
            this._applied = view;
        }
    };

    AnimationPanel.prototype.addKeyframe = function() {
        var st = this.state;
        var frame = Math.round(st.frame);
        var view = vt.captureView(this.viewer);

        // With "match quickflat size" ticked, a flat keyframe is framed for
        // the size this animation will render at -- quickflat's own, unless
        // someone has typed another one. Framed here rather than when the flat
        // view was applied because only the panel knows that size, and the
        // framing only reproduces quickflat.make_png at the aspect ratio it is
        // rendered at.
        if (vt.isFlat(view) && this.matchesFlat() &&
                this.viewer.fitFlatView !== undefined) {
            this.useFlatSize();
            var size = this.renderSize();
            if (size !== null) {
                this.viewer.fitFlatView(size[0] / size[1]);
                view = vt.captureView(this.viewer);
            }
        }
        view.frame = frame;
        // The dropdown is the source of truth: sync() has already pointed it at
        // the mode of any keyframe sitting here, so re-adding over one keeps
        // that keyframe's smoothing instead of silently resetting it.
        var chosen = this._el("anim-interp").val();
        view.interpolation = (hasInterpolation() &&
                              jsplot.interpolation.isValidMode(chosen)) ?
            chosen : st.mode;

        for (var i = 0; i < st.keyframes.length; i++) {
            if (st.keyframes[i].frame === frame) {
                st.keyframes[i] = view;
                this.invalidate();
                this.status("Replaced keyframe at frame " + frame +
                            " (" + modeLabel(view.interpolation) + ")");
                return;
            }
        }
        st.keyframes.push(view);
        this.invalidate();
        this.status("Added keyframe at frame " + frame + " (" +
                    modeLabel(view.interpolation) + ", " +
                    st.keyframes.length + " total)");
    };

    AnimationPanel.prototype.clearKeyframe = function() {
        var st = this.state;
        var frame = Math.round(st.frame);
        for (var i = 0; i < st.keyframes.length; i++) {
            if (st.keyframes[i].frame === frame) {
                st.keyframes.splice(i, 1);
                this.invalidate();
                this.status("Removed keyframe at frame " + frame);
                return;
            }
        }
        this.status("No keyframe at frame " + frame);
    };

    AnimationPanel.prototype.stop = function() {
        this.playing = false;
        this._el("anim-play").text("play animation");
    };

    // Closing the panel abandons whatever it was in the middle of.
    AnimationPanel.prototype.close = function() {
        this.stop();
        this.rendering = false;
    };

    AnimationPanel.prototype.playPause = function() {
        if (this.playing) {
            this.stop();
            this.status("Stopped");
            return;
        }
        var st = this.state;
        if (st.keyframes.length < 2) {
            this.status("Need at least two keyframes to play");
            return;
        }

        // Driven by our own clock rather than viewer.animate(): animate() works
        // in seconds, only builds segments for properties that change, and would
        // linearly interpolate the boolean and string properties.
        var self = this;
        var from = st.frame >= st.last ? st.first : st.frame;
        var start = new Date();
        this.playing = true;
        this._el("anim-play").text("stop");
        this.status("Playing");

        function step() {
            if (!self.playing)
                return;
            var frame = from + ((new Date()) - start) / 1000 * st.fps;
            if (frame >= st.last) {
                self.setFrame(st.last);
                self.sync();
                self.stop();
                self.status("Done");
                return;
            }
            self.setFrame(frame);
            self._el("anim-slider").val(frame);
            self._el("anim-frame").val(Math.round(frame));
            requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    };

    // The size frames are rendered at, as [width, height], or null if what is
    // in the size fields is not a size.
    AnimationPanel.prototype.renderSize = function() {
        var width = parseInt(this._el("anim-width").val(), 10);
        var height = parseInt(this._el("anim-height").val(), 10);
        if (!isFinite(width) || !isFinite(height) || width < 1 || height < 1)
            return null;
        return [width, height];
    };

    // ------------------------------------------------------------------
    // Rendering to a download
    // ------------------------------------------------------------------
    //
    // A render is packaged in the page and handed to the browser as one
    // download, the way the viewer's "Save image" button works: the frames land
    // on the machine running the browser, and the server writes nothing. One
    // download per frame is not an option -- browsers throttle a page that
    // starts hundreds of them, or ask whether to allow it -- so the frames go
    // into a single file, either a .zip of PNGs or an .mp4.

    // Whether this page can encode video. That takes WebCodecs, which browsers
    // only offer in a secure context: localhost, https or a local file, but not
    // plain http from another machine.
    vt.canEncodeVideo = function() {
        return typeof window.VideoEncoder !== "undefined" &&
               window.isSecureContext === true;
    };

    // Hand `blob` to the browser as a download called `filename`.
    vt.download = function(blob, filename) {
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        // The click only starts the download. Releasing the blob straight away
        // can cut it short in some browsers, so hold on to it for a while.
        setTimeout(function() { URL.revokeObjectURL(url); }, 60000);
    };

    // A movie name safe to use as a file name and as the folder inside a zip:
    // nothing that could climb out of that folder when it is unpacked.
    function movieName(name) {
        var safe = String(name || "").trim().replace(/[^A-Za-z0-9 _.-]/g, "_")
                                             .replace(/^[.\s]+/, "");
        return safe.length > 0 ? safe : "brainmovie";
    }

    function pad5(n) {
        var s = String(n);
        while (s.length < 5)
            s = "0" + s;
        return s;
    }

    function formatBytes(n) {
        if (n < 1024 * 1024)
            return (n / 1024).toFixed(0) + " KB";
        return (n / (1024 * 1024)).toFixed(1) + " MB";
    }

    function canvasToBlob(canvas, type) {
        return new Promise(function(resolve, reject) {
            canvas.toBlob(function(blob) {
                if (blob)
                    resolve(blob);
                else
                    reject(new Error("The browser could not encode the frame"));
            }, type);
        });
    }

    // One PNG per frame, in a zip. Lossless and transparent outside the brain,
    // so these are the frames to use when they have to match a flatmap from
    // quickflat.make_png. Entries are named after the animation's own frame
    // numbers, inside a folder named after the movie.
    function PngZipWriter(name) {
        this.name = name;
        this.zip = new jsplot.zipstore.ZipWriter();
    }
    PngZipWriter.prototype.extension = "zip";
    PngZipWriter.prototype.start = function() {
        return Promise.resolve();
    };
    PngZipWriter.prototype.addFrame = function(canvas, frame) {
        var zip = this.zip;
        var entry = this.name + "/" + this.name + "_" + pad5(frame) + ".png";
        return canvasToBlob(canvas, "image/png").then(function(blob) {
            return zip.add(entry, blob);
        });
    };
    PngZipWriter.prototype.finish = function() {
        return Promise.resolve(this.zip.finish());
    };
    PngZipWriter.prototype.abort = function() {};

    // An H.264 video, encoded by the browser (WebCodecs) and packed by
    // jsplot.mp4mux. Lossy, and opaque: H.264 has no alpha channel, so frames
    // are laid on black, which is how the viewer shows them on screen. H.264
    // also stores colour at half resolution and so needs even dimensions; an
    // odd size gets one extra column or row of black rather than being
    // rescaled.
    function Mp4Writer(width, height, fps) {
        this.width = width + (width % 2);
        this.height = height + (height % 2);
        this.padded = this.width !== width || this.height !== height;
        this.fps = fps;
        this.keyEvery = Math.max(1, Math.round(2 * fps));   // a keyframe every 2 s
        this.canvas = document.createElement("canvas");
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.ctx = this.canvas.getContext("2d");
        this.muxer = null;
        this.encoder = null;
        this.error = null;
    }
    Mp4Writer.prototype.extension = "mp4";

    // Settle on an encoder configuration before the first frame is rendered,
    // so a size the browser cannot encode fails straight away.
    Mp4Writer.prototype.start = function() {
        var self = this;
        return jsplot.mp4mux.encoderConfig(this.width, this.height, this.fps)
            .then(function(config) {
                self.muxer = new jsplot.mp4mux.Mp4Muxer(self.width, self.height,
                                                        self.fps);
                self.encoder = new VideoEncoder({
                    output: function(chunk, metadata) {
                        try {
                            self.muxer.addChunk(chunk, metadata);
                        } catch (e) {
                            self.error = e;
                        }
                    },
                    error: function(e) { self.error = e; },
                });
                self.encoder.configure(config);
            });
    };

    // `index` counts frames from the start of the render, so the video's
    // clock starts at zero whatever the animation's first frame is.
    Mp4Writer.prototype.addFrame = function(canvas, frame, index) {
        if (this.error)
            return Promise.reject(this.error);

        this.ctx.fillStyle = "#000";
        this.ctx.fillRect(0, 0, this.width, this.height);
        this.ctx.drawImage(canvas, 0, 0);

        var video = new VideoFrame(this.canvas, {
            timestamp: Math.round(index * 1e6 / this.fps),
            duration: Math.round(1e6 / this.fps),
        });
        this.encoder.encode(video, {keyFrame: index % this.keyEvery === 0});
        video.close();

        // Rendering is faster than encoding at these sizes; wait for the
        // encoder to catch up rather than queueing the whole movie as raw
        // frames.
        var encoder = this.encoder, self = this;
        return new Promise(function(resolve, reject) {
            (function wait() {
                if (self.error)
                    reject(self.error);
                else if (encoder.encodeQueueSize <= 2)
                    resolve();
                else
                    setTimeout(wait, 5);
            }());
        });
    };
    Mp4Writer.prototype.finish = function() {
        var self = this;
        return this.encoder.flush().then(function() {
            self.encoder.close();
            if (self.error)
                throw self.error;
            return self.muxer.finish();
        });
    };
    Mp4Writer.prototype.abort = function() {
        if (this.encoder !== null && this.encoder.state !== "closed")
            this.encoder.close();
    };

    // Choose the render format ("png" or "mp4") from code, as the select
    // would. Returns false for a format this browser cannot produce.
    AnimationPanel.prototype.setRenderFormat = function(format) {
        var select = this._el("anim-format");
        var option = select.find("option[value='" + format + "']");
        if (option.length === 0 || option.prop("disabled"))
            return false;
        select.val(format);
        this.updateFormatHint();
        return true;
    };

    // Set the render size from code, as typing it would.
    AnimationPanel.prototype.setRenderSize = function(width, height) {
        this._el("anim-width").val(width);
        this._el("anim-height").val(height);
        return this.renderSize();
    };

    AnimationPanel.prototype.updateFormatHint = function() {
        var hint = this._el("anim-format-hint");
        if (this._el("anim-format").val() === "mp4")
            hint.text("H.264: lossy, and without transparency. Use PNG " +
                      "frames to match quickflat.make_png() exactly.");
        else if (!vt.canEncodeVideo())
            hint.text("MP4 needs a browser that can encode video, on " +
                      "localhost, https or a local file.");
        else
            hint.text("");
    };

    AnimationPanel.prototype.render = function() {
        var st = this.state, self = this;

        if (this.rendering) {
            this.status("Already rendering");
            return;
        }
        if (st.keyframes.length === 0) {
            this.status("Add at least one keyframe first");
            return;
        }

        var name = movieName(this._el("anim-name").val());
        var format = this._el("anim-format").val();
        var size = this.renderSize();
        if (size === null) {
            this.status("Bad image size");
            return;
        }
        if (format === "mp4" && !vt.canEncodeVideo()) {
            this.status("This browser cannot encode MP4 here; render PNG frames");
            return;
        }
        var width = size[0], height = size[1];
        var writer = format === "mp4" ? new Mp4Writer(width, height, st.fps) :
                                        new PngZipWriter(name);

        this.stop();
        this.rendering = true;
        this._el("anim-render-form").hide();

        var total = st.last - st.first + 1;

        function finish(msg) {
            self.rendering = false;
            self.sync();
            self.status(msg);
        }

        function fail(msg) {
            writer.abort();
            finish(msg);
        }

        // Strictly one frame at a time: the webgl readback, the PNG or video
        // encoding and the packing are all asynchronous, so a plain loop would
        // race.
        function renderFrame(frame) {
            if (!self.rendering) {
                fail("Rendering cancelled");
                return;
            }
            self.setFrame(frame);
            self._el("anim-slider").val(frame);
            self._el("anim-frame").val(frame);
            self.status("Rendering frame " + (frame - st.first + 1) + " of " + total);

            // Let the viewer redraw before grabbing the framebuffer.
            requestAnimationFrame(function() {
                var image;
                try {
                    image = self.viewer.getImage(width, height);
                } catch (e) {
                    fail("Could not render frame " + frame + ": " + e.message);
                    return;
                }
                writer.addFrame(image, frame, frame - st.first).then(function() {
                    if (frame < st.last)
                        renderFrame(frame + 1);
                    else
                        deliver();
                }, function(e) {
                    fail("Frame " + frame + " failed: " + e.message);
                });
            });
        }

        function deliver() {
            self.status("Packing " + total + " frames");
            writer.finish().then(function(blob) {
                var filename = name + "." + writer.extension;
                vt.download(blob, filename);
                finish("Saved " + filename + " (" + total + " frames, " +
                       formatBytes(blob.size) +
                       (writer.padded ? ", padded to " + writer.width + " × " +
                                        writer.height : "") + ")");
            }, function(e) {
                fail("Could not finish " + name + "." + writer.extension +
                     ": " + e.message);
            });
        }

        writer.start().then(function() {
            renderFrame(st.first);
        }, function(e) {
            fail(e.message);
        });
    };

    // ------------------------------------------------------------------
    // The "save view" prompt
    // ------------------------------------------------------------------

    var SAVE_HTML = [
        "<div class='pycortex-row'><label>name</label>",
        "  <input type='text' id='viewsave-name' class='viewsave-name'></div>",
        "<div class='pycortex-row pycortex-buttons'>",
        "  <button class='viewsave-ok'>OK</button>",
        "  <button class='viewsave-cancel'>cancel</button></div>",
        "<div class='pycortex-hint'>Kept in the browser until python stores it",
        "  with save_new_views().</div>",
        "<div class='pycortex-status viewsave-status'></div>",
    ].join("\n");

    function SavePrompt(viewer, onSaved) {
        var self = this;
        this.viewer = viewer;
        this.onSaved = onSaved;
        this.panel = makePanel("viewsave", "Save view", SAVE_HTML);
        this.panel.find(".viewsave-cancel").click(function() { self.panel.hide(); });
        this.panel.find(".viewsave-ok").click(function() { self.save(); });
        this.panel.find(".viewsave-name").on("keypress", function(evt) {
            if (evt.which == 13)
                self.save();
        });
    }

    SavePrompt.prototype.show = function() {
        this.panel.find(".viewsave-status").text("");
        this.panel.show();
        this.panel.find(".viewsave-name").val("").focus();
    };

    SavePrompt.prototype.save = function() {
        var name = $.trim(this.panel.find(".viewsave-name").val());
        if (name.length === 0) {
            this.panel.find(".viewsave-status").text("Please give the view a name");
            return;
        }
        this.viewer.saveNewView(name);
        this.panel.hide();
        if (this.onSaved !== undefined)
            this.onSaved(name);
    };

    // ------------------------------------------------------------------
    // Wiring it into the camera menu
    // ------------------------------------------------------------------

    vt.installCameraUI = function(viewer, cam_ui) {
        // Views saved through the GUI. Deliberately separate from the views
        // loaded out of the filestore: python pulls these out with
        // JSMixer.retrieve_new_views() and decides whether to keep them.
        viewer._new_views = {};

        viewer.getNewViews = function() {
            return this._new_views;
        };

        // Capture the current view under `name`. Called by the save prompt, and
        // usable directly from python or a test.
        viewer.saveNewView = function(name) {
            this._new_views[name] = vt.captureView(this);
            return this._new_views[name];
        };

        viewer.applyView = function(view) {
            vt.applyView(this, view);
        };

        // Built up front, rather than letting addFolder make it, so that the
        // init override below is in place before dat.GUI can materialize the
        // folder.
        var views_ui = new module.Menu();
        var _views_init = views_ui.init;
        views_ui.init = function(gui) {
            _views_init.call(this, gui);
            // dat.GUI wraps a folder's element in <li class="folder">; tag that
            // so the stylesheet can widen and centre the view names without
            // touching the rest of the controls.
            if (gui && gui.domElement && gui.domElement.parentNode)
                $(gui.domElement.parentNode).addClass("pycortex-views");
        };
        cam_ui.addFolder("views", true, views_ui);

        var saved = (typeof viewopts !== "undefined" && viewopts.saved_views) ?
                    viewopts.saved_views : {};
        var subjects = Object.keys(saved);

        // jsplot.Menu stores each entry as a property of the folder, so a view
        // named after one of its own methods would break the menu.
        var RESERVED = {get: 1, set: 1, add: 1, addFolder: 1, remove: 1, init: 1};

        // What each button applies, looked up when it is clicked rather than
        // captured, so that re-adding a name replaces the view behind an
        // existing button. That happens when a view is saved over one of the
        // defaults every subject gets: without this the button would go on
        // applying the default until the page was reloaded.
        var view_registry = {};

        function addViewButton(label, view) {
            if (RESERVED[label] !== undefined) {
                console.warn("Skipping view '" + label + "': that name is " +
                             "reserved by the controls menu. Rename the file.");
                return;
            }
            view_registry[label] = view;
            if (label in views_ui._desc)   // never add the same row twice
                return;
            var desc = {};
            desc[label] = {action: function() {
                vt.applyView(viewer, view_registry[label]);
            }};
            views_ui.add(desc);
        }

        // Called by JSMixer.save_new_views once a view is on disk: it is no
        // longer "new", so it moves out of _new_views and joins the views
        // folder. The button's closure holds the view object itself, so it
        // keeps working after the entry is deleted.
        viewer.promoteNewView = function(name) {
            if (!(name in this._new_views))
                return false;
            addViewButton(name, this._new_views[name]);
            delete this._new_views[name];
            return true;
        };

        for (var i = 0; i < subjects.length; i++) {
            var subject = subjects[i];
            var names = Object.keys(saved[subject]).sort();
            for (var j = 0; j < names.length; j++) {
                // Disambiguate only when it could actually be ambiguous.
                var label = subjects.length > 1 ?
                            subject + ": " + names[j] : names[j];
                addViewButton(label, saved[subject][names[j]]);
            }
        }

        var prompt = null;
        var panel = null;

        cam_ui.add({
            "save view": {action: function() {
                if (prompt === null)
                    prompt = new SavePrompt(viewer, function(name) {
                        addViewButton(name, viewer._new_views[name]);
                    });
                prompt.show();
            }},
            "create animation": {action: function() {
                if (panel === null)
                    panel = new AnimationPanel(viewer);
                // Reachable from python (and from the console) the way the
                // keyframe state itself is, as viewer._anim.
                viewer._animPanel = panel;
                panel.show();
            }},
        });
    };

    vt.AnimationPanel = AnimationPanel;

    return vt;
    }(module.viewtools || {}));

    return module;
}(jsplot || {}));
