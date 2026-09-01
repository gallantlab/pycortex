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
        delete params['frame'];   // animation bookkeeping, not a menu path
        delete params['time'];    // written by _capture_view(frame_time=...)

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
    // same rule JSMixer._get_anim_seq uses. Numbers go through the viewer's own
    // _animInterp so that camera.azimuth still takes the short way around.
    vt.interpolate = function(viewer, a, b, t) {
        var subjects = vt.subjects(viewer);
        var subject = subjects.length > 0 ? subjects[0] : null;
        var out = {};

        for (var prop in a) {
            if (prop === 'frame')
                continue;
            var av = a[prop], bv = b[prop];
            var leaf = prop.split('.').pop();

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
        "<div class='pycortex-row pycortex-buttons'>",
        "  <button class='anim-play'>play animation</button>",
        "  <button class='anim-render'>render animation</button>",
        "</div>",
        "<div class='anim-render-form'>",
        "  <div class='pycortex-row'><label>folder</label>",
        "    <input type='text' id='anim-dir' class='anim-dir' placeholder='(root)'></div>",
        "  <div class='pycortex-hint anim-root'></div>",
        "  <div class='pycortex-row'><label>name</label>",
        "    <input type='text' id='anim-name' class='anim-name' value='brainmovie'></div>",
        "  <div class='pycortex-row'><label>size</label>",
        "    <input type='number' id='anim-width' class='anim-width' step='1'>",
        "    <span>&times;</span>",
        "    <input type='number' id='anim-height' class='anim-height' step='1'></div>",
        "  <div class='pycortex-row pycortex-buttons'>",
        "    <button class='anim-render-ok'>OK</button>",
        "    <button class='anim-render-cancel'>cancel</button></div>",
        "</div>",
        "<div class='pycortex-status anim-status'></div>",
    ].join("\n");

    function AnimationPanel(viewer) {
        this.viewer = viewer;
        viewer._anim = {frame: 0, first: 0, last: 30, fps: 30, keyframes: []};
        this.state = viewer._anim;
        this.playing = false;
        this.rendering = false;
        this._applied = undefined;

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

        var cfg = (typeof viewopts !== "undefined") ? viewopts.movie_post : undefined;
        var render = this._el("anim-render");
        if (cfg === undefined) {
            // No python behind this viewer (a static export), so there is
            // nowhere to write frames.
            render.prop("disabled", true)
                  .attr("title", "Rendering needs a viewer started from python");
        } else {
            this._el("anim-root").text("under " + cfg.root);
            render.click(function() { self._el("anim-render-form").toggle(); });
            this._el("anim-render-cancel").click(function() {
                self._el("anim-render-form").hide();
            });
            this._el("anim-render-ok").click(this.render.bind(this));
        }
        this._el("anim-render-form").hide();
        this._el("anim-width").val(this.viewer.imageWidth || 2400);
        this._el("anim-height").val(this.viewer.imageHeight || 1200);
    };

    AnimationPanel.prototype.show = function() {
        this.panel.show();
        this.sync();
    };

    AnimationPanel.prototype.status = function(msg) {
        this._el("anim-status").text(msg === undefined ? "" : msg);
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
        this.drawTicks();
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
                .attr("title", "keyframe at frame " + frame)
                .appendTo(ticks);
        }
    };

    AnimationPanel.prototype.sorted = function() {
        return this.state.keyframes.slice().sort(function(a, b) {
            return a.frame - b.frame;
        });
    };

    // The interpolated view at (possibly fractional) frame `f`.
    AnimationPanel.prototype.viewAt = function(f) {
        var kfs = this.sorted();
        if (kfs.length === 0)
            return null;
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
        view.frame = frame;

        for (var i = 0; i < st.keyframes.length; i++) {
            if (st.keyframes[i].frame === frame) {
                st.keyframes[i] = view;
                this._applied = undefined;
                this.drawTicks();
                this.status("Replaced keyframe at frame " + frame);
                return;
            }
        }
        st.keyframes.push(view);
        this._applied = undefined;
        this.drawTicks();
        this.status("Added keyframe at frame " + frame +
                    " (" + st.keyframes.length + " total)");
    };

    AnimationPanel.prototype.clearKeyframe = function() {
        var st = this.state;
        var frame = Math.round(st.frame);
        for (var i = 0; i < st.keyframes.length; i++) {
            if (st.keyframes[i].frame === frame) {
                st.keyframes.splice(i, 1);
                this._applied = undefined;
                this.drawTicks();
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

    AnimationPanel.prototype.render = function() {
        var st = this.state, self = this;
        var cfg = viewopts.movie_post;

        if (this.rendering) {
            this.status("Already rendering");
            return;
        }
        if (st.keyframes.length === 0) {
            this.status("Add at least one keyframe first");
            return;
        }

        var name = this._el("anim-name").val();
        var dir = this._el("anim-dir").val();
        var width = parseInt(this._el("anim-width").val(), 10);
        var height = parseInt(this._el("anim-height").val(), 10);
        if (!isFinite(width) || !isFinite(height) || width < 1 || height < 1) {
            this.status("Bad image size");
            return;
        }

        this.stop();
        this.rendering = true;
        this._el("anim-render-form").hide();

        var total = st.last - st.first + 1;

        function finish(msg) {
            self.rendering = false;
            self.sync();
            self.status(msg);
        }

        // Strictly one frame at a time: both the webgl readback and the upload
        // are asynchronous, so a plain loop would race.
        function renderFrame(frame) {
            if (!self.rendering) {
                finish("Rendering cancelled");
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
                    finish("Could not render frame " + frame + ": " + e.message);
                    return;
                }
                $.post(cfg.url, {token: cfg.token, dir: dir, name: name,
                                 frame: frame, png: image.toDataURL()})
                 .done(function() {
                     if (frame < st.last)
                         renderFrame(frame + 1);
                     else
                         finish("Rendered " + total + " frames");
                 })
                 .fail(function(xhr) {
                     finish("Frame " + frame + " failed: " +
                            (xhr.responseText || xhr.statusText));
                 });
            });
        }

        renderFrame(st.first);
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
        "<div class='pycortex-hint'>Kept in the browser until python calls",
        "  retrieve_new_views().</div>",
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

        var views_ui = cam_ui.addFolder("views", true);
        var saved = (typeof viewopts !== "undefined" && viewopts.saved_views) ?
                    viewopts.saved_views : {};
        var subjects = Object.keys(saved);

        // jsplot.Menu stores each entry as a property of the folder, so a view
        // named after one of its own methods would break the menu.
        var RESERVED = {get: 1, set: 1, add: 1, addFolder: 1, remove: 1, init: 1};

        function addViewButton(label, view) {
            if (RESERVED[label] !== undefined) {
                console.warn("Skipping view '" + label + "': that name is " +
                             "reserved by the controls menu. Rename the file.");
                return;
            }
            if (label in views_ui._desc)   // never add the same row twice
                return;
            var desc = {};
            desc[label] = {action: function() { vt.applyView(viewer, view); }};
            views_ui.add(desc);
        }

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
                panel.show();
            }},
        });
    };

    vt.AnimationPanel = AnimationPanel;

    return vt;
    }(module.viewtools || {}));

    return module;
}(jsplot || {}));
