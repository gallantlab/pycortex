var mriview = (function(module) {

    //Fetch a binary payload (one of the tractogram's buffers) as an
    //ArrayBuffer. Same XMLHttpRequest style as surfload.js / CTMLoader.js.
    function loadBuffer(url, callback, errback) {
        var xhr = new XMLHttpRequest();
        xhr.onreadystatechange = function() {
            if (xhr.readyState == 4) {
                if (xhr.status == 200 || xhr.status == 206 || xhr.status == 0) {
                    callback(xhr.response);
                } else {
                    console.error("mriview.Tractogram: couldn't load " + url +
                                  " (" + xhr.status + ")");
                    if (errback !== undefined)
                        errback(xhr.status);
                }
            }
        };
        xhr.open("GET", url, true);
        xhr.responseType = "arraybuffer";
        xhr.send(null);
    }

    //Three.js r69 only enables 32-bit element indices when the
    //OES_element_index_uint extension is present (see the THREE.Line branch of
    //renderBufferDirect, three.js:20451, which picks UNSIGNED_INT purely from
    //the array type -- an unsupported extension means a GL error rather than a
    //fallback). Without it we duplicate the vertices instead, which also avoids
    //needing r69's `geometry.offsets` drawcall chunking for >65535 vertices.
    module.supportsUint32Index = function(renderer) {
        var gl = null;
        if (renderer !== undefined && renderer !== null)
            gl = renderer.context;
        else if (window.viewer !== undefined && window.viewer.renderer !== undefined)
            gl = window.viewer.renderer.context;

        if (gl === null || gl === undefined)
            return false;
        try {
            return !!gl.getExtension("OES_element_index_uint");
        } catch (e) {
            return false;
        }
    };

    //A bundle of streamlines, rendered as GL_LINES.
    //
    //`meta` is one entry of the `tracts` dict of the metadata package built by
    //cortex/webgl/data.py: {subject, n_points, n_streamlines, alpha, linewidth,
    //visible, color, groups: {name: [start, stop]}, description,
    //urls:{points, offsets, colors, groups}}.
    module.Tractogram = function(name, meta, renderer) {
        this.name = name;
        this.meta = meta;
        this.renderer = renderer;

        this._visible = (meta.visible === undefined) ? true : !!meta.visible;
        this._opacity = (meta.alpha === undefined) ? 1 : meta.alpha;
        this._mix = 0;

        this.n_points = 0;
        this.n_streamlines = 0;
        //Number of vertices actually uploaded to the GPU: equals n_points
        //for the indexed geometry, 2 * (number of segments) for the
        //duplicated-vertex fallback. Plain numbers are also what the python
        //JSProxy can read back (typed-array lengths are not enumerable).
        this.n_vertices = 0;
        this.geometry = null;
        this.material = null;
        this.line = null;

        //Per-group visibility (name -> bool), including a synthetic
        //"(ungrouped)" entry when at least one streamline belongs to no
        //named group. Populated once the "groups" buffer has loaded (see
        //_build); empty until then, and left empty forever when the
        //tractogram has no groups at all (n.b. distinct from "has groups
        //but is entirely covered by them", which does not create the
        //pseudo-group). this._hasGroups records whether group filtering
        //applies at all, so setVisible-only tractograms skip it entirely.
        this._hasGroups = false;
        this._groupVisible = {};
        //Uint32Array: concatenation of every real group's streamline
        //indices (server "groups" buffer) followed by the synthetic
        //"(ungrouped)" group's indices, if any -- see _build.
        this._groupIndices = null;
        //name -> [start, stop] slice into _groupIndices.
        this._groupSlices = {};
        //Uint8Array of length n_streamlines, recomputed by
        //_updateStreamlineVisibility whenever _groupVisible changes: 1 if
        //the streamline is a member of >=1 visible group (or, with no
        //groups at all, always 1).
        this._streamlineVisible = null;
        //Raw (unfiltered) buffers, kept around so _rebuildGeometry can
        //recompute the segment index / duplicated arrays without re-fetching.
        this._rawPoints = null;
        this._rawOffsets = null;
        this._rawColors = null;

        //The viewer's own `loaded` Deferred must NOT wait on this one: tracts
        //are an overlay on top of a viewer that is usable without them.
        this.loaded = $.Deferred();

        this.object = new THREE.Group();
        this.object.name = "Tractogram:" + name;
        this.object.visible = this._visible;
        //Passes that render the scene with a surface-specific override
        //material (the SVG label depth pass in svgoverlay.js) must skip us:
        //those shaders expect surface attributes this geometry doesn't have.
        this.object.userData.skipOverrideMaterial = true;

        //DOM element for this tractogram's controls, built once _build has
        //run (see _buildElement); appended under #tracts by
        //Viewer.addTracts. References to the individual inputs are kept so
        //_syncControls can update them after a programmatic state change
        //(e.g. a Python call through the JSProxy).
        this.element = null;
        this._visibleCheckbox = null;
        this._opacitySlider = null;
        this._opacityBox = null;
        this._groupCheckboxes = {};

        var buffers = {}, names = ["points", "offsets", "colors", "groups"];
        var pending = names.length;
        var failed = false;
        var ondone = function(bufname) {
            return function(data) {
                buffers[bufname] = data;
                if (--pending === 0 && !failed)
                    this._build(buffers);
            }.bind(this);
        }.bind(this);
        var onfail = function(status) {
            if (!failed) {
                failed = true;
                this.loaded.reject(status);
            }
        }.bind(this);

        for (var i = 0; i < names.length; i++)
            loadBuffer(meta.urls[names[i]], ondone(names[i]), onfail);
    };

    //Turn the four raw buffers into a THREE.Line of segments, and set up
    //per-group visibility (this._groupVisible / this._groupSlices /
    //this._groupIndices) plus the "groups" dat.gui sub-folder, if any.
    module.Tractogram.prototype._build = function(buffers) {
        var points = new Float32Array(buffers.points);
        var offsets = new Uint32Array(buffers.offsets);
        var rawcolors = new Uint8Array(buffers.colors);

        var npts = points.length / 3;
        this.n_points = npts;
        //The offsets array carries a trailing sentinel equal to npts, so there
        //is one streamline per pair of consecutive entries.
        var nstream = Math.max(offsets.length - 1, 0);
        this.n_streamlines = nstream;

        //r69 always uploads vertex attributes as gl.FLOAT (three.js:20276
        //hardcodes it in setupVertexAttributes), so the uint8 colors must be
        //expanded to normalized floats -- normalized uint8 attributes are not
        //an option in this version.
        var colors = new Float32Array(rawcolors.length);
        for (var i = 0; i < rawcolors.length; i++)
            colors[i] = rawcolors[i] / 255;

        this._rawPoints = points;
        this._rawOffsets = offsets;
        this._rawColors = colors;

        this._setupGroups(buffers.groups, nstream);
        this._updateStreamlineVisibility();

        var alpha = this._opacity;
        this.material = new THREE.LineBasicMaterial({
            vertexColors: THREE.VertexColors,
            transparent: alpha < 1,
            opacity: alpha,
            //Always write depth, at every opacity -- see setOpacity for why.
            depthWrite: true,
            linewidth: (this.meta.linewidth === undefined) ? 1 : this.meta.linewidth,
        });

        this._buildLine(this._computeGeometryArrays());

        this._buildElement();

        this.loaded.resolve(this);
    };

    //Parse the "groups" buffer (uint32 streamline indices, the
    //concatenation of every group in metadata order -- see
    //Tractogram.groups_wire in cortex/dataset/tractogram.py) plus
    //`meta.groups` ({name: [start, stop]}, same order) into
    //this._groupSlices / this._groupIndices, and append a synthetic
    //"(ungrouped)" group covering any streamline that is not a member of a
    //real group. Initializes every group (including the pseudo one) to
    //visible. A tractogram with no groups at all leaves this._hasGroups
    //false and group filtering out of the picture entirely.
    module.Tractogram.prototype._setupGroups = function(groupsBuffer, nstream) {
        var groupIndices = new Uint32Array(groupsBuffer);
        var groupSlices = {};
        //Object key order for non-integer-like string keys follows
        //insertion order in all JS engines we target, and JSON.parse
        //preserves the order the metadata was serialized in -- but a group
        //named e.g. "2" would be reordered numerically by the JS engine;
        //not handled here (group names are expected to be bundle names).
        for (var gname in this.meta.groups)
            groupSlices[gname] = this.meta.groups[gname];
        var hasGroups = Object.keys(groupSlices).length > 0;

        if (hasGroups) {
            var inGroup = new Uint8Array(nstream);
            for (var gi = 0; gi < groupIndices.length; gi++)
                inGroup[groupIndices[gi]] = 1;
            var ungrouped = [];
            for (var s = 0; s < nstream; s++)
                if (!inGroup[s])
                    ungrouped.push(s);
            if (ungrouped.length > 0) {
                var combined = new Uint32Array(groupIndices.length + ungrouped.length);
                combined.set(groupIndices, 0);
                combined.set(ungrouped, groupIndices.length);
                groupSlices["(ungrouped)"] = [groupIndices.length, combined.length];
                groupIndices = combined;
            }
        }

        this._hasGroups = hasGroups;
        this._groupIndices = groupIndices;
        this._groupSlices = groupSlices;
        this._groupVisible = {};
        for (var name in groupSlices)
            this._groupVisible[name] = true;
    };

    //Recompute this._streamlineVisible (Uint8Array, one entry per
    //streamline) from this._groupVisible. A streamline is visible iff it is
    //a member of >=1 visible group (ungrouped streamlines follow the
    //"(ungrouped)" pseudo-group); with no groups at all, every streamline
    //is visible.
    module.Tractogram.prototype._updateStreamlineVisibility = function() {
        var nstream = this.n_streamlines;
        var vis = new Uint8Array(nstream);
        if (!this._hasGroups) {
            vis.fill(1);
        } else {
            for (var name in this._groupSlices) {
                if (!this._groupVisible[name])
                    continue;
                var se = this._groupSlices[name];
                for (var k = se[0]; k < se[1]; k++)
                    vis[this._groupIndices[k]] = 1;
            }
        }
        this._streamlineVisible = vis;
    };

    //Build the position/color/index arrays for the currently-visible
    //streamlines only, in the shape _buildLine expects. Shared by the first
    //_build and every subsequent _rebuildGeometry.
    module.Tractogram.prototype._computeGeometryArrays = function() {
        var points = this._rawPoints, offsets = this._rawOffsets, colors = this._rawColors;
        var nstream = this.n_streamlines;
        var visible = this._streamlineVisible;
        var npts = points.length / 3;

        //Number of segments among visible streamlines: every streamline of
        //L points yields L-1 segments.
        var nseg = 0;
        for (var s = 0; s < nstream; s++) {
            if (visible[s])
                nseg += Math.max(offsets[s+1] - offsets[s] - 1, 0);
        }

        if (module.supportsUint32Index(this.renderer) || npts <= 65535) {
            //Uint16 is enough (and universally supported) for small tractograms.
            var index = (npts <= 65535) ? new Uint16Array(2 * nseg) : new Uint32Array(2 * nseg);
            var k = 0;
            for (var s = 0; s < nstream; s++) {
                if (!visible[s])
                    continue;
                for (var j = offsets[s]; j + 1 < offsets[s+1]; j++) {
                    index[k++] = j;
                    index[k++] = j + 1;
                }
            }
            //Position/color stay the full (unfiltered) buffers -- only the
            //index changes with visibility.
            return {indexed: true, position: points, color: colors, index: index};
        } else {
            //Fallback: duplicate the endpoints of every segment so no element
            //index buffer is needed at all.
            var pos = new Float32Array(6 * nseg);
            var col = new Float32Array(6 * nseg);
            var k = 0;
            for (var s = 0; s < nstream; s++) {
                if (!visible[s])
                    continue;
                for (var j = offsets[s]; j + 1 < offsets[s+1]; j++) {
                    for (var d = 0; d < 3; d++) {
                        pos[6*k + d] = points[3*j + d];
                        pos[6*k + 3 + d] = points[3*(j+1) + d];
                        col[6*k + d] = colors[3*j + d];
                        col[6*k + 3 + d] = colors[3*(j+1) + d];
                    }
                    k++;
                }
            }
            return {indexed: false, position: pos, color: col};
        }
    };

    //Replace this.geometry/this.line with a fresh geometry built from
    //`arrays` (as returned by _computeGeometryArrays), disposing the old
    //ones. this.material is reused across rebuilds (only the geometry
    //changes), so opacity/depthWrite state carries over unchanged.
    module.Tractogram.prototype._buildLine = function(arrays) {
        var geometry = new THREE.BufferGeometry();
        geometry.addAttribute("position", new THREE.BufferAttribute(arrays.position, 3));
        geometry.addAttribute("color", new THREE.BufferAttribute(arrays.color, 3));
        if (arrays.indexed)
            geometry.addAttribute("index", new THREE.BufferAttribute(arrays.index, 1));
        geometry.computeBoundingSphere();

        if (this.line !== null)
            this.object.remove(this.line);
        if (this.geometry !== null)
            this.geometry.dispose();

        this.geometry = geometry;
        this.line = new THREE.Line(geometry, this.material, THREE.LinePieces);
        this.line.name = "Tractogram:" + this.name + ":lines";
        this.n_vertices = geometry.attributes.position.array.length / 3;
        //Segments actually drawn: shrinks when groups are hidden on both
        //the indexed path (index halves) and the fallback path (vertices
        //are duplicated per segment). n_vertices only shrinks on the latter.
        this.n_segments = arrays.indexed
            ? arrays.index.length / 2
            : this.n_vertices / 2;
        this._updateRenderOrder();
        this.object.add(this.line);
    };

    //Rebuild the rendered geometry from the current this._streamlineVisible
    //(called after any change to group visibility) and ask the viewer to
    //redraw. A no-op before the buffers have loaded.
    module.Tractogram.prototype._rebuildGeometry = function() {
        if (this._rawPoints === null)
            return;
        this._buildLine(this._computeGeometryArrays());
        this._syncControls();
        if (window.viewer !== undefined && window.viewer.schedule !== undefined)
            window.viewer.schedule();
    };

    //Build this.element: the DOM controls for this tractogram, appended
    //under #tracts by Viewer.addTracts once this.loaded resolves. Header
    //row (visibility checkbox, name, collapse toggle) always present; body
    //(opacity slider, and a per-group "bundles" section when the
    //tractogram has groups) can be collapsed. Collapsed by default when
    //there are more than 8 groups, to keep the panel from taking over the
    //screen for tractograms with many bundles.
    module.Tractogram.prototype._buildElement = function() {
        var names = this.groupNames();
        var collapsed = this._hasGroups && names.length > 8;

        var el = $("<div class='tract-item'></div>");
        var header = $("<div class='tract-header'></div>");
        var toggle = $("<span class='tract-toggle'></span>").text(collapsed ? "▸" : "▾");
        var visibleCheckbox = $("<input type='checkbox'>");
        var nameLabel = $("<span class='tract-name'></span>").text(this.name);
        header.append(visibleCheckbox, toggle, nameLabel);

        var body = $("<div class='tract-body'></div>");
        if (collapsed)
            body.hide();

        var opacityRow = $("<div class='tract-opacity-row'></div>");
        var opacitySlider = $("<input type='range' min='0' max='1' step='0.01'>");
        var opacityBox = $("<input type='number' class='tract-opacity-value' " +
                           "min='0' max='1' step='0.01'>");
        opacityRow.append($("<label>opacity</label>"), opacitySlider, opacityBox);
        body.append(opacityRow);

        this._groupCheckboxes = {};
        if (this._hasGroups) {
            var groupsSection = $("<div class='tract-groups'></div>");
            var actions = $("<div class='tract-groups-actions'></div>");
            var allLink = $("<a>all</a>");
            var noneLink = $("<a>none</a>");
            actions.append(allLink, noneLink);
            groupsSection.append(actions);

            for (var i = 0; i < names.length; i++) {
                (function(tract, gname) {
                    var slice = tract._groupSlices[gname];
                    var count = slice[1] - slice[0];
                    var row = $("<div class='tract-group-row'></div>");
                    var cb = $("<input type='checkbox'>");
                    var label = $("<span class='tract-group-name'></span>")
                        .text(gname + " ")
                        .append($("<span class='tract-group-count'></span>").text("(" + count + ")"));
                    row.append(cb, label);
                    groupsSection.append(row);
                    tract._groupCheckboxes[gname] = cb;

                    cb.on("change", function() {
                        tract.setGroupVisible(gname, cb.prop("checked"));
                    });
                }(this, names[i]));
            }

            allLink.on("click", function() { this.showAllGroups(); }.bind(this));
            noneLink.on("click", function() { this.hideAllGroups(); }.bind(this));

            body.append(groupsSection);
        }

        el.append(header, body);

        //Clicks/drags inside the panel must not reach the WebGL canvas
        //handlers underneath (camera rotation, picking, etc).
        el.on("mousedown click", function(e) { e.stopPropagation(); });

        visibleCheckbox.on("change", function() {
            this.setVisible(visibleCheckbox.prop("checked"));
        }.bind(this));
        opacitySlider.on("input change", function() {
            this.setOpacity(parseFloat(opacitySlider.val()));
        }.bind(this));
        //The box takes typed values, so it only commits on change/Enter --
        //reacting to "input" would fight the user mid-keystroke ("0.0" while
        //they are on their way to "0.05"). A value outside 0-1 or an empty
        //box is clamped or ignored by setOpacity, and _syncControls then puts
        //the accepted value back in the box.
        opacityBox.on("change", function() {
            this.setOpacity(opacityBox.val());
            this._syncControls();
        }.bind(this));
        opacityBox.on("keydown", function(e) {
            if (e.which === 13)
                opacityBox.trigger("change");
        });
        toggle.on("click", function() {
            collapsed = !collapsed;
            toggle.text(collapsed ? "▸" : "▾");
            body.toggle(!collapsed);
        });

        this.element = el;
        this._visibleCheckbox = visibleCheckbox;
        this._opacitySlider = opacitySlider;
        this._opacityBox = opacityBox;

        this._syncControls();
    };

    //Keep the DOM controls in step with this tractogram's state, whether it
    //changed via a click in the panel or programmatically (showAllGroups /
    //hideAllGroups / setGroupVisible / setVisible / setOpacity called
    //directly, e.g. from Python through the JSProxy). Called at the end of
    //every setter.
    module.Tractogram.prototype._syncControls = function() {
        if (this._visibleCheckbox !== null)
            this._visibleCheckbox.prop("checked", this._visible);
        if (this._opacitySlider !== null)
            this._opacitySlider.val(this._opacity);
        if (this._opacityBox !== null)
            this._opacityBox.val(this._opacity);
        for (var name in this._groupCheckboxes)
            this._groupCheckboxes[name].prop("checked", !!this._groupVisible[name]);
    };

    //Getter/setter pair, in the shape jsplot.Menu expects (called with no
    //argument it returns the current value, so dat.gui can initialize itself).
    module.Tractogram.prototype.setVisible = function(value) {
        if (value === undefined)
            return this._visible;
        this._visible = !!value;
        this._updateVisible();
        this._syncControls();
        this._requestRedraw();
    };

    module.Tractogram.prototype.setOpacity = function(value) {
        if (value === undefined)
            return this._opacity;
        value = parseFloat(value);
        if (isNaN(value))
            return;
        value = Math.min(1, Math.max(0, value));
        this._opacity = value;
        if (this.material !== null) {
            this.material.opacity = value;
            this.material.transparent = value < 1;
            //depthWrite stays on at every opacity. Turning it off (the
            //obvious thing to do for a transparent material) leaves the
            //streamlines with nothing to depth-test against each other, so
            //they blend in buffer order instead of depth order: the bundle
            //that happens to sit last in the geometry paints over the ones
            //in front of it, and which bundle looks nearest changes the
            //moment opacity drops below 1. Writing depth keeps the
            //occlusion identical at every opacity, at the cost of not
            //seeing one translucent streamline through another -- the
            //alternative is re-sorting every segment back-to-front on each
            //camera move, which r69 will not do for us.
            this.material.depthWrite = true;
            this.material.needsUpdate = true;
            this._updateRenderOrder();
        }
        this._syncControls();
        this._requestRedraw();
    };

    //The panel's inputs are plain DOM controls (not dat.gui, whose menu used
    //to dispatch an "update" the viewer redraws on), so every state change
    //has to ask the viewer for a frame itself.
    module.Tractogram.prototype._requestRedraw = function() {
        if (window.viewer !== undefined && window.viewer.schedule !== undefined)
            window.viewer.schedule();
    };

    //Getter/setter pair for one group's visibility, in the shape jsplot.Menu
    //expects. Rebuilds the rendered geometry (see _rebuildGeometry).
    module.Tractogram.prototype.setGroupVisible = function(name, value) {
        if (value === undefined)
            return !!this._groupVisible[name];
        this._groupVisible[name] = !!value;
        this._updateStreamlineVisibility();
        this._rebuildGeometry();
    };

    //Make every group (including "(ungrouped)") visible.
    module.Tractogram.prototype.showAllGroups = function() {
        for (var name in this._groupVisible)
            this._groupVisible[name] = true;
        this._updateStreamlineVisibility();
        this._rebuildGeometry();
    };

    //Hide every group (including "(ungrouped)"): no streamlines render.
    module.Tractogram.prototype.hideAllGroups = function() {
        for (var name in this._groupVisible)
            this._groupVisible[name] = false;
        this._updateStreamlineVisibility();
        this._rebuildGeometry();
    };

    //Names of every group this tractogram knows about, in metadata order,
    //including the synthetic "(ungrouped)" entry if present. Empty when the
    //tractogram has no groups.
    module.Tractogram.prototype.groupNames = function() {
        return Object.keys(this._groupSlices || {});
    };

    //Three.js r69 draws opaque objects first, then transparent ones sorted by
    //their (projected) center depth. Opaque tracts are therefore always
    //covered by a translucent surface, but as soon as the tracts themselves
    //become translucent the depth sort can put them *after* the surface --
    //drawn on top of it, undimmed, so lowering tract opacity from 1 to 0.9
    //made them brighter. Pinning renderDepth keeps translucent tracts first:
    //r69 sorts the transparent list ascending by z and then walks it from the
    //END (renderObjects iterates backwards), so the largest renderDepth is
    //drawn first, i.e. always under the surface, and the surface's own
    //opacity attenuates the tracts consistently.
    module.Tractogram.prototype._updateRenderOrder = function() {
        if (this.line === null)
            return;
        this.line.renderDepth = (this._opacity < 1) ? 1e6 : null;
    };

    //Streamlines are defined in the fiducial (unmorphed) space, so they only
    //make sense while the surface is not inflating/flattening.
    module.Tractogram.prototype.setMix = function(mix) {
        if (mix === undefined)
            return this._mix;
        this._mix = mix;
        this._updateVisible();
    };

    module.Tractogram.prototype._updateVisible = function() {
        this.object.visible = this._visible && this._mix === 0;
    };

    module.Tractogram.prototype.dispose = function() {
        if (this.line !== null)
            this.object.remove(this.line);
        if (this.geometry !== null)
            this.geometry.dispose();
        if (this.material !== null)
            this.material.dispose();
        this.geometry = null;
        this.material = null;
        this.line = null;
    };

    return module;
}(mriview || {}));
