/* Cortical depth ("laminar") profiles for the webgl viewer.
 *
 * This is the browser-side twin of cortex/quickflat/laminar.py: given a line
 * drawn across the flatmap, it builds the W x H matrix of anatomical points
 * that spans that line (gamma, 0..1 along the line) and cortical depth
 * (alpha, 0 = pial .. 1 = white matter), then paints it with the volume data.
 *
 * Everything is computed here, in the browser -- nothing is round-tripped to
 * python. The split is:
 *
 *   CPU (this file)  For each of the W+1 profile columns, locate the flatmap
 *                    point inside a flatmap triangle and barycentrically
 *                    interpolate the pial position, the white matter position
 *                    and the two surface areas. This is exactly
 *                    _column_geometry() from laminar.py, and like there it
 *                    only has to happen once per column, not once per pixel.
 *
 *   GPU (Shaders.laminar) Per fragment, turn alpha into an equivolume depth
 *                    (_depth_blend), mix pial/wm to get this pixel's
 *                    anatomical coordinate, push it through volxfm and sample
 *                    + colormap it through the same colorlut() the surface
 *                    shader uses. So every pixel of the panel maps to its own
 *                    point in the volume.
 */
var laminar = (function(module) {

    // Matches mriview_surface.js: flat vertex coordinates are scaled by this
    // when they are baked into the flat morph target.
    var flatscale = 0.3;

    var HEMIS = ["left", "right"];

    /*************************************************************************
     * FlatIndex -- point location over one hemisphere's flatmap
     *
     * A uniform grid over the flat triangles, so that locating the triangle
     * under a flatmap coordinate is O(1)-ish. Stands in for the scipy Delaunay
     * / trifinder used by laminar.py; we can do better than Delaunay here
     * because the flatmap already comes with its own triangulation.
     *************************************************************************/
    module.FlatIndex = function(hemi) {
        // The `uv` attribute holds the flatmap coordinates of both hemispheres
        // in one shared, nudged-apart space (normalized to [0,1] by
        // Surface._makeFlat). That is the same merged space laminar.py
        // triangulates, and unlike the surface's own local frame it does not
        // move when the brain is folded, inflated or shifted.
        var uv = hemi.attributes.uv;
        this.pts = uv.array;
        var pts = this.pts;

        // Only the culled index holds real flatmap faces; the full index also
        // contains the medial wall, whose flat coordinates are meaningless.
        var index = hemi.culled.index.array, offsets = hemi.culled.offsets;
        var ntri = 0;
        for (var j = 0; j < offsets.length; j++)
            ntri += offsets[j].count / 3;

        var tris = new Uint32Array(ntri * 3), t = 0;
        for (var j = 0; j < offsets.length; j++) {
            var start = offsets[j].start, count = offsets[j].count, base = offsets[j].index;
            for (var i = start, il = start + count; i < il; i += 3) {
                tris[t++] = base + index[i];
                tris[t++] = base + index[i+1];
                tris[t++] = base + index[i+2];
            }
        }
        this.tris = tris;
        this.ntri = ntri;

        var minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
        for (var k = 0; k < tris.length; k++) {
            var x = pts[tris[k]*2], y = pts[tris[k]*2+1];
            if (x < minx) minx = x;
            if (x > maxx) maxx = x;
            if (y < miny) miny = y;
            if (y > maxy) maxy = y;
        }
        this.bounds = {minx:minx, miny:miny, maxx:maxx, maxy:maxy};

        // Roughly one triangle per cell.
        var n = Math.max(1, Math.min(1024, Math.round(Math.sqrt(Math.max(ntri, 1)))));
        this.n = n;
        this.ox = minx;
        this.oy = miny;
        this.sx = n / Math.max(maxx - minx, 1e-9);
        this.sy = n / Math.max(maxy - miny, 1e-9);

        // Triangles are bucketed by their bounding box, so a point inside a
        // triangle is always found in that point's own cell.
        var starts = new Uint32Array(n*n + 1);
        var self = this;
        var visit = function(cb) {
            for (var ti = 0; ti < ntri; ti++) {
                var a = tris[ti*3], b = tris[ti*3+1], c = tris[ti*3+2];
                var ax = pts[a*2], ay = pts[a*2+1];
                var bx = pts[b*2], by = pts[b*2+1];
                var cx = pts[c*2], cy = pts[c*2+1];
                var i0 = self._cellx(Math.min(ax, bx, cx)), i1 = self._cellx(Math.max(ax, bx, cx));
                var j0 = self._celly(Math.min(ay, by, cy)), j1 = self._celly(Math.max(ay, by, cy));
                for (var jy = j0; jy <= j1; jy++)
                    for (var jx = i0; jx <= i1; jx++)
                        cb(jy*n + jx, ti);
            }
        };
        visit(function(cell) { starts[cell + 1]++; });
        for (var i = 0; i < n*n; i++)
            starts[i+1] += starts[i];

        var items = new Uint32Array(starts[n*n]);
        var cursor = new Uint32Array(n*n);
        for (var i = 0; i < n*n; i++)
            cursor[i] = starts[i];
        visit(function(cell, ti) { items[cursor[cell]++] = ti; });

        this.starts = starts;
        this.items = items;
    };
    module.FlatIndex.prototype._cellx = function(x) {
        var i = Math.floor((x - this.ox) * this.sx);
        return i < 0 ? 0 : (i > this.n - 1 ? this.n - 1 : i);
    };
    module.FlatIndex.prototype._celly = function(y) {
        var j = Math.floor((y - this.oy) * this.sy);
        return j < 0 ? 0 : (j > this.n - 1 ? this.n - 1 : j);
    };

    /* Locate (x, y) in the flatmap. Returns {a, b, c, l0, l1, l2} -- the three
     * vertex indices and their barycentric weights -- or null if the point
     * falls outside the flatmap. */
    module.FlatIndex.prototype.locate = function(x, y) {
        var bb = this.bounds;
        if (x < bb.minx || x > bb.maxx || y < bb.miny || y > bb.maxy)
            return null;

        var cell = this._celly(y) * this.n + this._cellx(x);
        var pts = this.pts, tris = this.tris, items = this.items;
        for (var k = this.starts[cell], kl = this.starts[cell+1]; k < kl; k++) {
            var ti = items[k];
            var a = tris[ti*3], b = tris[ti*3+1], c = tris[ti*3+2];
            var cx = pts[c*2], cy = pts[c*2+1];
            var v0x = pts[a*2] - cx, v0y = pts[a*2+1] - cy;
            var v1x = pts[b*2] - cx, v1y = pts[b*2+1] - cy;
            var det = v0x*v1y - v1x*v0y;
            if (det === 0)
                continue;
            var px = x - cx, py = y - cy;
            var l0 = (px*v1y - v1x*py) / det;
            var l1 = (v0x*py - px*v0y) / det;
            var l2 = 1 - l0 - l1;
            if (l0 >= -1e-6 && l1 >= -1e-6 && l2 >= -1e-6)
                return {a:a, b:b, c:c, l0:l0, l1:l1, l2:l2};
        }
        return null;
    };

    /* Nearest flatmap vertex to (x, y). Only used to seed the default line, so
     * a brute force scan is fine. */
    module.FlatIndex.prototype.nearestVertex = function(x, y) {
        var pts = this.pts, tris = this.tris;
        var best = -1, bestd = Infinity;
        for (var k = 0; k < tris.length; k++) {
            var v = tris[k];
            var dx = pts[v*2] - x, dy = pts[v*2+1] - y;
            var d = dx*dx + dy*dy;
            if (d < bestd) {
                bestd = d;
                best = v;
            }
        }
        return best < 0 ? null : {x:pts[best*2], y:pts[best*2+1]};
    };

    /*************************************************************************
     * Profile -- the depth profile panel, its line, and the UI around them
     *************************************************************************/
    module.Profile = function(viewer) {
        this.viewer = viewer;

        this._enabled = false;
        this._equivolume = true;
        this._width = 1024;
        this._height = 128;

        // Endpoints of the flatmap line, as {hemi, x, y} in the shared,
        // normalized flatmap space (see FlatIndex). `hemi` only records which
        // side the point landed on, for the readout.
        this.endpoints = null;

        this._index = null;         // {left: FlatIndex, right: FlatIndex}
        this._indexSurf = null;     // the Surface those indices were built for

        this._shader = null;
        this._shaderkey = null;
        this._target = null;
        this._pixels = null;
        this._sig = null;
        this._dirty = true;
        this._drag = null;

        // The line itself is drawn into a flatmap-space texture that the
        // surface shader samples at vUv, so it lies on the cortex in every
        // view rather than being pasted on the screen.
        this._lineTarget = null;
        this._lineScene = null;
        this._lineSig = null;

        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(0, 1, 1, 0, -1, 1);
        this.scene.add(this.camera);
        this._makeGeometry();

        var root = $(viewer.object);
        this.panel = root.find("#laminar_panel");
        this.canvas = root.find("#laminar_canvas");
        this.message = root.find("#laminar_message");
        this.readout = root.find("#laminar_readout");
        this.overlay = root.find("#laminar_overlay");

        this._buildOverlay();
        this._bindUI();

        viewer.addEventListener("draw", this._ondraw.bind(this));
        viewer.addEventListener("resize", this._onresize.bind(this));
    };

    /*** state ***************************************************************/

    module.Profile.prototype.setEnabled = function(val) {
        if (val === undefined)
            return this._enabled;

        this._enabled = !!val;
        if (this._enabled) {
            // The line is seeded lazily by _ondraw() once the surfaces are in.
            this.panel.show();
            this._placePanel();
        } else {
            this.panel.hide();
            this.overlay.hide();
        }
        this._applyLineLayer();
        this._dirty = true;
        this.viewer.schedule();
    };

    module.Profile.prototype.toggle = function() {
        this.setEnabled(!this._enabled);
    };

    module.Profile.prototype.setEquivolume = function(val) {
        if (val === undefined)
            return this._equivolume;
        this._equivolume = !!val;
        this._dirty = true;
        this.viewer.schedule();
    };

    module.Profile.prototype.setWidth = function(val) {
        if (val === undefined)
            return this._width;
        val = Math.max(8, Math.round(val));
        if (val === this._width)
            return;
        this._width = val;
        this._makeGeometry();
        this._dirty = true;
        this.viewer.schedule();
    };

    module.Profile.prototype.setHeight = function(val) {
        if (val === undefined)
            return this._height;
        this._height = Math.max(4, Math.round(val));
        this._dirty = true;
        this.viewer.schedule();
    };

    /*** surface plumbing ****************************************************/

    module.Profile.prototype._surface = function() {
        for (var i = 0; i < this.viewer.surfs.length; i++) {
            if (this.viewer.surfs[i].surf !== undefined)
                return this.viewer.surfs[i].surf;
        }
        return null;
    };

    module.Profile.prototype._ensureIndex = function() {
        var surf = this._surface();
        if (surf === null || surf.loaded.state() !== "resolved")
            return false;
        if (this._indexSurf === surf)
            return this._index !== null;

        this._indexSurf = surf;
        this._index = null;

        if (surf.flatlims === undefined || surf.hemis.left.attributes.wm === undefined)
            return false;   // needs both a flatmap and pial/wm surfaces

        this._index = {};
        for (var i = 0; i < HEMIS.length; i++)
            this._index[HEMIS[i]] = new module.FlatIndex(surf.hemis[HEMIS[i]]);

        this._lineSig = null;    // a new surface needs the line texture re-bound
        this._applyLineLayer();
        return true;
    };

    /* How flat is the current view? 1 when fully flattened. */
    module.Profile.prototype._flatness = function() {
        var surf = this._surface();
        if (surf === null || surf.flatlims === undefined)
            return 0;
        var nm = surf.names.length;
        var factor = 1 - Math.abs(surf.uniforms.surfmix.value * (nm - 1) - (nm - 1));
        return factor < 0 ? 0 : (factor > 1 ? 1 : factor);
    };

    module.Profile.prototype._isFlat = function() {
        return this._flatness() > 0.999;
    };

    module.Profile.prototype._mesh = function(hemi) {
        var surf = this._surface();
        if (surf === null || surf.sheets.length === 0)
            return null;
        return surf.sheets[0][hemi];
    };

    /* Shared flatmap coordinates -> the raw, un-normalized flatmap units that
     * cortex.quickflat.laminar.make_laminar_profile() takes. Its (u, v) are the
     * second and first flatmap columns respectively. */
    module.Profile.prototype._rawUV = function(x, y) {
        var lims = this._surface().flatlims;
        return {
            v: x * lims[1][0] - lims[0][0],
            u: y * lims[1][1] - lims[0][1],
        };
    };

    /* Shared flatmap coordinates -> a position in one hemisphere's own frame,
     * inverting Surface._makeFlat. */
    module.Profile.prototype._toLocal = function(hemi, x, y, out) {
        var surf = this._surface();
        var raw = this._rawUV(x, y);
        var sign = hemi === "right" ? 1 : -1;
        out = out || new THREE.Vector3();
        return out.set(0, flatscale * sign * raw.v + surf.flatoff[1], flatscale * raw.u);
    };

    /* ...and back again. */
    module.Profile.prototype._toShared = function(hemi, ly, lz) {
        var surf = this._surface();
        var lims = surf.flatlims;
        var sign = hemi === "right" ? 1 : -1;
        var rawv = sign * (ly - surf.flatoff[1]) / flatscale;
        var rawu = lz / flatscale;
        return {x:(rawv + lims[0][0]) / lims[1][0], y:(rawu + lims[0][1]) / lims[1][1]};
    };

    /* Shared flatmap coordinates -> world, for drawing the line overlay. Only
     * meaningful while the surface is flattened. */
    module.Profile.prototype._toWorld = function(ep, out) {
        var mesh = this._mesh(ep.hemi);
        if (mesh === null)
            return null;
        out = this._toLocal(ep.hemi, ep.x, ep.y, out);
        return out.applyMatrix4(mesh.matrixWorld);
    };

    /*** the profile line ****************************************************/

    module.Profile.prototype.resetLine = function() {
        if (!this._ensureIndex())
            return;

        // Seed the line across the middle of whichever hemisphere has a
        // flatmap, along its longer axis.
        var hemi = null;
        for (var i = 0; i < HEMIS.length; i++) {
            if (this._index[HEMIS[i]].ntri > 0) {
                hemi = HEMIS[i];
                break;
            }
        }
        if (hemi === null)
            return;

        var bb = this._index[hemi].bounds;
        var cx = (bb.minx + bb.maxx) / 2, cy = (bb.miny + bb.maxy) / 2;
        var dx = bb.maxx - bb.minx, dy = bb.maxy - bb.miny;
        var offx = 0, offy = 0;
        if (dx > dy)
            offx = 0.15 * dx;
        else
            offy = 0.15 * dy;

        // Snap to real vertices so both ends are guaranteed to sit inside the
        // flatmap.
        var p0 = this._index[hemi].nearestVertex(cx - offx, cy - offy);
        var p1 = this._index[hemi].nearestVertex(cx + offx, cy + offy);
        if (p0 === null || p1 === null)
            return;

        this.endpoints = [
            {hemi:hemi, x:p0.x, y:p0.y},
            {hemi:hemi, x:p1.x, y:p1.y},
        ];
        this._dirty = true;
        this.viewer.schedule();
    };


    /*************************************************************************
     * The line, drawn into a flatmap-space texture
     *
     * The surface shader already maps a texture onto the cortex through vUv,
     * which is the same normalized flatmap space the endpoints live in. So the
     * line goes into a small render target and the shader paints it onto the
     * surface -- flattened, inflated or folded, with the right occlusion and
     * no per-frame projection work.
     *************************************************************************/

    // Line and marker sizes, in flatmap units (roughly mm of cortex).
    var LINE_HALFWIDTH = 1.0;
    var LINE_HALO = 1.0;        // extra half-width of the dark outline
    var MARKER_RADIUS = 3.0;
    var MARKER_HALO = 1.2;
    var DISC_SEGMENTS = 24;

    var CORE_COLOR = [1.0, 0.8, 0.2, 1.0];
    var HALO_COLOR = [0.06, 0.06, 0.06, 1.0];

    /* Largest power of two <= v, at least 64: WebGL 1 only mipmaps
     * power-of-two textures, and mipmaps are what keep the line from
     * shimmering when the flatmap is zoomed out. */
    function _pot(v) {
        var p = 64;
        while (p * 2 <= v)
            p *= 2;
        return p;
    }

    module.Profile.prototype._makeLineScene = function() {
        // Two passes' worth of geometry in one buffer: halo first, core second.
        // Drawing is unblended and in primitive order, so the core simply
        // overwrites the middle of the halo.
        var nvert = 2 * (4 + 2 * (DISC_SEGMENTS + 1));
        var ntri = 2 * (2 + 2 * DISC_SEGMENTS);

        var position = new Float32Array(nvert * 3);
        var lcolor = new Float32Array(nvert * 4);
        var indices = new Uint16Array(ntri * 3);

        var v = 0, t = 0;
        for (var pass = 0; pass < 2; pass++) {
            var color = pass === 0 ? HALO_COLOR : CORE_COLOR;
            var base = v;

            // quad: 0,1 at the A end, 2,3 at the B end
            for (var i = 0; i < 4; i++, v++)
                lcolor.set(color, v * 4);
            indices.set([base, base+1, base+2, base, base+2, base+3], t * 3);
            t += 2;

            // one triangle fan per endpoint marker
            for (var e = 0; e < 2; e++) {
                var centre = v;
                for (var i = 0; i <= DISC_SEGMENTS; i++, v++)
                    lcolor.set(color, v * 4);
                for (var i = 0; i < DISC_SEGMENTS; i++) {
                    indices.set([centre, centre + 1 + i,
                                 centre + 1 + (i + 1) % DISC_SEGMENTS], t * 3);
                    t++;
                }
            }
        }

        var geom = new THREE.BufferGeometry();
        geom.addAttribute("index", new THREE.BufferAttribute(indices, 1));
        geom.addAttribute("position", new THREE.BufferAttribute(position, 3));
        geom.addAttribute("lcolor", new THREE.BufferAttribute(lcolor, 4));
        geom.dynamic = true;

        var shader = new THREE.ShaderMaterial({
            vertexShader: [
                "attribute vec4 lcolor;",
                "varying vec4 vLColor;",
                "void main() {",
                    "vLColor = lcolor;",
                    "gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);",
                "}",
            ].join("\n"),
            fragmentShader: [
                "varying vec4 vLColor;",
                "void main() { gl_FragColor = vLColor; }",
            ].join("\n"),
            attributes: {lcolor: {type:'v4', value:null}},
            // Opaque colors written straight through: the surface shader
            // composites this texture as premultiplied alpha, and for fully
            // opaque texels premultiplied and straight are the same thing.
            // That also keeps mipmapping of the edges correct.
            blending: THREE.NoBlending,
            side: THREE.DoubleSide,
            depthTest: false,
            depthWrite: false,
        });

        this._lineScene = new THREE.Scene();
        this._lineCamera = new THREE.OrthographicCamera(0, 1, 1, 0, -1, 1);
        this._lineGeom = geom;
        this._lineMesh = new THREE.Mesh(geom, shader);
        this._lineMesh.frustumCulled = false;
        this._lineScene.add(this._lineMesh);
    };

    /* Lay out the quad and the two markers for the current endpoints. Sizes are
     * given in flatmap units, so they have to be converted through the
     * flatmap's own extent to stay round (and evenly wide) in that space. */
    module.Profile.prototype._layoutLine = function() {
        var lims = this._surface().flatlims[1];
        var sx = lims[0], sy = lims[1];   // uv 0..1 spans this much flatmap
        var e0 = this.endpoints[0], e1 = this.endpoints[1];

        var ax = e0.x * sx, ay = e0.y * sy;
        var bx = e1.x * sx, by = e1.y * sy;
        var dx = bx - ax, dy = by - ay;
        var len = Math.sqrt(dx*dx + dy*dy);
        if (len < 1e-9) {
            dx = 1;
            dy = 0;
        } else {
            dx /= len;
            dy /= len;
        }

        var pos = this._lineGeom.attributes.position.array;
        var v = 0;
        var put = function(px, py) {
            pos[v*3]   = px / sx;
            pos[v*3+1] = py / sy;
            pos[v*3+2] = 0;
            v++;
        };

        for (var pass = 0; pass < 2; pass++) {
            var half = pass === 0 ? LINE_HALFWIDTH + LINE_HALO : LINE_HALFWIDTH;
            var rad = pass === 0 ? MARKER_RADIUS + MARKER_HALO : MARKER_RADIUS;
            var nx = -dy * half, ny = dx * half;

            put(ax + nx, ay + ny);
            put(ax - nx, ay - ny);
            put(bx - nx, by - ny);
            put(bx + nx, by + ny);

            var ends = [[ax, ay], [bx, by]];
            for (var e = 0; e < 2; e++) {
                put(ends[e][0], ends[e][1]);
                for (var i = 0; i < DISC_SEGMENTS; i++) {
                    var th = 2 * Math.PI * i / DISC_SEGMENTS;
                    put(ends[e][0] + rad * Math.cos(th), ends[e][1] + rad * Math.sin(th));
                }
            }
        }
        this._lineGeom.attributes.position.needsUpdate = true;
    };

    /* Point the surface's laminarline sampler at our texture (or unbind it and
     * recompile without it when the profile is off). */
    module.Profile.prototype._applyLineLayer = function() {
        var surf = this._surface();
        if (surf === null)
            return;
        var on = this._enabled && this._lineTarget !== null;
        surf.uniforms.laminarline.value = on ? this._lineTarget : null;
        surf.setLaminarLine(on);
    };

    module.Profile.prototype._updateLineTexture = function() {
        var surf = this._surface();
        if (surf === null || this.endpoints === null || surf.flatlims === undefined)
            return;

        var sig = [this.endpoints[0].x, this.endpoints[0].y,
                   this.endpoints[1].x, this.endpoints[1].y].join(",");
        if (sig === this._lineSig)
            return;

        if (this._lineScene === null)
            this._makeLineScene();

        if (this._lineTarget === null) {
            var lims = surf.flatlims[1];
            var tw = 2048;
            var th = _pot(Math.round(tw * lims[1] / lims[0]));
            this._lineTarget = new THREE.WebGLRenderTarget(tw, th, {
                minFilter: THREE.LinearMipMapLinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                stencilBuffer: false,
                depthBuffer: false,
                generateMipmaps: true,
            });
            this._applyLineLayer();
        }

        this._layoutLine();

        var renderer = this.viewer.renderer;
        var clearColor = renderer.getClearColor().clone();
        var clearAlpha = renderer.getClearAlpha();
        renderer.enableScissorTest(false);
        renderer.setClearColor(new THREE.Color(0, 0, 0), 0);
        renderer.render(this._lineScene, this._lineCamera, this._lineTarget, true);
        renderer.setClearColor(clearColor, clearAlpha);
        renderer.setRenderTarget(null);

        this._lineSig = sig;
    };

    /*** profile geometry ****************************************************/

    module.Profile.prototype._makeGeometry = function() {
        var W = this._width, n = W + 1;

        if (this.mesh !== undefined) {
            this.scene.remove(this.mesh);
            this.geometry.dispose();
        }

        var position = new Float32Array(n * 2 * 3);
        var alpha = new Float32Array(n * 2);
        for (var i = 0; i < n; i++) {
            var g = i / W;
            // top row: pial (alpha 0); bottom row: white matter (alpha 1)
            position[i*3] = g;         position[i*3+1] = 1;  position[i*3+2] = 0;
            position[(n+i)*3] = g;     position[(n+i)*3+1] = 0; position[(n+i)*3+2] = 0;
            alpha[i] = 0;
            alpha[n+i] = 1;
        }

        // Wound counter-clockwise in profile space so the strip is front
        // facing (the shader material doesn't render back faces).
        var indices = new Uint16Array(W * 6);
        for (var i = 0; i < W; i++) {
            indices[i*6+0] = n + i;
            indices[i*6+1] = n + i + 1;
            indices[i*6+2] = i + 1;
            indices[i*6+3] = n + i;
            indices[i*6+4] = i + 1;
            indices[i*6+5] = i;
        }

        var geom = new THREE.BufferGeometry();
        geom.addAttribute("index", new THREE.BufferAttribute(indices, 1));
        geom.addAttribute("position", new THREE.BufferAttribute(position, 3));
        geom.addAttribute("alpha", new THREE.BufferAttribute(alpha, 1));
        geom.addAttribute("pialpos", new THREE.BufferAttribute(new Float32Array(n*2*3), 3));
        geom.addAttribute("wmpos", new THREE.BufferAttribute(new Float32Array(n*2*3), 3));
        geom.addAttribute("pialarea", new THREE.BufferAttribute(new Float32Array(n*2), 1));
        geom.addAttribute("wmarea", new THREE.BufferAttribute(new Float32Array(n*2), 1));
        geom.addAttribute("valid", new THREE.BufferAttribute(new Float32Array(n*2), 1));
        geom.dynamic = true;

        this.geometry = geom;
        this.mesh = new THREE.Mesh(geom, this._shader);
        this.mesh.frustumCulled = false;
        this.scene.add(this.mesh);
    };

    /* Resolve every column of the profile: the CPU half of the algorithm.
     * Mirrors _column_geometry() in quickflat/laminar.py. */
    module.Profile.prototype._computeColumns = function() {
        if (!this._ensureIndex() || this.endpoints === null)
            return false;

        var surf = this._surface();
        var W = this._width, n = W + 1;
        var attrs = this.geometry.attributes;
        var pialpos = attrs.pialpos.array, wmpos = attrs.wmpos.array;
        var pialarea = attrs.pialarea.array, wmarea = attrs.wmarea.array;
        var valid = attrs.valid.array;

        var e0 = this.endpoints[0], e1 = this.endpoints[1];

        // Walk the line in the shared flatmap space, exactly as laminar.py
        // does. Because both hemispheres live in that one space, a line may
        // cross from one to the other; columns that land in the gap between
        // them simply come out invalid.

        // Try the endpoints' own hemispheres first, then the other one.
        var order = [this.endpoints[0].hemi];
        if (this.endpoints[1].hemi !== order[0])
            order.push(this.endpoints[1].hemi);
        for (var i = 0; i < HEMIS.length; i++)
            if (order.indexOf(HEMIS[i]) < 0)
                order.push(HEMIS[i]);

        var nvalid = 0;
        for (var i = 0; i < n; i++) {
            var g = i / W;
            var px = e0.x + (e1.x - e0.x) * g;
            var py = e0.y + (e1.y - e0.y) * g;

            var hit = null, hemi = null;
            for (var h = 0; h < order.length; h++) {
                hit = this._index[order[h]].locate(px, py);
                if (hit !== null) {
                    hemi = order[h];
                    break;
                }
            }

            if (hit === null) {
                valid[i] = valid[n+i] = 0;
                for (var k = 0; k < 3; k++) {
                    pialpos[i*3+k] = pialpos[(n+i)*3+k] = 0;
                    wmpos[i*3+k] = wmpos[(n+i)*3+k] = 0;
                }
                pialarea[i] = pialarea[n+i] = 1;
                wmarea[i] = wmarea[n+i] = 1;
                continue;
            }
            nvalid++;

            var hemig = surf.hemis[hemi];
            var pia = hemig.attributes.position.array;
            var wm = hemig.attributes.wm.array;
            var wmstride = hemig.attributes.wm.itemSize;
            var pareas = hemig.attributes.pialarea.array;
            var wareas = hemig.attributes.wmarea.array;

            var a = hit.a, b = hit.b, c = hit.c, l0 = hit.l0, l1 = hit.l1, l2 = hit.l2;
            for (var k = 0; k < 3; k++) {
                var pv = l0*pia[a*3+k] + l1*pia[b*3+k] + l2*pia[c*3+k];
                var wv = l0*wm[a*wmstride+k] + l1*wm[b*wmstride+k] + l2*wm[c*wmstride+k];
                pialpos[i*3+k] = pialpos[(n+i)*3+k] = pv;
                wmpos[i*3+k] = wmpos[(n+i)*3+k] = wv;
            }
            // These are the viewer's own smoothed per-vertex areas, not the
            // raw triangle areas laminar.py uses. Interpolating them keeps the
            // profile's equivolume depths identical to the ones the surface
            // shader draws at the same value of the depth slider.
            var pa = l0*pareas[a] + l1*pareas[b] + l2*pareas[c];
            var wa = l0*wareas[a] + l1*wareas[b] + l2*wareas[c];
            pialarea[i] = pialarea[n+i] = pa;
            wmarea[i] = wmarea[n+i] = wa;
            valid[i] = valid[n+i] = 1;
        }

        attrs.pialpos.needsUpdate = true;
        attrs.wmpos.needsUpdate = true;
        attrs.pialarea.needsUpdate = true;
        attrs.wmarea.needsUpdate = true;
        attrs.valid.needsUpdate = true;
        return nvalid > 0;
    };

    /*** rendering ***********************************************************/

    module.Profile.prototype._ensureShader = function() {
        var dv = this.viewer.active;
        var key = [dv.uuid, this._equivolume, dv.filter, dv.data.length, dv.data[0].raw].join("|");
        if (this._shaderkey === key && this._shader !== null)
            return this._shader;

        if (this._shader !== null)
            this._shader.dispose();

        this._shader = dv.getShader(Shaders.laminar, {}, {
            equivolume: this._equivolume,
            lights: false,
            depthTest: false,
            depthWrite: false,
            transparent: true,
            // Render straight into an empty target: no blending keeps the
            // colors un-premultiplied so they can be copied out as-is.
            blending: THREE.NoBlending,
        })[0];
        this._shaderkey = key;
        this.mesh.material = this._shader;
        return this._shader;
    };

    module.Profile.prototype._showMessage = function(msg) {
        if (msg)
            this.message.text(msg).css("display", "flex");
        else
            this.message.css("display", "none");
    };

    /* Returns false while something we are waiting on hasn't arrived yet, so
     * the caller knows to try again on the next frame rather than treating the
     * current state as final. */
    module.Profile.prototype.render = function() {
        var viewer = this.viewer, dv = viewer.active;
        var surf = this._surface();

        if (surf === null || surf.loaded.state() !== "resolved") {
            this._showMessage("Loading surfaces\u2026");
            return false;
        }
        if (!this._ensureIndex()) {
            this._showMessage("Depth profiles need a flatmap and pial/white matter surfaces.");
            return true;
        }
        if (!dv || dv.vertex) {
            this._showMessage("Depth profiles need a volume dataset.");
            return true;
        }
        if (!mriview.dataBuffersReady(dv.data))
            return false;
        if (this.endpoints === null) {
            this._showMessage("Couldn't place a profile line on this flatmap.");
            return true;
        }
        if (!this._computeColumns()) {
            this._showMessage("The profile line is off the flatmap.");
            return true;
        }
        this._showMessage(null);
        this._ensureShader();

        var W = this._width, H = this._height;
        if (this._target === null || this._target.width !== W || this._target.height !== H) {
            if (this._target !== null)
                this._target.dispose();
            this._target = new THREE.WebGLRenderTarget(W, H, {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat,
                stencilBuffer: false,
                depthBuffer: false,
                generateMipmaps: false,
            });
        }

        var renderer = viewer.renderer;
        var clearColor = renderer.getClearColor().clone();
        var clearAlpha = renderer.getClearAlpha();
        renderer.enableScissorTest(false);
        renderer.setClearColor(new THREE.Color(0, 0, 0), 0);
        renderer.render(this.scene, this.camera, this._target, true);
        renderer.setClearColor(clearColor, clearAlpha);

        this._blit();
        this._updateReadout();
        return true;
    };

    /* Copy the render target into the panel's 2D canvas, flipping the rows
     * (GL is bottom-up) so the pial surface ends up on top. */
    module.Profile.prototype._blit = function() {
        var gl = this.viewer.renderer.context;
        var W = this._target.width, H = this._target.height;
        var nbytes = W * H * 4;
        if (this._pixels === null || this._pixels.length !== nbytes)
            this._pixels = new Uint8Array(nbytes);

        gl.bindFramebuffer(gl.FRAMEBUFFER, this._target.__webglFramebuffer);
        gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, this._pixels);
        // Go back through the renderer rather than binding null directly: it
        // caches which framebuffer is bound (and the viewport that goes with
        // it), so a raw unbind here would leave it convinced our target is
        // still current and send the next render to the wrong buffer.
        this.viewer.renderer.setRenderTarget(null);

        var canvas = this.canvas[0];
        if (canvas.width !== W || canvas.height !== H) {
            canvas.width = W;
            canvas.height = H;
        }
        var ctx = canvas.getContext("2d");
        var img = ctx.createImageData(W, H);
        var rowbytes = W * 4;
        for (var row = 0; row < H; row++)
            img.data.set(this._pixels.subarray((H-1-row)*rowbytes, (H-row)*rowbytes), row*rowbytes);
        ctx.putImageData(img, 0, 0);
    };

    module.Profile.prototype._updateReadout = function() {
        if (this.endpoints === null)
            return;
        if (this._surface() === null)
            return;
        var parts = [];
        for (var i = 0; i < 2; i++) {
            var uv = this._rawUV(this.endpoints[i].x, this.endpoints[i].y);
            parts.push((i === 0 ? "A" : "B") + " " +
                       this.endpoints[i].hemi.charAt(0).toUpperCase() + "H " +
                       "(" + uv.u.toFixed(1) + ", " + uv.v.toFixed(1) + ")");
        }
        this.readout.text(parts.join("  \u2022  ") + "  \u2022  " +
                          this._width + "\u00d7" + this._height);
    };

    /*** per-frame update ****************************************************/

    module.Profile.prototype._signature = function() {
        var dv = this.viewer.active;
        var sig = [this._enabled, this._equivolume, this._width, this._height];
        if (this.endpoints !== null) {
            for (var i = 0; i < 2; i++)
                sig.push(this.endpoints[i].hemi, this.endpoints[i].x, this.endpoints[i].y);
        } else {
            sig.push(null);
        }
        if (dv) {
            sig.push(dv.uuid, dv.frame, dv.filter, dv.cmap[0].value,
                     dv.vmin[0].value[0], dv.vmin[0].value[1],
                     dv.vmax[0].value[0], dv.vmax[0].value[1],
                     dv.uniforms.framemix ? dv.uniforms.framemix.value : 0,
                     dv.uniforms.data ? dv.uniforms.data.value[0] : null);
        }
        return sig;
    };

    function _sigequal(a, b) {
        if (a === null || b === null || a.length !== b.length)
            return false;
        for (var i = 0; i < a.length; i++)
            if (a[i] !== b[i])
                return false;
        return true;
    }

    module.Profile.prototype._ondraw = function() {
        if (this._enabled && this.endpoints === null)
            this.resetLine();      // no-op until the surfaces have loaded

        this._updateOverlay();
        if (!this._enabled)
            return;

        this._updateLineTexture();

        var sig = this._signature();
        if (!this._dirty && _sigequal(sig, this._sig))
            return;
        if (this.render()) {
            this._sig = sig;
            this._dirty = false;
        }
    };

    module.Profile.prototype._onresize = function(evt) {
        this.overlay.attr({width:evt.width, height:evt.height})
            .css({width:evt.width + "px", height:evt.height + "px"});
    };

    /*** the drag handles ****************************************************
     *
     * The line is painted on the surface (above); this screen-space overlay
     * only carries what has to be grabbable, and only while the brain is
     * flattened -- there is no sensible way to drag a flatmap coordinate
     * around on a folded brain.
     *************************************************************************/

    var SVGNS = "http://www.w3.org/2000/svg";

    module.Profile.prototype._buildOverlay = function() {
        var svg = this.overlay[0];
        // Invisible: it exists to be grabbed, not seen.
        var hit = document.createElementNS(SVGNS, "line");
        hit.setAttribute("class", "laminar-hit");
        svg.appendChild(hit);
        this._svghit = hit;

        this._svghandles = [];
        for (var i = 0; i < 2; i++) {
            var g = document.createElementNS(SVGNS, "g");
            g.setAttribute("class", "laminar-handle");
            var circle = document.createElementNS(SVGNS, "circle");
            circle.setAttribute("r", 8);
            g.appendChild(circle);
            var label = document.createElementNS(SVGNS, "text");
            label.setAttribute("dy", -12);
            label.setAttribute("text-anchor", "middle");
            label.textContent = i === 0 ? "A" : "B";
            g.appendChild(label);
            svg.appendChild(g);
            this._svghandles.push({group:g, circle:circle, label:label});
            g.addEventListener("mousedown", this._startDrag.bind(this, i), true);
        }
        hit.addEventListener("mousedown", this._startDrag.bind(this, "line"), true);
    };

    module.Profile.prototype._updateOverlay = function() {
        var show = this._enabled && this.endpoints !== null && this._isFlat() &&
                   this._mesh("left") !== null;
        if (!show) {
            this.overlay.hide();
            return;
        }

        var screen = [];
        for (var i = 0; i < 2; i++) {
            var w = this._toWorld(this.endpoints[i], new THREE.Vector3());
            if (w === null) {
                this.overlay.hide();
                return;
            }
            var p = w.project(this.viewer.camera);
            screen.push([
                (p.x * 0.5 + 0.5) * this.viewer.width,
                (-p.y * 0.5 + 0.5) * this.viewer.height,
            ]);
        }

        this._svghit.setAttribute("x1", screen[0][0]);
        this._svghit.setAttribute("y1", screen[0][1]);
        this._svghit.setAttribute("x2", screen[1][0]);
        this._svghit.setAttribute("y2", screen[1][1]);
        for (var i = 0; i < 2; i++) {
            this._svghandles[i].circle.setAttribute("cx", screen[i][0]);
            this._svghandles[i].circle.setAttribute("cy", screen[i][1]);
            this._svghandles[i].label.setAttribute("x", screen[i][0]);
            this._svghandles[i].label.setAttribute("y", screen[i][1]);
        }
        this.overlay.show();
    };

    /* Intersect the mouse ray with the plane the flatmap lives in, in `hemi`'s
     * local frame. Returns {u, v} whether or not the point is on the flatmap. */
    module.Profile.prototype._mouseToPlane = function(evt, hemi) {
        var mesh = this._mesh(hemi);
        if (mesh === null)
            return null;

        var canvas = this.viewer.canvas[0];
        var rect = canvas.getBoundingClientRect();
        var nx = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
        var ny = -((evt.clientY - rect.top) / rect.height) * 2 + 1;

        var camera = this.viewer.camera;
        var far = new THREE.Vector3(nx, ny, 0.5).unproject(camera);
        var inv = new THREE.Matrix4().getInverse(mesh.matrixWorld);
        var o = camera.position.clone().applyMatrix4(inv);
        var d = far.applyMatrix4(inv).sub(o);
        if (Math.abs(d.x) < 1e-12)
            return null;

        var t = -o.x / d.x;
        var p = o.add(d.multiplyScalar(t));
        var uv = this._toShared(hemi, p.y, p.z);
        return {x:uv.x, y:uv.y, t:t};
    };

    /* Same, but only accepts points that actually land on the flatmap; picks
     * whichever hemisphere is hit first. */
    module.Profile.prototype._mouseToFlat = function(evt) {
        if (!this._ensureIndex())
            return null;
        var best = null;
        for (var i = 0; i < HEMIS.length; i++) {
            var hit = this._mouseToPlane(evt, HEMIS[i]);
            if (hit === null || hit.t <= 0)
                continue;
            if (this._index[HEMIS[i]].locate(hit.x, hit.y) === null)
                continue;
            if (best === null || hit.t < best.t)
                best = {hemi:HEMIS[i], x:hit.x, y:hit.y, t:hit.t};
        }
        return best;
    };

    module.Profile.prototype._startDrag = function(which, evt) {
        if (!this._enabled || !this._isFlat())
            return;
        evt.preventDefault();
        evt.stopPropagation();

        this._drag = {which:which, prev:evt};
        this._dragmove = this._onDragMove.bind(this);
        this._dragup = this._onDragUp.bind(this);
        window.addEventListener("mousemove", this._dragmove, true);
        window.addEventListener("mouseup", this._dragup, true);
        this.overlay.addClass("dragging");
    };

    module.Profile.prototype._onDragMove = function(evt) {
        if (this._drag === null)
            return;
        evt.preventDefault();
        evt.stopPropagation();

        if (this._drag.which === "line") {
            // Translate both ends by the mouse delta, measured in each end's
            // own hemisphere frame (the two frames are mirrored).
            var moved = [];
            for (var i = 0; i < 2; i++) {
                var ep = this.endpoints[i];
                var was = this._mouseToPlane(this._drag.prev, ep.hemi);
                var now = this._mouseToPlane(evt, ep.hemi);
                if (was === null || now === null)
                    return;
                var cand = {hemi:ep.hemi, x:ep.x + (now.x - was.x), y:ep.y + (now.y - was.y)};
                if (this._index[ep.hemi].locate(cand.x, cand.y) === null)
                    return;    // would drag an end off the flatmap; ignore
                moved.push(cand);
            }
            this.endpoints = moved;
        } else {
            var hit = this._mouseToFlat(evt);
            if (hit === null)
                return;
            this.endpoints[this._drag.which] = {hemi:hit.hemi, x:hit.x, y:hit.y};
        }

        this._drag.prev = evt;
        this._dirty = true;
        this.viewer.schedule();
    };

    module.Profile.prototype._onDragUp = function(evt) {
        if (this._drag === null)
            return;
        evt.preventDefault();
        evt.stopPropagation();
        window.removeEventListener("mousemove", this._dragmove, true);
        window.removeEventListener("mouseup", this._dragup, true);
        this._drag = null;
        this.overlay.removeClass("dragging");
    };

    /*** panel chrome ********************************************************/

    /* jQuery UI's draggable writes left/top, which fights the right/bottom
     * anchoring the panel starts out with. Pin it down once, the first time it
     * becomes visible and has a measurable size. */
    module.Profile.prototype._placePanel = function() {
        if (this._placed)
            return;
        this._placed = true;
        var pos = this.panel.position();
        this.panel.css({left:pos.left, top:pos.top, right:"auto", bottom:"auto"});
        if ($.fn.draggable !== undefined)
            this.panel.draggable({handle:"#laminar_header", containment:"parent"});
    };

    module.Profile.prototype._bindUI = function() {
        this.panel.find("#laminar_close").click(function() {
            this.viewer.ui.set("depth profile.show", false);
        }.bind(this));

        this.panel.find("#laminar_flip").click(function() {
            if (this.endpoints !== null) {
                this.endpoints = [this.endpoints[1], this.endpoints[0]];
                this._dirty = true;
                this.viewer.schedule();
            }
        }.bind(this));

    };

    return module;
}(laminar || {}));
