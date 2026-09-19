//Manual aligner: moves the anatomical surfaces in the space of a functional
//reference volume, the way the old mayavi aligner did. The volume stays on
//its own voxel grid, so its slices are displayed without resampling, and the
//pial and white matter surfaces are cut off at the displayed slices, which
//draws their outline on top of the anatomy in the image. A second view mode
//paints the volume onto the surface instead, for judging the alignment from
//the pattern the data makes on the cortex.
//
//The scene lives in the "world" frame handed over by cortex/webgl/aligner.py:
//the voxel grid of the reference volume in millimeters, with the axes
//permuted and flipped so that x, y and z point right, anterior and superior.
//The transform being edited is the pycortex "coord" transform (anatomical
//millimeters to voxel indices); in the scene it appears as the world matrix
//of the surface object, world <- anatomical = config.world * coord.
var aligner = (function(module) {

    //One slice view per world axis. `look` is the viewing direction and `up`
    //the screen-up direction. The coronal and axial views follow the
    //radiological convention (the subject's right on the left of the screen,
    //as in cortex.volume.mosaic); the sagittal view is seen from the right.
    var VIEWS = {
        x: {axis: 0, title: "sagittal", look: [-1, 0, 0], up: [0, 0, 1]},
        y: {axis: 1, title: "coronal",  look: [ 0,-1, 0], up: [0, 0, 1]},
        z: {axis: 2, title: "axial",    look: [ 0, 0, 1], up: [0, 1, 0]},
    };

    //Placement of the four views in the 2x2 grid, matching the mayavi aligner
    var LAYOUT = [
        {name: "y",  col: 0, row: 0},
        {name: "z",  col: 0, row: 1},
        {name: "x",  col: 1, row: 0},
        {name: "3d", col: 1, row: 1},
    ];

    var MODES = {outline: "mesh + slices", projected: "data on surface"};
    module.MODES = MODES;

    //Slice setters, one per world axis, so that the slice controls in the
    //menu and the views stay in sync
    var SLICE_SETTERS = ["setSagittal", "setCoronal", "setAxial"];

    var UNDO_LIMIT = 200;
    //Wheel movement (in normalized pixels) that steps one slice
    var WHEEL_STEP = 30;

    //Nested 4x4 array in row-major order to a THREE.Matrix4, and back
    module.matrixFromRows = function(rows) {
        var m = new THREE.Matrix4();
        m.set(rows[0][0], rows[0][1], rows[0][2], rows[0][3],
              rows[1][0], rows[1][1], rows[1][2], rows[1][3],
              rows[2][0], rows[2][1], rows[2][2], rows[2][3],
              rows[3][0], rows[3][1], rows[3][2], rows[3][3]);
        return m;
    };
    module.matrixToRows = function(m) {
        var e = m.elements, rows = [];
        for (var i = 0; i < 4; i++) {
            var row = [];
            for (var j = 0; j < 4; j++)
                row.push(e[i + 4 * j]);
            rows.push(row);
        }
        return rows;
    };

    //Wheel deltas are in pixels, lines or pages depending on the browser and
    //the device; normalize them all to pixels (as in movement.js)
    function wheelDelta(event) {
        var delta = event.deltaY;
        if (event.deltaMode == 1)
            delta *= 18;
        else if (event.deltaMode == 2)
            delta *= 180;
        return delta;
    }

    //Resolves once every image has decoded, so that the colormap textures
    //and the list of 1D colormaps can be built from them
    function whenImagesLoaded(images) {
        var waiting = [];
        for (var i = 0; i < images.length; i++) {
            var img = images[i];
            if (img.complete && img.naturalHeight > 0)
                continue;
            var deferred = $.Deferred();
            img.addEventListener("load", deferred.resolve);
            img.addEventListener("error", deferred.resolve);
            waiting.push(deferred);
        }
        return $.when.apply($, waiting);
    }

    //Unique triangle edges of an indexed BufferGeometry, as a line index
    //buffer with the same chunking (offsets) as the triangle index, so that
    //the mesh can be drawn as a wireframe. three.js r69 draws wireframes of
    //indexed BufferGeometries by reinterpreting the triangle index as line
    //pairs, which does not give the triangle edges.
    module.buildEdges = function(geometry) {
        var index = geometry.attributes.index.array;
        var offsets = geometry.offsets;
        var edges = new Uint16Array(index.length * 2);
        var edgeOffsets = [];
        var total = 0;
        for (var j = 0; j < offsets.length; j++) {
            var start = offsets[j].start, count = offsets[j].count;
            var seen = new Set();
            var chunkStart = total;
            var add = function(a, b) {
                var key = a < b ? a * 65536 + b : b * 65536 + a;
                if (seen.has(key))
                    return;
                seen.add(key);
                edges[total++] = a;
                edges[total++] = b;
            };
            for (var i = start, il = start + count; i < il; i += 3) {
                add(index[i], index[i+1]);
                add(index[i+1], index[i+2]);
                add(index[i+2], index[i]);
            }
            edgeOffsets.push({start: chunkStart, count: total - chunkStart, index: offsets[j].index});
        }
        var edgeGeom = new THREE.BufferGeometry();
        edgeGeom.addAttribute("index", new THREE.BufferAttribute(edges.subarray(0, total), 2));
        edgeGeom.addAttribute("position", geometry.attributes.position);
        edgeGeom.addAttribute("wm", geometry.attributes.wm);
        edgeGeom.offsets = edgeOffsets;
        return edgeGeom;
    };

    module.Aligner = function(figure, config) {
        jsplot.Axes.call(this, figure);
        this.config = config;
        this.loaded = $.Deferred();
        this.ready = false;
        this._ready = {volume: false, mesh: false};
        this._scheduled = false;
        this._draw = this.draw.bind(this);
        this.nframes = 0;

        $(this.object).html($("#aligner_html").html());
        this.canvas = $(this.object).find("#aligner-canvas");

        //Frames: voxel -> world, and anatomical -> world for the surfaces
        this.world = module.matrixFromRows(config.world);
        this.worldInv = new THREE.Matrix4().getInverse(this.world);
        this.xfm = new THREE.Matrix4().multiplyMatrices(this.world, module.matrixFromRows(config.xfm));
        this._undo = [];

        //Voxel dimensions in voxel axis order, and for each world axis the
        //voxel axis it is drawn from and the voxel size along it
        var shape = config.volume.shape;
        this.dims = [shape[2], shape[1], shape[0]];
        this.vax = [0, 0, 0];
        this.vscale = [1, 1, 1];
        var e = this.world.elements;
        for (var a = 0; a < 3; a++) {
            var best = -1;
            for (var j = 0; j < 3; j++) {
                var v = Math.abs(e[a + 4 * j]);
                if (v > best) {
                    best = v;
                    this.vax[a] = j;
                }
            }
            this.vscale[a] = best;
        }
        //The cursor (voxel coordinates) picks the displayed slices and is the
        //pivot for rotations
        this.cursor = [(this.dims[0] - 1) / 2, (this.dims[1] - 1) / 2, (this.dims[2] - 1) / 2];
        this.planeCoord = [0, 0, 0];

        this._mode = MODES.outline;
        this._cmapName = config.cmap;
        this._showSurf = [true, true];
        this._translateStep = 1;
        this._rotateStep = 1;
        this.hasWM = true;
        this.hoverView = null;
        this._drag = null;
        this._wheelAcc = 0;

        //Renderer and scene. Objects are drawn in scene order (planes, then
        //the surfaces, then the crosshairs): the slice views draw their plane
        //without a depth test so that the surface outline always shows on top
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas[0],
            antialias: true,
            preserveDrawingBuffer: true,
            alpha: false,
        });
        this.renderer.setClearColor(new THREE.Color(0x000000), 1);
        this.renderer.sortObjects = false;

        this.scene = new THREE.Scene();
        this.light = new THREE.DirectionalLight(0xffffff, 1.0);
        this.scene.add(this.light);
        this.scene.add(this.light.target);

        this.planeGroup = new THREE.Object3D();
        this.brain = new THREE.Object3D();
        this.brain.matrixAutoUpdate = false;
        this.brain.matrix.copy(this.xfm);
        this.brain.matrixWorldNeedsUpdate = true;
        this.crossGroup = new THREE.Object3D();
        this.scene.add(this.planeGroup);
        this.scene.add(this.brain);
        this.scene.add(this.crossGroup);

        this.camera3d = new THREE.PerspectiveCamera(45, 1, 1, 4000);
        this.camera3d.up.set(0, 0, 1);
        this.controls = new jsplot.LandscapeControls();
        this.controls.addEventListener("change", this.schedule.bind(this));

        this.views = {};
        this.viewlist = [];
        for (var i = 0; i < LAYOUT.length; i++) {
            var name = LAYOUT[i].name;
            var div = $(this.object).find("#view-" + name)[0];
            var view = {
                name: name,
                div: div,
                label: $(div).find(".aligner-label")[0],
                col: LAYOUT[i].col,
                row: LAYOUT[i].row,
                rect: {left: 0, top: 0, width: 1, height: 1},
                is2d: name != "3d",
            };
            if (view.is2d) {
                var spec = VIEWS[name];
                view.axis = spec.axis;
                view.title = spec.title;
                view.look = new THREE.Vector3().fromArray(spec.look);
                view.up = new THREE.Vector3().fromArray(spec.up);
                view.right = new THREE.Vector3().crossVectors(view.look, view.up);
                view.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 4000);
                view.center = new THREE.Vector3();
                view.height = 200;
                view.fitted = false;
            } else {
                view.camera = this.camera3d;
            }
            this.views[name] = view;
            this.viewlist.push(view);
            this._bindView(view);
        }
        window.addEventListener("mousemove", this._onMouseMove.bind(this), false);
        window.addEventListener("mouseup", this._onMouseUp.bind(this), false);
        window.addEventListener("keydown", this._onKeyDown.bind(this), true);

        //Uniforms shared by the planes and the painted surface
        this.volUniforms = {
            data:       {type:'tv',  value:[null, null, null, null]},
            mosaic:     {type:'v2v', value:[new THREE.Vector2(1, 1), new THREE.Vector2(1, 1)]},
            dshape:     {type:'v2v', value:[new THREE.Vector2(1, 1), new THREE.Vector2(1, 1)]},
            nslices:    {type:'f',   value:1},
            volxfm:     {type:'m4',  value:this.worldInv},
            colormap:   {type:'t',   value:null},
            vmin:       {type:'f',   value:config.vmin},
            vmax:       {type:'f',   value:config.vmax},
            brightness: {type:'f',   value:0},
            contrast:   {type:'f',   value:1},
            gamma:      {type:'f',   value:1},
            flip:       {type:'i',   value:0},
            outside:    {type:'v3',  value:new THREE.Vector3(.25, .25, .25)},
        };
        //Uniforms shared by the surface outlines
        this.meshUniforms = {
            color:    {type:'c',  value:new THREE.Color(config.mesh_color)},
            opacity:  {type:'f',  value:config.mesh_opacity},
            slabLo:   {type:'v3', value:new THREE.Vector3()},
            slabHi:   {type:'v3', value:new THREE.Vector3()},
            slabMask: {type:'v3', value:new THREE.Vector3(1, 1, 1)},
        };
        this.planeMaterials = {
            flat:  this._makeVolumeMaterial({lights: false, depthTest: false}),
            solid: this._makeVolumeMaterial({lights: false, depthTest: true}),
        };
        this.projectedMaterial = this._makeVolumeMaterial({lights: true, depthmix: true});
        this.outlineMaterials = [this._makeMeshMaterial(0, true), this._makeMeshMaterial(1, true)];
        this.surfaceMaterials = [this._makeMeshMaterial(0, false), this._makeMeshMaterial(1, false)];

        this.planes2d = [];
        this.planes3d = [];
        this.planeGeoms = [];
        this.crosshairs = [];
        this.hemis = [];

        //Control panel on the right, holding the menu, the save status and
        //the list of controls
        this.ui = new jsplot.Menu();
        this.ui.addEventListener("update", this.schedule.bind(this));
        this._setupPanel(figure);

        //Colormap textures, and the menu once the colormap images are ready
        //(the 1D colormaps are told apart from the 2D ones by their height)
        this.colormaps = {};
        var images = $(this.object).find(".cmap img").toArray();
        whenImagesLoaded(images).done(function() {
            for (var i = 0; i < images.length; i++) {
                var img = images[i];
                var tex = new THREE.Texture(img);
                tex.minFilter = THREE.LinearFilter;
                tex.magFilter = THREE.LinearFilter;
                tex.flipY = true;
                tex.needsUpdate = true;
                this.colormaps[img.parentNode.id] = tex;
                if (typeof(colormaps) !== "undefined")
                    colormaps[img.parentNode.id] = tex;
            }
            this._buildUI();
        }.bind(this));

        //The reference volume, through the viewer's mosaic loader
        this.volume = new dataset.VolumeData(config.volume, config.images);
        this.volume.loaded.done(this._volumeReady.bind(this));

        //The surfaces, from the CTM pack the viewer uses: the base positions
        //are the pial surface and the `wm` attribute the white matter
        var loader = new THREE.CTMLoader(false);
        loader.loadParts(config.ctm, function(geometries, materials, json) {
            this._meshReady(geometries);
        }.bind(this), {useWorker: true});
    };
    module.Aligner.prototype = Object.create(jsplot.Axes.prototype);
    module.Aligner.prototype.constructor = module.Aligner;

    module.Aligner.prototype._setupPanel = function(figure) {
        var wrapper = document.createElement("div");
        wrapper.innerHTML = $("#aligner_panel_html").html();
        var panel = wrapper.querySelector("#aligner-panel");
        $(panel).find("#aligner-controls")[0].appendChild(figure.ui_element);
        if (this.config.view_only)
            $(panel).find("#aligner-viewonly").show();
        this.statusElement = $(panel).find("#aligner-status");
        figure.gui.open();
        try {
            figure.gui.width = 284;
        } catch (e) {}
        figure.w2obj.show("right", true);
        figure.setSize("right", 300);
        figure.w2obj.content("right", panel);
    };

    module.Aligner.prototype.showStatus = function(message, error) {
        this.statusElement.text(message);
        this.statusElement.toggleClass("error", !!error);
    };

    module.Aligner.prototype._makeVolumeMaterial = function(opts) {
        var shaders = Shaders.aligner_volume({
            sampler: "nearest",
            lights: opts.lights,
            depthmix: opts.depthmix,
        });
        var uniforms = THREE.UniformsUtils.merge([
            THREE.UniformsLib["lights"],
            {
                diffuse:    {type:'v3', value:new THREE.Vector3(.7, .7, .7)},
                specular:   {type:'v3', value:new THREE.Vector3(0, 0, 0)},
                emissive:   {type:'v3', value:new THREE.Vector3(.35, .35, .35)},
                shininess:  {type:'f',  value:1},
                specularStrength: {type:'f', value:0},
                depth:      {type:'f',  value:0.5},
            }
        ]);
        for (var name in this.volUniforms)
            uniforms[name] = this.volUniforms[name];
        var depth = opts.depthTest !== false;
        return new THREE.ShaderMaterial({
            vertexShader: shaders.vertex,
            fragmentShader: shaders.fragment,
            uniforms: uniforms,
            attributes: shaders.attrs,
            lights: !!opts.lights,
            side: opts.depthmix ? THREE.FrontSide : THREE.DoubleSide,
            depthTest: depth,
            depthWrite: depth,
        });
    };

    module.Aligner.prototype._makeMeshMaterial = function(depth, clip) {
        var shaders = Shaders.aligner_mesh({});
        var uniforms = {
            color:    this.meshUniforms.color,
            slabLo:   this.meshUniforms.slabLo,
            slabHi:   this.meshUniforms.slabHi,
            slabMask: this.meshUniforms.slabMask,
            depth:    {type:'f', value:depth},
            doClip:   {type:'i', value:clip ? 1 : 0},
            opacity:  clip ? {type:'f', value:1} : this.meshUniforms.opacity,
        };
        return new THREE.ShaderMaterial({
            vertexShader: shaders.vertex,
            fragmentShader: shaders.fragment,
            uniforms: uniforms,
            attributes: shaders.attrs,
            transparent: !clip,
            depthWrite: clip,
            side: THREE.DoubleSide,
        });
    };

    //-------------------------------------------------------------------------
    // Loading
    //-------------------------------------------------------------------------
    module.Aligner.prototype._volumeReady = function() {
        var vol = this.volume;
        this.volUniforms.data.value[0] = vol.textures[0];
        for (var i = 0; i < 2; i++) {
            this.volUniforms.mosaic.value[i].set(vol.mosaic[0], vol.mosaic[1]);
            this.volUniforms.dshape.value[i].set(vol.shape[0], vol.shape[1]);
        }
        this.volUniforms.nslices.value = vol.numslices;

        //World bounding box of the volume
        var min = new THREE.Vector3(Infinity, Infinity, Infinity);
        var max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
        for (var c = 0; c < 8; c++) {
            var corner = new THREE.Vector3(
                (c & 1) ? this.dims[0] - 0.5 : -0.5,
                (c & 2) ? this.dims[1] - 0.5 : -0.5,
                (c & 4) ? this.dims[2] - 0.5 : -0.5).applyMatrix4(this.world);
            min.min(corner);
            max.max(corner);
        }
        this.bbox = {min: min, max: max};
        this.bboxCenter = min.clone().add(max).multiplyScalar(0.5);

        this._buildPlanes();
        this._updateSliceGeometry();

        this.controls.setTarget(this.bboxCenter.toArray());
        this.controls.setRadius(1.1 * min.distanceTo(max));

        this._ready.volume = true;
        this._checkReady();
    };

    module.Aligner.prototype._buildPlanes = function() {
        var cross_material = new THREE.LineBasicMaterial({color: 0x44ccff, depthTest: false, depthWrite: false});
        var makeQuad = function() {
            var geom = new THREE.Geometry();
            for (var i = 0; i < 4; i++)
                geom.vertices.push(new THREE.Vector3());
            geom.faces.push(new THREE.Face3(0, 1, 2), new THREE.Face3(2, 1, 3));
            geom.dynamic = true;
            return geom;
        };
        for (var a = 0; a < 3; a++) {
            //One quad for the slice views (drawn without depth test) and one
            //for the 3D view (drawn with it), so that each mesh owns its
            //geometry buffers
            var geoms = [makeQuad(), makeQuad()];
            this.planeGeoms.push(geoms);

            var flat = new THREE.Mesh(geoms[0], this.planeMaterials.flat);
            flat.frustumCulled = false;
            this.planes2d.push(flat);
            this.planeGroup.add(flat);

            var solid = new THREE.Mesh(geoms[1], this.planeMaterials.solid);
            solid.frustumCulled = false;
            this.planes3d.push(solid);
            this.planeGroup.add(solid);

            var cross = new THREE.Geometry();
            for (var i = 0; i < 4; i++)
                cross.vertices.push(new THREE.Vector3());
            cross.dynamic = true;
            var lines = new THREE.Line(cross, cross_material, THREE.LinePieces);
            lines.frustumCulled = false;
            this.crosshairs.push(lines);
            this.crossGroup.add(lines);
        }
    };

    module.Aligner.prototype._meshReady = function(geometries) {
        for (var i = 0; i < geometries.length; i++) {
            var geom = geometries[i];
            if (geom.attributes.wm === undefined) {
                //Only a fiducial surface: draw it once, as both surfaces
                this.hasWM = false;
                geom.addAttribute("wm", geom.attributes.position);
            }
            geom.addAttribute("wmnorm", mriview.computeNormal(geom.attributes.wm, geom.attributes.index, geom.offsets));
            var edges = module.buildEdges(geom);

            var hemi = {geometry: geom, edges: edges, outlines: [], surfaces: []};
            for (var s = 0; s < 2; s++) {
                var line = new THREE.Line(edges, this.outlineMaterials[s], THREE.LinePieces);
                line.frustumCulled = false;
                hemi.outlines.push(line);
                this.brain.add(line);

                var surf = new THREE.Mesh(geom, this.surfaceMaterials[s]);
                surf.frustumCulled = false;
                hemi.surfaces.push(surf);
                this.brain.add(surf);
            }
            hemi.projected = new THREE.Mesh(geom, this.projectedMaterial);
            hemi.projected.frustumCulled = false;
            this.brain.add(hemi.projected);
            this.hemis.push(hemi);
        }
        this._ready.mesh = true;
        this._checkReady();
    };

    module.Aligner.prototype._checkReady = function() {
        if (!this._ready.volume || !this._ready.mesh)
            return;
        $("#dataload").hide();
        $(this.object).find("#aligner-load").hide();
        this.ready = true;
        this.resize();
        this.schedule();
        this.loaded.resolve();
    };

    module.Aligner.prototype._buildUI = function() {
        var names = [];
        for (var name in this.colormaps) {
            if (this.colormaps[name].image.height == 1)
                names.push(name);
        }
        names.sort();
        if (names.indexOf(this._cmapName) < 0)
            this._cmapName = names.indexOf("gray") >= 0 ? "gray" : names[0];
        this.volUniforms.colormap.value = this.colormaps[this._cmapName];

        var top = {};
        if (!this.config.view_only)
            top.save = {action: this.save.bind(this)};
        top.undo = {action: this.undo.bind(this)};
        top.view = {action: [this, "setMode", [MODES.outline, MODES.projected]]};
        this.ui.add(top);

        var vol = this.config.volume;
        this.ui.addFolder("image", false).add({
            colormap:   {action: [this, "setColormap", names]},
            flip:       {action: [this, "setFlip"]},
            vmin:       {action: [this, "setVmin", vol.min, vol.max]},
            vmax:       {action: [this, "setVmax", vol.min, vol.max]},
            brightness: {action: [this, "setBrightness", -1, 1, 0.01]},
            contrast:   {action: [this, "setContrast", 0, 4, 0.01]},
            gamma:      {action: [this, "setGamma", 0.1, 4, 0.01]},
        });
        this.ui.addFolder("mesh", false).add({
            color:   {action: [this, "setMeshColor"], color: true},
            opacity: {action: [this, "setMeshOpacity", 0, 1, 0.01]},
            pial:    {action: [this, "setShowPial"]},
            white:   {action: [this, "setShowWhite"]},
            depth:   {action: [this, "setDepth", 0, 1, 0.01]},
        });

        var slices = {};
        var titles = ["sagittal", "coronal", "axial"];
        for (var a = 0; a < 3; a++)
            slices[titles[a]] = {action: [this, SLICE_SETTERS[a], 0, this.dims[this.vax[a]] - 1, 1]};
        this.ui.addFolder("slices", false).add(slices);

        this.ui.addFolder("steps", true).add({
            "translate (mm)": {action: [this, "setTranslateStep", 0.05, 10, 0.05]},
            "rotate (deg)":   {action: [this, "setRotateStep", 0.05, 10, 0.05]},
        });
        this.schedule();
    };

    //Updates the value shown by a menu control without running its action
    module.Aligner.prototype._syncControl = function(folder, name, value) {
        var menu = this.ui[folder];
        if (menu === undefined || menu._controls[name] === undefined)
            return;
        menu[name] = value;
        menu._controls[name].updateDisplay();
    };

    //-------------------------------------------------------------------------
    // Slices and cursor
    //-------------------------------------------------------------------------
    module.Aligner.prototype.getSlice = function(axis) {
        var va = this.vax[axis];
        return Math.min(Math.max(Math.round(this.cursor[va]), 0), this.dims[va] - 1);
    };
    module.Aligner.prototype._setSlice = function(axis, slice) {
        var va = this.vax[axis];
        this.cursor[va] = Math.min(Math.max(Math.round(slice), 0), this.dims[va] - 1);
        this._cursorChanged();
    };
    module.Aligner.prototype.setSagittal = function(slice) {
        if (slice === undefined)
            return this.getSlice(0);
        this._setSlice(0, slice);
    };
    module.Aligner.prototype.setCoronal = function(slice) {
        if (slice === undefined)
            return this.getSlice(1);
        this._setSlice(1, slice);
    };
    module.Aligner.prototype.setAxial = function(slice) {
        if (slice === undefined)
            return this.getSlice(2);
        this._setSlice(2, slice);
    };

    //The cursor in voxel coordinates of the reference volume
    module.Aligner.prototype.getCursor = function() {
        return this.cursor.slice();
    };
    module.Aligner.prototype.setCursor = function(voxel) {
        for (var i = 0; i < 3; i++)
            this.cursor[i] = Math.min(Math.max(voxel[i], -0.5), this.dims[i] - 0.5);
        this._cursorChanged();
    };
    //The cursor in world coordinates
    module.Aligner.prototype.cursorWorld = function() {
        return new THREE.Vector3().fromArray(this.cursor).applyMatrix4(this.world);
    };

    module.Aligner.prototype._cursorChanged = function() {
        var titles = ["sagittal", "coronal", "axial"];
        for (var a = 0; a < 3; a++)
            this._syncControl("slices", titles[a], this.getSlice(a));
        if (this._ready.volume)
            this._updateSliceGeometry();
        this.schedule();
    };

    //Moves the slice planes to the cursor's slices, the slabs that cut the
    //surfaces to half a voxel around them, and the crosshairs to the cursor
    module.Aligner.prototype._updateSliceGeometry = function() {
        var e = this.world.elements;
        var cw = this.cursorWorld();
        for (var a = 0; a < 3; a++) {
            var va = this.vax[a];
            var slice = this.getSlice(a);
            var b = (a + 1) % 3, c = (a + 2) % 3;
            var vb = this.vax[b], vc = this.vax[c];

            var lo = new THREE.Vector3(Infinity, Infinity, Infinity);
            var hi = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
            for (var i = 0; i < 4; i++) {
                var voxel = new THREE.Vector3();
                voxel.setComponent(va, slice);
                voxel.setComponent(vb, (i & 1) ? this.dims[vb] - 0.5 : -0.5);
                voxel.setComponent(vc, (i & 2) ? this.dims[vc] - 0.5 : -0.5);
                voxel.applyMatrix4(this.world);
                lo.min(voxel);
                hi.max(voxel);
                for (var g = 0; g < 2; g++)
                    this.planeGeoms[a][g].vertices[i].copy(voxel);
            }
            for (var g = 0; g < 2; g++) {
                var geom = this.planeGeoms[a][g];
                geom.verticesNeedUpdate = true;
                geom.computeFaceNormals();
                geom.computeVertexNormals();
                geom.normalsNeedUpdate = true;
            }

            var coord = e[a + 4 * va] * slice + e[a + 12];
            this.planeCoord[a] = coord;
            this.meshUniforms.slabLo.value.setComponent(a, coord - 0.5 * this.vscale[a]);
            this.meshUniforms.slabHi.value.setComponent(a, coord + 0.5 * this.vscale[a]);

            //Crosshair: one line along each in-plane axis through the cursor
            var cross = this.crosshairs[a].geometry;
            for (var i = 0; i < 4; i++) {
                cross.vertices[i].copy(cw);
                cross.vertices[i].setComponent(a, coord);
            }
            cross.vertices[0].setComponent(b, lo.getComponent(b));
            cross.vertices[1].setComponent(b, hi.getComponent(b));
            cross.vertices[2].setComponent(c, lo.getComponent(c));
            cross.vertices[3].setComponent(c, hi.getComponent(c));
            cross.verticesNeedUpdate = true;

            var view = this.views["xyz"[a]];
            view.label.textContent = view.title + "  " + (slice + 1) + " / " + this.dims[va];
        }
    };

    //-------------------------------------------------------------------------
    // The transform
    //-------------------------------------------------------------------------
    //The pycortex coord transform (anatomical mm -> voxel indices) as a nested
    //4x4 array, and its setter
    module.Aligner.prototype.getXfm = function() {
        var coord = new THREE.Matrix4().multiplyMatrices(this.worldInv, this.xfm);
        return module.matrixToRows(coord);
    };
    module.Aligner.prototype.setXfm = function(rows) {
        this.pushUndo();
        this.xfm.multiplyMatrices(this.world, module.matrixFromRows(rows));
        this._xfmChanged();
    };
    module.Aligner.prototype.pushUndo = function() {
        this._undo.push(this.xfm.clone());
        if (this._undo.length > UNDO_LIMIT)
            this._undo.shift();
    };
    module.Aligner.prototype.undo = function() {
        if (this._undo.length == 0) {
            this.showStatus("nothing to undo");
            return;
        }
        this.xfm.copy(this._undo.pop());
        this._xfmChanged();
    };
    module.Aligner.prototype._xfmChanged = function() {
        this.brain.matrix.copy(this.xfm);
        this.brain.matrixWorldNeedsUpdate = true;
        this.dispatchEvent({type: "xfm"});
        this.schedule();
    };

    //Translates the surfaces by a world vector (mm)
    module.Aligner.prototype.translate = function(vector) {
        this.pushUndo();
        this._translate(new THREE.Vector3().fromArray(vector));
    };
    module.Aligner.prototype._translate = function(vector) {
        var trans = new THREE.Matrix4().makeTranslation(vector.x, vector.y, vector.z);
        this.xfm.multiplyMatrices(trans, this.xfm);
        this._xfmChanged();
    };
    //Rotates the surfaces by `angle` degrees about a world axis through the
    //cursor (or through `pivot`, a world point)
    module.Aligner.prototype.rotate = function(axis, angle, pivot) {
        this.pushUndo();
        this._rotate(new THREE.Vector3().fromArray(axis), angle * Math.PI / 180,
                     pivot === undefined ? this.cursorWorld() : new THREE.Vector3().fromArray(pivot));
    };
    module.Aligner.prototype._rotate = function(axis, radians, pivot) {
        var rot = new THREE.Matrix4().makeRotationAxis(axis.clone().normalize(), radians);
        var toPivot = new THREE.Matrix4().makeTranslation(pivot.x, pivot.y, pivot.z);
        var fromPivot = new THREE.Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z);
        var xfm = new THREE.Matrix4().multiplyMatrices(fromPivot, this.xfm);
        xfm.multiplyMatrices(rot, xfm);
        this.xfm.multiplyMatrices(toPivot, xfm);
        this._xfmChanged();
    };

    module.Aligner.prototype.save = function() {
        if (this.config.view_only) {
            this.showStatus("view only: the transform is not saved", true);
            return "view only";
        }
        this.showStatus("saving...");
        $.ajax({
            type: "POST",
            url: "save",
            data: {xfm: JSON.stringify(this.getXfm())},
            dataType: "json",
        }).done(function(resp) {
            this.showStatus(resp.message, resp.status != "ok");
        }.bind(this)).fail(function() {
            this.showStatus("saving failed: no answer from the server", true);
        }.bind(this));
        return "saving";
    };

    //The canvas as drawn last, as a PNG data url. Drawing is left to the
    //scheduled animation frames, so that a slow (software) render cannot hold
    //up the reply to the python side; getFrames tells when a frame landed.
    module.Aligner.prototype.snapshot = function() {
        return this.renderer.domElement.toDataURL("image/png");
    };
    //Number of frames drawn so far
    module.Aligner.prototype.getFrames = function() {
        return this.nframes;
    };
    //Runs a method for the python side and tags the result with the token
    //of the request. The websocket protocol pairs replies with requests by
    //their order and gives up on a reply after two seconds, so replies held
    //up by a busy page (parsing the surfaces, a slow frame) would otherwise
    //be taken for those of later requests.
    //The reply also carries the number of frames drawn so far and whether a
    //redraw is pending, so that the python side can wait for the frame that
    //shows the effect of the call before taking a snapshot.
    module.Aligner.prototype.call = function(token, name, args) {
        var reply = {token: token, frames: this.nframes};
        if (!(this[name] instanceof Function)) {
            reply.error = "no method " + name;
        } else {
            try {
                reply.value = this[name].apply(this, args);
            } catch (e) {
                reply.error = e.message;
            }
        }
        reply.scheduled = this._scheduled;
        return reply;
    };
    //A menu control by its dotted path, for instance image.vmin or mesh.color
    module.Aligner.prototype.getControl = function(name) {
        return this.ui.get(name);
    };
    module.Aligner.prototype.setControl = function(name, value) {
        this.ui.set(name, value);
    };

    //-------------------------------------------------------------------------
    // Display settings
    //-------------------------------------------------------------------------
    module.Aligner.prototype.setMode = function(mode) {
        if (mode === undefined)
            return this._mode;
        this._mode = mode;
        this.schedule();
    };
    module.Aligner.prototype.toggleMode = function() {
        this.setMode(this._mode == MODES.outline ? MODES.projected : MODES.outline);
    };
    module.Aligner.prototype.setColormap = function(name) {
        if (name === undefined)
            return this._cmapName;
        if (this.colormaps[name] === undefined)
            return;
        this._cmapName = name;
        this.volUniforms.colormap.value = this.colormaps[name];
        this.schedule();
    };
    module.Aligner.prototype.setFlip = function(flip) {
        if (flip === undefined)
            return this.volUniforms.flip.value == 1;
        this.volUniforms.flip.value = flip ? 1 : 0;
        this.schedule();
    };
    module.Aligner.prototype.setVmin = function(value) {
        if (value === undefined)
            return this.volUniforms.vmin.value;
        this.volUniforms.vmin.value = value;
        this.schedule();
    };
    module.Aligner.prototype.setVmax = function(value) {
        if (value === undefined)
            return this.volUniforms.vmax.value;
        this.volUniforms.vmax.value = value;
        this.schedule();
    };
    module.Aligner.prototype.setBrightness = function(value) {
        if (value === undefined)
            return this.volUniforms.brightness.value;
        this.volUniforms.brightness.value = value;
        this.schedule();
    };
    module.Aligner.prototype.setContrast = function(value) {
        if (value === undefined)
            return this.volUniforms.contrast.value;
        this.volUniforms.contrast.value = value;
        this.schedule();
    };
    module.Aligner.prototype.setGamma = function(value) {
        if (value === undefined)
            return this.volUniforms.gamma.value;
        this.volUniforms.gamma.value = value;
        this.schedule();
    };
    module.Aligner.prototype.setMeshColor = function(color) {
        if (color === undefined)
            return "#" + this.meshUniforms.color.value.getHexString();
        this.meshUniforms.color.value.set(color);
        this.schedule();
    };
    //Opacity of the whole surfaces in the 3D view (0 hides them, leaving
    //the outlines on the slices)
    module.Aligner.prototype.setMeshOpacity = function(value) {
        if (value === undefined)
            return this.meshUniforms.opacity.value;
        this.meshUniforms.opacity.value = value;
        this.schedule();
    };
    module.Aligner.prototype.setShowPial = function(show) {
        if (show === undefined)
            return this._showSurf[0];
        this._showSurf[0] = show;
        this.schedule();
    };
    module.Aligner.prototype.setShowWhite = function(show) {
        if (show === undefined)
            return this._showSurf[1];
        this._showSurf[1] = show;
        this.schedule();
    };
    //Cortical depth the data is painted at, from pial (0) to white matter (1)
    module.Aligner.prototype.setDepth = function(value) {
        if (value === undefined)
            return this.projectedMaterial.uniforms.depth.value;
        this.projectedMaterial.uniforms.depth.value = value;
        this.schedule();
    };
    module.Aligner.prototype.setTranslateStep = function(value) {
        if (value === undefined)
            return this._translateStep;
        this._translateStep = value;
    };
    module.Aligner.prototype.setRotateStep = function(value) {
        if (value === undefined)
            return this._rotateStep;
        this._rotateStep = value;
    };

    //-------------------------------------------------------------------------
    // Drawing
    //-------------------------------------------------------------------------
    module.Aligner.prototype.resize = function() {
        var w = $(this.object).width(), h = $(this.object).height();
        if (!w || !h)
            return;
        this.width = w;
        this.height = h;
        this.renderer.setSize(w, h);
        for (var i = 0; i < this.viewlist.length; i++) {
            var view = this.viewlist[i];
            view.rect = {
                left: Math.floor(view.col * w / 2),
                top: Math.floor(view.row * h / 2),
                width: Math.floor(w / 2),
                height: Math.floor(h / 2),
            };
        }
        this.schedule();
    };

    module.Aligner.prototype.schedule = function() {
        if (!this._scheduled) {
            this._scheduled = true;
            requestAnimationFrame(this._draw);
        }
    };

    module.Aligner.prototype.draw = function() {
        this._scheduled = false;
        if (!this.ready || !this.width || !this.height)
            return;
        this.controls.update(this.camera3d);
        this.renderer.enableScissorTest(true);
        for (var i = 0; i < this.viewlist.length; i++) {
            var view = this.viewlist[i], r = view.rect;
            if (r.width < 1 || r.height < 1)
                continue;
            var bottom = this.height - r.top - r.height;
            this.renderer.setViewport(r.left, bottom, r.width, r.height);
            this.renderer.setScissor(r.left, bottom, r.width, r.height);
            this._prepareView(view);
            if (view.is2d && this._mode == MODES.outline) {
                //Two passes: the slice first, then the outlines and the
                //crosshair on top of it whatever their depth. Drawing them in
                //one pass would leave the order to three.js, which draws its
                //list of opaque objects back to front through the scene.
                this._showLayer(view, "plane");
                this.renderer.render(this.scene, view.camera);
                this._showLayer(view, "lines");
                this.renderer.autoClear = false;
                this.renderer.render(this.scene, view.camera);
                this.renderer.autoClear = true;
            } else {
                this._showLayer(view, "all");
                this.renderer.render(this.scene, view.camera);
            }
        }
        this.renderer.enableScissorTest(false);
        this.nframes++;
        this.dispatchEvent({type: "draw"});
    };

    //Sets the visibility of the objects for a view: a slice view shows its
    //plane, the outline of the surfaces cut to that slice and the crosshair;
    //the 3D view shows all three planes, the outlines on all of them and,
    //when opaque enough, the whole surfaces. In the painted mode every view
    //shows the surfaces colored by the volume instead. `layer` restricts the
    //visible objects to the "plane" or the "lines" of a slice view.
    module.Aligner.prototype._showLayer = function(view, layer) {
        var outline = this._mode == MODES.outline;
        var plane = layer != "lines", lines = layer != "plane";
        for (var a = 0; a < 3; a++) {
            this.planes2d[a].visible = plane && outline && view.is2d && view.axis == a;
            this.planes3d[a].visible = plane && outline && !view.is2d;
            this.crosshairs[a].visible = lines && outline && view.is2d && view.axis == a;
        }
        var surfaces = outline && !view.is2d && this.meshUniforms.opacity.value > 0;
        for (var i = 0; i < this.hemis.length; i++) {
            var hemi = this.hemis[i];
            for (var s = 0; s < 2; s++) {
                var shown = this._showSurf[s] && (s == 0 || this.hasWM);
                hemi.outlines[s].visible = lines && outline && shown;
                hemi.surfaces[s].visible = lines && surfaces && shown;
            }
            hemi.projected.visible = lines && !outline;
        }
    };

    //Sets the slabs that cut the surfaces, the camera and the light of a view
    module.Aligner.prototype._prepareView = function(view) {
        var mask = this.meshUniforms.slabMask.value;
        if (view.is2d)
            mask.set(view.axis == 0 ? 1 : 0, view.axis == 1 ? 1 : 0, view.axis == 2 ? 1 : 0);
        else
            mask.set(1, 1, 1);

        if (view.is2d) {
            if (!view.fitted)
                this._fitView(view);
            this._updateOrthoCamera(view);
            this.light.position.copy(view.camera.position);
            this.light.target.position.copy(view.center);
        } else {
            this.camera3d.aspect = view.rect.width / view.rect.height;
            this.camera3d.updateProjectionMatrix();
            this.light.position.copy(this.camera3d.position);
            this.light.target.position.copy(this.controls.target);
        }
    };

    //Frames the whole volume in a slice view
    module.Aligner.prototype._fitView = function(view) {
        var size = this.bbox.max.clone().sub(this.bbox.min);
        var upAxis = this._majorAxis(view.up), rightAxis = this._majorAxis(view.right);
        var aspect = view.rect.width / view.rect.height;
        view.height = 1.1 * Math.max(size.getComponent(upAxis), size.getComponent(rightAxis) / aspect);
        view.center.copy(this.bboxCenter);
        view.fitted = true;
    };
    module.Aligner.prototype._majorAxis = function(vector) {
        var arr = vector.toArray(), best = 0;
        for (var i = 1; i < 3; i++)
            if (Math.abs(arr[i]) > Math.abs(arr[best]))
                best = i;
        return best;
    };

    module.Aligner.prototype._updateOrthoCamera = function(view) {
        var cam = view.camera;
        var aspect = view.rect.width / view.rect.height;
        var h = view.height, w = h * aspect;
        cam.left = -w / 2;
        cam.right = w / 2;
        cam.top = h / 2;
        cam.bottom = -h / 2;
        cam.updateProjectionMatrix();
        cam.up.copy(view.up);
        cam.position.copy(view.center).sub(view.look.clone().multiplyScalar(2000));
        cam.lookAt(view.center);
        cam.updateMatrixWorld();
    };

    //-------------------------------------------------------------------------
    // Interaction
    //-------------------------------------------------------------------------
    module.Aligner.prototype._bindView = function(view) {
        view.div.addEventListener("mouseenter", function() {
            this.hoverView = view;
        }.bind(this), false);
        view.div.addEventListener("contextmenu", function(event) {
            event.preventDefault();
        }, false);
        if (!view.is2d) {
            this.controls.bind(view.div);
            return;
        }
        view.div.addEventListener("mousedown", this._onMouseDown.bind(this, view), false);
        view.div.addEventListener("wheel", this._onWheel.bind(this, view), {passive: false});
    };

    module.Aligner.prototype._localPos = function(view, event) {
        var r = view.div.getBoundingClientRect();
        return {x: event.clientX - r.left, y: event.clientY - r.top};
    };

    //The world point under the mouse in a slice view, on the slice plane
    module.Aligner.prototype._mouseWorld = function(view, pos) {
        this._updateOrthoCamera(view);
        var nx = (pos.x / view.rect.width) * 2 - 1;
        var ny = 1 - (pos.y / view.rect.height) * 2;
        var point = new THREE.Vector3(nx, ny, 0).unproject(view.camera);
        point.setComponent(view.axis, this.planeCoord[view.axis]);
        return point;
    };
    module.Aligner.prototype._worldToScreen = function(view, point) {
        this._updateOrthoCamera(view);
        var ndc = point.clone().project(view.camera);
        return {x: (ndc.x + 1) / 2 * view.rect.width, y: (1 - ndc.y) / 2 * view.rect.height};
    };
    //Millimeters per screen pixel in a slice view
    module.Aligner.prototype._mmPerPixel = function(view) {
        return view.height / view.rect.height;
    };

    //Moves the cursor, in the plane of a slice view, to the mouse position
    module.Aligner.prototype._setCursorFromMouse = function(view, pos) {
        var voxel = this._mouseWorld(view, pos).applyMatrix4(this.worldInv);
        for (var a = 0; a < 3; a++) {
            if (a == view.axis)
                continue;
            var va = this.vax[a];
            this.cursor[va] = Math.min(Math.max(voxel.getComponent(va), -0.5), this.dims[va] - 0.5);
        }
        this._cursorChanged();
    };

    module.Aligner.prototype._onMouseDown = function(view, event) {
        event.preventDefault();
        this.hoverView = view;
        var pos = this._localPos(view, event);
        var state;
        if (event.button === 0 && event.shiftKey) {
            state = "pan";
        } else if (event.button === 0) {
            state = "cursor";
            this._setCursorFromMouse(view, pos);
        } else if (event.button === 1) {
            state = "pan";
        } else if (event.button === 2) {
            state = (event.ctrlKey || event.altKey || event.metaKey) ? "rotate" : "translate";
            this.pushUndo();
        } else {
            return;
        }
        this._drag = {view: view, state: state, last: pos};
        if (state == "rotate") {
            this._drag.pivot = this._worldToScreen(view, this.cursorWorld());
            this._drag.angle = Math.atan2(-(pos.y - this._drag.pivot.y), pos.x - this._drag.pivot.x);
        }
    };

    module.Aligner.prototype._onMouseMove = function(event) {
        var drag = this._drag;
        if (drag === null)
            return;
        event.preventDefault();
        var view = drag.view;
        var pos = this._localPos(view, event);
        var dx = pos.x - drag.last.x, dy = pos.y - drag.last.y;
        var scale = this._mmPerPixel(view);
        if (drag.state == "cursor") {
            this._setCursorFromMouse(view, pos);
        } else if (drag.state == "pan") {
            view.center.sub(view.right.clone().multiplyScalar(dx * scale));
            view.center.add(view.up.clone().multiplyScalar(dy * scale));
        } else if (drag.state == "translate") {
            var move = view.right.clone().multiplyScalar(dx * scale);
            move.sub(view.up.clone().multiplyScalar(dy * scale));
            this._translate(move);
        } else if (drag.state == "rotate") {
            var angle = Math.atan2(-(pos.y - drag.pivot.y), pos.x - drag.pivot.x);
            var delta = angle - drag.angle;
            if (delta > Math.PI)
                delta -= 2 * Math.PI;
            else if (delta < -Math.PI)
                delta += 2 * Math.PI;
            drag.angle = angle;
            //counterclockwise on the screen is a positive rotation about the
            //axis pointing at the viewer
            this._rotate(view.look.clone().negate(), delta, this.cursorWorld());
        }
        drag.last = pos;
        this.schedule();
    };

    module.Aligner.prototype._onMouseUp = function(event) {
        this._drag = null;
    };

    //Wheel: steps through the slices; with ctrl (or a trackpad pinch) zooms
    //the view about the mouse position
    module.Aligner.prototype._onWheel = function(view, event) {
        event.preventDefault();
        var delta = wheelDelta(event);
        if (event.ctrlKey || event.metaKey) {
            var pos = this._localPos(view, event);
            var before = this._mouseWorld(view, pos);
            var factor = Math.exp(Math.min(Math.max(delta, -100), 100) * 0.005);
            view.height *= factor;
            //keep the point under the mouse where it is
            view.center.sub(before).multiplyScalar(factor).add(before);
        } else {
            this._wheelAcc += delta;
            while (Math.abs(this._wheelAcc) >= WHEEL_STEP) {
                var step = this._wheelAcc > 0 ? 1 : -1;
                this._setSlice(view.axis, this.getSlice(view.axis) + step);
                this._wheelAcc -= step * WHEEL_STEP;
            }
        }
        this.schedule();
    };

    module.Aligner.prototype._onKeyDown = function(event) {
        var tag = event.target.tagName;
        if (tag == "INPUT" || tag == "TEXTAREA" || tag == "SELECT")
            return;
        var key = event.key;
        if ((event.ctrlKey || event.metaKey) && (key == "z" || key == "Z")) {
            this.undo();
            event.preventDefault();
            return;
        }
        if (event.ctrlKey || event.metaKey || event.altKey)
            return;
        if (key == "m" || key == "M") {
            this.toggleMode();
            event.preventDefault();
            return;
        }

        var view = this.hoverView;
        if (view === null || !view.is2d)
            return;
        var fine = event.shiftKey ? 0.1 : 1;
        var tstep = this._translateStep * fine;
        var rstep = this._rotateStep * fine * Math.PI / 180;
        var toViewer = view.look.clone().negate();
        var handled = true;
        switch (key) {
            case "ArrowLeft":
                this.translate(view.right.clone().multiplyScalar(-tstep).toArray());
                break;
            case "ArrowRight":
                this.translate(view.right.clone().multiplyScalar(tstep).toArray());
                break;
            case "ArrowUp":
                this.translate(view.up.clone().multiplyScalar(tstep).toArray());
                break;
            case "ArrowDown":
                this.translate(view.up.clone().multiplyScalar(-tstep).toArray());
                break;
            case "q": case "Q":
                this.pushUndo();
                this._rotate(toViewer, rstep, this.cursorWorld());
                break;
            case "e": case "E":
                this.pushUndo();
                this._rotate(toViewer, -rstep, this.cursorWorld());
                break;
            case "[": case "{":
                this._setSlice(view.axis, this.getSlice(view.axis) - 1);
                break;
            case "]": case "}":
                this._setSlice(view.axis, this.getSlice(view.axis) + 1);
                break;
            default:
                handled = false;
        }
        if (handled)
            event.preventDefault();
    };

    return module;
}(aligner || {}));
