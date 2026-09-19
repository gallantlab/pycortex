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

    //What the page shows. The first two keep the three slice views and differ
    //in the fourth panel: the slice planes in space, or the surface with the
    //reference data painted on it through the alignment being edited. The
    //third gives that surface the whole window, where it inflates and
    //flattens the way the viewer's does.
    var DISPLAY = {
        slices:  "3 ortho + 3D slices",
        brain:   "3 ortho + 3D brain",
        surface: "data on the surface",
    };
    var DISPLAY_ORDER = [DISPLAY.slices, DISPLAY.brain, DISPLAY.surface];
    module.DISPLAY = DISPLAY;

    //What one view draws: the surfaces outlined on the slices, the data on
    //the surfaces where they sit in the volume, or the data on the surfaces
    //in the anatomy's own frame, where they inflate and flatten
    var OUTLINE = 0, PROJECTED = 1, MORPHED = 2;

    //Millimeters of the flatmap per unit of its 2D coordinates, as in
    //mriview_surface.js, so that a flattened surface comes out brain-sized
    var FLATSCALE = 0.3;

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
        //Everything the alignment has been through since the page opened: the
        //one it was loaded with, then an entry per edit holding what the edit
        //was and the alignment it left behind. Picking one puts that
        //alignment back.
        this._history = [{kind: "loaded", xfm: this.xfm.clone(), label: ""}];
        this._historyIndex = 0;
        this._historyRows = [];
        this._restoring = false;
        //the point whose movement measures an edit, until the surfaces load
        this._meshCenter = new THREE.Vector3();

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

        this._display = DISPLAY.slices;
        this._mix = 0;
        this._pivot = 0;
        this._culled = false;
        this._framedFor = null;
        //The transform the page saves to. It starts as the one being edited
        //and can be changed, which saves the alignment as a new transform.
        this._xfmName = config.xfmname;
        this._savedXfm = this.xfm.clone();
        this._savedName = this._xfmName;
        this._dirtyShown = false;
        this._saveState = null;
        this._title = document.title;
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
        //The data view holds the surfaces in the anatomy's own frame, so that
        //they keep their place while the alignment is edited and while they
        //inflate; the volume is sampled through the alignment instead.
        this.surfGroup = new THREE.Object3D();
        this.scene.add(this.planeGroup);
        this.scene.add(this.brain);
        this.scene.add(this.crossGroup);
        this.scene.add(this.surfGroup);

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
            //so that the keys move the mesh before the mouse has been over
            //any view, rather than doing nothing for no visible reason
            if (view.is2d && this.hoverView === null)
                this.hoverView = view;
        }
        window.addEventListener("mousemove", this._onMouseMove.bind(this), false);
        window.addEventListener("mouseup", this._onMouseUp.bind(this), false);
        window.addEventListener("keydown", this._onKeyDown.bind(this), true);
        window.addEventListener("beforeunload", this._onBeforeUnload.bind(this), false);

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
            this._meshReady(geometries, json);
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
        //The masks of this transform were cut through the alignment being
        //edited, so saving deletes them
        var masks = this.config.masks || [];
        if (masks.length > 0 && !this.config.view_only) {
            $(panel).find("#aligner-masks").text(
                "Saving over " + this.config.xfmname + " deletes its " + masks.length +
                " cached mask" + (masks.length == 1 ? "" : "s") + " (" + masks.join(", ") +
                "). Data already masked with them has to be masked again from the " +
                "volumes. Saving under another name leaves them alone.").show();
        }
        this.statusElement = $(panel).find("#aligner-status");
        this.historyElement = $(panel).find("#aligner-history-list")[0];
        this._renderHistory();
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
            morphs: opts.morphs,
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

    module.Aligner.prototype._meshReady = function(geometries, json) {
        //The surfaces the data view morphs between: the anatomical surface,
        //whatever the CTM pack carries (the inflated one) and the flatmap.
        //Only a surface every hemisphere has can be morphed to.
        var names = [];
        var packed = (json && json.names) || [];
        for (var k = 0; k < packed.length; k++) {
            var everywhere = geometries.length > 0;
            for (var i = 0; i < geometries.length; i++)
                everywhere = everywhere && geometries[i].attributes[packed[k]] !== undefined;
            if (everywhere)
                names.push(packed[k]);
        }
        var hasFlat = !!(json && json.flatlims) && geometries.length > 1 &&
                      geometries[0].attributes.uv !== undefined;
        this.surfNames = ["anatomical"].concat(names);
        if (hasFlat)
            this.surfNames.push("flat");

        for (var i = 0; i < geometries.length; i++)
            geometries[i].computeBoundingBox();
        //The halves of the flatmap unfold from either side of this offset,
        //as mriview lays them out
        var offx = 0, offy = Infinity;
        for (var i = 0; i < geometries.length; i++) {
            var box = geometries[i].boundingBox;
            offx = Math.max(offx, Math.abs(box.min.x), Math.abs(box.max.x));
            offy = Math.min(offy, box.min.y);
        }
        this.flatoff = [offx / 3, offy];
        this._buildMorphMaterial(this.surfNames.length);

        for (var i = 0; i < geometries.length; i++) {
            var geom = geometries[i];
            if (geom.attributes.wm === undefined) {
                //Only a fiducial surface: draw it once, as both surfaces
                this.hasWM = false;
                geom.addAttribute("wm", geom.attributes.position);
            }
            geom.addAttribute("wmnorm", mriview.computeNormal(geom.attributes.wm, geom.attributes.index, geom.offsets));
            var edges = module.buildEdges(geom);

            var hemi = {geometry: geom, edges: edges, bounds: geom.boundingBox,
                        outlines: [], surfaces: []};
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

            this._buildMorph(hemi, i, names, hasFlat);
            this.hemis.push(hemi);
        }
        this._updateMorphXfm();
        //an edit is measured by how far this point travels
        if (this.hemis.length > 0) {
            var box = new THREE.Box3();
            for (var i = 0; i < this.hemis.length; i++)
                box.union(this.hemis[i].bounds);
            this._meshCenter = box.center();
        }
        this._ready.mesh = true;
        this._checkReady();
    };

    //The material of the data view. It paints the volume on the surfaces
    //where the alignment being edited puts them in it, while they are drawn
    //in the anatomy's own frame, so that editing the alignment moves the
    //colors over a surface that stays where it is.
    module.Aligner.prototype._buildMorphMaterial = function(morphs) {
        this.morphUniforms = {surfmix: {type:'f', value: 0}};
        var material = this._makeVolumeMaterial({lights: true, depthmix: true, morphs: morphs});
        material.uniforms.volxfm = {type:'m4', value: new THREE.Matrix4()};
        material.uniforms.surfmix = this.morphUniforms.surfmix;
        //one cortical depth for both ways of painting the data
        material.uniforms.depth = this.projectedMaterial.uniforms.depth;
        this.morphMaterial = material;
    };

    //The morphing copy of one hemisphere, hung from the pair of pivots that
    //swing the flatmap open, as mriview does it
    module.Aligner.prototype._buildMorph = function(hemi, index, names, hasFlat) {
        var geom = hemi.geometry;
        //the shader reads the surfaces as mixSurfs<i>, and needs a normal
        //for each of them
        for (var k = 0; k < names.length; k++) {
            geom.attributes["mixSurfs" + k] = geom.attributes[names[k]];
            geom.addAttribute("mixNorms" + k, mriview.computeNormal(
                geom.attributes[names[k]], geom.attributes.index, geom.offsets));
            delete geom.attributes[names[k]];
        }
        if (hasFlat) {
            var flat = this._makeFlat(geom.attributes.uv.array, index == 1);
            geom.addAttribute("mixSurfs" + names.length, new THREE.BufferAttribute(flat.pos, 4));
            geom.addAttribute("mixNorms" + names.length, new THREE.BufferAttribute(flat.norms, 3));
            //the medial wall is not part of the flatmap, so its polygons are
            //dropped while the surface is flattened
            var culled = mriview._cull_flatmap_vertices(
                geom.attributes.index.array, geom.attributes.auxdat.array, geom.offsets);
            hemi.culled = {index: new THREE.BufferAttribute(culled.indices, 3), offsets: culled.offsets};
            hemi.full = {index: geom.attributes.index, offsets: geom.offsets};
        }

        var box = hemi.bounds;
        var pivots = {back: new THREE.Object3D(), front: new THREE.Object3D()};
        pivots.front.add(pivots.back);
        pivots.back.position.y = box.min.y - box.max.y;
        pivots.front.position.y = box.max.y - box.min.y + this.flatoff[1];
        var mesh = new THREE.Mesh(geom, this.morphMaterial);
        mesh.position.y = -this.flatoff[1];
        mesh.frustumCulled = false;
        mesh.visible = false;
        pivots.back.add(mesh);
        this.surfGroup.add(pivots.front);
        hemi.pivots = pivots;
        hemi.morph = mesh;
    };

    //The flatmap as a surface to morph to: its 2D coordinates laid in the
    //plane that faces the camera when the halves have swung open
    module.Aligner.prototype._makeFlat = function(uv, right) {
        var count = uv.length / 2;
        var flat = new Float32Array(count * 4);
        var norms = new Float32Array(count * 3);
        for (var i = 0; i < count; i++) {
            flat[i*4+1] = FLATSCALE * (right ? uv[i*2] : -uv[i*2]) + this.flatoff[1];
            flat[i*4+2] = FLATSCALE * uv[i*2+1];
            norms[i*3] = right ? 1 : -1;
        }
        return {pos: flat, norms: norms};
    };

    //The alignment as the data view uses it: anatomical millimeters to the
    //voxels of the reference volume, which is the pycortex coord transform
    module.Aligner.prototype._updateMorphXfm = function() {
        if (this.morphMaterial === undefined)
            return;
        this.morphMaterial.uniforms.volxfm.value.multiplyMatrices(this.worldInv, this.xfm);
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

        //the name field sits above the save button, so that the alignment can
        //be saved as a new transform
        var top = {};
        if (!this.config.view_only) {
            top.transform = {action: [this, "setXfmName"]};
            top.save = {action: this.save.bind(this)};
        }
        top.undo = {action: this.undo.bind(this)};
        top.display = {action: [this, "setDisplay", DISPLAY_ORDER]};
        this.ui.add(top);
        this._updateDirty();

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
            unfold:  {action: [this, "setMix", 0, 1, 0.01]},
            pivot:   {action: [this, "setPivot", -180, 180, 1]},
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
        this._previewColormaps();
        this.schedule();
    };

    //Draws a strip of each colormap beside its name in the colormap dropdown.
    //dat.GUI renders a plain select, so select2 takes it over to draw the
    //options, the same library the viewer's colormap picker uses. Picking one
    //writes the value into the select and fires a jQuery event; dat.GUI
    //listens for the native one, so the pick is passed on as such and the
    //control goes on working the way the others do.
    module.Aligner.prototype._previewColormaps = function() {
        var folder = this.ui["image"];
        var control = folder === undefined ? undefined : folder._controls.colormap;
        if (control === undefined || control.__select === undefined || $.fn.select2 === undefined)
            return;

        var select = control.__select;
        var colormaps = this.colormaps;
        var draw = function(state) {
            if (!state.id)
                return state.text;
            var texture = colormaps[state.id];
            var row = $("<span class='aligner-cmap'></span>");
            if (texture !== undefined && texture.image !== undefined)
                row.append($("<img>").attr("src", texture.image.src));
            row.append($("<span class='aligner-cmap-name'></span>").text(state.text));
            return row;
        };

        this._cmapSelect = $(select).select2({
            templateResult: draw,
            templateSelection: draw,
            width: "100%",
        });
        //The open dropdown hangs off the body rather than off the control,
        //and the width mriview.css gives it there reaches past the right
        //edge of the window, which scrolls the page sideways. This marks it
        //as the aligner's, for the width aligner.css gives it instead.
        var instance = this._cmapSelect.data("select2");
        if (instance != null && instance.$dropdown !== undefined) {
            //select2 hangs the dropdown in a container of its own when it
            //attaches it to the body, so the list itself is a level down
            var list = instance.$dropdown.hasClass("select2-dropdown")
                     ? instance.$dropdown : instance.$dropdown.find(".select2-dropdown");
            list.addClass("aligner-cmap-dropdown");
        }
        this._cmapSelect.on("select2:select", function() {
            select.dispatchEvent(new Event("change"));
        });
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
        this.beginEdit("set");
        this.xfm.multiplyMatrices(this.world, module.matrixFromRows(rows));
        this._xfmChanged();
    };

    //Opens an entry in the history for the edit that is about to be made. A
    //drag holds one open while it runs, so that the entry grows with it
    //rather than leaving one behind per frame. An edit made after stepping
    //back drops the entries that followed, as an editor does.
    module.Aligner.prototype.beginEdit = function(kind) {
        this._history.length = this._historyIndex + 1;
        this._history.push({kind: kind || "edit", xfm: this.xfm.clone(), label: ""});
        //the alignment the page opened with stays as the first entry, so it
        //is the oldest edit that gives way once the history is full
        while (this._history.length > UNDO_LIMIT)
            this._history.splice(1, 1);
        this._historyIndex = this._history.length - 1;
        this._describeEdit(this._historyIndex);
        this._renderHistory();
    };
    module.Aligner.prototype.undo = function() {
        if (this._historyIndex == 0) {
            this.showStatus("nothing to undo");
            return;
        }
        this.selectHistory(this._historyIndex - 1);
    };

    //The history, as one entry per edit: what the edit was and how far it
    //moved and turned the surfaces
    module.Aligner.prototype.getHistory = function() {
        var out = [];
        for (var i = 0; i < this._history.length; i++)
            out.push({kind: this._history[i].kind, label: this._history[i].label});
        return out;
    };
    //Puts back the alignment as it stood at one entry of the history
    module.Aligner.prototype.selectHistory = function(index) {
        if (index === undefined)
            return this._historyIndex;
        index = Math.min(Math.max(Math.round(index), 0), this._history.length - 1);
        this._historyIndex = index;
        //the entries are what was, so going back writes none of them
        this._restoring = true;
        this.xfm.copy(this._history[index].xfm);
        this._xfmChanged();
        this._restoring = false;
    };

    //A gesture that ended where it started left no edit, so it leaves no
    //entry either
    module.Aligner.prototype._dropEmptyEdit = function() {
        var i = this._historyIndex;
        if (i == 0 || i != this._history.length - 1)
            return;
        var now = this._history[i].xfm.elements, before = this._history[i - 1].xfm.elements;
        for (var k = 0; k < 16; k++) {
            if (now[k] !== before[k])
                return;
        }
        this._history.pop();
        this._historyIndex = i - 1;
        this._renderHistory();
    };

    //How far the surfaces moved and turned between two alignments. The
    //distance is the one their center covers, which is what a rotation about
    //a far-off pivot amounts to on screen; the world frame is millimeters.
    module.Aligner.prototype._describeEdit = function(index) {
        var entry = this._history[index];
        if (index == 0) {
            entry.label = "";
            return;
        }
        var from = this._history[index - 1].xfm, to = entry.xfm;
        var center = this._meshCenter;
        var moved = center.clone().applyMatrix4(to).distanceTo(center.clone().applyMatrix4(from));
        var relative = new THREE.Matrix4().multiplyMatrices(
            to, new THREE.Matrix4().getInverse(from));
        var e = relative.elements;
        var trace = Math.min(Math.max((e[0] + e[5] + e[10] - 1) / 2, -1), 1);
        var angle = Math.acos(trace) * 180 / Math.PI;
        var parts = [];
        if (moved >= 0.005)
            parts.push(moved.toFixed(2) + " mm");
        if (angle >= 0.005)
            parts.push(angle.toFixed(2) + "°");
        entry.label = parts.join(", ");
    };

    //Draws the history in the panel. Only the entry being edited changes
    //while a drag runs, so the rows are built again only when there are
    //different ones to show.
    module.Aligner.prototype._renderHistory = function() {
        var list = this.historyElement;
        if (list === undefined || list === null)
            return;
        if (this._historyRows.length != this._history.length) {
            list.innerHTML = "";
            this._historyRows = [];
            for (var i = 0; i < this._history.length; i++) {
                var row = document.createElement("li");
                row.appendChild(document.createElement("span")).className = "aligner-history-kind";
                row.appendChild(document.createElement("span")).className = "aligner-history-size";
                row.addEventListener("click", this.selectHistory.bind(this, i), false);
                list.appendChild(row);
                this._historyRows.push(row);
            }
        }
        for (var i = 0; i < this._history.length; i++) {
            var row = this._historyRows[i];
            var entry = this._history[i];
            var kind = i == 0 ? "loaded" : entry.kind;
            if (row.firstChild.textContent != kind)
                row.firstChild.textContent = kind;
            if (row.lastChild.textContent != entry.label)
                row.lastChild.textContent = entry.label;
            row.className = i == this._historyIndex ? "current" : "";
        }
        //keep the entry in view inside the list, rather than scrolling the
        //whole panel down to it and taking the controls off the screen
        if (this._historyRows.length > 0) {
            var row = this._historyRows[this._historyIndex];
            var top = row.offsetTop, bottom = top + row.offsetHeight;
            if (top < list.scrollTop)
                list.scrollTop = top;
            else if (bottom > list.scrollTop + list.clientHeight)
                list.scrollTop = bottom - list.clientHeight;
        }
    };

    module.Aligner.prototype._xfmChanged = function() {
        this.brain.matrix.copy(this.xfm);
        this.brain.matrixWorldNeedsUpdate = true;
        this._updateMorphXfm();
        //the open entry of the history follows the edit that is being made
        if (!this._restoring && this._historyIndex > 0) {
            this._history[this._historyIndex].xfm.copy(this.xfm);
            this._describeEdit(this._historyIndex);
        }
        this._renderHistory();
        this._updateDirty();
        this.dispatchEvent({type: "xfm"});
        this.schedule();
    };

    //The name of the transform the page saves to. Changing it saves the
    //alignment as a new transform, leaving the one it was loaded from alone.
    module.Aligner.prototype.setXfmName = function(name) {
        if (name === undefined)
            return this._xfmName;
        this._xfmName = String(name).trim();
        this._updateDirty();
    };

    //Whether the alignment on screen is the one that was last saved. An undo
    //back to that alignment counts as saved again, which is why this compares
    //the matrices rather than merely noting that something was moved.
    module.Aligner.prototype.isDirty = function() {
        if (this._xfmName !== this._savedName)
            return true;
        var now = this.xfm.elements, saved = this._savedXfm.elements;
        for (var i = 0; i < 16; i++) {
            if (now[i] !== saved[i])
                return true;
        }
        return false;
    };

    //Marks unsaved changes with an asterisk, on the save button and in the
    //title, so that a page left open does not look like a saved alignment.
    module.Aligner.prototype._updateDirty = function() {
        var dirty = this.isDirty();
        if (dirty === this._dirtyShown)
            return;
        this._dirtyShown = dirty;
        document.title = (dirty ? "* " : "") + this._title;
        var button = this.ui._controls.save;
        if (button !== undefined)
            button.name(dirty ? "save *" : "save");
    };

    //Translates the surfaces by a world vector (mm)
    module.Aligner.prototype.translate = function(vector) {
        this.beginEdit("translate");
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
        this.beginEdit("rotate");
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
            var refused = "view only: the transform is not saved";
            this.showStatus(refused, true);
            this._saveState = {status: "error", message: refused, name: this._xfmName};
            return "view only";
        }
        //what is being saved, rather than what is on screen when the answer
        //comes back: the alignment can be moved on while the request is out
        var name = this._xfmName;
        var saved = this.xfm.clone();
        this.showStatus("saving " + name + "...");
        //the state the python side reads to know whether the save landed,
        //since the request is answered long after this returns
        this._saveState = {status: "saving", message: "saving " + name + "...", name: name};
        $.ajax({
            type: "POST",
            url: "save",
            //the server only takes saves from the page it served, which this
            //token is what makes it
            data: {xfm: JSON.stringify(this.getXfm()), name: name, token: this.config.save_token},
            dataType: "json",
        }).done(function(resp) {
            this.showStatus(resp.message, resp.status != "ok");
            this._saveState = {status: resp.status, message: resp.message, name: name};
            if (resp.status == "ok") {
                this._savedXfm = saved;
                this._savedName = name;
                this._updateDirty();
            }
        }.bind(this)).fail(function(xhr) {
            //an error the server answered with says more than the request
            //having failed does
            var message = "saving failed: no answer from the server";
            if (xhr && xhr.responseJSON && xhr.responseJSON.message)
                message = xhr.responseJSON.message;
            this.showStatus(message, true);
            this._saveState = {status: "error", message: message, name: name};
        }.bind(this));
        return "saving";
    };
    //What became of the last save: "saving" while the request is out, then
    //"ok" or "error" with the message the server answered with
    module.Aligner.prototype.getSaveState = function() {
        return this._saveState;
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
    module.Aligner.prototype.setColormap = function(name) {
        if (name === undefined)
            return this._cmapName;
        if (this.colormaps[name] === undefined)
            return;
        this._cmapName = name;
        this.volUniforms.colormap.value = this.colormaps[name];
        //keep the dropdown showing the colormap that is in effect when the
        //change came from somewhere else, a menu set from python for instance
        if (this._cmapSelect !== undefined && this._cmapSelect.val() != name)
            this._cmapSelect.val(name).trigger("change.select2");
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
        var single = this._display == DISPLAY.surface;
        for (var i = 0; i < this.viewlist.length; i++) {
            var view = this.viewlist[i];
            if (single) {
                //the 3D view takes the window; draw() skips the empty ones
                view.rect = view.is2d ? {left: 0, top: 0, width: 0, height: 0}
                                      : {left: 0, top: 0, width: w, height: h};
            } else {
                view.rect = {
                    left: Math.floor(view.col * w / 2),
                    top: Math.floor(view.row * h / 2),
                    width: Math.floor(w / 2),
                    height: Math.floor(h / 2),
                };
            }
        }
        this.schedule();
    };

    //What the page shows: the three slice views with the planes in the fourth
    //panel, the same with the surface there instead, or that surface on its
    //own filling the window. The data view keeps every control, so the
    //reference data can be looked at on the surface, through the alignment
    //being edited, without saving it and opening the viewer.
    module.Aligner.prototype.setDisplay = function(name) {
        if (name === undefined)
            return this._display;
        if (DISPLAY_ORDER.indexOf(name) < 0)
            return;
        this._display = name;
        var single = name == DISPLAY.surface;
        $(this.object).find("#aligner").toggleClass("single", single);
        //the surfaces stand in the anatomy's own frame in the data view and
        //in the volume's in the other two, so the camera is pointed at
        //whichever of them is on screen
        this.controls.setMix(single ? this._flatness() : 0);
        this._frameFor(single ? "anatomical" : "world");
        this._applyIndex();
        this.resize();
    };
    module.Aligner.prototype.toggleDisplay = function() {
        var next = (DISPLAY_ORDER.indexOf(this._display) + 1) % DISPLAY_ORDER.length;
        this.setDisplay(DISPLAY_ORDER[next]);
    };

    //What one view draws
    module.Aligner.prototype._viewMode = function(view) {
        if (this._display == DISPLAY.surface)
            return MORPHED;
        if (this._display == DISPLAY.brain && !view.is2d)
            return PROJECTED;
        return OUTLINE;
    };

    //Points the camera at the surfaces in the frame they are drawn in, and
    //only when that frame changes, so that a view the user set up survives a
    //trip through the other displays
    module.Aligner.prototype._frameFor = function(frame) {
        if (this._framedFor == frame)
            return;
        this._framedFor = frame;
        this._frameSurface(frame == "world" ? this.xfm : null);
    };

    //Points the camera at the surface and sits it back far enough to see all
    //of it, the way the viewer opens on a surface. The angle it is seen from
    //is left as it was.
    module.Aligner.prototype._frameSurface = function(matrix) {
        if (this.hemis.length == 0)
            return;
        var min = new THREE.Vector3(Infinity, Infinity, Infinity);
        var max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
        for (var i = 0; i < this.hemis.length; i++) {
            var box = this.hemis[i].bounds;
            if (box === undefined)
                continue;
            for (var c = 0; c < 8; c++) {
                var corner = new THREE.Vector3(
                    (c & 1) ? box.max.x : box.min.x,
                    (c & 2) ? box.max.y : box.min.y,
                    (c & 4) ? box.max.z : box.min.z);
                if (matrix)
                    corner.applyMatrix4(matrix);
                min.min(corner);
                max.max(corner);
            }
        }
        if (!isFinite(min.x))
            return;
        //far enough for the widest side to fit the height of the view: the
        //diagonal would be the safe distance for any angle, but a brain seen
        //from any of them covers much less than its diagonal
        var size = max.clone().sub(min);
        var half = Math.max(size.x, size.y, size.z) / 2;
        var fov = this.camera3d.fov * Math.PI / 360;
        this.controls.setTarget(min.clone().add(max).multiplyScalar(0.5).toArray());
        this.controls.setRadius(1.1 * half / Math.sin(fov));
    };

    //How far the surface has been unfolded, from the anatomical surface (0)
    //through the inflated one to the flatmap (1), as the viewer's `unfold`
    module.Aligner.prototype.setMix = function(mix) {
        if (mix === undefined)
            return this._mix;
        this._mix = Math.min(Math.max(mix, 0), 1);
        if (this.morphUniforms !== undefined)
            this.morphUniforms.surfmix.value = this._mix;
        var flat = this._flatness();
        this._applyIndex();
        this.setPivot(180 * flat);
        for (var i = 0; i < this.hemis.length; i++) {
            if (this.hemis[i].pivots !== undefined)
                this.hemis[i].pivots.back.rotation.x = flat * -Math.PI / 2;
        }
        //A flatmap faces the light head on, which washes the data out, so the
        //shading gives way to flat illumination as it opens; the viewer's
        //lighting follows the flatmap the same way.
        if (this.morphMaterial !== undefined) {
            var uniforms = this.morphMaterial.uniforms;
            var lit = 0.7 * (1 - flat);
            uniforms.diffuse.value.set(lit, lit, lit);
            var ambient = 0.35 + 0.65 * flat;
            uniforms.emissive.value.set(ambient, ambient, ambient);
        }
        //the camera swings round to face the flatmap as it opens
        if (this._display == DISPLAY.surface)
            this.controls.setMix(flat);
        this.schedule();
    };

    //How much of the flatmap is showing, which is what the medial wall and
    //the camera follow: 1 once the surface is all the way unfolded
    module.Aligner.prototype._flatness = function() {
        if (this.surfNames === undefined || this.surfNames.indexOf("flat") < 0)
            return 0;
        var last = this.surfNames.length - 1;
        return Math.min(Math.max(1 - Math.abs(this._mix * last - last), 0), 1);
    };

    //How far apart the two halves of the surface are swung, in degrees
    module.Aligner.prototype.setPivot = function(value) {
        if (value === undefined)
            return this._pivot;
        this._pivot = value;
        var radians = value * Math.PI / 360;
        for (var i = 0; i < this.hemis.length; i++) {
            var pivots = this.hemis[i].pivots;
            if (pivots === undefined)
                continue;
            var sign = i == 0 ? 1 : -1;
            pivots.front.rotation.z = value > 0 ? 0 : radians * sign;
            pivots.back.rotation.z = value > 0 ? radians * sign : 0;
        }
        this.schedule();
    };

    //The medial wall has no place on the flatmap, so its polygons are dropped
    //while one is showing. The surfaces of the other displays share these
    //geometries, so they are given the whole mesh back on the way out.
    module.Aligner.prototype._applyIndex = function() {
        var cull = this._display == DISPLAY.surface && this._flatness() > 0;
        if (cull === this._culled)
            return;
        this._culled = cull;
        for (var i = 0; i < this.hemis.length; i++) {
            var hemi = this.hemis[i];
            if (hemi.culled === undefined)
                continue;
            var source = cull ? hemi.culled : hemi.full;
            hemi.geometry.attributes.index = source.index;
            hemi.geometry.offsets = source.offsets;
        }
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
            if (view.is2d && this._viewMode(view) == OUTLINE) {
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
    //when opaque enough, the whole surfaces. The other two modes show the
    //surfaces colored by the volume instead, where the alignment puts them in
    //it or in the anatomy's own frame. `layer` restricts the visible objects
    //to the "plane" or the "lines" of a slice view.
    module.Aligner.prototype._showLayer = function(view, layer) {
        var mode = this._viewMode(view);
        var outline = mode == OUTLINE;
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
            hemi.projected.visible = lines && mode == PROJECTED;
            if (hemi.morph !== undefined)
                hemi.morph.visible = lines && mode == MORPHED;
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
            //A control in the panel keeps the keyboard once it has been
            //used, and the browser would hand it the keys meant for the mesh.
            //Pointing at a view is the aligner's way of saying which view the
            //keys act on, so it takes them back here.
            var focused = document.activeElement;
            if (focused && focused.blur && $(focused).closest("#figure_ui").length > 0)
                focused.blur();
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
            //the whole drag is one edit, so its entry is opened here and
            //grows until the button comes back up
            this.beginEdit(state);
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
        if (this._drag !== null &&
            (this._drag.state == "translate" || this._drag.state == "rotate"))
            this._dropEmptyEdit();
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

    //A page closed with the alignment unsaved loses it, so the browser is
    //asked to put the question to the user. Browsers word it themselves, and
    //only ask at all once the page has been interacted with.
    module.Aligner.prototype._onBeforeUnload = function(event) {
        if (this.config.view_only || !this.isDirty())
            return;
        event.preventDefault();
        //browsers from before the standard settled read the message off the
        //event rather than from the return value
        event.returnValue = "The alignment has changes that have not been saved.";
        return event.returnValue;
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
            this.toggleDisplay();
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
        //WASD moves the mesh the same way the arrows do. Both upper and lower
        //case, since shift is the fine-step modifier and shift+w arrives as W.
        switch (key) {
            case "ArrowLeft": case "a": case "A":
                this.translate(view.right.clone().multiplyScalar(-tstep).toArray());
                break;
            case "ArrowRight": case "d": case "D":
                this.translate(view.right.clone().multiplyScalar(tstep).toArray());
                break;
            case "ArrowUp": case "w": case "W":
                this.translate(view.up.clone().multiplyScalar(tstep).toArray());
                break;
            case "ArrowDown": case "s": case "S":
                this.translate(view.up.clone().multiplyScalar(-tstep).toArray());
                break;
            case "q": case "Q":
                this.beginEdit("rotate");
                this._rotate(toViewer, rstep, this.cursorWorld());
                break;
            case "e": case "E":
                this.beginEdit("rotate");
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
