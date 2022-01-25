var mriview = (function(module) {
    var flatscale = 0.3;

    function makeAxes(length, color) {
        function v(x,y,z){ 
            return new THREE.Vector3(x,y,z); 
        }
        var lineGeo = new THREE.Geometry();
        lineGeo.vertices.push(
            v(-length, 0, 0), v(length, 0, 0),
            v(0, -length, 0), v(0, length, 0),
            v(0, 0, -length), v(0, 0, length)
        );
        var lineMat = new THREE.LineBasicMaterial({ color: color, linewidth: 2});
        var axes = new THREE.Line(lineGeo, lineMat);
        axes.type = THREE.Lines;

        var vox = new THREE.BoxGeometry(1, 1, 1);
        var voxMat = new THREE.MeshLambertMaterial({color:0xff0000, transparent:true, opacity:0.75});
        var voxmesh = new THREE.Mesh(vox, voxMat);
        return {axes:axes, vox:voxmesh};
    }

    module.Surface = function(ctminfo) {
        this.loaded = $.Deferred();

        this.sheets = [];
        this.pivots = {};
        this.hemis = {};
        this.volume = 0;
        this._layers = 1;
        this._dither = false;
        this._pivot = 0;
        this._shift = 0;
        this._specular = parseFloat(viewopts.specularity);
        this._leftvis = true;
        this._rightvis = true;
        this.shaders = {};
        this._active = null;
        this._sampler = "nearest";
        this._equivolume = false;
        //this.rotation = [ 0, 0, 200 ]; //azimuth, altitude, radius

        this.object = new THREE.Group();
        this.object.name = 'Surface';

        this.uniforms = THREE.UniformsUtils.merge( [
            THREE.UniformsLib[ "lights" ],
            {
                diffuse:    { type:'v3', value:new THREE.Vector3( .8,.8,.8 )},
                specular:   { type:'v3', value:new THREE.Vector3( .005,.005,.005 )},
                emissive:   { type:'v3', value:new THREE.Vector3( .2,.2,.2 )},
                shininess:  { type:'f',  value:1000},
                specularStrength:{ type:'f',  value:this._specular},

                thickmix:   { type:'f',  value:0.5},
                surfmix:    { type:'f',  value:0},
                bumpyflat:  { type:'i',  value:viewopts.bumpy_flatmap == 'true'},
                allowtilt:  { type:'i',  value:viewopts.allow_tilt == 'true'},
                // equivolume:  { type:'i',  value:viewopts.equivolume == 'true'},

                //hatch:      { type:'t',  value:0, texture: module.makeHatch() },
                //hatchrep:   { type:'v2', value:new THREE.Vector2(108, 40) },
                hatchAlpha: { type:'f', value:1.},
                hatchColor: { type:'v3', value:new THREE.Vector3( 0,0,0 )},

                dataAlpha:  { type:'f', value:1.},
                overlay:    { type:'t', value:null },
                brightness:  { type:'f', value:parseFloat(viewopts.brightness)},
                smoothness:  { type:'f', value:parseFloat(viewopts.smoothness)},
                contrast:    { type:'f', value:parseFloat(viewopts.contrast)},
                extratex:   { type:'t', value:null},

                // screen:     { type:'t', value:this.volumebuf},
                // screen_size:{ type:'v2', value:new THREE.Vector2(100, 100)},
            }
        ]);

        this.ui = (new jsplot.Menu()).add({
            unfold: {action:[this, "setMix", 0., 1.]},
            pivot: {action:[this, "setPivot", -180, 180]},
            shift: {action:[this, "setShift", 0, 200]},
            depth: {action:[this.uniforms.thickmix, "value", 0, 1]},
            "pial surface": {action: this.to_pial_surface.bind(this), key: 'p', help: "Pial surface"},
            "fiducial surface": {action: this.to_fiducial_surface.bind(this), key: 'u', help: "Fiducial surface"},
            "WM surface": {action: this.to_white_matter_surface.bind(this), key: 'y', help: "White matter surface"},
            bumpy_flatmap: {action:[this.uniforms.bumpyflat, "value"]},
            allow_tilt: {action:[this.uniforms.allowtilt, "value"]},
            equivolume: {action:[this, "setEquivolume"]},
            changeDepth: {action: this.changeDepth.bind(this), wheel: true, modKeys: ['altKey'], hidden: true, help:'Change depth'},
            changeInflation: {action: this.changeInflation.bind(this), wheel: true, modKeys: ['shiftKey'], hidden: true, help:'Change inflation'},
            colorbar: {action:[this, "toggleColorbar"]},
            opacity: {action:[this.uniforms.dataAlpha, "value", 0, 1]},
            toggleOpacity: {action: this.toggleOpacity.bind(this), key: 'o', hidden: true, help:'Toggle data opacity'},
            left: {action:[this, "setLeftVis"]},
            leftToggle: {action: this.toggleLeftVis.bind(this), key: 'L', modKeys: ['shiftKey'], hidden: true, help:'Toggle left hemisphere'},
            right: {action:[this, "setRightVis"]},
            rightToggle: {action: this.toggleRightVis.bind(this), key: 'R', modKeys: ['shiftKey'], hidden: true, help:'Toggle right hemisphere'},
	        specularity: {action:[this, "setSpecular", 0, 1]},
            layers: {action:[this, "setLayers", {1:1, 4:4, 8:8, 16:16, 32:32}]},
            toggleMultipleLayers: {action: this.toggleMultipleLayers.bind(this), key: 'm', hidden: true, help: "Toggle multiple layers"},
            dither: {action:[this, "setDither"]},
            sampler: {action:[this, "setSampler", ["nearest", "trilinear"]]},
        });
        

        this.ui.addFolder("curvature", true).add({
            brightness: {action:[this.uniforms.brightness, "value", 0, 1]},
            contrast: {action:[this.uniforms.contrast, "value", 0, 1]},
            smoothness: {action:[this.uniforms.smoothness, "value", 0, 1]},
        });

        var loader = new THREE.CTMLoader(false);
        loader.loadParts( ctminfo, function( geometries, materials, json ) {
            geometries[0].computeBoundingBox();
            geometries[1].computeBoundingBox();

            this.flatlims = json.flatlims;
            this.flatoff = [
                Math.max(
                    Math.abs(geometries[0].boundingBox.min.x),
                    Math.abs(geometries[1].boundingBox.max.x)
                ) / 3, Math.min(
                    geometries[0].boundingBox.min.y, 
                    geometries[1].boundingBox.min.y
                )];

            this.names = json.names;
            var gb0 = geometries[0].boundingBox, gb1 = geometries[1].boundingBox;
            var center = [
                ((gb1.max.x - gb0.min.x) / 2) + gb0.min.x,
                (Math.max(gb0.max.y, gb1.max.y) - Math.min(gb0.min.y, gb1.min.y)) / 2 + Math.min(gb0.min.y, gb1.min.y),
                (Math.max(gb0.max.z, gb1.max.z) - Math.min(gb0.min.z, gb1.min.z)) / 2 + Math.min(gb0.min.z, gb1.min.z),
            ];
            this.center = center;
            //this.object.position.set(0, -center[1], -center[2]);

            var names = {left:0, right:1};
            var posdata = {
                left :{positions:[], normals:[]}, 
                right:{positions:[], normals:[]},
            };
            for (var name in names) {
                var hemi = geometries[names[name]];
                posdata[name].map = hemi.indexMap;
                posdata[name].positions.push(hemi.attributes.position);
                posdata[name].normals.push(hemi.attributes.normal);
                hemi.name = "hemi_"+name;

                //Put attributes in the correct locations for the shader
                if (hemi.attributesKeys.indexOf('wm') != -1) {
                    this.volume = 1;
                    hemi.addAttribute("wmnorm", module.computeNormal(hemi.attributes['wm'], hemi.attributes.index, hemi.offsets) );
                    posdata[name].wm = hemi.attributes.wm;
                    posdata[name].wmnorm = hemi.attributes.wmnorm;
                }
                //Rename the actual surfaces to match shader variable names
                for (var i = 0; i < json.names.length; i++ ) {
                    hemi.attributes['mixSurfs'+i] = hemi.attributes[json.names[i]];
                    hemi.addAttribute('mixNorms'+i, module.computeNormal(hemi.attributes[json.names[i]], hemi.attributes.index, hemi.offsets) );
                    posdata[name].positions.push(hemi.attributes['mixSurfs'+i]);
                    posdata[name].normals.push(hemi.attributes['mixNorms'+i])
                    delete hemi.attributes[json.names[i]];
                }
                //Setup flatmap mix
		var wmareas = module.computeAreas(hemi.attributes.wm, hemi.attributes.index, hemi.offsets);
                wmareas = module.iterativelySmoothVertexData(hemi.attributes.wm, hemi.attributes.index, hemi.offsets, wmareas, smoothfactor, smoothiter);
                hemi.wmareas = wmareas;

		var pialareas = module.computeAreas(hemi.attributes.position, hemi.attributes.index, hemi.offsets);
                pialareas = module.iterativelySmoothVertexData(hemi.attributes.position, hemi.attributes.index, hemi.offsets, pialareas, smoothfactor, smoothiter);
                hemi.pialareas = pialareas;

		var pialarea_attr = new THREE.BufferAttribute(pialareas, 1);
                pialarea_attr.needsUpdate = true;
                var wmarea_attr = new THREE.BufferAttribute(wmareas, 1);
                wmarea_attr.needsUpdate = true;
		
                hemi.addAttribute('pialarea', pialarea_attr);
                hemi.addAttribute('wmarea', wmarea_attr);

                if (this.flatlims !== undefined) {
                    var flats = this._makeFlat(hemi.attributes.uv.array, json.flatlims, names[name]);
                    hemi.addAttribute('mixSurfs'+json.names.length, new THREE.BufferAttribute(flats.pos, 4));
                    hemi.addAttribute('mixNorms'+json.names.length, new THREE.BufferAttribute(flats.norms, 3));
                    hemi.attributes['mixSurfs'+json.names.length].needsUpdate = true;
                    hemi.attributes['mixNorms'+json.names.length].needsUpdate = true;
                    posdata[name].positions.push(hemi.attributes['mixSurfs'+json.names.length]);
                    posdata[name].normals.push(hemi.attributes['mixNorms'+json.names.length]);


                    var smoothfactor = 0.1;
                    var smoothiter = 50;

                    // var flatareas = module.computeAreas(hemi.attributes.mixSurfs1, hemi.culled.index, hemi.culled.offsets);
                    // var flatareascale = flatscale ** 2;
                    // flatareas = flatareas.map(function (a) { return a / flatareascale;});
                    // flatareas = module.iterativelySmoothVertexData(hemi.attributes.position, hemi.attributes.index, hemi.offsets, flatareas, smoothfactor, smoothiter);
                    // hemi.flatareas = flatareas;

                    var dists = module.computeDist(hemi.attributes.position, hemi.attributes.wm);
                    dists = module.iterativelySmoothVertexData(hemi.attributes.position, hemi.attributes.index, hemi.offsets, dists, smoothfactor, smoothiter);

                    var vertexvolumes = module.computeVertexPrismVolume(wmareas, pialareas, dists);
                    hemi.vertexvolumes = vertexvolumes;
                    var flatheights = module.computeFlatVolumeHeight(wmareas, vertexvolumes);
                    flatheights.array = flatheights.array.map(function (h) {return h * flatscale;});
                    // flatheights.array = flatheights.array.map(Math.sqrt);
                    
                    var flat_offset_verts;
                    if ( name == "left" ) {
                        flat_offset_verts = module.offsetVerts(hemi.attributes.mixSurfs1, flatheights, 0, -1);
                    } else {
                        flat_offset_verts = module.offsetVerts(hemi.attributes.mixSurfs1, flatheights, 0, 1);
                    }
                    // // hemi.addAttribute('offsetflat', flat_offset_verts);
                    // var flatoff_geom = new THREE.BufferGeometry();
                    // flatoff_geom.addAttribute('position', flat_offset_verts);
                    // flatoff_geom.addAttribute('index', hemi.attributes.index);
                    // flatoff_geom.computeVertexNormals();

                    // console.log(flatoff_geom);
                    // this.flatoff = flatoff_geom;

                    // hemi.addAttribute('flatBumpNorms', flatoff_geom.attributes.normal);
                    hemi.addAttribute('flatheight', flatheights);
                    hemi.addAttribute('flatBumpNorms', module.computeNormal(flat_offset_verts, hemi.attributes.index, hemi.offsets) );
                } else {
                    // Fill these attributes so the shader doesn't choke, even though
                    // there's no flatmap
                    // just set flatheight to 1 everywhere.
                    // var flatheight_arr = new Float32Array(hemi.attributes.position.position / hemi.attributes.position.itemSize);
                    // flatheight_arr = flatheight_arr.map(function (x) {return 1.0;});
                    // var flatheight = new THREE.BufferAttribute(flatheight_arr, 1);
                    // hemi.addAttribute('flatheight', flatheight);

                    // // and set the flatBumpNorms to the 
                    // var flatBumpNorms = module.computeNormal()
                }

                //Generate an index list that has culled non-flatmap vertices
                var culled = module._cull_flatmap_vertices(hemi.attributes.index.array, hemi.attributes.auxdat.array, hemi.offsets);
                hemi.culled = { index: new THREE.BufferAttribute(culled.indices, 3), offsets:culled.offsets };
                hemi.fullind = { index: hemi.attributes.index, offsets:hemi.offsets };

                //Queue blank data attributes for vertexdata
                hemi.addAttribute("data0", new THREE.BufferAttribute(new Float32Array(), 1));
                hemi.addAttribute("data1", new THREE.BufferAttribute(new Float32Array(), 1));
                hemi.addAttribute("data2", new THREE.BufferAttribute(new Float32Array(), 1));
                hemi.addAttribute("data3", new THREE.BufferAttribute(new Float32Array(), 1));

                hemi.dynamic = true;
                var pivots = {back:new THREE.Group(), front:new THREE.Group()};
                pivots.front.add(pivots.back);
                pivots.back.position.y = hemi.boundingBox.min.y - hemi.boundingBox.max.y;
                pivots.front.position.y = hemi.boundingBox.max.y - hemi.boundingBox.min.y + this.flatoff[1];
                this.pivots[name] = pivots;
                this.hemis[name] = hemi;
                this.object.add(pivots.front);
            }
            this.setHalo(1);

            //Add anatomical and flat names
            this.names.unshift("anatomical");
            if (this.flatlims !== undefined) {
                this.names.push("flat");
            }

            //create picker
            this.picker = new PickPosition(this, posdata);
            this.picker.markers.left.position.y = -this.flatoff[1];
            this.picker.markers.right.position.y = -this.flatoff[1];
            this.pivots.left.back.add(this.picker.markers.left);
            this.pivots.right.back.add(this.picker.markers.right);
            this.addEventListener("mix", this.picker.setMix.bind(this.picker));

            //generate rois
            if (this.flatlims !== undefined) {
                var path = loader.extractUrlBase(ctminfo) + json.rois;
                this.svg = new svgoverlay.SVGOverlay(path, posdata, this);
                this.svg.addEventListener("update", function(evt) {
                    if (this.uniforms.overlay.value && this.uniforms.overlay.value.dispose)
                        this.uniforms.overlay.value.dispose();
                    this.uniforms.overlay.value = evt.texture;
                    this.loaded.resolve();
                    this.dispatchEvent({type:"update"});
                }.bind(this));
                this.pivots.left.back.add(this.svg.labels.left);
                this.pivots.right.back.add(this.svg.labels.right);
                this.svg.labels.left.position.y = -this.flatoff[1];
                this.svg.labels.right.position.y = -this.flatoff[1];
                this.addEventListener("mix", this.svg.setMix.bind(this.svg));
                this.addEventListener("resize", function(evt) {
                    this.resize(evt.width, evt.height);
                }.bind(this.svg));
                this.svg.loaded.done(function() { 
                    this.ui.addFolder("overlays", true, this.svg.ui);
                }.bind(this));
            } else if (json.extratex !== undefined) { //extratex
                this.uniforms.extratex.value = THREE.ImageUtils.loadTexture(json.extratex);
            } else {
                this.loaded.resolve();
            }

        }.bind(this), {useWorker:true});
    };
    THREE.EventDispatcher.prototype.apply(module.Surface.prototype);
    module.Surface.prototype.resize = function(evt) {
    //     this.volumebuf = new THREE.WebGLRenderTarget(width, height, {
    //         minFilter: THREE.LinearFilter,
    //         magFilter: THREE.LinearFilter,
    //         format:THREE.RGBAFormat,
    //         stencilBuffer:false,
    //     });
    //     this.uniforms.screen.value = this.volumebuf;
    //     this.uniforms.screen_size.value.set(width, height);
        this.width = evt.width;
        this.height = evt.height;
        this.loaded.done(function() {
            this.picker.resize(evt.width, evt.height);
        }.bind(this));
        this.dispatchEvent({type:"resize", width:evt.width, height:evt.height});
    };
    module.Surface.prototype.init = function(dataview) {

        this._active = dataview;

        this.loaded.done(function() {
            var shaders = [];
            // Halo rendering code, ignore for now
            // if (this.sheets.length > 1) { //setup meshes for halo rendering
            //     for (var i = 0; i < this.sheets.length; i++) {
            //         var shaders = dataview.getShader(Shaders.surface, this.uniforms, {
            //             morphs:this.names.length, volume:1, rois: false, halo: true });
            //         for (var j = 0; j < shaders.length; j++) {
            //             shaders[j].transparent = i != 0;
            //             shaders[j].depthTest = true;
            //             shaders[j].depthWrite = i == 0;
            //             shaders[j].uniforms.thickmix = {type:'f', value: 1 - i / (this.sheets.length-1)};
            //             shaders[j].blending = THREE.CustomBlending;
            //             shaders[j].blendSrc = THREE.OneFactor;
            //             shaders[j].blendDst = THREE.OneFactor;
            //         }
            //         this.preshaders.push(shaders);
            //     }
            //     var shaders = dataview.getShader(Shaders.surface, this.uniforms, {
            //         morphs:this.names.length, volume:1, rois:false, halo:false });
            //     for (var j = 0; j < shaders.length; j++) {
            //         shaders[j].uniforms.thickmix = {type:'f', value:1};
            //         shaders[j].uniforms.dataAlpha = {type:'f', value:0};
            //     }

            //     this.quadshade = dataview.getShader(Shaders.cmap_quad, this.uniforms);
            //     this.quadshade.transparent = true;
            //     this.quadshade.blending = THREE.CustomBlending
            //     this.quadshade.blendSrc = THREE.OneFactor
            //     this.quadshade.blendDst = THREE.OneMinusSrcAlphaFactor
            //     this.quadshade.depthWrite = false;
            // } else {
            if (dataview.vertex) {
                var shade_cls = Shaders.surface_vertex;
            } else {
                var shade_cls = Shaders.surface_pixel;
            }
            var shaders = dataview.getShader(shade_cls, this.uniforms, {
                hasflat: this.flatlims !== undefined,
                morphs: this.names.length, 
                volume: this.volume, 
                layers: this._layers,
                rois: this.svg instanceof svgoverlay.SVGOverlay,
                extratex: this.uniforms.extratex.value !== null,
                halo: false,
                dither: this._dither,
                equivolume: this._equivolume,
                sampler: this._sampler,
            });
            this.shaders[dataview.uuid] = shaders[0];
        }.bind(this));
    };
    module.Surface.prototype.pick = function(renderer, camera, x, y) {
        // console.log(intersects[0].object.geometry.name);
        if (this.svg !== undefined) {
            this.svg.labels.left.visible = false;
            this.svg.labels.right.visible = false;
        }
        result = this.picker.pick(renderer, camera, x, y, false);
        if (this.svg !== undefined) {
            this.svg.labels.left.visible = true;
            this.svg.labels.right.visible = true;
        }
        return result
    }
    module.Surface.prototype.clearShaders = function() {
        for (var name in this.shaders) {
            this.shaders[name].dispose();
        }
    }
    module.Surface.prototype.prerender = function(renderer, scene, camera) {
        if (this.svg !== undefined) {
            this.svg.prerender(renderer, scene, camera);
        }
    }

    // var oldcolor, black = new THREE.Color(0,0,0);
    // module.Surface.prototype._prerender_halosurf = function(evt) {
    //     var idx = evt.idx, renderer = evt.renderer, scene = evt.scene, camera = evt.camera;
    //     camera.add(scene.fsquad);
    //     scene.fsquad.material = this.quadshade[idx];
    //     scene.fsquad.visible = false;
    //     for (var i = 0; i < this.sheets.length; i++) {
    //         this.sheets[i].left.material = this.preshaders[i][idx];
    //         this.sheets[i].right.material = this.preshaders[i][idx];
    //         this.sheets[i].left.visible = true;
    //         this.sheets[i].right.visible = true;
    //     }
    //     oldcolor = renderer.getClearColor()
    //     renderer.setClearColor(black, 0);
    //     //renderer.render(scene, camera);
    //     renderer.render(scene, camera, this.volumebuf);
    //     renderer.setClearColor(oldcolor, 1);
    //     for (var i = 1; i < this.sheets.length; i++) {
    //         this.sheets[i].left.visible = false;
    //         this.sheets[i].right.visible = false;
    //     }
    //     scene.fsquad.visible = true;
    // };

    module.Surface.prototype.apply = function(dataview) {
        this.loaded.done(function() {
            for (var i = 0; i < this.sheets.length; i++) {
                this.sheets[i].left.material = this.shaders[dataview.uuid];
                this.sheets[i].right.material = this.shaders[dataview.uuid];
            }

            this.picker.apply(dataview)
        }.bind(this));
    };

    module.Surface.prototype.resetShaders = function() {
        this.clearShaders();
        this.init(this._active);
        this.apply(this._active);
    }

    module.Surface.prototype.setHalo = function(layers) {
        var lmesh, rmesh;
        layers = Math.max(layers, 1);
        for (var i = 0; i < this.sheets.length; i++) {
            this.pivots.left.back.remove(this.sheets[i].left);
            this.pivots.right.back.remove(this.sheets[i].right);
        }
        this.sheets = [];
        for (var i = 0; i < layers; i++) {
            lmesh = this._makeMesh(this.hemis.left);
            rmesh = this._makeMesh(this.hemis.right);
            this.sheets.push({left:lmesh, right:rmesh});
            this.pivots.left.back.add(lmesh);
            this.pivots.right.back.add(rmesh);
        }
    };

    var _last_clipped = 0;
    module.Surface.prototype.setMix = function(mix) {
        if (mix === undefined)
            return this.uniforms.surfmix.value;

        this.uniforms.surfmix.value = mix;
        var smix = mix * (this.names.length - 1);
        // check if there is a flatmap
        if (this.flatlims !== undefined) {
            var factor = 1 - Math.abs(smix - (this.names.length-1));
            var clipped = 0 <= factor ? (factor <= 1 ? factor : 1) : 0;
        } else {
            var factor = 0;
            var clipped = 0;
        }

        //Swap out the polys array between the culled and full when flattening
        if (clipped > 0 && _last_clipped == 0) {
            this.hemis.left.attributes.index = this.hemis.left.culled.index;
            this.hemis.left.offsets = this.hemis.left.culled.offsets;
            this.hemis.right.attributes.index = this.hemis.right.culled.index;
            this.hemis.right.offsets = this.hemis.right.culled.offsets;
        } else if (clipped == 0 && _last_clipped > 0) {
            this.hemis.left.attributes.index = this.hemis.left.fullind.index;
            this.hemis.left.offsets = this.hemis.left.fullind.offsets;
            this.hemis.right.attributes.index = this.hemis.right.fullind.index;
            this.hemis.right.offsets = this.hemis.right.fullind.offsets;
        }
        _last_clipped = clipped;

        // this.uniforms.specularStrength.value = this._specular * (1-clipped);
        this.uniforms.specularStrength.value = this._specular;
        this.setPivot( 180 * clipped);

        this.pivots.left.back.rotation.x = clipped * -Math.PI/2;
        this.pivots.right.back.rotation.x = clipped * -Math.PI/2;
        
        this.dispatchEvent({type:'mix', flat:clipped, mix:mix, thickmix:this.uniforms.thickmix.value});
    };
    module.Surface.prototype.setThickMix = function(val) {
        this.uniforms.thickmix.value = val;

        let mix = this.uniforms.surfmix.value;
        var smix = mix * (this.names.length - 1);
        var factor = 1 - Math.abs(smix - (this.names.length-1));
        let clipped = 0 <= factor ? (factor <= 1 ? factor : 1) : 0;
        this.dispatchEvent({type:'mix', flat:clipped, mix:mix, thickmix:this.uniforms.thickmix.value});
        viewer.schedule()

        var gui = this.ui._gui
        for (var i in gui.__controllers) {
            gui.__controllers[i].updateDisplay();
        }
    };
    module.Surface.prototype.changeDepth = function(direction) {
        let inc;
        if (direction > 0) {
            inc = .05
        } else {
            inc = -.05
        }

        let newVal = this.uniforms.thickmix.value + inc;
        if (-.01 <= newVal && newVal <= 1.01) {
            this.setThickMix(newVal);
        }
    }
    module.Surface.prototype.to_pial_surface = function() {
        this.setThickMix(0.0)
    };
    module.Surface.prototype.to_fiducial_surface = function() {
        this.setThickMix(0.5)
    };
    module.Surface.prototype.to_white_matter_surface = function() {
        this.setThickMix(1.0)
    };
    module.Surface.prototype.changeInflation = function(direction) {
        let inc;
        if (direction > 0) {
            inc = .01
        } else {
            inc = -.01
        }

        let newVal = this.uniforms.surfmix.value + inc;
        if (0.0 <= newVal && newVal <= 1.0) {
            this.setMix(newVal);
            viewer.schedule();
        }
    }
    module.Surface.prototype.setPivot = function(val) {
        if (val === undefined)
            return this._pivot;

        this._pivot = val;
        var names = {left:1, right:-1}
        if (val > 0) {
            for (var name in names) {
                this.pivots[name].front.rotation.z = 0;
                this.pivots[name].back.rotation.z = val*Math.PI/180 * names[name]/ 2;
            }
        } else {
            for (var name in names) {
                this.pivots[name].back.rotation.z = 0;
                this.pivots[name].front.rotation.z = val*Math.PI/180 * names[name] / 2;
            }
        }
    };
    module.Surface.prototype.setShift = function(val) {
        if (val === undefined)
            return this._shift;
        this._shift = val;
        this.pivots.left.front.position.x = -val;
        this.pivots.right.front.position.x = val;
    };
    module.Surface.prototype.setSpecular = function(val) {
        if (val === undefined)
            return this._specular;

        this._specular = val;
	this.uniforms.specularStrength.value = this._specular;// * (1-_last_clipped);
    
    };
    module.Surface.prototype.setLeftVis = function(val) {
        if (val === undefined)
            return this._leftvis
        this._leftvis = val;
        //this.surfs[0].surf.pivots.left.front.visible = val;
        this.pivots.left.front.visible = val;
    };
    module.Surface.prototype.toggleColorbar = function(val) {
        if (val === true || val === undefined) {
            $('#colorlegend').css('display', 'block')
            return true
        } else {
            $('#colorlegend').css('display', 'none')
            return false
        }

        // necessary due to datgui library's extremely hacky introspection of functions
        return true
    };
    module.Surface.prototype.toggleLeftVis = function() {
        this.setLeftVis(!this._leftvis);
        viewer.schedule();
    };
    module.Surface.prototype.setRightVis = function(val) {
        if (val === undefined)
            return this._rightvis
        this._rightvis = val;
        //this.surfs[0].surf.pivots.right.front.visible = val;
        this.pivots.right.front.visible = val;
    };
    module.Surface.prototype.toggleRightVis = function() {
        this.setRightVis(!this._rightvis);
        viewer.schedule();
    };
    module.Surface.prototype.toggleOpacity = function() {
        let newValue = 1 - Math.round(this.uniforms.dataAlpha.value)
        surface = Object.keys(viewer.ui._desc.surface).filter((key) => key[0] != '_')[0]
        viewer.ui.set('surface.' + surface + '.opacity', newValue)
        viewer.schedule();
    };
    module.Surface.prototype.setLayers = function(val) {
        if (val === undefined)
            return this._layers;
        this._layers = val;
        this.resetShaders();
    }
    module.Surface.prototype.toggleMultipleLayers = function() {
        if (this._layers == 1) {
            this._layers = 32;
        } else {
            this._layers = 1;
        }
        this.resetShaders();
        viewer.schedule();
        var gui = this.ui._gui
        for (var i in gui.__controllers) {
            gui.__controllers[i].updateDisplay();
        }
    }
    module.Surface.prototype.setDither = function(val) {
        if (val === undefined)
            return this._dither;
        this._dither = val;
        this.resetShaders();
    }
    module.Surface.prototype.setSampler = function(val) {
        if (val === undefined)
            return this._sampler;
        this._sampler = val;
        this._active.setFilter(val);
        this.resetShaders();
    }
    module.Surface.prototype.setEquivolume = function(val) {
        if (val === undefined)
            return this._equivolume;
        this._equivolume = val;
        this.resetShaders();
    }

    module.Surface.prototype._makeMesh = function(geom, shader) {
        //Creates the mesh object given the geometry and shader
        var mesh = new THREE.Mesh(geom, shader);
        mesh.position.y = -this.flatoff[1];
        return mesh;
    };

    module.Surface.prototype._makeFlat = function(uv, flatlims, right) {
        var fmin = flatlims[0], fmax = flatlims[1];
        var flat = new Float32Array(uv.length / 2 * 4);
        var norms = new Float32Array(uv.length / 2 * 3);
        for (var i = 0, il = uv.length / 2; i < il; i++) {
            if (right) {
                flat[i*4+1] = flatscale*uv[i*2] + this.flatoff[1];
                norms[i*3] = 1;
                // flat[i*4+0] = -flatscale*uv[i*2+1];
            } else {
                flat[i*4+1] = flatscale*-uv[i*2] + this.flatoff[1];
                norms[i*3] = -1;
                // flat[i*4+0] = flatscale*uv[i*2+1];
            }
            flat[i*4+2] = flatscale*uv[i*2+1];
            uv[i*2]   = (uv[i*2]   + fmin[0]) / fmax[0];
            uv[i*2+1] = (uv[i*2+1] + fmin[1]) / fmax[1];
        }

        return {pos:flat, norms:norms};
    };

    module.SurfDelegate = function(dataview) {
        this.object = new THREE.Group();
        this.object.name = "SurfDelegate";

        this.ui = new jsplot.Menu();

        this.update(dataview);
        this._update = this.update.bind(this);
        this._attrib = this.setAttribute.bind(this);
        this._resize = this.resize.bind(this);
        this._listeners = {};
    }
    module.SurfDelegate.prototype.update = function(dataview) {
        if (this.surf !== undefined) {
            this.object.remove(this.surf.object);
            this.surf.clearShaders();
            for (var name in this._listeners)
                this.surf.removeEventListener(name, this._listeners[name]);
            this.ui.remove(this.surf.ui);
        }
        var subj = dataview.data[0].subject;
        this.surf = subjects[subj];
        this.surf.init(dataview);
        this.object.add(this.surf.object);
        
        this.ui.addFolder(subj, false, this.surf.ui);

        for (var name in this._listeners)
            this.surf.addEventListener(name, this._listeners[name]);
    }
    module.SurfDelegate.prototype.setAttribute = function(event) {
        var name = event.name, left = event.value[0], right = event.value[1];
        var hemis = this.surf.hemis;
        this.surf.loaded.done(function() {
            hemis.left.attributes[name] = left;
            hemis.right.attributes[name] = right;
        });
    }
    module.SurfDelegate.prototype.pick = function(renderer, camera, x, y) {
        return this.surf.pick(renderer, camera, x, y);
    }
    module.SurfDelegate.prototype.setMix = function(mix) {
        return this.surf.setMix(mix);
    }
    module.SurfDelegate.prototype.setPivot = function(pivot) {
        return this.surf.setPivot(pivot);
    }
    module.SurfDelegate.prototype.setShift = function(shift) {
        return this.surf.setShift(shift);
    }
    module.SurfDelegate.prototype.apply = function(dataview) {
        return this.surf.apply(dataview);
    }
    module.SurfDelegate.prototype.resize = function(evt) {
        return this.surf.resize(evt);
    }
    module.SurfDelegate.prototype.addEventListener = function(name, func) {
        this._listeners[name] = func;
        this.surf.addEventListener(name, func);
    }
    module.SurfDelegate.prototype.removeEventListener = function(name, func) {
        this.surf.removeEventListener(name, this._listeners[name]);
        delete this._listeners[name];
    }
    module.SurfDelegate.prototype.prerender = function(renderer, scene, camera) {
        this.surf.prerender(renderer, scene, camera);
    }


    module.VolumeSheets = function(dataview) {
        if (dataview.vertex)
            throw "Cannot show volume integration for vertex dataview"

        this.shaders = {};
        this.uniforms = THREE.UniformsUtils.merge( [
            THREE.UniformsLib[ "lights" ],
            {
                diffuse:    { type:'v3', value:new THREE.Vector3( 0,0,0 )},
                specular:   { type:'v3', value:new THREE.Vector3( 0,0,0 )},
                emissive:   { type:'v3', value:new THREE.Vector3( 1,1,1 )},
                shininess:  { type:'f',  value:0},
                specularStrength:{ type:'f',  value:0},
                dataAlpha:  { type:'f', value:.1},
            }]);

        this.object = new THREE.Group();
        this.object.name = 'VolumeSheets'
        this.setSheets(30);
        this.update(dataview);
    }
    THREE.EventDispatcher.prototype.apply(module.VolumeSheets.prototype);
    module.VolumeSheets.prototype.update = function(dataview) {
        var shaders = dataview.getShader(Shaders.main, this.uniforms, {
            viewspace:true,
            depthTest:true,
            depthWrite:false,
            blending:THREE.AdditiveBlending,
            transparent:true,
            lights:false,
        });
        this.shaders[dataview.uuid] = shaders[0];

        //compute dataview extents to scale the sheets
        var xfm = new THREE.Matrix4();
        xfm.set.apply(xfm, dataview.xfm);
        var ixfm = (new THREE.Matrix4()).getInverse(xfm);
        var zsize = dataview.data[0].mosaic;
        var shape = dataview.data[0].shape;
        var near = (new THREE.Vector3(0,0,0)).applyMatrix4(ixfm);
        var far = (new THREE.Vector3(shape[0], shape[1], zsize[0]*zsize[1])).applyMatrix4(ixfm);
        var mid = near.clone().multiplyScalar(0.5).add(far.clone().multiplyScalar(.5));

        var scale = near.distanceTo(far);
        this.object.scale.set(scale/2, scale/2, scale/2);
        this.object.position.set(mid.x, mid.y, mid.z);
    }
    module.VolumeSheets.prototype.apply = function(dataview) {
        this.mesh.material = this.shaders[dataview.uuid];
        //this.sheets.material = this.debug_shade;
    }
    module.VolumeSheets.prototype.setSheets = function(n) {
        this.n_sheets = n;
        this.uniforms.dataAlpha.value = 1 / n;
        var position = new THREE.BufferAttribute(new Float32Array(4*3*n), 3);
        var index = new THREE.BufferAttribute(new Uint16Array(2*3*n), 3);
        var normal = new THREE.BufferAttribute(new Float32Array(4*3*n), 3);
        for (var i = 0; i < n; i++) {
            var z = ((2*i+1) / (2*n))*2 - 1;
            position.setXYZ(i*4  , -1, -1, z);
            position.setXYZ(i*4+1,  1, -1, z);
            position.setXYZ(i*4+2,  1,  1, z);
            position.setXYZ(i*4+3, -1,  1, z);
            normal.setXYZ(i*4  , 0, 0, 1);
            normal.setXYZ(i*4+1, 0, 0, 1);
            normal.setXYZ(i*4+2, 0, 0, 1);
            normal.setXYZ(i*4+3, 0, 0, 1);
            index.setXYZ(i*2  , i*4, i*4+1, i*4+2);
            index.setXYZ(i*2+1, i*4, i*4+2, i*4+3);
        }
        this.sheets = new THREE.BufferGeometry();
        this.sheets.addAttribute("position", position);
        this.sheets.addAttribute("index", index);
        this.sheets.addAttribute("normal", normal);

        this.debug_shade = new THREE.MeshBasicMaterial({color:0xffcccc, 
            side:THREE.DoubleSide,
            transparent:true,
            opacity:.1,
            blending:THREE.AdditiveBlending,
            depthTest:true,
            depthWrite:false,
        });
        this.mesh = new THREE.Mesh(this.sheets, null);
        this.object.add(this.mesh);
    }

    return module;
}(mriview || {}));
