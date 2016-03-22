// make sure canvas size is set properly for high DPI displays
// From: http://www.khronos.org/webgl/wiki/HandlingHighDPI
var dpi_ratio = window.devicePixelRatio || 1;

var mriview = (function(module) {
    module.flatscale = .35;

    module.Viewer = function(figure) { 
        //Allow objects to listen for mix updates
        THREE.EventTarget.call( this );
        jsplot.Axes.call(this, figure);

        var blanktex = document.createElement("canvas");
        blanktex.width = 16;
        blanktex.height = 16;
        this.blanktex = new THREE.Texture(blanktex);
        this.blanktex.magFilter = THREE.NearestFilter;
        this.blanktex.minFilter = THREE.NearestFilter;
        this.blanktex.needsUpdate = true;

        //Initialize all the html
        $(this.object).html($("#mriview_html").html())

        this.canvas = $(this.object).find("#brain");

        // scene and camera
        this.camera = new THREE.CombinedCamera( this.canvas.width(), this.canvas.height(), 45, 1.0, 1000, 1., 1000. );
        this.camera.position.set(300, 300, 300);
        this.camera.up.set(0,0,1);

        this.scene = new THREE.Scene();
        this.scene.add( this.camera );
        this.controls = new THREE.LandscapeControls( $(this.object).find("#braincover")[0], this.camera );
        this.controls.addEventListener("change", this.schedule.bind(this));
        this.addEventListener("resize", function(event) {
            this.controls.resize(event.width, event.height);
        }.bind(this));
        this.controls.addEventListener("mousedown", function(event) {
            if (this.active && this.active.fastshader) {
                this.meshes.left.material = this.active.fastshader;
                this.meshes.right.material = this.active.fastshader;
            }
        }.bind(this));
        this.controls.addEventListener("mouseup", function(event) {
            if (this.active && this.active.fastshader) {
                this.meshes.left.material = this.active.shader;
                this.meshes.right.material = this.active.shader;
                this.schedule();
            }
        }.bind(this));
        
        this.light = new THREE.DirectionalLight( 0xffffff );
        this.light.position.set( -200, -200, 1000 ).normalize();
        this.camera.add( this.light );
        this.flatmix = 0;
        this.specular = 0.5;
        this.frame = 0;

        // renderer
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            preserveDrawingBuffer:true, 
            canvas:this.canvas[0],
        });
        this.renderer.sortObjects = true;
        this.renderer.setClearColorHex( 0x0, 1 );
        this.renderer.context.getExtension("OES_texture_float");
        this.renderer.context.getExtension("OES_texture_float_linear");
        this.renderer.context.getExtension("OES_standard_derivatives");
        this.renderer.setSize( this.canvas.width(), this.canvas.height() );

        this.uniforms = THREE.UniformsUtils.merge( [
            THREE.UniformsLib[ "lights" ],
            {
                diffuse:    { type:'v3', value:new THREE.Vector3( .8,.8,.8 )},
                specular:   { type:'v3', value:new THREE.Vector3( this.specular, this.specular, this.specular )},
                emissive:   { type:'v3', value:new THREE.Vector3( .2,.2,.2 )},
                shininess:  { type:'f',  value:200},

                thickmix:   { type:'f',  value:0.5},
                framemix:   { type:'f',  value:0},
                hatchrep:   { type:'v2', value:new THREE.Vector2(108, 40) },
                offsetRepeat:{type:'v4', value:new THREE.Vector4( 0, 0, 1, 1 ) },
                
                //hatch:      { type:'t',  value:0, texture: module.makeHatch() },
                colormap:   { type:'t',  value:0, texture: this.blanktex },
                map:        { type:'t',  value:1, texture: this.blanktex },
                data:       { type:'tv', value:2, texture: [this.blanktex, this.blanktex, this.blanktex, this.blanktex]},
                mosaic:     { type:'v2v', value:[new THREE.Vector2(6, 6), new THREE.Vector2(6, 6)]},
                dshape:     { type:'v2v', value:[new THREE.Vector2(100, 100), new THREE.Vector2(100, 100)]},
                volxfm:     { type:'m4v', value:[new THREE.Matrix4(), new THREE.Matrix4()] },
                nsamples:   { type:'i', value:0},

                vmin:       { type:'fv1',value:[0,0]},
                vmax:       { type:'fv1',value:[1,1]},

                curvAlpha:  { type:'f', value:1.},
                curvScale:  { type:'f', value:2.0},//0.5 TEMP FIX!!
                curvLim:    { type:'f', value:.2},
		curvMean:   { type:'f', value:0.5},
                dataAlpha:  { type:'f', value:1.0},
                hatchAlpha: { type:'f', value:1.},
                hatchColor: { type:'v3', value:new THREE.Vector3( 0,0,0 )},
                voxlineColor:{type:'v3', value:new THREE.Vector3( 0,0,0 )},
                voxlineWidth:{type:'f', value:viewopts.voxline_width},

                hide_mwall: { type:'i', value:0},
            }
        ]);
        
        this.state = "pause";
        this._startplay = null;
        this._animation = null;
        this.loaded = $.Deferred().done(function() {
            this.schedule();
            $(this.object).find("#ctmload").hide();
            this.canvas.css("opacity", 1);
        }.bind(this));
        this.postsvg = $.Deferred();
        this.cmapload = $.Deferred();
        this.labelshow = true;
        this._pivot = 0;

        this.planes = [new sliceplane.Plane(this, 0), 
            new sliceplane.Plane(this, 1), 
            new sliceplane.Plane(this, 2)]
        this.planes[0].mesh.visible = false;
        this.planes[1].mesh.visible = false;
        this.planes[2].mesh.visible = false;

        this.dataviews = {};
        this.active = null;

        var cmapnames = {};
        $(this.object).find("#colormap option").each(function(idx) {
            cmapnames[$(this).text()] = idx;
        });
        this.cmapnames = cmapnames;

        this._bindUI();

        this.figure.register("setdepth", this, function(depth) { this.setState("depth", depth);}.bind(this));
        this.figure.register("setmix", this, this.setMix.bind(this));
        this.figure.register("setpivot", this, this.setPivot.bind(this));
        this.figure.register("setshift", this, this.setShift.bind(this));
        this.figure.register("setcamera", this, function(az, alt, rad, target) {
            this.setState('azimuth', az);
            this.setState('altitude', alt);
            this.setState('radius', rad);
            this.setState('target', target);
            this.schedule();
        }.bind(this));
        this.figure.register("playsync", this, function(time) {
            if (this._startplay != null)
                this._startplay = (new Date()) - (time * 1000);
        });
        this.figure.register("playtoggle", this, this.playpause.bind(this));
        this.figure.register("setFrame", this, this.setFrame.bind(this));
    }
    module.Viewer.prototype = Object.create(jsplot.Axes.prototype);
    module.Viewer.prototype.constructor = module.Viewer;
    
    module.Viewer.prototype.schedule = function() {
        if (!this._scheduled) {
            this._scheduled = true;
            var target = this.getState('target');
            var az = this.getState('azimuth'), 
                alt = this.getState('altitude'),
                rad = this.getState('radius');
            this.figure.notify("setcamera", this, [az, alt, rad, target]);

            requestAnimationFrame( function() {
                this.draw();
                if (this.state == "play" || this._animation != null) {
                    this.schedule();
                }
            }.bind(this));
            // THIS NEXT LINE IS SUPER HACK BUT IT DON'T WORK ELSEWHERE FUCK ME
            $("#overlay_fieldset legend").css("width", $("#overlay_fieldset ul").width()+"px");
        }
    };
    module.Viewer.prototype.draw = function () {
        if (this.state == "play") {
            var sec = ((new Date()) - this._startplay) / 1000;
            this.setFrame(sec);
        } 
        if (this._animation) {
            var sec = ((new Date()) - this._animation.start) / 1000;
            if (!this._animate(sec)) {
                this.meshes.left.material = this.active.shader;
                this.meshes.right.material = this.active.shader;
                delete this._animation;
            }
        }
        this.controls.update(this.flatmix);
        if (this.labelshow && this.roipack) {
            this.roipack.labels.render(this, this.renderer);
        } else {
            this.renderer.render(this.scene, this.camera);
        }
        this._scheduled = false;
        this.dispatchEvent({type:"draw"});
    };
    module.Viewer.prototype.load = function(ctminfo, callback) {
        var loader = new THREE.CTMLoader(false);
        loader.loadParts( ctminfo, function( geometries, materials, header, json ) {
            geometries[0].computeBoundingBox();
            geometries[1].computeBoundingBox();

            this.meshes = {};
            this.pivot = {};
            this.flatlims = json.flatlims;
            this.flatoff = [
                Math.max(
                    Math.abs(geometries[0].boundingBox.min.x),
                    Math.abs(geometries[1].boundingBox.max.x)
                ) / 3, Math.min(
                    geometries[0].boundingBox.min.y, 
                    geometries[1].boundingBox.min.y
                )];
            this._makeBtns(json.names);
            //this.controls.flatsize = module.flatscale * this.flatlims[1][0];
            this.controls.flatoff = this.flatoff[1];
            var gb0 = geometries[0].boundingBox, gb1 = geometries[1].boundingBox;
            this.surfcenter = [
                ((gb1.max.x - gb0.min.x) / 2) + gb0.min.x,
                (Math.max(gb0.max.y, gb1.max.y) - Math.min(gb0.min.y, gb1.min.y)) / 2 + Math.min(gb0.min.y, gb1.min.y),
                (Math.max(gb0.max.z, gb1.max.z) - Math.min(gb0.min.z, gb1.min.z)) / 2 + Math.min(gb0.min.z, gb1.min.z),
            ];

            if (geometries[0].attributes.wm !== undefined) {
                //We have a thickness volume!
                this.uniforms.nsamples.value = 1;
                $(this.object).find("#thickmapping").show();
            }

            var names = {left:0, right:1};
            if (this.flatlims !== undefined) {
                //Do not attempt to create flatmaps or load ROIs if there are no flats
                var posdata = {left:[], right:[]};
                for (var name in names) {
                    var right = names[name];
                    var flats = module.makeFlat(geometries[right].attributes.uv.array, json.flatlims, this.flatoff, right);
                    geometries[right].morphTargets.push({itemSize:3, stride:3, array:flats.pos});
                    geometries[right].morphNormals.push(flats.norms);
                    posdata[name].push(geometries[right].attributes.position);
                    for (var i = 0, il = geometries[right].morphTargets.length; i < il; i++) {
                        posdata[name].push(geometries[right].morphTargets[i]);
                    }
                    geometries[right].reorderVertices();
                    geometries[right].dynamic = true;

                    //var geom = splitverts(geometries[right], name == "right" ? leftlen : 0);
                    var meshpiv = this._makeMesh(geometries[right], this.shader);
                    this.meshes[name] = meshpiv.mesh;
                    this.pivot[name] = meshpiv.pivots;
                    this.scene.add(meshpiv.pivots.front);
                }
                //Load in whole SVG file (if we are going to add a sulcal layer, we need to then
                //choose which elements within ROIpack to display.)
                $.get(loader.extractUrlBase(ctminfo)+json.rois, null, function(svgdoc) {
                    this.roipack = new ROIpack(svgdoc, this.renderer, posdata);
                    this.addEventListener("mix", function(evt) {
                         this.roipack.labels.setMix(evt.mix);
                    }.bind(this));
                    this.addEventListener("resize", function(event) {
                        this.roipack.resize(event.width, event.height);
                    }.bind(this));
                    this.roipack.update(this.renderer).done(function(tex) {
                        this.uniforms.map.texture = tex; // New texture gotten here.
                        this.postsvg.resolve();
                        this.schedule();
                    }.bind(this));
                }.bind(this));
            } else {
                for (var name in names) {
                    var right = names[name];
                    var len = geometries[right].attributes.position.array.length / 3;
                    geometries[right].reorderVertices();
                    geometries[right].dynamic = true;
                    var meshpiv = this._makeMesh(geometries[right], this.shader);
                    this.meshes[name] = meshpiv.mesh;
                    this.pivot[name] = meshpiv.pivots;
                    this.scene.add(meshpiv.pivots.front);
                }
            }
            
            this.picker = new FacePick(this, 
                this.meshes.left.geometry.attributes.position.array, 
                this.meshes.right.geometry.attributes.position.array);
            this.addEventListener("mix", this.picker.setMix.bind(this.picker));
            this.addEventListener("resize", function(event) {
                this.picker.resize(event.width, event.height);
            }.bind(this));
            this.controls.addEventListener("change", function() {
                this.picker._valid = false;
            }.bind(this));
            this.controls.addEventListener("pick", function(event) {
                this.picker.pick(event.x, event.y, event.keep);
            }.bind(this));
            this.controls.addEventListener("dblpick", function(event) {
                this.picker.dblpick(event.x, event.y, event.keep);
            }.bind(this));
            this.controls.addEventListener("undblpick", function(event) {
                this.picker.undblpick();
            }.bind(this));

            this.setState("target", this.surfcenter);
            
            if (typeof(callback) == "function")
                this.loaded.done(callback);
            this.loaded.resolve();
        }.bind(this), true, true );
    };
    module.Viewer.prototype.resize = function(width, height) {
        if (width !== undefined) {
            if (width.width !== undefined) {
                height = width.height;
                width = width.width;
            }
            $(this.object).find("#brain").css("width", width);
            this.canvas[0].width = width;
            //width = $(this.object).width();
        }
        var w = width === undefined ? $(this.object).width()  : width;
        var h = height === undefined ? $(this.object).height()  : height;
        var aspect = w / h;

        this.renderer.setSize( w * dpi_ratio, h * dpi_ratio );
        this.renderer.domElement.style.width = w + 'px'; 
        this.renderer.domElement.style.height = h + 'px'; 

        // this.renderer.setSize(w, h);
        this.camera.setSize(aspect * 100, 100);
        this.camera.updateProjectionMatrix();
        this.dispatchEvent({ type:"resize", width:w, height:h});
        this.loaded.done(this.schedule.bind(this));
    };
    module.Viewer.prototype.getState = function(state) {
        switch (state) {
            case 'mix':
                return $(this.object).find("#mix").slider("value");
            case 'pivot':
                //return $("#pivot").slider("value");
                return this._pivot;
            case 'frame':
                return this.frame;
            case 'azimuth':
                return this.controls.azimuth;
            case 'altitude':
                return this.controls.altitude;
            case 'radius':
                return this.controls.radius;
            case 'target':
                var t = this.controls.target;
                return [t.x, t.y, t.z];
            case 'depth':
                return this.uniforms.thickmix.value;
            case 'visL':
                return this.meshes.left.visible;
            case 'visR':
                return this.meshes.right.visible;
            case 'rotationL':
                var rot = this.meshes.left.rotation
                return [rot.x,rot.y,rot.z];
            case 'rotationR':
                var rot = this.meshes.right.rotation
                return [rot.x,rot.y,rot.z];
            case 'alpha':
                return this.renderer.getClearAlpha;
            case 'projection':
                if (this.camera.inOrthographicMode) {
                    return 'orthographic'}
                else if (this.camera.inPerspectiveMode) {
                    return 'perspective'}
            case 'slices':
                return [this.planes[0].slice, this.planes[1].slice, this.planes[2].slice]
        };
    };
    module.Viewer.prototype.setState = function(state, value) {
        switch (state) {
            case 'mix':
                return this.setMix(value);
            case 'pivot':
                return this.setPivot(value);
            case 'frame':
                this.setFrame(value);
                return this.figure.notify("setFrame", this, [value]);
            case 'azimuth':
                return this.controls.setCamera(value);
            case 'altitude':
                return this.controls.setCamera(undefined, value);
            case 'radius':
                return this.controls.setCamera(undefined, undefined, value);
            case 'target':
                if (this.roipack) this.roipack._updatemove = true;
                return this.controls.target.set(value[0], value[1], value[2]);
            case 'depth':
                return this.uniforms.thickmix.value = value;
            case 'visL':
                if (this.roipack) this.roipack._updatemove = true;
                return this.meshes.left.visible = value;
            case 'visR':
                if (this.roipack) this.roipack._updatemove = true;
                return this.meshes.right.visible = value;
            //case 'volume_vis':
            //    this.planes[0].mesh.visible = value[0]
            //    this.planes[1].mesh.visible = value[1]
            //    this.planes[2].mesh.visible = value[2]
            case 'rotationL':
                if (this.roipack) this.roipack._updatemove = true;
                return this.meshes.left.rotation.set(value[0], value[1], value[2]);
            case 'rotationR':
                if (this.roipack) this.roipack._updatemove = true;
                return this.meshes.right.rotation.set(value[0], value[1], value[2]);
            case 'alpha':
                return this.renderer.setClearColor(0,value);
            case 'specularity':
                return this.specularity = value
            case 'data':
                return this.setData(value)
            case 'labels':
                return this.labelshow = value;
            case 'pivot':
                return 'SORRY NOT YET'
            case 'projection':
                if (value=='perspective'){
                    return this.controls.camera.toPerspective()}
                else if (value=='orthographic'){
                    return this.controls.camera.toOrthographic()}
            case 'slices':
                return [this.planes[0].update(value[0]), this.planes[1].update(value[1]), this.planes[2].update(value[2])];
        };
    };
    module.Viewer.prototype.animate = function(animation) {
        var state = {};
        var anim = [];
        animation.sort(function(a, b) { return a.idx - b.idx});
        for (var i = 0, il = animation.length; i < il; i++) {
            var f = animation[i];
            if (f.idx == 0) {
                this.setState(f.state, f.value);
                state[f.state] = {idx:0, val:f.value};
            } else {
                if (state[f.state] === undefined)
                    state[f.state] = {idx:0, val:this.getState(f.state)};
                var start = {idx:state[f.state].idx, state:f.state, value:state[f.state].val}
                var end = {idx:f.idx, state:f.state, value:f.value};
                state[f.state].idx = f.idx;
                state[f.state].val = f.value;
                if (start.value instanceof Array) {
                    var test = true;
                    for (var j = 0; test && j < start.value.length; j++)
                        test = test && (start.value[j] == end.value[j]);
                    if (!test)
                        anim.push({start:start, end:end, ended:false});
                } else if (start.value != end.value)
                    anim.push({start:start, end:end, ended:false});
            }
        }
        if (this.active.fastshader) {
            this.meshes.left.material = this.active.fastshader;
            this.meshes.right.material = this.active.fastshader;
        }
        this._animation = {anim:anim, start:new Date()};
        this.schedule();
    };
    module.Viewer.prototype._animate = function(sec) {
        var state = false;
        var idx, val, f, i, j;
        for (i = 0, il = this._animation.anim.length; i < il; i++) {
            f = this._animation.anim[i];
            if (!f.ended) {
                if (f.start.idx <= sec && sec < f.end.idx) {
                    idx = (sec - f.start.idx) / (f.end.idx - f.start.idx);
                    if (f.start.value instanceof Array) {
                        val = [];
                        for (j = 0; j < f.start.value.length; j++) {
                            //val.push(f.start.value[j]*(1-idx) + f.end.value[j]*idx);
                            val.push(this._animInterp(f.start.state, f.start.value[j], f.end.value[j], idx));
                        }
                    } else {
                        //val = f.start.value * (1-idx) + f.end.value * idx;
                        val = this._animInterp(f.start.state, f.start.value, f.end.value, idx);
                    }
                    this.setState(f.start.state, val);
                    state = true;
                } else if (sec >= f.end.idx) {
                    this.setState(f.end.state, f.end.value);
                    f.ended = true;
                } else if (sec < f.start.idx) {
                    state = true;
                }
            }
        }
        return state;
    };
    module.Viewer.prototype._animInterp = function(state, startval, endval, idx) {
        switch (state) {
            case 'azimuth':
                // Azimuth is an angle, so we need to choose which direction to interpolate
                if (Math.abs(endval - startval) >= 180) { // wrap
                    if (startval > endval) {
                        return (startval * (1-idx) + (endval+360) * idx + 360) % 360;
                    }
                    else {
                        return (startval * (1-idx) + (endval-360) * idx + 360) % 360;
                    }
                } 
                else {
                    return (startval * (1-idx) + endval * idx);
                }
            default:
                // Everything else can be linearly interpolated
                return startval * (1-idx) + endval * idx;
        }
    };
    module.Viewer.prototype.reset_view = function(center, height) {
        var asp = this.flatlims[1][0] / this.flatlims[1][1];
        var camasp = height !== undefined ? asp : this.camera.cameraP.aspect;
        var size = [module.flatscale*this.flatlims[1][0], module.flatscale*this.flatlims[1][1]];
        var min = [module.flatscale*this.flatlims[0][0], module.flatscale*this.flatlims[0][1]];
        var xoff = center ? 0 : size[0] / 2 - min[0];
        var zoff = center ? 0 : size[1] / 2 - min[1];
        var h = size[0] / 2 / camasp;
        h /= Math.tan(this.camera.fov / 2 * Math.PI / 180);
        this.controls.target.set(xoff, this.flatoff[1], zoff);
        this.controls.setCamera(180, 90, h);
        this.setMix(1);
        this.setShift(0);
    };
    module.Viewer.prototype.update_volvis = function() {
        for (var i = 0; i < 3; i++) {
            this.planes[i].update();
            this.planes[i].mesh.visible = $("#volvis").prop("checked");
        }
        this.schedule();
    };
    module.Viewer.prototype.update_leftvis = function() {
        this.meshes.left.visible = $("#leftvis").prop("checked");
        this.schedule();
    };
    module.Viewer.prototype.update_rightvis = function() {
        this.meshes.right.visible = $("#rightvis").prop("checked");
        this.schedule();
    };
    module.Viewer.prototype.update_projection = function() {
        if ($("#projpersp").prop("checked")) {
            this.setState("projection", "perspective");
        } else {
            this.setState("projection", "orthographic");
        }
        this.schedule();
    };
    // module.Viewer.prototype.saveflat = function(height, posturl) {
    //     var width = height * this.flatlims[1][0] / this.flatlims[1][1];;
    //     var roistate = $(this.object).find("#roishow").attr("checked");
    //     this.screenshot(width, height, function() { 
    //         this.reset_view(false, height); 
    //         $(this.object).find("#roishow").attr("checked", false);
    //         $(this.object).find("#roishow").change();
    //     }.bind(this), function(png) {
    //         $(this.object).find("#roishow").attr("checked", roistate);
    //         $(this.object).find("#roishow").change();
    //         this.controls.target.set(0,0,0);
    //         this.roipack.saveSVG(png, posturl);
    //     }.bind(this));
    // }; 
    module.Viewer.prototype.setMix = function(val) {
        var num = this.meshes.left.geometry.morphTargets.length;
        var flat = num - 1;
        var n1 = Math.floor(val * num)-1;
        var n2 = Math.ceil(val * num)-1;

        for (var h in this.meshes) {
            var hemi = this.meshes[h];
            if (hemi !== undefined) {
                for (var i=0; i < num; i++) {
                    hemi.morphTargetInfluences[i] = 0;
                }

                if (this.flatlims !== undefined)
                    this.uniforms.hide_mwall.value = (n2 == flat);

                hemi.morphTargetInfluences[n2] = (val * num)%1;
                if (n1 >= 0)
                    hemi.morphTargetInfluences[n1] = 1 - (val * num)%1;
            }
        }
        if (this.flatlims !== undefined) {
            this.flatmix = n2 == flat ? (val*num-.000001)%1 : 0;
            this.setPivot(this._pivot);
            this.update_spec();
            if (n2 == flat) {
                for (var i = 0; i < 3; i++) {
                    this.planes[i].update();
                    this.planes[i].mesh.visible = false;
                }
                $("#volvis").attr("disabled", true);
            } else {
                $("#volvis").removeAttr("disabled");
                this.update_volvis();
            }
        }
        $(this.object).find("#mix").slider("value", val);
        
        this.dispatchEvent({type:"mix", flat:this.flatmix, mix:val});
        this.figure.notify("setmix", this, [val]);
        this.schedule();
    }; 
    module.Viewer.prototype.setPivot = function (val, fromuser) {
        this._pivot = val;
        val = this.flatmix * 180 + (1-this.flatmix) * this._pivot;
        $(this.object).find("#pivot").slider("option", "value", val);
        var names = {left:1, right:-1};
        var frac = Math.abs(val/180) * (1-this.flatmix);
        if (val > 0) {
            for (var name in names) {
                this.pivot[name].front.rotation.z = 0;
                this.pivot[name].back.rotation.z = val*Math.PI/180 * names[name]/ 2;
                // Move hemispheres so that focus stays at center
                this.pivot[name].back.position.y = frac * this.meshes.right.geometry.boundingBox.min.y + ((1-frac) * this.pivot[name].back.orig_position.y);

            }
        } else {
            for (var name in names) {
                this.pivot[name].back.rotation.z = 0;
                // Make sure back position is reset before front pivoting
                this.pivot[name].back.position.y = this.pivot[name].back.orig_position.y;
                this.pivot[name].front.rotation.z = val*Math.PI/180 * names[name] / 2;

                // Move hemispheres so that focus stays at center
                this.pivot[name].front.position.y = frac * (this.meshes.right.geometry.boundingBox.max.y - this.pivot[name].front.orig_position.y) + ((1-frac) * this.pivot[name].front.orig_position.y);
            }
        }
        this.figure.notify("setpivot", this, [val]);
        this.schedule();
    };
    module.Viewer.prototype.setShift = function(val) {
        this.pivot.left.front.position.x = -val;
        this.pivot.right.front.position.x = val;
        this.figure.notify('setshift', this, [val]);
        this.schedule();
    };
    module.Viewer.prototype.update_spec = function() {
        var s = this.specular * (1 - this.flatmix);
        this.uniforms.specular.value.set(s, s, s);
    };
    module.Viewer.prototype.addData = function(data) {
        if (!(data instanceof Array))
            data = [data];

        var name, view;

        var handle = "<div class='handle'><span class='ui-icon ui-icon-carat-2-n-s'></span></div>";
        for (var i = 0; i < data.length; i++) {
            view = data[i];
            name = view.name;
            this.dataviews[name] = view;

            var found = false;
            $(this.object).find("#datasets li").each(function() {
                found = found || ($(this).text() == name);
            })
            if (!found)
                $(this.object).find("#datasets").append("<li class='ui-corner-all'>"+handle+name+"</li>");
        }
        
        this.setData(data[0].name);
    };
    module.Viewer.prototype.setData = function(name) {
        if (this.state == "play")
            this.playpause();

        if (name instanceof Array) {
            if (name.length == 1) {
                name = name[0];
            } else if (name.length == 2) {
                var dv1 = this.dataviews[name[0]];
                var dv2 = this.dataviews[name[1]];
                //Can't create 2D data view when the view is already 2D!
                if (dv1.data.length > 1 || dv2.data.length > 1)
                    return false;

                return this.addData(dataset.makeFrom(dv1, dv2));
            } else {
                return false;
            }
        }

        this.active = this.dataviews[name];
        this.dispatchEvent({type:"setData", data:this.active});

        if (this.active.data[0].raw) {
            $("#color_fieldset").fadeTo(0.15, 0);
        } else {
            if (this.active.cmap !== undefined)
                this.setColormap(this.active.cmap);
            $("#color_fieldset").fadeTo(0.15, 1);
        }

        $.when(this.cmapload, this.loaded).done(function() {
            this.setVoxView(this.active.filter, viewopts.voxlines);
            //this.active.init(this.uniforms, this.meshes);
            $(this.object).find("#vrange").slider("option", {min: this.active.data[0].min, max:this.active.data[0].max});
            if (this.active.data.length > 1) {
                $(this.object).find("#vrange2").slider("option", {min: this.active.data[1].min, max:this.active.data[1].max});
                this.setVminmax(this.active.vmin[0], this.active.vmax[0], 0);
                this.setVminmax(this.active.vmin[1], this.active.vmax[1], 1);
                $(this.object).find("#vminmax2").show();
            } else {
                $(this.object).find("#vminmax2").hide();
                this.setVminmax(this.active.vmin, this.active.vmax, 0);
            }

            this.setupStim();
            
            $(this.object).find("#datasets li").each(function() {
                if ($(this).text() == name)
                    $(this).addClass("ui-selected");
                else
                    $(this).removeClass("ui-selected");
            })

            $(this.object).find("#datasets").val(name);
            if (typeof(this.active.description) == "string") {
                var html = name+"<div class='datadesc'>"+this.active.description+"</div>";
                $(this.object).find("#dataname").html(html).show();
            } else {
                $(this.object).find("#dataname").text(name).show();
            }
            this.schedule();
        }.bind(this));
    };
    module.Viewer.prototype.nextData = function(dir) {
        var i = 0, found = false;
        var datasets = [];
        $(this.object).find("#datasets li").each(function() {
            if (!found) {
                if (this.className.indexOf("ui-selected") > 0)
                    found = true;
                else
                    i++;
            }
            datasets.push($(this).text())
        });
        if (dir === undefined)
            dir = 1

        this.setData([datasets[(i+dir).mod(datasets.length)]]);
    };
    module.Viewer.prototype.rmData = function(name) {
        delete this.datasets[name];
        $(this.object).find("#datasets li").each(function() {
            if ($(this).text() == name)
                $(this).remove();
        })
    };
    module.Viewer.prototype.setupStim = function() {
        if (this.active.data[0].movie) {
            $(this.object).find("#moviecontrols").show();
            $(this.object).find("#bottombar").addClass("bbar_controls");
            $(this.object).find("#movieprogress>div").slider("option", {min:0, max:this.active.length});
            this.active.data[0].loaded.progress(function(idx) {
                var pct = idx / this.active.frames * 100;
                $(this.object).find("#movieprogress div.ui-slider-range").width(pct+"%");
            }.bind(this)).done(function() {
                $(this.object).find("#movieprogress div.ui-slider-range").width("100%");
            }.bind(this));
            
            this.active.loaded.done(function() {
                this.setFrame(0);
            }.bind(this));

            if (this.active.stim && figure) {
                figure.setSize("right", "30%");
                this.movie = figure.add(jsplot.MovieAxes, "right", false, this.active.stim);
                this.movie.setFrame(0);
            }
        } else {
            $(this.object).find("#moviecontrols").hide();
            $(this.object).find("#bottombar").removeClass("bbar_controls");
            this.active.set(this.uniforms, 0);
        }
    };

    module.Viewer.prototype.setColormap = function(cmap) {
        if (typeof(cmap) == 'string') {
            if (this.cmapnames[cmap] !== undefined)
                $(this.object).find("#colormap").ddslick('select', {index:this.cmapnames[cmap]});

            return;
        } else if (cmap instanceof Image || cmap instanceof Element) {
            this.colormap = new THREE.Texture(cmap);
        } else if (cmap instanceof NParray) {
            var ncolors = cmap.shape[cmap.shape.length-1];
            if (ncolors != 3 && ncolors != 4)
                throw "Invalid colormap shape"
            var height = cmap.shape.length > 2 ? cmap.shape[1] : 1;
            var fmt = ncolors == 3 ? THREE.RGBFormat : THREE.RGBAFormat;
            this.colormap = new THREE.DataTexture(cmap.data, cmap.shape[0], height, fmt);
        }

        if (this.active !== null)
            this.active.cmap = $(this.object).find("#colormap .dd-selected label").text();

        this.cmapload = $.Deferred();
        this.cmapload.done(function() {
            this.colormap.needsUpdate = true;
            this.colormap.premultiplyAlpha = true;
            this.colormap.flipY = true;
            this.colormap.minFilter = THREE.LinearFilter;
            this.colormap.magFilter = THREE.LinearFilter;
            this.uniforms.colormap.texture = this.colormap;
            this.loaded.done(this.schedule.bind(this));
            
            if (this.colormap.image.height > 8) {
                $(this.object).find(".vcolorbar").show();
            } else {
                $(this.object).find(".vcolorbar").hide();
            }
        }.bind(this));

        if ((cmap instanceof Image || cmap instanceof Element) && !cmap.complete) {
            cmap.addEventListener("load", function() { this.cmapload.resolve(); }.bind(this));
        } else {
            this.cmapload.resolve();
        }
    };

    module.Viewer.prototype.setVminmax = function(vmin, vmax, dim) {
        if (dim === undefined)
            dim = 0;
        this.uniforms.vmin.value[dim] = vmin;
        this.uniforms.vmax.value[dim] = vmax;

        var range, min, max;
        if (dim == 0) {
            range = "#vrange"; min = "#vmin"; max = "#vmax";
        } else {
            range = "#vrange2"; min = "#vmin2"; max = "#vmax2";
        }

        if (vmax > $(this.object).find(range).slider("option", "max")) {
            $(this.object).find(range).slider("option", "max", vmax);
            this.active.data[dim].max = vmax;
        } else if (vmin < $(this.object).find(range).slider("option", "min")) {
            $(this.object).find(range).slider("option", "min", vmin);
            this.active.data[dim].min = vmin;
        }
        $(this.object).find(range).slider("values", [vmin, vmax]);
        $(this.object).find(min).val(vmin);
        $(this.object).find(max).val(vmax);
        if (this.active.vmin instanceof Array) {
            this.active.vmin[dim] = vmin;
            this.active.vmax[dim] = vmax;
        } else {
            this.active.vmin = vmin;
            this.active.vmax = vmax;
        }

        this.schedule();
    };

    module.Viewer.prototype.startCmapSearch = function() {
        var sr = $(this.object).find("#cmapsearchresults"),
        cm = $(this.object).find("#colormap"),
        sb = $(this.object).find("#cmapsearchbox"),
        v = this;

        sb.val("");
        sb.css("width", cm.css("width"));
        sb.css("height", cm.css("height"));
        sr.show();
        sb.keyup(function(e) {
            if (e.keyCode == 13) { // enter
                try {this.setColormap($(sr[0]).find(".selected_sr input")[0].value);}
                finally {this.stopCmapSearch();}
            } if (e.keyCode == 27) { // escape
                this.stopCmapSearch();
            }
            var value = sb[0].value.trim();
            sr.empty();
            if (value.length > 0) {
                for (var k in this.cmapnames) {
                    if (k.indexOf(value) > -1) {
                        sr.append($($(this.object).find("#colormap li")[this.cmapnames[k]]).clone());
                    }
                }
                $(sr[0].firstChild).addClass("selected_sr");
                sr.children().mousedown(function() {
                    try {v.setColormap($(this).find("input")[0].value);}
                    finally {v.stopCmapSearch();}
                });
            }
        }.bind(this)).show();
        sb.focus();
        sb.blur(function() {this.stopCmapSearch();}.bind(this));
    };

    module.Viewer.prototype.stopCmapSearch = function() {
        var sr = $(this.object).find("#cmapsearchresults"),
        sb = $(this.object).find("#cmapsearchbox");
        sr.hide().empty();
        sb.hide();
    };

    module.Viewer.prototype.setFrame = function(frame) {
        if (frame > this.active.length) {
            frame -= this.active.length;
            this._startplay += this.active.length;
        }

        this.frame = frame;
        this.active.set(this.uniforms, frame);
        $(this.object).find("#movieprogress div").slider("value", frame);
        $(this.object).find("#movieframe").attr("value", frame);
        this.schedule();
    };

    module.Viewer.prototype.setVoxView = function(interp, lines) {
        viewopts.voxlines = lines;
        if (this.active.filter != interp) {
            this.active.setFilter(interp);
        }
        this.active.init(this.uniforms, this.meshes, this.flatlims !== undefined);
        $(this.object).find("#datainterp option").attr("selected", null);
        $(this.object).find("#datainterp option[value="+interp+"]").attr("selected", "selected");
    };

    module.Viewer.prototype.playpause = function() {
        if (this.state == "pause") {
            //Start playing
            this._startplay = (new Date()) - (this.frame * this.active.rate) * 1000;
            this.state = "play";
            this.schedule();
            $(this.object).find("#moviecontrols img").attr("src", "resources/images/control-pause.png");
        } else {
            this.state = "pause";
            $(this.object).find("#moviecontrols img").attr("src", "resources/images/control-play.png");
        }
        this.figure.notify("playtoggle", this);
    };
    module.Viewer.prototype.getImage = function(width, height, post) {
        if (width === undefined)
            width = this.canvas.width();
        
        if (height === undefined)
            height = width * this.canvas.height() / this.canvas.width();

        console.log(width, height);
        var renderbuf = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format:THREE.RGBAFormat,
            stencilBuffer:false,
        });

        var clearAlpha = this.renderer.getClearAlpha();
        var clearColor = this.renderer.getClearColor();
        var oldw = this.canvas.width(), oldh = this.canvas.height();
        this.camera.setSize(width, height);
        this.camera.updateProjectionMatrix();
        this.controls._zoom(1.0) // To assure orthographic zoom is set correctly
        //this.renderer.setSize(width, height);
        this.renderer.setClearColorHex(0x0, 0);
        this.renderer.render(this.scene, this.camera, renderbuf);
        //this.renderer.setSize(oldw, oldh);
        this.renderer.setClearColorHex(clearColor, clearAlpha);
        this.camera.setSize(oldw, oldh);
        this.camera.updateProjectionMatrix();

        var img = mriview.getTexture(this.renderer.context, renderbuf)
        if (post !== undefined)
            $.post(post, {png:img.toDataURL()});
        return img;
    };
    var _bound = false;
    module.Viewer.prototype._bindUI = function() {
        $(window).scrollTop(0);
        $(window).resize(function() { this.resize(); }.bind(this));
        this.canvas.resize(function() { this.resize(); }.bind(this));
        //These are events that should only happen once, regardless of multiple views
        if (!_bound) {
            _bound = true;
            window.addEventListener( 'keydown', function(e) {
                btnspeed = 0.5;
                if (e.target.tagName == "INPUT" && e.target.type == "text")
                    return;
                if (e.keyCode == 32) {         //space
                    if (this.active.data[0].movie)
                        this.playpause();
                    e.preventDefault();
                    e.stopPropagation();
                } else if (e.keyCode == 82) { //r
                    this.animate([{idx:btnspeed, state:"target", value:this.surfcenter},
                                  {idx:btnspeed, state:"mix", value:0.0}]);
                } else if (e.keyCode == 73) { //i
                    this.animate([{idx:btnspeed, state:"mix", value:0.5}]);
                } else if (e.keyCode == 70) { //f
                    this.animate([{idx:btnspeed, state:"target", value:[0,0,0]},
                                  {idx:btnspeed, state:"mix", value:1.0}]);
                } else if (e.keyCode == 37) { //left
		    this.animate([{idx:btnspeed, state:"azimuth", value:(Math.floor(this.getState("azimuth")/90)+1)*90.5}]);
		} else if (e.keyCode == 39) { //right
		    this.animate([{idx:btnspeed, state:"azimuth", value:(Math.floor(this.getState("azimuth")/90)-1)*90.5}]);
		} else if (e.keyCode == 38) { //up
		    this.animate([{idx:btnspeed, state:"altitude", value:(Math.round(this.getState("altitude")/90)-1)*90.5}]);
		} else if (e.keyCode == 40) { //down
		    this.animate([{idx:btnspeed, state:"altitude", value:(Math.round(this.getState("altitude")/90)+1)*90.5}]);
		}
            }.bind(this));
        }
        window.addEventListener( 'keydown', function(e) {
            if (e.target.tagName == "INPUT" && e.target.type == "text")
                return;
            if (e.keyCode == 107 || e.keyCode == 187) { //+
                this.nextData(1);
            } else if (e.keyCode == 109 || e.keyCode == 189) { //-
                this.nextData(-1);
            } else if (e.keyCode == 68) { //d
                if (this.uniforms.dataAlpha.value < 1)
                    this.uniforms.dataAlpha.value = 1;
                else
                    this.uniforms.dataAlpha.value = 0;
                this.schedule();
            } else if (e.keyCode == 76) { //l
                var box = $(this.object).find("#labelshow");
                box.attr("checked", box.attr("checked") == "checked" ? null : "checked");
                this.labelshow = !this.labelshow;
                this.schedule();
                e.stopPropagation();
                e.preventDefault();
            } else if (e.keyCode == 81) { //q
                this.planes[0].next();
            } else if (e.keyCode == 87) { //w
                this.planes[0].prev();
            } else if (e.keyCode == 65) { //a
                this.planes[1].next();
            } else if (e.keyCode == 83) { //s
                this.planes[1].prev();
            } else if (e.keyCode == 90) { //z
                this.planes[2].next();
            } else if (e.keyCode == 88) { //x
                this.planes[2].prev();
            }
        }.bind(this));
        var _this = this;
        $(this.object).find("#mix").slider({
            min:0, max:1, step:.001,
            slide: function(event, ui) { this.setMix(ui.value); }.bind(this)
        });
        $(this.object).find("#pivot").slider({
            min:-180, max:180, step:.01,
            slide: function(event, ui) { this.setPivot(ui.value); }.bind(this)
        });

        $(this.object).find("#shifthemis").slider({
            min:0, max:100, step:.01,
            slide: function(event, ui) { this.setShift(ui.value); }.bind(this)
        });

        if ($(this.object).find("#color_fieldset").length > 0) {
            $(this.object).find("#colormap").ddslick({ width:296, height:350, 
                onSelected: function() { 
                    this.setColormap($(this.object).find("#colormap .dd-selected-image")[0]);
                }.bind(this)
            });
            $(this.object).find("#cmapsearch").click(function() {
                this.startCmapSearch();
            }.bind(this));

            $(this.object).find("#vrange").slider({ 
                range:true, width:200, min:0, max:1, step:.001, values:[0,1],
                slide: function(event, ui) { this.setVminmax(ui.values[0], ui.values[1]); }.bind(this)
            });
            $(this.object).find("#vmin").change(function() { 
                this.setVminmax(
                    parseFloat($(this.object).find("#vmin").val()), 
                    parseFloat($(this.object).find("#vmax").val())
                ); 
            }.bind(this));
            $(this.object).find("#vmax").change(function() { 
                this.setVminmax(
                    parseFloat($(this.object).find("#vmin").val()), 
                    parseFloat($(this.object).find("#vmax").val())
                    ); 
            }.bind(this));

            $(this.object).find("#vrange2").slider({ 
                range:true, width:200, min:0, max:1, step:.001, values:[0,1], orientation:"vertical",
                slide: function(event, ui) { this.setVminmax(ui.values[0], ui.values[1], 1); }.bind(this)
            });
            $(this.object).find("#vmin2").change(function() { 
                this.setVminmax(
                    parseFloat($(this.object).find("#vmin2").val()), 
                    parseFloat($(this.object).find("#vmax2").val()),
                    1);
            }.bind(this));
            $(this.object).find("#vmax2").change(function() { 
                this.setVminmax(
                    parseFloat($(this.object).find("#vmin2").val()), 
                    parseFloat($(this.object).find("#vmax2").val()), 
                    1); 
            }.bind(this));            
        }

        //this fires when the radio button for selecting coord systems is changed
        $(this.object).find(".radio").change(function() { //the .radio class is specifically for coord space selection, not all radio buttons
            space = $(this.object).find(".radio:checked").val();
            ptidx = $(this.object).find("#ptidx").val();
            pthem = $(this.object).find("#pthem").val();
            if (pthem==="left")
                h = this.meshes.left.geometry ;
            else if (pthem==="right")
                h = this.meshes.right.geometry ;

            if (typeof h !== 'undefined') {
                if (space==="magnet") {
                    coordarray = h.attributes.position ;
                    $(this.object).find("#coordsys_mag").prop('checked',true) ;
                    mniidx = (ptidx)*coordarray.itemSize  ;
                    px = coordarray.array[mniidx] ;
                    py = coordarray.array[mniidx+1] ;
                    pz = coordarray.array[mniidx+2] ;
                    console.log('mm coords: '+px+', '+py+', '+pz) ;
                    var coord = new THREE.Vector3(px, py, pz);
                    var vec = this.uniforms.volxfm.value[0].multiplyVector3(coord);
                    mnix = vec.x ;
                    mniy = vec.y ;
                    mniz = vec.z ;
                    $(this.object).find(".units").text("vox");
                }
                else { //mni or undefined
                    coordarray = h.attributes.mnicoords ;
                    $(this.object).find("#coordsys_mni").prop('checked',true) ;
                    mniidx = (ptidx)*coordarray.itemSize  ;
                    mnix = coordarray.array[mniidx] ;
                    mniy = coordarray.array[mniidx+1] ;
                    mniz = coordarray.array[mniidx+2] ;
                    $(this.object).find(".units").text("mm");
                }
                $(this.object).find("#mniX").val(mnix.toFixed(2)) ;
                $(this.object).find("#mniY").val(mniy.toFixed(2)) ;
                $(this.object).find("#mniZ").val(mniz.toFixed(2)) ;
            }
        }.bind(this));

        //this fires when the coordinate entry form is submitted -- it's time to look up the nearest vertex
        $(this.object).find("#mniform").submit(function() {
                x = $(this.object).find("#mniX").val();
                y = $(this.object).find("#mniY").val();
                z = $(this.object).find("#mniZ").val();
                space = $(this.object).find(".radio:checked").val();
                if (space==="magnet") {
                    //do some conversions to eventually get mm coords, which are what's stored in the kdtree
                    var coord = new THREE.Vector3(x, y, z);
                    var xfm = this.uniforms.volxfm.value[0] ;
                    var xfm_inv = new THREE.Matrix4() ;
                    xfm_inv = xfm_inv.getInverse(xfm);
                    xfm_inv.multiplyVector3(coord) ;
                    var left = this.picker.lkdt.nearest([coord.x, coord.y, coord.z], 1)[0];
                    var right = this.picker.rkdt.nearest([coord.x, coord.y, coord.z], 1)[0];
                }
                else { //mni or undefined
                    var left = this.picker.mni_lkdt.nearest([x, y, z], 1)[0];
                    var right = this.picker.mni_rkdt.nearest([x, y, z], 1)[0];
                }
                
                if (left[1] < right[1]) {
                    ptidx = left[0][3] ;
                    h = this.meshes.left.geometry ;
                    hemi = "left" ;
                }
                else {
                    ptidx = right[0][3] ;
                    h = this.meshes.right.geometry ;
                    hemi = "right" ;
                }

                this.picker.addMarker(hemi, ptidx, false);
                
                if (space==="magnet")
                    coordarray = h.attributes.position ;
                else //mni or undefined
                    coordarray = h.attributes.mnicoords ;

                mniidx = (ptidx)*coordarray.itemSize  ;
                mnix = coordarray.array[mniidx] ;
                mniy = coordarray.array[mniidx+1] ;
                mniz = coordarray.array[mniidx+2] ;

                //if we're in MNI space, we can just update the box with the coords. if not though, we need to get vox (not mm) coords...
                if (space==="magnet") {
                    var coord2 = new THREE.Vector3(mnix, mniy, mniz);
                    var vec2 = this.uniforms.volxfm.value[0].multiplyVector3(coord2);
                    mnix = vec2.x ;
                    mniy = vec2.y ;
                    mniz = vec2.z ;
                }

                //update the hidden fields ptidx and pthem with the picked vertex's info
                $(this.object).find("#ptidx").val(ptidx) ;
                $(this.object).find("#pthem").val(hemi) ;

                //update coords
                $(this.object).find("#mniX").val(mnix.toFixed(2)) ;
                $(this.object).find("#mniY").val(mniy.toFixed(2)) ;
                $(this.object).find("#mniZ").val(mniz.toFixed(2)) ;
                return(0); //do not reload page
        }.bind(this));


        // Setup controls for multiple overlays
        var updateOverlays = function() {
            this.roipack.update(this.renderer).done(function(tex){
                this.uniforms.map.texture = tex;
                this.schedule();
            }.bind(this));
        }.bind(this);

        for (var li=0; li<disp_layers.length; li++) {
            var layername = disp_layers[li];

            $(this.object).find("#"+layername+"_linewidth").slider({
                min:.5, max:10, step:.1, 
		value: disp_defaults[layername]['line_width'],
                change: updateOverlays,
            });
            $(this.object).find("#"+layername+"_linealpha").slider({
                min:0, max:1, step:.001, 
		value: disp_defaults[layername]['line_alpha'],
                change: updateOverlays,
            });
            $(this.object).find("#"+layername+"_fillalpha").slider({
                min:0, max:1, step:.001, 
		value: disp_defaults[layername]['fill_alpha'],
                change: updateOverlays,
            });
            $(this.object).find("#"+layername+"_shadowalpha").slider({
                min:0, max:20, step:1, value:4,
                change: updateOverlays,
            });
            $(this.object).find("#"+layername+"_linecolor").minicolors({
		change: updateOverlays,
		defaultValue: disp_defaults[layername]['line_color']
	    });
            $(this.object).find("#"+layername+"_fillcolor").minicolors({
		change: updateOverlays,
		defaultValue: disp_defaults[layername]['fill_color']
	    });
            $(this.object).find("#"+layername+"_shadowcolor").minicolors({change: updateOverlays});

            var _this = this;
            $(this.object).find("#"+layername+"show").change(function() {
                console.log("Toggling: " + "#"+$(this).attr("layername"));
                var el = $(_this.roipack.svgroi).find("#"+$(this).attr("layername"));
                if (this.checked)
                    el.css('display','inline');
                else
                    el.css('display','none');

                updateOverlays();
            }).attr("layername", layername);

            $(this.object).find("#"+layername+"labelshow").change(function() {
                // this.labelshow = !this.labelshow;
                _this.roipack.layer_label_visibility[$(this).attr("layername")] = this.checked;
                _this.roipack.update_labels();
                _this.roipack.labels.setMix(_this.flatmix);
                _this.schedule();
            }).attr("layername", layername);
        }
        $(this.object).find("#overlay_fieldset").tabs();

        $(this.object).find("#layer_curvalpha").slider({ min:0, max:1, step:.001, value:1, slide:function(event, ui) {
            this.uniforms.curvAlpha.value = ui.value;
            this.schedule();
        }.bind(this)})
        $(this.object).find("#layer_curvmult").slider({ min:.001, max:2, step:.001, value:1, slide:function(event, ui) {
            this.uniforms.curvScale.value = ui.value;
            this.schedule();
        }.bind(this)})
        $(this.object).find("#layer_curvlim").slider({ min:0, max:.5, step:.001, value:.2, slide:function(event, ui) {
            this.uniforms.curvLim.value = ui.value;
            this.schedule();
        }.bind(this)})
	$(this.object).find("#layer_curvmean").slider({ min:0, max:1.0, step:.001, value:.5, slide:function(event, ui) {
            this.uniforms.curvMean.value = ui.value;
            this.schedule();
        }.bind(this)})
        $(this.object).find("#layer_dataalpha").slider({ min:0, max:1, step:.001, value:1.0, slide:function(event, ui) {
            this.uniforms.dataAlpha.value = ui.value;
            this.schedule();
        }.bind(this)})
        $(this.object).find("#layer_specularity").slider({ min:0, max:1, step:.001, value:this.specular, slide:function(event, ui) {
            this.specular = ui.value;
            this.update_spec();
            this.schedule();
        }.bind(this)})
        $(this.object).find("#layer_hatchalpha").slider({ min:0, max:1, step:.001, value:1, slide:function(event, ui) {
            this.uniforms.hatchAlpha.value = ui.value;
            this.schedule();
        }.bind(this)})
        $(this.object).find("#layer_hatchcolor").minicolors({close: function(hex, rgb) {
            this.uniforms.hatchColor.value.set(rgb.r / 255, rgb.g / 255, rgb.b / 255);
            this.schedule();
        }.bind(this)});

        $(this.object).find("#volvis").change(this.update_volvis.bind(this));
        $(this.object).find("#leftvis").change(this.update_leftvis.bind(this));
        $(this.object).find("#rightvis").change(this.update_rightvis.bind(this));
        $(this.object).find("#projpersp").change(this.update_projection.bind(this));
        $(this.object).find("#projortho").change(this.update_projection.bind(this));

        $(this.object).find("#voxline_show").change(function() {
            viewopts.voxlines = $(this.object).find("#voxline_show")[0].checked;
            this.setVoxView(this.active.filter, viewopts.voxlines);
            this.schedule();
        }.bind(this));
        $(this.object).find("#voxline_color").minicolors({ close: function(hex, rgb) {
            this.uniforms.voxlineColor.value.set(rgb.r / 255, rgb.g / 255, rgb.b/255);
            this.schedule();
        }.bind(this)});
        $(this.object).find("#voxline_width").slider({ min:.001, max:.1, step:.001, value:viewopts.voxline_width, slide:function(event, ui) {
            this.uniforms.voxlineWidth.value = ui.value;
            this.schedule();
        }.bind(this)});
        $(this.object).find("#datainterp").change(function() {
            this.setVoxView($(this.object).find("#datainterp").val(), viewopts.voxlines);
            this.schedule();
        }.bind(this));
        $(this.object).find("#thicklayers").slider({ min:1, max:32, step:1, value:1, slide:function(event, ui)  {
            if (ui.value == 1)
                $(this.object).find("#thickmix_row").show();
            else 
                $(this.object).find("#thickmix_row").hide();
            this.uniforms.nsamples.value = ui.value;
            this.active.init(this.uniforms, this.meshes, this.flatlims !== undefined, this.frames);
            this.schedule();
        }.bind(this)});
        $(this.object).find("#thickmix").slider({ min:0, max:1, step:.001, value:0.5, slide:function(event, ui) {
            this.figure.notify("setdepth", this, [ui.value]);
            this.uniforms.thickmix.value = ui.value;
            this.schedule();
        }.bind(this)})

        $(this.object).find("#resetflat").click(function() {
            this.reset_view();
        }.bind(this));

        //Dataset box
        var setdat = function(event, ui) {
            var names = [];
            $(this.object).find("#datasets li.ui-selected").each(function() { names.push($(this).text()); });
            this.setData(names);
        }.bind(this)
        $(this.object).find("#datasets")
            .sortable({ 
                handle: ".handle",
                stop: setdat,
             })
            .selectable({
                selecting: function(event, ui) {
                    var selected = $(this.object).find("#datasets li.ui-selected, #datasets li.ui-selecting");
                    if (selected.length > 2) {
                        $(ui.selecting).removeClass("ui-selecting");
                    }
                }.bind(this),
                unselected: function(event, ui) {
                    var selected = $(this.object).find("#datasets li.ui-selected, #datasets li.ui-selecting");
                    if (selected.length < 1) {
                        $(ui.unselected).addClass("ui-selected");
                    }
                }.bind(this),
                stop: setdat,
            });

        $(this.object).find("#moviecontrol").click(this.playpause.bind(this));

        $(this.object).find("#movieprogress>div").slider({min:0, max:1, step:.001,
            slide: function(event, ui) { 
                this.setFrame(ui.value); 
                this.figure.notify("setFrame", this, [ui.value]);
            }.bind(this)
        });
        $(this.object).find("#movieprogress>div").append("<div class='ui-slider-range ui-widget-header'></div>");

        $(this.object).find("#movieframe").change(function() { 
            _this.setFrame(this.value); 
            _this.figure.notify("setFrame", _this, [this.value]);
        });
    };
    module.Viewer.prototype._makeBtns = function(names) {
        var btnspeed = 0.5; // How long should folding/unfolding animations take?
        var td, btn, name;
        td = document.createElement("td");
        btn = document.createElement("button");
        btn.setAttribute("title", "Reset to fiducial view of the brain (Hotkey: R)");
        btn.innerHTML = "Fiducial";
        td.setAttribute("style", "text-align:left;width:150px;");
        btn.addEventListener("click", function() {
            this.animate([{idx:btnspeed, state:"target", value:[0,0,0]},
                          {idx:btnspeed, state:"mix", value:0.0}]);
        }.bind(this));
        td.appendChild(btn);
        $(this.object).find("#mixbtns").append(td);

        var nameoff = this.flatlims === undefined ? 0 : 1;
        for (var i = 0; i < names.length; i++) {
            name = names[i][0].toUpperCase() + names[i].slice(1);
            td = document.createElement("td");
            btn = document.createElement("button");
            btn.innerHTML = name;
            btn.setAttribute("title", "Switch to the "+name+" view of the brain");

            btn.addEventListener("click", function(j) {
                this.animate([{idx:btnspeed, state:"mix", value: (j+1) / (names.length+nameoff)}]);
            }.bind(this, i));
            td.appendChild(btn);
            $(this.object).find("#mixbtns").append(td);
        }

        if (this.flatlims !== undefined) {
            td = document.createElement("td");
            btn = document.createElement("button");
            btn.innerHTML = "Flat";
            btn.setAttribute("title", "Switch to the flattened view of the brain (Hotkey: F)");
            td.setAttribute("style", "text-align:right;width:150px;");
            btn.addEventListener("click", function() {
                this.animate([{idx:btnspeed, state:"mix", value:1.0}]);
            }.bind(this));
            td.appendChild(btn);
            $(this.object).find("#mixbtns").append(td);
        }

        $(this.object).find("#mix, #pivot, #shifthemis").parent().attr("colspan", names.length+2);
    };
    module.Viewer.prototype._makeMesh = function(geom, shader) {
        //Creates the pivots and the mesh object given the geometry and shader
        var mesh = new THREE.Mesh(geom, shader);
        mesh.doubleSided = true;
        mesh.position.y = -this.flatoff[1];
        mesh.updateMatrix();
        var pivots = {back:new THREE.Object3D(), front:new THREE.Object3D()};
        pivots.back.add(mesh);
        pivots.front.add(pivots.back);
        pivots.back.position.y = geom.boundingBox.min.y - geom.boundingBox.max.y;
        pivots.front.position.y = geom.boundingBox.max.y - geom.boundingBox.min.y + this.flatoff[1];
        pivots.back.orig_position = pivots.back.position.clone();
        pivots.front.orig_position = pivots.front.position.clone();

        return {mesh:mesh, pivots:pivots};
    };

    return module;
}(mriview || {}));
