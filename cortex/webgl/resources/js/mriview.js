var flatscale = .25;

var samplers = {
    trilinear: "trilinear",
    nearest: "nearest",
    nearlin: "nearest",
    debug: "debug",
}

function MRIview() { 
    //Allow objects to listen for mix updates
    THREE.EventTarget.call( this );
    // scene and camera
    this.camera = new THREE.CombinedCamera( $("#brain").width(), $("#brain").height(), 45, 1.0, 1000, 1., 1000. );
    this.camera.position.set(300, 300, 300);
    this.camera.up.set(0,0,1);

    this.scene = new THREE.Scene();
    this.scene.add( this.camera );
    this.controls = new THREE.LandscapeControls( this.camera );
    this.controls.addEventListener("change", this.schedule.bind(this));
    this.addEventListener("resize", function(event) {
        this.controls.resize(event.width, event.height);
    }.bind(this));
    this.controls.addEventListener("mousedown", function(event) {
        if (this.fastshader) {
            this.meshes.left.material = this.fastshader;
            this.meshes.right.material = this.fastshader;
        }
    }.bind(this));
    this.controls.addEventListener("mouseup", function(event) {
        if (this.fastshader) {
            this.meshes.left.material = this.shader;
            this.meshes.right.material = this.shader;
            this.schedule();
        }
    }.bind(this));
    
    this.light = new THREE.DirectionalLight( 0xffffff );
    this.light.position.set( -200, -200, 1000 ).normalize();
    this.camera.add( this.light );
    this.flatmix = 0;
    this.frame = 0;

    // renderer
    this.renderer = new THREE.WebGLRenderer({ 
        antialias: true, 
        preserveDrawingBuffer:true, 
        canvas:$("#brain")[0] 
    });
    this.renderer.sortObjects = true;
    this.renderer.setClearColorHex( 0x0, 1 );
    this.renderer.context.getExtension("OES_texture_float");
    this.renderer.context.getExtension("OES_standard_derivatives");
    this.renderer.setSize( $("#brain").width(), $("#brain").height() );

    this.uniforms = THREE.UniformsUtils.merge( [
        THREE.UniformsLib[ "lights" ],
        {
            diffuse:    { type:'v3', value:new THREE.Vector3( .8,.8,.8 )},
            specular:   { type:'v3', value:new THREE.Vector3( 1,1,1 )},
            emissive:   { type:'v3', value:new THREE.Vector3( .2,.2,.2 )},
            shininess:  { type:'f',  value:200},

            thickmix:   { type:'f',  value:0.5},
            framemix:   { type:'f',  value:0},
            hatchrep:   { type:'v2', value:new THREE.Vector2(108, 40) },
            offsetRepeat:{type:'v4', value:new THREE.Vector4( 0, 0, 1, 1 ) },
            
            hatch:      { type:'t',  value:0, texture: makeHatch() },
            colormap:   { type:'t',  value:1, texture: null },
            map:        { type:'t',  value:2, texture: null },
            data:       { type:'tv', value:3, texture: [null, null, null, null]},
            mosaic:     { type:'v2v', value:[new THREE.Vector2(6, 6), new THREE.Vector2(6, 6)]},
            dshape:     { type:'v2v', value:[new THREE.Vector2(100, 100), new THREE.Vector2(100, 100)]},
            volxfm:     { type:'m4v', value:[new THREE.Matrix4(), new THREE.Matrix4()] },

            vmin:       { type:'fv1',value:[0,0]},
            vmax:       { type:'fv1',value:[1,1]},

            curvAlpha:  { type:'f', value:1.},
            curvScale:  { type:'f', value:.5},
            curvLim:    { type:'f', value:.2},
            dataAlpha:  { type:'f', value:1.0},
            hatchAlpha: { type:'f', value:1.},
            hatchColor: { type:'v3', value:new THREE.Vector3( 0,0,0 )},
            voxlineColor:{type:'v3', value:new THREE.Vector3( 0,0,0 )},
            voxlineWidth:{type:'f', value:viewopts.voxline_width},

            hide_mwall: { type:'i', value:0},
        }
    ]);
    
    this.state = "pause";
    this.thick = 0;
    this.shadeAdapt = true;
    this._startplay = null;
    this._animation = null;
    this.loaded = $.Deferred().done(function() {
        this.schedule();
        $("#ctmload").hide();
        $("#brain").css("opacity", 1);
    }.bind(this));
    this.cmapload = $.Deferred();
    this.labelshow = true;
    this._pivot = 0;

    this.datasets = {}
    this.active = [];

    this._bindUI();
}
MRIview.prototype = { 
    setShader: function(ds) {
        if (ds === undefined)
            ds = this.active[0];
        var sampler = samplers[ds.filter];
        var shaders = Shaders.main(sampler, ds.raw, viewopts.voxlines, this.thick);
        this.shader = new THREE.ShaderMaterial( { 
            vertexShader:shaders.vertex,
            fragmentShader:shaders.fragment,
            uniforms: this.uniforms,
            attributes: { wm:true, auxdat:true },
            morphTargets:true, 
            morphNormals:true, 
            lights:true, 
            depthTest: true,
            transparent:false,
            blending:THREE.CustomBlending,
        });
        this.shader.map = true;
        this.shader.metal = true;

        if (this.thick > 1 && this.shadeAdapt) {
            shaders = Shaders.main(sampler, ds.raw, viewopts.voxlines, 1);
            this.fastshader = new THREE.ShaderMaterial({
                vertexShader:shaders.vertex,
                fragmentShader:shaders.fragment,
                uniforms: this.uniforms,
                attributes: { wm:true, auxdat:true },
                morphTargets:true, 
                morphNormals:true, 
                lights:true, 
                depthTest: true,
                transparent:false,
                blending:THREE.CustomBlending,                
            });
            this.fastshader.map = true;
            this.fastshader.metal = true;
        } else {
            this.fastshader = null;
        }

        this.uniforms.colormap.texture.needsUpdate = true;
        this.uniforms.hatch.texture.needsUpdate = true;
        if (this.meshes && this.meshes.left) {
            this.meshes.left.material = this.shader;
            this.meshes.right.material = this.shader;
        }
    }, 
    schedule: function() {
        if (!this._scheduled) {
            this._scheduled = true;
            requestAnimationFrame( function() {
                this.draw();
                if (this.state == "play" || this._animation != null) {
                    this.schedule();
                }
            }.bind(this));
        }
    },
    draw: function () {
        if (this.state == "play") {
            var rate = this.active[0].rate;
            var sec = ((new Date()) - this._startplay) / 1000 * rate;
            if (sec > this.active[0].textures.length) {
                this._startplay += this.active[0].textures.length;
                sec -= this.active[0].textures.length;
            }
            this.setFrame(sec);
        } 
        if (this._animation) {
            var sec = ((new Date()) - this._animation.start) / 1000;
            if (!this._animate(sec)) {
                this.meshes.left.material = this.shader;
                this.meshes.right.material = this.shader;
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
    },
    load: function(ctminfo, callback) {
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
            this.controls.flatsize = flatscale * this.flatlims[1][0];
            this.controls.flatoff = this.flatoff[1];

            if (geometries[0].attributes.wm !== undefined) {
                //We have a thickness volume!
                this.thick = 1;
                $("#thickmapping").show();
            }

            var posdata = {left:[], right:[]};
            var names = {left:0, right:1};
            for (var name in names) {
                var right = names[name];
                var flats = makeFlat(geometries[right].attributes.uv.array, json.flatlims, this.flatoff, right);
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

            $.get(loader.extractUrlBase(ctminfo)+json.rois, null, function(svgdoc) {
                this.roipack = new ROIpack(svgdoc, this.renderer, posdata);
                this.addEventListener("mix", function(evt) {
                     this.roipack.labels.setMix(evt.mix);
                }.bind(this));
                this.addEventListener("resize", function(event) {
                    this.roipack.resize(event.width, event.height);
                }.bind(this));
                this.roipack.update(this.renderer).done(function(tex) {
                    this.uniforms.map.texture = tex;
                    this.schedule();
                }.bind(this));
            }.bind(this));

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
            
            if (typeof(callback) == "function")
                this.loaded.done(callback);
            this.loaded.resolve();
        }.bind(this), true, true );
    },
    resize: function(width, height) {
        if (width !== undefined) {
            $("#brain, #roilabels").css("width", width);
            width = $("#brain").width();
        }
        var w = width === undefined ? $("#brain").width()  : width;
        var h = height === undefined ? $("#brain").height()  : height;
        var aspect = w / h;
        this.renderer.setSize(w, h);
        this.camera.setSize(aspect * 100, 100);
        this.camera.updateProjectionMatrix();
        this.dispatchEvent({ type:"resize", width:w, height:h});
        this.schedule();
    },
    getState: function(state) {
        switch (state) {
            case 'mix':
                return $("#mix").slider("value");
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
        };
    },
    setState: function(state, value) {
        switch (state) {
            case 'mix':
                return this.setMix(value);
            case 'pivot':
                return this.setPivot(value);
            case 'frame':
                return this.setFrame(value);
            case 'azimuth':
                return this.controls.setCamera(value);
            case 'altitude':
                return this.controls.setCamera(undefined, value);
            case 'radius':
                return this.controls.setCamera(undefined, undefined, value);
            case 'target':
                if (this.roipack) this.roipack._updatemove = true;
                return this.controls.target.set(value[0], value[1], value[2]);
        };
    },
    animate: function(animation) {
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
        if (this.fastshader) {
            this.meshes.left.material = this.fastshader;
            this.meshes.right.material = this.fastshader;
        }
        this._animation = {anim:anim, start:new Date()};
        this.schedule();
    },
    _animate: function(sec) {
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
                }
            }
        }
        return state;
    },
    _animInterp: function(state, startval, endval, idx) {
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
    },
    saveIMG: function(post) {
        var png = $("#brain")[0].toDataURL();
        if (post !== undefined)
            $.post(post, {png:png});
        else
            window.location.href = png.replace("image/png", "application/octet-stream");
    },
    screenshot: function(width, height, pre, post) {
        $("#main").css("opacity", 0);
        setTimeout(function() {
            if (typeof(pre) == "function")
                pre();
            this.resize(width, height);
            this.saveShot(post);
            this.resize();
            $("#main").css("opacity", 1);
        }.bind(this), 500);
    },
    reset_view: function(center, height) {
        var asp = this.flatlims[1][0] / this.flatlims[1][1];
        var camasp = height !== undefined ? asp : this.camera.cameraP.aspect;
        var size = [flatscale*this.flatlims[1][0], flatscale*this.flatlims[1][1]];
        var min = [flatscale*this.flatlims[0][0], flatscale*this.flatlims[0][1]];
        var xoff = center ? 0 : size[0] / 2 - min[0];
        var zoff = center ? 0 : size[1] / 2 - min[1];
        var h = size[0] / 2 / camasp;
        h /= Math.tan(this.camera.fov / 2 * Math.PI / 180);
        this.controls.target.set(xoff, this.flatoff[1], zoff);
        this.controls.setCamera(180, 90, h);
        this.setMix(1);
        this.setShift(0);
    },
    saveflat: function(height, posturl) {
        var width = height * this.flatlims[1][0] / this.flatlims[1][1];;
        var roistate = $("#roishow").attr("checked");
        this.screenshot(width, height, function() { 
            this.reset_view(false, height); 
            $("#roishow").attr("checked", false);
            $("#roishow").change();
        }.bind(this), function(png) {
            $("#roishow").attr("checked", roistate);
            $("#roishow").change();
            this.controls.target.set(0,0,0);
            this.roipack.saveSVG(png, posturl);
        }.bind(this));
    }, 
    setMix: function(val) {
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

                this.shader.uniforms.hide_mwall.value = (n2 == flat);

                hemi.morphTargetInfluences[n2] = (val * num)%1;
                if (n1 >= 0)
                    hemi.morphTargetInfluences[n1] = 1 - (val * num)%1;
            }
        }
        this.flatmix = n2 == flat ? (val*num-.000001)%1 : 0;
        this.setPivot(this.flatmix*180);
        this.shader.uniforms.specular.value.set(1-this.flatmix, 1-this.flatmix, 1-this.flatmix);
        $("#mix").slider("value", val);
        
        this.dispatchEvent({type:"mix", flat:this.flatmix, mix:val});
        this.schedule();
    }, 
    setPivot: function (val) {
        $("#pivot").slider("option", "value", val);
        this._pivot = val;
        var names = {left:1, right:-1}
        if (val > 0) {
            for (var name in names) {
                this.pivot[name].front.rotation.z = 0;
                this.pivot[name].back.rotation.z = val*Math.PI/180 * names[name]/ 2;
            }
        } else {
            for (var name in names) {
                this.pivot[name].back.rotation.z = 0;
                this.pivot[name].front.rotation.z = val*Math.PI/180 * names[name] / 2;
            }
        }
        this.schedule();
    },
    setShift: function(val) {
        this.pivot.left.front.position.x = -val;
        this.pivot.right.front.position.x = val;
        this.schedule();
    },
    addData: function(data) {
        if (data instanceof NParray || data instanceof Dataset)
            data = {'data0':data};

        var name, names = [];
        if (data.__order__ !== undefined) {
            names = data.__order__;
            delete data.__order__;
        } else {
            for (name in data) {
                names.push(name);
            }
            names.sort();
        }

        var handle = "<div class='handle'><span class='ui-icon ui-icon-carat-2-n-s'></span></div>";
        for (var i = 0; i < names.length; i++) {
            name = names[i];
            this.datasets[name] = data[name];

            var found = false;
            $("#datasets li").each(function() {
                found = found || ($(this).text() == name);
            })
            if (!found)
                $("#datasets").append("<li class='ui-corner-all'>"+handle+name+"</li>");
        }
        
        this.setData(names);
    },
    setData: function(names) {
        if (!(names instanceof Array))
            names = [names];

        var ds2, ds = this.datasets[names[0]];
        if (names.length > 1) {
            ds2 = this.datasets[names[1]];
            if (ds.frames != ds.frames)
                throw 'Invalid selection';
            if (ds.raw ^ ds2.raw)
                throw 'Invalid selection';

        }
        
        if (ds.cmap !== undefined)
            this.setColormap(this.dataset);

        $.when(this.cmapload, this.loaded).done(function() {
            if (names.length > (this.colormap.image.height > 8 ? 2 : 1))
                return false;

            this.setShader(ds);
            ds.set(this.uniforms, 0, 0, true);
            this.active = [ds];
            $("#vrange").slider("option", {min: ds.lmin, max:ds.lmax});
            this.setVminmax(ds.min, ds.max, 0);

            if (ds.frames > 1) {
                $("#moviecontrols").show();
                $("#bottombar").addClass("bbar_controls");
                $("#movieprogress div").slider("option", {min:0, max:ds.frames-1});
            } else {
                $("#moviecontrols").hide();
                $("#bottombar").removeClass("bbar_controls");
            }
            $("#datasets li").each(function() {
                if ($(this).text() == names[0])
                    $(this).addClass("ui-selected");
                else
                    $(this).removeClass("ui-selected");
            })
            if (names.length > 1) {
                ds2.set(this.uniforms, 1, 0, true);
                this.active.push(ds2);
                $("#vrange2").slider("option", {min: ds2.lmin, max:ds2.lmax});
                this.setVminmax(ds2.min, ds2.max, 1);
                $("#datasets li").each(function() {
                    if ($(this).text() == names[1])
                        $(this).addClass("ui-selected");
                });
            }

            $("#datasets").val(names);
            $("#dataname").text(names.join(" / ")).show();
            this.schedule();
        }.bind(this));
    },
    nextData: function(dir) {
        var i = 0, found = false;
        var datasets = [];
        $("#datasets li").each(function() {
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
        if (this.colormap.image.height > 8) {
            var idx = (i + dir * 2).mod(datasets.length);
            this.setData(datasets.slice(idx, idx+2));
        } else {
            this.setData([datasets[(i+dir).mod(datasets.length)]]);
        }
    },
    rmData: function(name) {
        delete this.datasets[name];
        $("#datasets li").each(function() {
            if ($(this).text() == name)
                $(this).remove();
        })
    },

    addPlugin: function(obj, static) {
        this._staticplugin = static === true;
        if (!static)
            $("#pluginload").show();

        $("#bar").css('display', 'table');
        $("#pluginbox").html(obj);
        var _lastwidth = "60%";
        var resizepanel = function() {
            var width = $("#bar").position().left / $(window).width() * 100 + "%";
            this.resize(width);
            $("#left").width(width);
            $("#bottombar").width(width);
            // $("#topbar").width(width);
            $("#sidepanel")
                .width(100-parseFloat(width)+"%")
                .css("left", width).css('display', 'table');
            $("#bar").css("left", width);
            _lastwidth = width;
        }.bind(this);
        // this.resizepanel();

        $("#bar")
            .draggable({axis:'x', drag: resizepanel, stop: resizepanel})
            .click(function() {
                var width = document.getElementById("brain").style.width;
                if (width == "100%" || width == "") {
                    width = _lastwidth;
                    $("#sidepanel").css('display', 'table');
                } else {
                    width = "100%";
                    $("#sidepanel").hide();
                }
                this.resize(width);
                $("#bar").css("left", width);
                $("#sidepanel").width(100-parseFloat(width)+"%").css('left', width);
                $("#bottombar").width(width);
                // $("#topbar").width(width);
            }.bind(this));
        $("#bar").click();
    },
    rmPlugin: function() {
        if (!this._staticplugin) {
            $("#bar").hide();
            this.resize("100%");
            $("sidepanel").width("0%").hide();
        }
    },

    setColormap: function(cmap) {
        if (typeof(cmap) == 'string') {
            if (cmapnames[cmap] !== undefined)
                $("#colormap").ddslick('select', {index:cmapnames[cmap]});

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

        if (this.active[0] !== undefined)
            this.active[0].cmap = $("#colormap .dd-selected label").text();

        this.cmapload = $.Deferred();
        this.cmapload.done(function() {
            this.colormap.needsUpdate = true;
            this.colormap.premultiplyAlpha = true;
            this.colormap.flipY = true;
            this.colormap.minFilter = THREE.LinearFilter;
            this.colormap.magFilter = THREE.LinearFilter;
            this.uniforms.colormap.texture = this.colormap;
            this.schedule();
            
            if (this.colormap.image.height > 8) {
                $(".vcolorbar").show();
            } else {
                $(".vcolorbar").hide();
            }
        }.bind(this));

        if ((cmap instanceof Image || cmap instanceof Element) && !cmap.complete) {
            cmap.addEventListener("load", function() { this.cmapload.resolve(); }.bind(this));
        } else {
            this.cmapload.resolve();
        }
    },

    setVminmax: function(vmin, vmax, dim) {
        if (dim === undefined)
            dim = 0;
        this.shader.uniforms.vmin.value[dim] = vmin;
        this.shader.uniforms.vmax.value[dim] = vmax;

        var range, min, max;
        if (dim == 0) {
            range = "#vrange"; min = "#vmin"; max = "#vmax";
        } else {
            range = "#vrange2"; min = "#vmin2"; max = "#vmax2";
        }

        if (vmax > $(range).slider("option", "max")) {
            $(range).slider("option", "max", vmax);
            this.active[dim].lmax = vmax;
        } else if (vmin < $(range).slider("option", "min")) {
            $(range).slider("option", "min", vmin);
            this.active[dim].lmin = vmin;
        }
        $(range).slider("values", [vmin, vmax]);
        $(min).val(vmin);
        $(max).val(vmax);
        this.active[dim].min = vmin;
        this.active[dim].max = vmax;

        this.schedule();
    },

    setFrame: function(frame) {
        this.frame = frame;
        var fframe = Math.floor(frame);
        this.active[0].set(this.uniforms, 0, fframe);
        if (this.active.length > 1)
            this.active[1].set(this.uniforms, 1, fframe);
        this.uniforms.framemix.value = frame - fframe;
        $("#movieprogress div").slider("value", frame);
        $("#movieframe").attr("value", frame);
        this.schedule();
    },

    setVoxView: function(interp, lines) {
        this.active[0].filter = interp;
        for (var i = 0, il = this.active[0].textures.length; i < il; i++) {
            this.active[0].textures[i].minFilter = filtertypes[interp];
            this.active[0].textures[i].magFilter = filtertypes[interp];
            this.active[0].textures[i].needsUpdate = true;
        }
        this.setShader();
        $("#datainterp option").attr("selected", null);
        $("#datainterp option[value="+interp+"]").attr("selected", "selected");
    },

    playpause: function() {
        var dataset = this.active[0];
        if (this.state == "pause") {
            //Start playing
            var func = function() {
                this._startplay = (new Date()) - (this.frame * dataset.rate) * 1000;
                this.schedule();
                this.state = "play";
                $("#moviecontrols img").attr("src", "resources/images/control-pause.png");
            }.bind(this);

            if (dataset.stim !== undefined) {
                dataset.stim.callback = function() {
                    dataset.stim.play();
                    func();
                    dataset.stim.callback = null;
                }
                dataset.stim.seekTo(this.frame-dataset.delay);
            } else {
                func();
            }
        } else {
            this.state = "pause";
            $("#moviecontrols img").attr("src", "resources/images/control-play.png");

            if (dataset.stim !== undefined) {
                dataset.stim.pause();
            }
        }
    },

    getImage: function(width, height) {
        if (width === undefined)
            width = $("#brain").width();
        if (height === undefined)
            height = width * $("#brain").height() / $("#brain").width();
        var renderbuf = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format:THREE.RGBAFormat,
            stencilBuffer:false,
        });
        var clearAlpha = this.renderer.getClearAlpha();
        var clearColor = this.renderer.getClearColor();
        this.renderer.setClearColorHex(0x0, 0);
        this.renderer.render(this.scene, this.camera, renderbuf);
        this.renderer.setClearColorHex(clearColor, clearAlpha);
        return getTexture(this.renderer.context, renderbuf);
    },

    _bindUI: function() {
        $(window).scrollTop(0);
        $(window).resize(function() { this.resize(); }.bind(this));
        $("#brain").resize(function() { this.resize(); }.bind(this));
        window.addEventListener( 'keydown', function(e) {
            btnspeed = 0.3;
            if (e.keyCode == 32) {         //space
                if (this.active[0].textures.length > 1)
                    this.playpause();
                e.preventDefault();
                e.stopPropagation();
            } else if (e.keyCode == 107) { //+
                this.nextData(1);
            } else if (e.keyCode == 109) { //-
                this.nextData(-1);
            } else if (e.keyCode == 82) { //r
                this.animate([{idx:btnspeed, state:"target", value:[0,0,0]},
                              {idx:btnspeed, state:"mix", value:0.0}]);
            } else if (e.keyCode == 70) { //f
                this.animate([{idx:btnspeed, state:"target", value:[0,0,0]},
                              {idx:btnspeed, state:"mix", value:1.0}]);
            } else if (e.keyCode == 76) { //l
                this.labelshow = !this.labelshow;
                this.schedule();
                e.stopPropagation();
                e.preventDefault();
            }
        }.bind(this))
        var _this = this;
        $("#mix").slider({
            min:0, max:1, step:.001,
            slide: function(event, ui) { this.setMix(ui.value); }.bind(this)
        });
        $("#pivot").slider({
            min:-180, max:180, step:.01,
            slide: function(event, ui) { this.setPivot(ui.value); }.bind(this)
        });

        $("#shifthemis").slider({
            min:0, max:100, step:.01,
            slide: function(event, ui) { this.setShift(ui.value); }.bind(this)
        });

        if ($("#color_fieldset").length > 0) {
            $("#colormap").ddslick({ width:296, height:400, 
                onSelected: function() { 
                    this.setColormap($("#colormap .dd-selected-image")[0]);
                }.bind(this)
            });

            $("#vrange").slider({ 
                range:true, width:200, min:0, max:1, step:.001, values:[0,1],
                slide: function(event, ui) { this.setVminmax(ui.values[0], ui.values[1]); }.bind(this)
            });
            $("#vmin").change(function() { this.setVminmax(parseFloat($("#vmin").val()), parseFloat($("#vmax").val())); }.bind(this));
            $("#vmax").change(function() { this.setVminmax(parseFloat($("#vmin").val()), parseFloat($("#vmax").val())); }.bind(this));

            $("#vrange2").slider({ 
                range:true, width:200, min:0, max:1, step:.001, values:[0,1], orientation:"vertical",
                slide: function(event, ui) { this.setVminmax(ui.values[0], ui.values[1], 1); }.bind(this)
            });
            $("#vmin2").change(function() { this.setVminmax(parseFloat($("#vmin2").val()), parseFloat($("#vmax2").val()), 1); }.bind(this));
            $("#vmax2").change(function() { this.setVminmax(parseFloat($("#vmin2").val()), parseFloat($("#vmax2").val()), 1); }.bind(this));            
        }
        var updateROIs = function() {
            this.roipack.update(this.renderer).done(function(tex){
                this.uniforms.map.texture = tex;
                this.schedule();
            }.bind(this));
        }.bind(this);
        $("#roi_linewidth").slider({
            min:.5, max:10, step:.1, value:3,
            change: updateROIs,
        });
        $("#roi_linealpha").slider({
            min:0, max:1, step:.001, value:1,
            change: updateROIs,
        });
        $("#roi_fillalpha").slider({
            min:0, max:1, step:.001, value:0,
            change: updateROIs,
        });
        $("#roi_shadowalpha").slider({
            min:0, max:20, step:1, value:4,
            change: updateROIs,
        });
        $("#roi_linecolor").miniColors({close: updateROIs});
        $("#roi_fillcolor").miniColors({close: updateROIs});
        $("#roi_shadowcolor").miniColors({close: updateROIs});

        var blanktex = new THREE.DataTexture(new Uint8Array(16*16*4), 16, 16);
        blanktex.needsUpdate = true;
        var _this = this;
        this.blanktex = blanktex;
        $("#roishow").change(function() {
            if (this.checked) 
                updateROIs();
            else {
                _this.uniforms.map.texture = blanktex;
                _this.schedule();
            }
        });

        $("#labelshow").change(function() {
            this.labelshow = !this.labelshow;
            this.schedule();
        }.bind(this));

        $("#layer_curvalpha").slider({ min:0, max:1, step:.001, value:1, slide:function(event, ui) {
            this.uniforms.curvAlpha.value = ui.value;
            this.schedule();
        }.bind(this)})
        $("#layer_curvmult").slider({ min:.001, max:2, step:.001, value:1, slide:function(event, ui) {
            this.uniforms.curvScale.value = ui.value;
            this.schedule();
        }.bind(this)})
        $("#layer_curvlim").slider({ min:0, max:.5, step:.001, value:.2, slide:function(event, ui) {
            this.uniforms.curvLim.value = ui.value;
            this.schedule();
        }.bind(this)})
        $("#layer_dataalpha").slider({ min:0, max:1, step:.001, value:1.0, slide:function(event, ui) {
            this.uniforms.dataAlpha.value = ui.value;
            this.schedule();
        }.bind(this)})
        $("#layer_hatchalpha").slider({ min:0, max:1, step:.001, value:1, slide:function(event, ui) {
            this.uniforms.hatchAlpha.value = ui.value;
            this.schedule();
        }.bind(this)})
        $("#layer_hatchcolor").miniColors({close: function(hex, rgb) {
            this.uniforms.hatchColor.value.set(rgb.r / 255, rgb.g / 255, rgb.b / 255);
            this.schedule();
        }.bind(this)});

        $("#voxline_show").change(function() {
            viewopts.voxlines = $("#voxline_show")[0].checked;
            this.setVoxView(this.active[0].filter, viewopts.voxlines);
            this.schedule();
        }.bind(this));
        $("#voxline_color").miniColors({ close: function(hex, rgb) {
            this.uniforms.voxlineColor.value.set(rgb.r / 255, rgb.g / 255, rgb.b/255);
            this.schedule();
        }.bind(this)});
        $("#voxline_width").slider({ min:.001, max:.1, step:.001, value:viewopts.voxline_width, slide:function(event, ui) {
            this.uniforms.voxlineWidth.value = ui.value;
            this.schedule();
        }.bind(this)});
        $("#datainterp").change(function() {
            this.setVoxView($("#datainterp").val(), viewopts.voxlines);
            this.schedule();
        }.bind(this));
        $("#thicklayers").slider({ min:1, max:32, step:1, value:1, slide:function(event, ui)  {
            if (ui.value == 1)
                $("#thickmix_row").show();
            else 
                $("#thickmix_row").hide();
            this.thick = ui.value;
            this.setShader();
            this.schedule();
        }.bind(this)});
        $("#thickmix").slider({ min:0, max:1, step:.001, value:0.5, slide:function(event, ui) {
            this.uniforms.thickmix.value = ui.value;
            this.schedule();
        }.bind(this)})

        $("#resetflat").click(function() {
            this.reset_view();
        }.bind(this));

        //Dataset box
        var setdat = function(event, ui) {
            var names = [];
            $("#datasets li.ui-selected").each(function() { names.push($(this).text()); });
            this.setData(names);
        }.bind(this)
        $("#datasets")
            .sortable({ 
                handle: ".handle",
                stop: setdat,
             })
            .selectable({
                selecting: function(event, ui) {
                    var max = this.colormap.image.height > 8 ? 2 : 1;
                    var selected = $("#datasets").find("li.ui-selected, li.ui-selecting");
                    if (selected.length > max) {
                        $(ui.selecting).removeClass("ui-selecting");
                    }
                }.bind(this),
                unselected: function(event, ui) {
                    var selected = $("#datasets").find("li.ui-selected, li.ui-selecting");
                    if (selected.length < 1) {
                        $(ui.unselected).addClass("ui-selected");
                    }
                },
                stop: setdat,
            });


        $("#moviecontrol").click(this.playpause.bind(this));

        $("#movieprogress div").slider({min:0, max:1, step:.01,
            slide: function(event, ui) { 
                this.setFrame(ui.value); 
                var dataset = this.active[0];
                if (dataset.stim !== undefined) {
                    dataset.stim.seekTo(ui.value - dataset.delay);
                }
            }.bind(this)
        });

        $("#movieframe").change(function() { 
            _this.setFrame(this.value); 
            var dataset = _this.active[0];
            if (dataset.stim !== undefined) {
                dataset.stim.seekTo(this.value - dataset.delay);
            }
        });
    },

    _makeBtns: function(names) {
        var btnspeed = 0.5; // How long should folding/unfolding animations take?
        var td, btn, name;
        td = document.createElement("td");
        btn = document.createElement("button");
        btn.setAttribute("title", "Reset to fiducial view of the brain");
        btn.innerHTML = "Fiducial";
        td.setAttribute("style", "text-align:left;");
        btn.addEventListener("click", function() {
            this.animate([{idx:btnspeed, state:"target", value:[0,0,0]},
                          {idx:btnspeed, state:"mix", value:0.0}]);
        }.bind(this));
        td.appendChild(btn);
        $("#mixbtns").append(td);

        for (var i = 0; i < names.length; i++) {
            name = names[i][0].toUpperCase() + names[i].slice(1);
            td = document.createElement("td");
            btn = document.createElement("button");
            btn.innerHTML = name;
            btn.setAttribute("title", "Switch to the "+name+" view of the brain");

            btn.addEventListener("click", function(j) {
                this.animate([{idx:btnspeed, state:"mix", value: (j+1) / (names.length+1)}]);
            }.bind(this, i));
            td.appendChild(btn);
            $("#mixbtns").append(td);
        }
        td = document.createElement("td");
        btn = document.createElement("button");
        btn.innerHTML = "Flat";
        btn.setAttribute("title", "Switch to the flattened view of the brain");
        td.setAttribute("style", "text-align:right;");
        btn.addEventListener("click", function() {
            this.animate([{idx:btnspeed, state:"mix", value:1.0}]);
        }.bind(this));
        td.appendChild(btn);
        $("#mixbtns").append(td);
        $("#mix, #pivot, #shifthemis").parent().attr("colspan", names.length+2);
    },

    _makeMesh: function(geom, shader) {
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

        return {mesh:mesh, pivots:pivots};
    },
}

