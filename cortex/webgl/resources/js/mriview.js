var flatscale = .2;
var vShadeHead = [
    "attribute vec4 auxdat;",

    "uniform sampler2D colormap;",
    "uniform float vmin[2];",
    "uniform float vmax[2];",

    "uniform float framemix;",

    "varying vec3 vViewPosition;",
    "varying vec3 vNormal;",
    "varying vec4 vColor;",
    "varying float vCurv;",
    "varying float vDrop;",
    "varying float vMedial;",

    "void main() {",

        "vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",

        THREE.ShaderChunk[ "map_vertex" ],

        "vDrop = auxdat.x;",
        "vCurv = auxdat.y;",
        "vMedial = auxdat.z;",
        "",
].join("\n");

var vShadeTail = [ "",
        "vViewPosition = -mvPosition.xyz;",

        THREE.ShaderChunk[ "morphnormal_vertex" ],

        "vNormal = transformedNormal;",

        THREE.ShaderChunk[ "lights_phong_vertex" ],
        THREE.ShaderChunk[ "morphtarget_vertex" ],
        THREE.ShaderChunk[ "default_vertex" ],

    "}"

].join("\n");

var cmapShader = ([
    THREE.ShaderChunk[ "map_pars_vertex" ], 
    THREE.ShaderChunk[ "lights_phong_pars_vertex" ],
    THREE.ShaderChunk[ "morphtarget_pars_vertex" ],
    "attribute float data0;",
    "attribute float data1;",
    "attribute float data2;",
    "attribute float data3;",
    ]).join("\n") + vShadeHead + ([
        'if (!(data0 <= 0. || 0. <= data0)) {',
            'vColor = vec4(0.);',
        '} else {',
            "float vnorm0 = (data0 - vmin[0]) / (vmax[0] - vmin[0]);",
            "float vnorm1 = (data1 - vmin[1]) / (vmax[1] - vmin[1]);",
            "float vnorm2 = (data2 - vmin[0]) / (vmax[0] - vmin[0]);",
            "float vnorm3 = (data3 - vmin[1]) / (vmax[1] - vmin[1]);",

            "float fnorm0 = (1. - framemix) * vnorm0 + framemix * vnorm2;",
            "float fnorm1 = (1. - framemix) * vnorm1 + framemix * vnorm3;",

            "vec2 cuv = vec2(clamp(fnorm0, 0., .999), clamp(fnorm1, 0., .999) );",

            "vColor  = texture2D(colormap, cuv);",
        '}',
].join("\n")) + vShadeTail;

var rawShader = ([
    THREE.ShaderChunk[ "map_pars_vertex" ], 
    THREE.ShaderChunk[ "lights_phong_pars_vertex" ],
    THREE.ShaderChunk[ "morphtarget_pars_vertex" ],
    "attribute vec4 data0;",
    "attribute vec4 data2;",
    ]).join("\n") + vShadeHead + ([
        "vColor  = (1. - framemix) * data0 * vec4(1. / 255.);",
        "vColor +=       framemix  * data2 * vec4(1. / 255.);",
        "vColor.rgb *= vColor.a;",
].join("\n")) + vShadeTail;

var fragmentShader = [
    THREE.ShaderChunk[ "map_pars_fragment" ],
    THREE.ShaderChunk[ "lights_phong_pars_fragment" ],

    "uniform int hide_mwall;",

    "uniform sampler2D hatch;",
    "uniform vec2 hatchrep;",

    "uniform float curvAlpha;",
    "uniform float curvScale;",
    "uniform float curvLim;",
    "uniform float dataAlpha;",
    "uniform float hatchAlpha;",
    "uniform vec3 hatchColor;",

    "uniform vec3 diffuse;",
    "uniform vec3 ambient;",
    "uniform vec3 emissive;",
    "uniform vec3 specular;",
    "uniform float shininess;",

    "varying vec4 vColor;",
    "varying float vCurv;",
    "varying float vDrop;",
    "varying float vMedial;",

    "void main() {",
        //Curvature Underlay
        "float curv = clamp(vCurv / curvScale  + .5, curvLim, 1.-curvLim);",
        "gl_FragColor = vec4(vec3(curv)*curvAlpha, curvAlpha);",

        "if (vMedial < .999) {",
            //Data overlay
            "gl_FragColor = vColor*dataAlpha + gl_FragColor * (1. - vColor.a * dataAlpha);",

            //Cross hatch / dropout layer
            "float hw = gl_FrontFacing ? hatchAlpha*vDrop : 1.;",
            "vec4 hcolor = hw * vec4(hatchColor, 1.) * texture2D(hatch, vUv*hatchrep);",
            "gl_FragColor = hcolor + gl_FragColor * (1. - hcolor.a);",
            //roi layer
            "vec4 roi = texture2D(map, vUv);",
            "gl_FragColor = roi + gl_FragColor * (1. - roi.a);",
        "} else if (hide_mwall == 1) {",
            "discard;",
        "}",

        THREE.ShaderChunk[ "lights_phong_fragment" ],
    "}"
].join("\n");

var buf = {
    pos: new Float32Array(3),
    norm: new Float32Array(3),
    p: new Float32Array(3),
    n: new Float32Array(3),
}

Number.prototype.mod = function(n) {
    return ((this%n)+n)%n;
}

function MRIview() { 
    //Allow objects to listen for mix updates
    THREE.EventTarget.call( this );
    // scene and camera
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera( 60, $("#brain").width() / $("#brain").height(), 0.5, 1000 );
    this.camera.position.set(200, 200, 200);
    this.camera.up.set(0,0,1);

    this.scene.add( this.camera );
    this.controls = new THREE.LandscapeControls( this.camera );
    this.controls.addEventListener("change", this.schedule.bind(this));
    this.addEventListener("resize", function(event) {
        this.controls.resize(event.width, event.height);
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
    this.renderer.sortObjects = false;
    this.renderer.setClearColorHex( 0x0, 1 );
    this.renderer.setSize( $("#brain").width(), $("#brain").height() );
    this.pixbuf = new Uint8Array($("#brain").width() * $("#brain").height() * 4);
    this.state = "pause";

    var uniforms = THREE.UniformsUtils.merge( [
        THREE.UniformsLib[ "lights" ],
        {
            diffuse:    { type:'v3', value:new THREE.Vector3( 1,1,1 )},
            specular:   { type:'v3', value:new THREE.Vector3( 1,1,1 )},
            emissive:   { type:'v3', value:new THREE.Vector3( 0,0,0 )},
            shininess:  { type:'f',  value:200},

            framemix:   { type:'f',  value:0},
            hatchrep:   { type:'v2', value:new THREE.Vector2(108, 40) },
            offsetRepeat:{type:'v4', value:new THREE.Vector4( 0, 0, 1, 1 ) },
            map:        { type:'t',  value:0, texture: null },
            hatch:      { type:'t',  value:1, texture: makeHatch() },
            colormap:   { type:'t',  value:2, texture: null },

            vmin:       { type:'fv1',value:[0,0]},
            vmax:       { type:'fv1',value:[1,1]},

            curvAlpha:  { type:'f', value:1.},
            curvScale:  { type:'f', value:.5},
            curvLim:    { type:'f', value:.2},
            dataAlpha:  { type:'f', value:1.0},
            hatchAlpha: { type:'f', value:1.},
            hatchColor: { type:'v3', value:new THREE.Vector3( 0,0,0 )},

            hide_mwall: { type:'i', value:0},
        }
    ])
    this.shader =  this.cmapshader = new THREE.ShaderMaterial( { 
        vertexShader:cmapShader,
        fragmentShader:fragmentShader,
        uniforms: uniforms,
        attributes: { data0:true, data1:true, data2:true, data3:true, auxdat:true },
        morphTargets:true, 
        morphNormals:true, 
        lights:true, 
        vertexColors:true,
        blending:THREE.CustomBlending,
        transparent:true,
    });
    this.shader.map = true;
    this.shader.metal = true;
    this.shader.uniforms.hatch.texture.needsUpdate = true;

    this.rawshader = new THREE.ShaderMaterial( {
        vertexShader:rawShader,
        fragmentShader:fragmentShader,
        uniforms: uniforms,
        attributes: { data0:true, data1:true, data2:true, data3:true, auxdat:true },
        morphTargets:true,
        morphNormals:true,
        lights:true,
        vertexColors:true,
        blending:THREE.CustomBlending,
        transparent:true,
    });
    this.rawshader.map = true;
    this.rawshader.metal = true;
    this.rawshader.uniforms.hatch.texture.needsUpdate = true;

    this.projector = new THREE.Projector();
    this._startplay = null;
    this._staticplugin = false;
    this._loaded = false;
    this.loaded = $.Deferred();
    this.cmapload = $.Deferred();
    this.labelshow = true;

    this.datasets = {}
    this.active = [];
    
    this._bindUI();
}
MRIview.prototype = { 
    schedule: function() {
        if (this._loaded && !this._scheduled && this.state == "pause") {
            this._scheduled = true;
            requestAnimationFrame( this.draw.bind(this) );
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
            requestAnimationFrame(this.draw.bind(this));
        } else if (this.state == "animate") {
            var sec = ((new Date()) - this._startplay) / 1000;
            if (this._animate(sec))
                requestAnimationFrame(this.draw.bind(this));
            else {
                this.state = "pause";
                delete this._anim;
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
            this.datamap = {};
            this.flatlims = json.flatlims;
            this.flatoff = [
                Math.max(
                    Math.abs(geometries[0].boundingBox.min.x),
                    Math.abs(geometries[1].boundingBox.max.x)
                ) / 3, Math.min(
                    geometries[0].boundingBox.min.y, 
                    geometries[1].boundingBox.min.y
                )];

            var names = {left:0, right:1};
            for (var name in names) {
                var right = names[name];
                //this.datamap[name] = geometries[right].attributes.datamap.array;
                this._makeFlat(geometries[right], right);
                var meshpiv = this._makeMesh(geometries[right], this.shader);
                this.meshes[name] = meshpiv.mesh;
                this.pivot[name] = meshpiv.pivots;
                this.scene.add(meshpiv.pivots.front);
            }

            $.get(loader.extractUrlBase(ctminfo)+json.rois, null, function(svgdoc) {
                this.roipack = new ROIpack(svgdoc, function(tex) {
                    this.shader.uniforms.map.texture = tex;
                    this.schedule();
                }.bind(this), this);
                this.roipack.update(this.renderer);
                this.addEventListener("mix", function() {
                    this.roipack.labels.setMix(this);
                }.bind(this));
                this.addEventListener("resize", function(event) {
                    this.roipack.resize(event.width, event.height);
                }.bind(this));
            }.bind(this));

            this.controls.flatsize = flatscale * this.flatlims[1][0];
            this.controls.flatoff = this.flatoff[1];

            this.picker = new FacePick(this);
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

            //Create the surface inflat/unfold buttons
            var btnspeed = 0.5; // How long should folding/unfolding animations take?
            var td, btn, name;
            td = document.createElement("td");
            btn = document.createElement("button");
            btn.setAttribute("title", "Return to the folded, fiducial view of the brain");
            btn.innerHTML = "Fiducial";
            td.setAttribute("style", "text-align:left;");
            btn.addEventListener("click", function() {
                this.animate([{idx:btnspeed, state:"target", value:[0,0,0]},
                              {idx:btnspeed, state:"mix", value:0.0}]);
            }.bind(this));
            td.appendChild(btn);
            $("#mixbtns").append(td);
            for (var i = 0; i < json.names.length; i++) {
                name = json.names[i][0].toUpperCase() + json.names[i].slice(1);
                td = document.createElement("td");
                btn = document.createElement("button");
                btn.innerHTML = name;
                btn.setAttribute("title", "Switch to the "+name+" view of the brain");

                btn.addEventListener("click", function(j) {
                    this.animate([{idx:btnspeed, state:"mix", value: (j+1) / (json.names.length+1)}]);
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
            $("#mix, #pivot, #shifthemis").parent().attr("colspan", json.names.length+2);

            this._loaded = true;
            this.schedule();
            $("#ctmload").hide();
            $("#brain").css("opacity", 1);
            
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
        this.renderer.setSize(w, h);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.pixbuf = new Uint8Array(w*h*4);
        this.dispatchEvent({ type:"resize", width:w, height:h});
        this.schedule();
    },
    getState: function(state) {
        switch (state) {
            case 'mix':
                return $("#mix").slider("value");
            case 'pivot':
                return $("#pivot").slider("value");
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
        this._anim = anim;
        this._startplay = new Date();
        this.schedule();
        this.state = "animate";
    },
    _animate: function(sec) {
        var state = false;
        var idx, val, f, i, j;
        for (i = 0, il = this._anim.length; i < il; i++) {
            f = this._anim[i];
            if (!f.ended) {
                if (f.start.idx <= sec && sec < f.end.idx) {
                    idx = (sec - f.start.idx) / (f.end.idx - f.start.idx);
                    if (f.start.value instanceof Array) {
                        val = [];
                        for (j = 0; j < f.start.value.length; j++) {
                            val.push(f.start.value[j]*(1-idx) + f.end.value[j]*idx);
                        }
                    } else {
                        val = f.start.value * (1-idx) + f.end.value * idx;
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
        var camasp = height !== undefined ? this.flatlims[1][0] / this.flatlims[1][1] : this.camera.aspect;
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
            if (data[name] instanceof Dataset) {
                this.datasets[name] = data[name];
            } else if (data[name] instanceof NParray) {
                this.datasets[name] = new Dataset(data[name]);
            }

            $.when(this.loaded, this.datasets[name].loaded).then(function() {
                if (this.meshes.left.geometry.indexMap !== undefined) {
                    this.datasets[name].rearrange(
                        this.meshes.left.geometry.attributes.position.array.length / 3, 
                        this.meshes.left.geometry.indexMap, 
                        this.meshes.right.geometry.indexMap);
                }
            }.bind(this));

            var found = false;
            $("#datasets li").each(function() {
                found = found || ($(this).text() == name);
            })
            if (!found)
                $("#datasets").append("<li class='ui-corner-all'>"+handle+name+"</li>");
        }

        $.when(this.cmapload, this.loaded).then(function() {
            this.setData(names.slice(0,this.colormap.image.height > 10 ? 2 : 1));
        }.bind(this));
    },
    setData: function(names) {
        if (names instanceof Array) {
            if (names.length > (this.colormap.image.height > 10 ? 2 : 1))
                return false;

            this.active = [this.datasets[names[0]]];
            this.datasets[names[0]].set(this, 0);
            $("#datasets li").each(function() {
                if ($(this).text() == names[0])
                    $(this).addClass("ui-selected");
                else
                    $(this).removeClass("ui-selected");
            })
            if (names.length > 1) {
                this.active.push(this.datasets[names[1]]);
                this.datasets[names[1]].set(this, 1);
                $("#datasets li").each(function() {
                    if ($(this).text() == names[1])
                        $(this).addClass("ui-selected");
                });
            }
        } else {
            return false;
        }
        
        this.schedule();
        $("#datasets").val(names);
        $("#dataname").text(names.join(" / ")).show();
        return true;
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
        if (this.colormap.image.height > 10) {
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
            this.shader.uniforms.colormap.texture = this.colormap;
            this.schedule();
            
            if (this.colormap.image.height > 10) {
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
        if (this.active[0].array !== undefined) {
            this._setVerts(this.active[0].textures[fframe], 0);
            this._setVerts(this.active[0].textures[fframe + 1], 2);
        }
        if (this.active.length > 1 && this.active[1].array !== undefined) {
            this._setVerts(this.active[1].textures[fframe], 1);
            this._setVerts(this.active[1].textures[fframe + 1], 3);
        }
        this.shader.uniforms.framemix.value = frame - fframe;
        $("#movieprogress div").slider("value", frame);
        $("#movieframe").attr("value", frame);
        this.schedule();
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
    getPos: function(idx) {
        //Returns the 2D screen coordinate of the given point index
        var vert = this.getVert(idx);
        var name = vert.name;

        vert.pos = this.pivot[name].back.matrix.multiplyVector3(vert.pos);
        vert.pos = this.pivot[name].front.matrix.multiplyVector3(vert.pos);
        vert.norm = this.pivot[name].back.matrix.multiplyVector3(vert.norm);
        vert.norm = this.pivot[name].front.matrix.multiplyVector3(vert.norm);

        var cpos = this.camera.position.clone().subSelf(vert.pos).normalize();
        var dot = -vert.norm.clone().subSelf(vert.pos).normalize().dot(cpos);

        var spos = this.projector.projectVector(vert.pos, this.camera);
        var snorm = this.projector.projectVector(vert.norm, this.camera);
        var w = this.renderer.domElement.width;
        var h = this.renderer.domElement.height;
        var x = (spos.x + 1) / 2 * w;
        var y = h - (spos.y + 1) / 2 * h;
        var xn = (snorm.x + 1) / 2 * w;
        var yn = h - (snorm.y + 1) / 2 * h;

        return {pos:[x, y], norm:[xn, yn], dot:dot, vert:vert};
    },
    getVert: function(idx) {
        //Returns a structure with the hemisphere name, position, and (normal+position) as norm
        var leftlen = this.meshes.left.geometry.attributes.position.array.length / 3;
        var name = idx < leftlen ? "left" : "right";
        var hemi = this.meshes[name].geometry;
        if (idx >= leftlen)
            idx -= leftlen;

        var influ = this.meshes[name].morphTargetInfluences;
        var pos = hemi.attributes.position.array;
        var norm = hemi.attributes.normal.array;
        buf.pos[0] = pos[idx*3+0]; buf.pos[1] = pos[idx*3+1]; buf.pos[2] = pos[idx*3+2];
        buf.norm[0] = norm[idx*3+0]; buf.norm[1] = norm[idx*3+1]; buf.norm[2] = norm[idx*3+2];
        
        var isum = 0, mt, mn, s;
        buf.p[0] = 0; buf.p[1] = 0; buf.p[2] = 0;
        buf.n[0] = 0; buf.n[1] = 0; buf.n[2] = 0;
        for (var i = 0, il = hemi.morphTargets.length; i < il; i++) {
            s = influ[i];
            mt = hemi.morphTargets[i];
            mn = hemi.morphNormals[i];
            isum += s;
            buf.p[0] += s * mt.array[mt.stride*idx+0];
            buf.p[1] += s * mt.array[mt.stride*idx+1];
            buf.p[2] += s * mt.array[mt.stride*idx+2];
            buf.n[0] += s * mn[3*idx+0];
            buf.n[1] += s * mn[3*idx+1];
            buf.n[2] += s * mn[3*idx+2];
        }

        influ = 1 - isum;
        buf.pos[0] = influ*buf.pos[0] + buf.p[0];
        buf.pos[1] = influ*buf.pos[1] + buf.p[1];
        buf.pos[2] = influ*buf.pos[2] + buf.p[2];
        influ *= .5;
        buf.norm[0] = influ*buf.norm[0] + buf.n[0] + buf.pos[0];
        buf.norm[1] = influ*buf.norm[1] + buf.n[1] + buf.pos[1];
        buf.norm[2] = influ*buf.norm[2] + buf.n[2] + buf.pos[2];
        
        var pos = new THREE.Vector3(buf.pos[0], buf.pos[1], buf.pos[2]);
        var norm = new THREE.Vector3(buf.norm[0], buf.norm[1], buf.norm[2]);        
        pos = this.meshes[name].matrix.multiplyVector3(pos);
        norm = this.meshes[name].matrix.multiplyVector3(norm);
        return {hemi:hemi, name:name, pos:pos, norm:norm}
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

    remap: function(idx) {
        var leftlen = this.meshes.left.geometry.attributes.position.array.length / 3;
        var offset = idx < leftlen ? 0 : leftlen;
        var name = idx < leftlen ? "left" : "right";
        var hemi = this.meshes[name].geometry;

        if (hemi.indexMap === undefined)
            return idx;

        if (idx >= leftlen)
            idx -= leftlen;

        return hemi.indexMap[idx] + offset;
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
            this.roipack.update(this.renderer);
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
                _this.shader.uniforms.map.texture = blanktex;
                _this.schedule();
            }
        });

        $("#labelshow").change(function() {
            this.labelshow = !this.labelshow;
            this.schedule();
        }.bind(this));

        $("#layer_curvalpha").slider({ min:0, max:1, step:.001, value:1, slide:function(event, ui) {
            this.shader.uniforms.curvAlpha.value = ui.value;
            this.schedule();
        }.bind(this)})
        $("#layer_curvmult").slider({ min:.001, max:2, step:.001, value:1, slide:function(event, ui) {
            this.shader.uniforms.curvScale.value = ui.value;
            this.schedule();
        }.bind(this)})
        $("#layer_curvlim").slider({ min:0, max:.5, step:.001, value:.2, slide:function(event, ui) {
            this.shader.uniforms.curvLim.value = ui.value;
            this.schedule();
        }.bind(this)})
        $("#layer_dataalpha").slider({ min:0, max:1, step:.001, value:1.0, slide:function(event, ui) {
            this.shader.uniforms.dataAlpha.value = ui.value;
            this.schedule();
        }.bind(this)})
        $("#layer_hatchalpha").slider({ min:0, max:1, step:.001, value:1, slide:function(event, ui) {
            this.shader.uniforms.hatchAlpha.value = ui.value;
            this.schedule();
        }.bind(this)})
        $("#layer_hatchcolor").miniColors({close: function(hex, rgb) {
            this.shader.uniforms.hatchColor.value.set(rgb.r / 255, rgb.g / 255, rgb.b / 255);
            this.schedule();
        }.bind(this)});

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
                    var max = this.colormap.image.height > 10 ? 2 : 1;
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

    _makeFlat: function(geom, right) {
        //Generates the UV and flatmap morphTarget using the unscaled UV found in the file
        geom.computeBoundingSphere();
        geom.dynamic = true;
        
        var uv = geom.attributes.uv.array;
        var fmin = this.flatlims[0], fmax = this.flatlims[1];
        var flat = new Float32Array(uv.length / 2 * 3);
        var norms = new Float32Array(uv.length / 2 * 3);
        for (var i = 0, il = uv.length / 2; i < il; i++) {
            if (right) {
                flat[i*3+1] = flatscale*uv[i*2] + this.flatoff[1];
                norms[i*3] = 1;
            } else {
                flat[i*3+1] = flatscale * -uv[i*2] + this.flatoff[1];
                norms[i*3] = -1;
            }
            flat[i*3+2] = flatscale*uv[i*2+1];
            uv[i*2]   = (uv[i*2]   + fmin[0]) / fmax[0];
            uv[i*2+1] = (uv[i*2+1] + fmin[1]) / fmax[1];
        }
        geom.morphTargets.push({ array:flat, stride:3 })
        geom.morphNormals.push( norms );

        var len = geom.attributes.position.array.length / 3;
        var arr = this.active.length > 0 ? this.active[0].textures[0] : new Float32Array(len);
        geom.attributes.data0 = {array:arr, itemSize:1, stride:1, numItems:len}
        geom.attributes.data1 = {array:arr, itemSize:1, stride:1, numItems:len}
        geom.attributes.data2 = {array:arr, itemSize:1, stride:1, numItems:len}
        geom.attributes.data3 = {array:arr, itemSize:1, stride:1, numItems:len}
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
    _setVerts: function(array, idx) {
        if (this.meshes && this.meshes.left) {
            var attributes = this.meshes.left.geometry.attributes;
            var length = attributes.position.array.length / 3;
            if (array instanceof Uint8Array)
                length *= 4;
            attributes['data'+idx].array = array.subarray(0, length);
            attributes['data'+idx].needsUpdate = true;
            attributes = this.meshes.right.geometry.attributes;
            attributes['data'+idx].array = array.subarray(length);
            attributes['data'+idx].needsUpdate = true;
        }
    },
    _setShader: function(raw) {
        if (this.meshes && this.meshes.left) {
            if (raw) {
                this.shader = this.rawshader;
                this.meshes.left.material = this.rawshader;
                this.meshes.right.material = this.rawshader;
            } else {
                this.shader = this.cmapshader;
                this.meshes.left.material = this.cmapshader;
                this.meshes.right.material = this.cmapshader;
            }

            var val, len;
            for (var i = 0; i < 2; i++) {
                for (var j = 0; j < 4; j++) {
                    var attributes = this.meshes[i ? 'left' : 'right'].geometry.attributes;
                    var dattr = attributes['data' + j];

                    len = attributes.position.numItems / 3;
                    if (raw) {
                        val = 4;
                        dattr.array = new Uint8Array(val*len);
                    } else {
                        val = 1;
                        dattr.array = new Float32Array(len);
                    }

                    dattr.itemSize = val;
                    dattr.stride = val;
                    dattr.numItems = val * len;
                }
            }
        }
    }, 
}

function makeHatch(size, linewidth, spacing) {
    //Creates a cross-hatching pattern in canvas for the dropout shading
    if (spacing === undefined)
        spacing = 32;
    if (size === undefined)
        size = 128;
    var canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    var ctx = canvas.getContext("2d");
    ctx.lineWidth = linewidth ? linewidth: 8;
    ctx.strokeStyle = "white";
    for (var i = 0, il = size*2; i < il; i+=spacing) {
        ctx.beginPath();
        ctx.moveTo(i-size, 0);
        ctx.lineTo(i, size);
        ctx.stroke();
        ctx.moveTo(i, 0)
        ctx.lineTo(i-size, size);
        ctx.stroke();
    }

    var tex = new THREE.Texture(canvas);
    tex.needsUpdate = true;
    tex.premultiplyAlpha = true;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearMipMapLinearFilter;
    return tex;
}