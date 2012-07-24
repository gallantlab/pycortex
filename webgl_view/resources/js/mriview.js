var flatscale = .4;

var vShadeHead = [
    "attribute vec2 datamap;",
    "uniform sampler2D data;",
    "uniform sampler2D data2;",
    "uniform vec2 datasize;",

    "uniform sampler2D colormap;",
    "uniform float vmin;",
    "uniform float vmax;",
    "uniform float vmin2;",
    "uniform float vmax2;",

    "varying vec3 vViewPosition;",
    "varying vec3 vNormal;",
    "varying vec4 vColor;",

    THREE.ShaderChunk[ "map_pars_vertex" ], 
    THREE.ShaderChunk[ "lights_phong_pars_vertex" ],
    THREE.ShaderChunk[ "morphtarget_pars_vertex" ],

    "void main() {",

        "vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",

        THREE.ShaderChunk[ "map_vertex" ],
        
        "vec2 dcoord = (2.*datamap+1.) / (2.*datasize);",
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

var cmapShader = vShadeHead + ([ 
        "float vdata = texture2D(data, dcoord).r;",
        "float vdata2 = texture2D(data2, dcoord).r;",
        "float vnorm = (vdata - vmin) / (vmax - vmin);",
        "float vnorm2 = (vdata2 - vmin2) / (vmax2 - vmin2);",
        "vColor = texture2D(colormap, vec2(clamp(vnorm, 0., .999), clamp(vnorm2, 0., .999) ));",
].join("\n")) + vShadeTail;

var rawShader = vShadeHead + "vColor = texture2D(data, dcoord);" + vShadeTail;

var fragmentShader = [

    "uniform vec3 diffuse;",
    "uniform float opacity;",

    "uniform vec3 ambient;",
    "uniform vec3 emissive;",
    "uniform vec3 specular;",
    "uniform float shininess;",

    "varying vec4 vColor;",
    THREE.ShaderChunk[ "map_pars_fragment" ],
    THREE.ShaderChunk[ "lights_phong_pars_fragment" ],

    "void main() {",
        "vec4 mapcolor = texture2D(map, vUv);",
        "gl_FragColor.a = mapcolor.a + vColor.a*(1.-mapcolor.a);",
        "gl_FragColor.rgb = mapcolor.rgb*mapcolor.a + vColor.rgb*vColor.a*(1.-mapcolor.a);",
        "gl_FragColor.rgb /= gl_FragColor.a;",

        THREE.ShaderChunk[ "lights_phong_fragment" ],
    "}"

].join("\n");

var flatVertShade = [
    "varying vec3 vColor;",
    "attribute float idx;",
    THREE.ShaderChunk[ "morphtarget_pars_vertex" ],
    "void main() {",
        "vColor.r = floor(idx / (256. * 256.)) / 255.;",
        "vColor.g = mod(idx / 256., 256.) / 255.;",
        "vColor.b = mod(idx, 256.) / 255.;",
        THREE.ShaderChunk[ "morphtarget_vertex" ],
        THREE.ShaderChunk[ "default_vertex" ],
    "}",
].join("\n");

var flatFragShade = [
    "varying vec3 vColor;",
    "void main() {",
        "gl_FragColor = vec4(vColor, 1.);",
    "}"
].join("\n");

function MRIview() { 
    // scene and camera
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera( 60, window.innerWidth / (window.innerHeight), 0.1, 5000 );
    this.camera.position.set(200, 200, 200);
    this.camera.up.set(0,0,1);
    $("#brain").css("opacity", 0);

    this.scene.add( this.camera );
    this.controls = new THREE.LandscapeControls( this.camera, $("#brain")[0], this.scene );
    
    this.light = new THREE.DirectionalLight( 0xffffff );
    this.light.position.set( -200, -200, 1000 ).normalize();
    this.camera.add( this.light );
    this.flatmix = 0;

    // renderer
    this.renderer = new THREE.WebGLRenderer({ 
        antialias: true, 
        preserveDrawingBuffer:true, 
        canvas:$("#brain")[0] 
    });
    this.renderer.setClearColorHex( 0x0, 1 );
    this.renderer.setSize( window.innerWidth,window.innerHeight);

    var uniforms = THREE.UniformsUtils.merge( [
        THREE.UniformsLib[ "lights" ],
        {
            diffuse:    { type:'v3', value:new THREE.Vector3( 1,1,1 )},
            specular:   { type:'v3', value:new THREE.Vector3( 1,1,1 )},
            emissive:   { type:'v3', value:new THREE.Vector3( 0,0,0 )},
            shininess:  { type:'f', value:200},

            map:        { type:'t', value:0, texture: null },
            colormap:   { type:'t', value:1, texture: null },
            data:       { type:'t', value:2, texture: null },
            data2:      { type:'t', value:3, texture: null },
            datasize:   { type:'v2', value:new THREE.Vector2(256, 0)},
            offsetRepeat:{ type: "v4", value: new THREE.Vector4( 0, 0, 1, 1 ) },

            vmin:{ type:'f', value: 0},
            vmax:{ type:'f', value: 1},
        }
    ])

    this.shader =  this.cmapshader = new THREE.ShaderMaterial( { 
        vertexShader:cmapShader,
        fragmentShader:fragmentShader,
        uniforms: uniforms,
        attributes: { datamap:true },
        morphTargets:true, 
        morphNormals:true, 
        lights:true, 
        vertexColors:true,
    });
    this.shader.map = true;
    this.shader.metal = true;
    this.shader.needsUpdate = true;

    this.rawshader = new THREE.ShaderMaterial( {
        vertexShader:rawShader,
        fragmentShader:fragmentShader,
        uniforms: THREE.UniformsUtils.clone(uniforms),
        attributes: { datamap:true },
        morphTargets:true,
        morphNormals:true,
        lights:true,
        vertexColors:true,
    });
    this.rawshader.map = true;
    this.rawshader.metal = true;
    this.rawshader.needsUpdate = true;
    
    this._bindUI();
}
MRIview.prototype = { 
    draw: function () {
        this.controls.update(this.flatmix);
        this.renderer.render( this.scene, this.camera );
        this._scheduled = false;
    },

    load: function(subj) {
        if (this.meshes) {
            for (var hemi in this.meshes) {
                this.scene.remove(this.meshes[hemi]);
                delete this.meshes[hemi];
            }
            delete this.meshes;
        }
        
        var loader = new THREE.CTMLoader(true);
        $("#threeDview").append(loader.statusDomElement);
        var ctminfo = "resources/ctm/AH_AH_huth2_[inflated,superinflated].json";

        loader.loadParts( ctminfo, function( geometries, materials, header, json ) {
            var rawdata = new Uint32Array(header.length / 4);
            var charview = new Uint8Array(rawdata.buffer);
            for (var i = 0, il = header.length; i < il; i++) {
                charview[i] = header.charCodeAt(i);
            }

            var polyfilt = {};
            polyfilt.left = rawdata.subarray(2, rawdata[0]+2);
            polyfilt.right = rawdata.subarray(rawdata[0]+2);

            geometries[0].computeBoundingBox();
            geometries[1].computeBoundingBox();

            this.meshes = {};
            this.pivot = {};
            this.datamap = {};
            this.polys = { norm:{}, flat:{}}
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

            $.get(loader.extractUrlBase(ctminfo)+json.rois, null, function(svgdoc) {
                this.rois = new ROIpack(svgdoc, function(tex) {
                    this.shader.uniforms.map.texture = tex;
                    this.controls.dispatchEvent({type:"change"});
                }.bind(this));
                this.rois.update(this.renderer);
            }.bind(this));
            for (var name in names) {
                var right = names[name];
                this.datamap[name] = geometries[right].attributes.datamap.array;
                this._makeFlat(geometries[right], polyfilt[name], right);
                var meshpiv = this._makeMesh(geometries[right], this.shader);
                this.meshes[name] = meshpiv.mesh;
                this.pivot[name] = meshpiv.pivots;
                this.scene.add(meshpiv.pivots.front);

                this.polys.norm[name] =  geometries[right].attributes.index;
                this.polys.flat[name] = geometries[right].attributes.flatindex;
            }
            this.controls.picker = new FacePick(this);
            this.controls.addEventListener("change", function() {
                if (!this._scheduled) {
                    this._scheduled = true;
                    requestAnimationFrame( this.draw.bind(this) );
                }
            }.bind(this));
            this.draw();
            $("#brain").css("opacity", 1);
        }.bind(this), true, true );
    },
    resize: function(width, height) {
        var w = width === undefined ? window.innerWidth : width;
        var h = height === undefined ? window.innerHeight : height;
        this.renderer.setSize(w, h);
        this.camera.aspect = w / h;
        this.controls.picker.resize(w, h);
        this.camera.updateProjectionMatrix();
        this.controls.dispatchEvent({type:"change"});
    },
    screenshot: function(width, height, callback) {
        $("#brain").css("opacity", 0);
        setTimeout(function() {
            if (typeof(callback) == "function")
                callback();
            this.resize(width, height);
            this.draw();
            window.location.href = $("#brain")[0].toDataURL().replace('image/png', 'image/octet-stream');
            this.resize();
            $("#brain").css("opacity", 1);
        }.bind(this), 1000);
    },
    reset_view: function(center, height) {
        var flatasp = this.flatlims[1][0] / this.flatlims[1][1];
        var camasp = height ? flatasp : this.camera.aspect;
        var size = [flatscale*this.flatlims[1][0], flatscale*this.flatlims[1][1]];
        var min = [flatscale*this.flatlims[0][0], flatscale*this.flatlims[0][1]];
        var xoff = center ? 0 : size[0] / 2 - min[0];
        var zoff = center ? 0 : size[1] / 2 - min[1];
        var h = size[0] / 2 / camasp;
        h /= Math.tan(this.camera.fov / 2 * Math.PI / 180);
        this.controls.target.set(xoff, this.flatoff[1], zoff);
        this.controls.set(180, 90, h);
        this.setMix(1);
    },
    saveflat: function(height) {
        var flatasp = this.flatlims[1][0] / this.flatlims[1][1];
        var width = height * flatasp;
        this.screenshot(width, height, function() { 
            this.reset_view(false, height); 
        }.bind(this));
    }, 
    setMix: function(val) {
        var num = this.meshes.left.geometry.morphTargets.length;
        var flat = num - 1;
        var n1 = Math.floor(val * num)-1;
        var n2 = Math.ceil(val * num)-1;

        for (var h in this.meshes) {
            var hemi = this.meshes[h];
            
            for (var i=0; i < num; i++) {
                hemi.morphTargetInfluences[i] = 0;
            }

            if ((this.lastn2 == flat) ^ (n2 == flat)) {
                this.setPoly(n2 == flat ? "flat" : "norm");
            }

            hemi.morphTargetInfluences[n2] = (val * num)%1;
            if (n1 >= 0)
                hemi.morphTargetInfluences[n1] = 1 - (val * num)%1;
        }
        this.flatmix = n2 == flat ? (val*num-.000001)%1 : 0;
        this.setPivot(this.flatmix*180);
        this.shader.uniforms.specular.value.set(1-this.flatmix, 1-this.flatmix, 1-this.flatmix);
        this.lastn2 = n2;
        this.controls.setCamera(this.flatmix);
        $("#mix").slider("value", val);
        this.controls.dispatchEvent({type:"change"});
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
        this.controls.dispatchEvent({type:"change"});
    },
    setPoly: function(polyvar) {
        for (var name in this.meshes) {
            this.meshes[name].geometry.attributes.index = this.polys[polyvar][name];
            this.meshes[name].geometry.offsets[0].count = this.polys[polyvar][name].numItems;
        }
        this.controls.dispatchEvent({type:"change"});
    },
    setData: function(dataset) {
        if (dataset.raw) {
            this.shader = this.rawshader;
            if (this.meshes && this.meshes.left) {
                this.meshes.left.material = this.rawshader;
                this.meshes.right.material = this.rawshader;
            }
        } else {
            this.shader = this.cmapshader;
            if (this.meshes && this.meshes.left) {
                this.meshes.left.material = this.cmapshader;
                this.meshes.right.material = this.cmapshader;
            }
        }
        this.shader.uniforms.data.texture = dataset.textures[0];
        this.shader.uniforms.datasize.value = dataset.shape;
        this.dataset = dataset;
        this.setVminmax(0, 1);
        this.controls.dispatchEvent({type:"change"});
    },

    setColormap: function(cmap) {
        cmap.needsUpdate = true;
        cmap.flipY = false;
        this.shader.uniforms.colormap.texture = cmap;
        this.controls.dispatchEvent({type:"change"});
    },

    setVminmax: function(vrmin, vrmax) {
        var amin = this.dataset.minmax[0];
        var amax = this.dataset.minmax[1];

        var vmin = (vrmin * (amax - amin)) + amin;
        var vmax = (vrmax * (amax - amin)) + amin;

        $("#vmin").val(vmin);
        $("#vmax").val(vmax);

        this.shader.uniforms.vmin.value = vrmin;
        this.shader.uniforms.vmax.value = vrmax;
        this.controls.dispatchEvent({type:"change"});
    },

    setVrange: function(vmin, vmax) {
        var amin = this.dataset.minmax[0];
        var amax = this.dataset.minmax[1];

        var vrmax = (vmax - amin) / (amax - amin);
        var vrmin = (vmin - amin) / (amax - amin);

        if (vrmax > 1) {
            $("#vrange").slider("option", "max", vrmax);
        }
        if (vrmin < 0) {
            $("#vrange").slider("option", "min", vrmin);
        }
        $("#vrange").slider("option", "values", [vrmin, vrmax]);
        this.shader.uniforms.vmin.value = vrmin;
        this.shader.uniforms.vmax.value = vrmax;
        this.controls.dispatchEvent({type:"change"});
    },

    _bindUI: function() {
        $(window).resize(function() { this.resize(); }.bind(this));
        var _this = this;
        $("#mix").slider({
            min:0, max:1, step:.001,
            slide: function(event, ui) { this.setMix(ui.value); }.bind(this)
        });
        $("#pivot").slider({
            min:-180, max:180, step:.01,
            slide: function(event, ui) { this.setPivot(ui.value); }.bind(this)
        });
        $("#vrange").slider({ 
            range:true, width:200, min:0, max:1, step:.001, values:[0,1],
            slide: function(event, ui) { this.setVminmax(ui.values[0], ui.values[1]); }.bind(this)
        });
        $("#vmin").change(function() { this.setVrange($("#vmin").val(), $("#vmax").val()); }.bind(this));
        $("#vmax").change(function() { this.setVrange($("#vmin").val(), $("#vmax").val()); }.bind(this));
        $("#roi_linewidth").slider({
            min:.5, max:10, step:.1, value:3,
            change: this._updateROIs.bind(this),
        });
        $("#roi_linealpha").slider({
            min:0, max:1, step:.001, value:1,
            change: this._updateROIs.bind(this),
        });
        $("#roi_fillalpha").slider({
            min:0, max:1, step:.001, value:0,
            change: this._updateROIs.bind(this),
        });
        $("#roi_shadowalpha").slider({
            min:0, max:20, step:1, value:4,
            change: this._updateROIs.bind(this),
        });
        $("#roi_linecolor").miniColors({close: this._updateROIs.bind(this)});
        $("#roi_fillcolor").miniColors({close: this._updateROIs.bind(this)});
        $("#roi_shadowcolor").miniColors({close: this._updateROIs.bind(this)});

        var blanktex = new THREE.DataTexture(new Uint8Array(16*16*4), 16, 16);
        blanktex.needsUpdate = true;
        var _this = this;
        $("#roishow").change(function() {
            if (this.checked) 
                _this._updateROIs();
            else {
                _this.shader.uniforms.map.texture = blanktex;
                _this.controls.dispatchEvent({type:"change"});
            }
        })

        $("#colormap").ddslick({ width:296, height:400, 
            onSelected: function() { 
                setTimeout(function() {
                    this.setColormap(new THREE.Texture($("#colormap .dd-selected-image")[0]));
                }.bind(this), 5);
            }.bind(this)
        });
        this.setColormap(new THREE.Texture($("#colormap .dd-selected-image")[0]));
    },

    _makeFlat: function(geom, polyfilt, right) {
        geom.computeBoundingSphere();
        geom.dynamic = true;
        
        var fmin = this.flatlims[0], fmax = this.flatlims[1];
        var uv = geom.attributes.uv.array;
        var flat = new Float32Array(uv.length / 2 * 3);
        var norms = new Float32Array(uv.length / 2 * 3);
        for (var i = 0, il = uv.length / 2; i < il; i++) {
            if (!right) {
                //flat[i*3] = -this.flatoff[0];
                flat[i*3+1] = flatscale * -uv[i*2] + this.flatoff[1];
                norms[i*3] = -1;
            } else {
                //flat[i*3] = this.flatoff[0];
                flat[i*3+1] = flatscale*uv[i*2] + this.flatoff[1];
                norms[i*3] = 1;
            }
            flat[i*3+2] = flatscale*uv[i*2+1];
            uv[i*2]   = (uv[i*2]   + fmin[0]) / fmax[0];
            uv[i*2+1] = (uv[i*2+1] + fmin[1]) / fmax[1];
        }
        geom.morphTargets.push({ array:flat, stride:3 })
        geom.morphNormals.push( norms );

        //Make the triangle indicies with cuts
        var polys = new Uint16Array(geom.attributes.index.array.length - polyfilt.length*3);
        var j = 0;
        for (var i = 0, il = geom.attributes.index.array.length / 3; i < il; i++) {
            if (i != polyfilt[i-j]) {
                polys[j*3]   = geom.attributes.index.array[i*3];
                polys[j*3+1] = geom.attributes.index.array[i*3+1];
                polys[j*3+2] = geom.attributes.index.array[i*3+2];
                j++;
            }
        }

        geom.attributes.flatindex = {itemsize:1, array:polys, numItems:polys.length, stride:1};

        var voxidx = new Uint8Array(uv.length / 2 * 3);
        for (var i = 0, il = uv.length / 2; i < il; i ++) {
            voxidx[i*3+0] = Math.floor(i / (256*256));
            voxidx[i*3+1] = Math.floor(i / 256);
            voxidx[i*3+2] = i %256;
        }

        geom.attributes.voxidx = {itemsize:3, array:voxidx};
    },
    _makeMesh: function(geom, shader) {
        var mesh = new THREE.Mesh(geom, shader);
        mesh.doubleSided = true;
        mesh.position.y = -this.flatoff[1];
        var pivots = {back:new THREE.Object3D(), front:new THREE.Object3D()};
        pivots.back.add(mesh);
        pivots.front.add(pivots.back);
        pivots.back.position.y = geom.boundingBox.min.y - geom.boundingBox.max.y;
        pivots.front.position.y = geom.boundingBox.max.y - geom.boundingBox.min.y + this.flatoff[1];
        return {mesh:mesh, pivots:pivots};
    }, 
    _updateROIs: function() {
        this.rois.update(this.renderer);
    }
}

function Dataset(url, callback) {
    this.textures = [];
    this.minmax = null;
    this.raw = false;

    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';

    var _this = this;
    xhr.onload = function(e) {
        if (this.status == 200) {
            _this.parse(this.response);
            if (typeof(callback) == "function")
                callback(_this);
        } else {
            console.log(this.status);
        }
    };
    xhr.send();
}
Dataset.prototype.parse = function (data) {
    var minmax, shape, dtex, tex, nnum = 3, raw = false;

    this.length = (new Uint32Array(data, 0, 1))[0];
    this.minmax = new Float32Array(this.length*2);
    shape = new Uint32Array(data, 4, 2);

    for (var i = 0; i < this.length; i++) {
        minmax = new Float32Array(data, nnum*4, 2);
        nnum += 2;

        raw = minmax[0] == 0 && minmax[1] == 0

        if (raw) {
            dtex = new Uint8Array(data, nnum*4, shape[0]*shape[1]*4);
            tex = new THREE.DataTexture(dtex, shape[0], shape[1], THREE.RGBAformat, THREE.UnsignedByteType);
        } else {
            dtex = new Float32Array(data, nnum*4, shape[0]*shape[1]);
            tex = new THREE.DataTexture(dtex, shape[0], shape[1], THREE.LuminanceFormat, THREE.FloatType);
        } 
        nnum += shape[0]*shape[1];

        tex.needsUpdate = true;
        tex.flipY = false;
        tex.minFilter = THREE.NearestFilter;
        tex.maxFilter = THREE.NearestFilter;
        this.textures.push(tex);
        this.minmax[i*2] = minmax[0];
        this.minmax[i*2+1] = minmax[1];
    }
    //Assume the dataset is sanitized, and the shapes are equal
    this.shape = new THREE.Vector2(shape[0], shape[1]);
    this.raw = raw;

    if (this.length != this.textures.length)
        throw "Invalid dataset";
}

function FacePick(viewer, callback) {
    this.viewer = viewer;
    this.pivot = {};
    this.meshes = {};
    this.idxrange = {};
    if (typeof(callback) == "function") {
        this.callback = callback;
    } else {
        this.callback = function(pos, idx) {
            console.log(pos, idx);
        }
    }

    this.scene = new THREE.Scene();
    this.shader = new THREE.ShaderMaterial({
        vertexShader: flatVertShade,
        fragmentShader: flatFragShade,
        attributes: { idx: true },
        morphTargets:true
    });

    this.camera = new THREE.PerspectiveCamera( 60, window.innerWidth / (window.innerHeight), 0.1, 5000 )
    this.camera.position = this.viewer.camera.position;
    this.camera.up.set(0,0,1);
    this.scene.add(this.camera);
    this.renderbuf = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, {
        minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter
    })
    this.height = window.innerHeight;

    var nface, nfaces = 0;
    for (var name in this.viewer.meshes) {
        var hemi = this.viewer.meshes[name];
        var morphs = [];
        for (var i = 0, il = hemi.geometry.morphTargets.length; i < il; i++) {
            morphs.push(hemi.geometry.morphTargets[i].array);
        }
        nface = hemi.geometry.attributes.flatindex.array.length / 3;
        this.idxrange[name] = [nfaces, nfaces + nface];

        var worker = new Worker("resources/js/mriview_worker.js");
        worker.onmessage = this.handleworker.bind(this);
        worker.postMessage({
            func:   "genFlatGeom",
            name:   name,
            ppts:   hemi.geometry.attributes.position.array,
            ppolys: hemi.geometry.attributes.flatindex.array,
            morphs: morphs,
            faceoff: nfaces,
        });
        nfaces += nface;
    }

    this._valid = false;
}
FacePick.prototype = {
    resize: function(w, h) {
        this.camera.aspect = w / h;
        this.height = h;
        delete this.renderbuf;
        this.renderbuf = new THREE.WebGLRenderTarget(w, h, {
            minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter
        })
    },

    draw: function(debug) {
        this.camera.lookAt(this.viewer.controls.target);
        for (var name in this.meshes) {
            this.pivot[name].front.rotation.z = this.viewer.pivot[name].front.rotation.z;
            this.pivot[name].back.rotation.z = this.viewer.pivot[name].back.rotation.z;
            this.meshes[name].morphTargetInfluences = this.viewer.meshes[name].morphTargetInfluences;
        }
        if (debug)
            this.viewer.renderer.render(this.scene, this.camera);
        else
            this.viewer.renderer.render(this.scene, this.camera, this.renderbuf);
    },

    handleworker: function(event) {
        var msg = event.data;
        var i, il, geom = new THREE.BufferGeometry();
        geom.attributes.position = {itemSize:3, array:msg.pts, stride:3};
        geom.attributes.index = {itemSize:1, array:msg.polys, stride:1};
        geom.attributes.idx = {itemSize:1, array:msg.color, stride:1};
        geom.morphTargets = [];
        for (i = 0, il = msg.morphs.length; i < il; i++) {
            geom.morphTargets.push({itemSize:3, array:msg.morphs[i], stride:3});
        }
        geom.offsets = []
        for (i = 0, il = msg.polys.length; i < il; i += 65535) {
            geom.offsets.push({start:i, index:i, count:Math.min(65535, il - i)});
        }
        geom.computeBoundingBox();
        var meshpiv = this.viewer._makeMesh(geom, this.shader);
        this.meshes[msg.name] = meshpiv.mesh;
        this.pivot[msg.name] = meshpiv.pivots;
        this.scene.add(meshpiv.pivots.front);
    }, 

    pick: function(x, y) {
        if (!this._valid)
            this.draw();
        var gl = this.viewer.renderer.context;
        var pix = new Uint8Array(4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.renderbuf.__webglFramebuffer);
        gl.readPixels(x, this.height - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pix);
        var faceidx = (pix[0] << 16) + (pix[1] << 8) + pix[2];
        if (faceidx > 0) {
            //adjust for clicking on black area
            faceidx -= 1;
            for (var name in this.idxrange) {
                var lims = this.idxrange[name];
                if (lims[0] <= faceidx && faceidx < lims[1]) {
                    faceidx -= lims[0];
                    var polys = this.viewer.polys.flat[name].array;
                    var map = this.viewer.datamap[name];

                    var ptidx = polys[faceidx*3];
                    var dataidx = map[ptidx*2] + (map[ptidx*2+1] << 8);

                    var pts = this.viewer.meshes[name].geometry.attributes.position.array;
                    var pos = [pts[ptidx*3], pts[ptidx*3+1], pts[ptidx*3+2]];

                    this.callback(pos, dataidx);
                }
            }
        } 
    }
}

function Cursors() {
    this.elements = [];
}
Cursors.prototype = {
    create: function(pos, contents) {
    }
}