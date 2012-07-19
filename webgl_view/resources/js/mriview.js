var flatscale = .8;
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
        "gl_FragColor.rgb = mapcolor.rgb + vec3(1.-mapcolor.a)*vColor.rgb*vec3(vColor.a);",
        "gl_FragColor.a = mapcolor.a + vColor.a*(1. - mapcolor.a);",

        THREE.ShaderChunk[ "lights_phong_fragment" ],
    "}"

].join("\n");

function MRIview() {
    this.container = $("#threeDview")
    // scene and camera
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera( 60, window.innerWidth / (window.innerHeight), 0.1, 5000 );
    this.camera.position.set(200, 200, 200);
    this.camera.up.set(0,0,1);

    this.scene.add( this.camera );
    this.controls = new THREE.LandscapeControls( this.camera, this.container[0] );
    
    this.light = new THREE.DirectionalLight( 0xffffff );
    this.light.position.set( 200, 200, 1000 ).normalize();
    this.camera.add( this.light );

    // renderer
    this.renderer = new THREE.WebGLRenderer( { antialias: true, preserveDrawingBuffer:true } );
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
        $(this.renderer.domElement).remove();
        this.container.html("Loading...");
        if (this.meshes) {
            for (var hemi in this.meshes) {
                this.scene.remove(this.meshes[hemi]);
                delete this.meshes[hemi];
            }
            delete this.meshes;
        }
        
        var loader = new THREE.CTMLoader(true);
        $("#threeDview").append(loader.statusDomElement);
        var ctminfo = "resources/ctm/AH_AH_huth_[inflated,superinflated].json";

        loader.loadParts( ctminfo, function( geometries, materials, header, json ) {
            var rawdata = new Uint32Array(header.length / 4);
            var charview = new Uint8Array(rawdata.buffer);
            for (var i = 0, il = header.length; i < il; i++) {
                charview[i] = header.charCodeAt(i);
            }

            var polyfilt = {};
            polyfilt.left = rawdata.subarray(2, rawdata[0]+2);
            polyfilt.right = rawdata.subarray(rawdata[0]+2);
            
            this.meshes = {};
            this.pivot = {};
            this.polys = { norm:{}, flat:{}}
            var names = {left:0, right:1};
            $.get(loader.extractUrlBase(ctminfo)+json.rois, null, this._loadROIs.bind(this));

            for (var name in names) {
                var right = names[name];
                geometries[right].doubleSided = true;
                this._makeFlat(geometries[right], json.flatlims[0], json.flatlims[1], 
                    polyfilt[name], right);
                this._makeMesh(geometries[right], name);

                this.polys.norm[name] =  geometries[right].attributes.index;
                this.polys.flat[name] = geometries[right].attributes.flatindex;
            }

            this.container.html(this.renderer.domElement);

            this.controls.addEventListener("change", function() {
                if (!this._scheduled) {
                    this._scheduled = true;
                    requestAnimationFrame( this.draw.bind(this) );
                }
            }.bind(this));
            this.controls.dispatchEvent({type:"change"});

        }.bind(this), true, true );

    },
    resize: function() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.camera.aspect = window.innerWidth / (window.innerHeight);
        this.camera.updateProjectionMatrix();
        this.controls.dispatchEvent({type:"change"});
    },
    screenshot: function(width, height) {
        window.location.href = this.renderer.domElement.toDataURL().replace('image/png', 'image/octet-stream');
    },
    reset_view: function(center) {

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
            this.flatmix = n2 == flat ? (val*num-.000001)%1 : 0;
            this.setPivot(this.flatmix*180);
            this.shader.uniforms.specular.value.set(1-this.flatmix, 1-this.flatmix, 1-this.flatmix);

            hemi.morphTargetInfluences[n2] = (val * num)%1;
            if (n1 >= 0)
                hemi.morphTargetInfluences[n1] = 1 - (val * num)%1;
        }
        this.lastn2 = n2;
        this.controls.setCamera(this.flatmix);
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
        $(window).resize(this.resize.bind(this));
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

    _makeFlat: function(geom, fmin, fmax, polyfilt, right) {
        geom.computeBoundingBox();
        geom.computeBoundingSphere();
        geom.dynamic = true;
        
        var uv = geom.attributes.uv.array;
        var flat = new Float32Array(uv.length / 2 * 3);
        var norms = new Float32Array(uv.length / 2 * 3);
        for (var i = 0, il = uv.length / 2; i < il; i++) {
            if (!right) {
                flat[i*3] = geom.boundingBox.min.x / 3;
                flat[i*3+1] = fmax[0] - (uv[i*2] + fmin[0]);
                norms[i*3] = -1;
            } else {
                flat[i*3] = geom.boundingBox.max.x / 3;
                flat[i*3+1] = uv[i*2];
                norms[i*3] = 1;
            }
            flat[i*3+2] = uv[i*2+1];
            uv[i*2]   = (uv[i*2]   - fmin[0]) / fmax[0];
            uv[i*2+1] = (uv[i*2+1] - fmin[1]) / fmax[1];
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

        geom.attributes.flatindex = {itemsize:1, array:polys, numItems:polys.length};
    },
    _makeMesh: function(geom, name) {
        var mesh = new THREE.Mesh(geom, this.shader);
        mesh.position.y = geom.boundingBox.min.y;
        this.meshes[name] = mesh;
        this.pivot[name] = {back:new THREE.Object3D(), front:new THREE.Object3D()};
        this.pivot[name].back.add(mesh);
        this.pivot[name].front.add(this.pivot[name].back);
        this.pivot[name].back.position.y = geom.boundingBox.min.y - geom.boundingBox.max.y;
        this.pivot[name].front.position.y = geom.boundingBox.max.y;
        this.scene.add(this.pivot[name].front);
    }, 
    _loadROIs: function(svgdoc) {
        this.svgroi = svgdoc.getElementsByTagName("svg")[0];
        this.svgroi.id = "svgroi";
        document.getElementById("hiderois").appendChild(this.svgroi);
        this.rois = $(this.svgroi).find("path");
        this._updateROIs();
    }, 
    _updateROIs: function() {
        var fo = "fill-opacity:"+$("#roi_fillalpha").slider("option", "value");
        var lo = "stroke-opacity:"+$("#roi_linealpha").slider("option", "value");
        var fc = "fill:"+$("#roi_fillcolor").attr("value");
        var lc = "stroke:"+$("#roi_linecolor").attr("value");
        var lw = "stroke-width:"+$("#roi_linewidth").slider("option", "value") + "px";
        var sw = parseInt($("#roi_shadowalpha").slider("option", "value"));

        this.rois.attr("style", [fo, lo, fc, lc, lw].join(";"));
        var svg_xml = (new XMLSerializer()).serializeToString(this.svgroi);

        var canvas = document.getElementById("canvasrois");

        var texfunc = function() {
            var img = new Image();
            img.src = canvas.toDataURL();

            var tex = new THREE.Texture(canvas);
            tex.needsUpdate = true;
            tex.premultiplyAlpha = true;
            this.shader.uniforms.map.texture = tex;
            this.controls.dispatchEvent({type:"change"});
        }.bind(this);

        if (sw > 0) {
            var sc = "stroke:"+$("#roi_shadowcolor").val();
            this.rois.attr("style", [sc, fc, fo, lo, lw].join(";"));
            var shadow_xml = (new XMLSerializer()).serializeToString(this.svgroi);

            canvg(canvas, shadow_xml, {
                ignoreMouse:true, 
                ignoreAnimation:true,
                ignoreClear:false,
                renderCallback: function() {
                    stackBoxBlurCanvasRGBA("canvasrois", 0,0, canvas.width, canvas.height, sw, 1);
                    canvg(canvas, svg_xml, {
                        ignoreMouse:true,
                        ignoreAnimation:true,
                        ignoreDimensions:true,
                        ignoreClear:true,
                        renderCallback: function() { setTimeout(texfunc, 100); },
                    });
                }.bind(this),
            });

        } else {
            canvg(canvas, svg_xml, {
                ignoreMouse:true,
                ignoreAnimation:true,
                renderCallback:texfunc,
            });
        }
    },
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