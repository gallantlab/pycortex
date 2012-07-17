var vertexShader = [
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

        "float vdata = texture2D(data, dcoord).r;",
        "float vdata2 = texture2D(data2, dcoord).r;",
        "float vnorm = (vdata - vmin) / (vmax - vmin);",
        "float vnorm2 = (vdata2 - vmin2) / (vmax2 - vmin2);",
        "vColor = texture2D(colormap, vec2(vnorm, vnorm2 ));",

        "vViewPosition = -mvPosition.xyz;",

        THREE.ShaderChunk[ "morphnormal_vertex" ],

        "vNormal = transformedNormal;",

        THREE.ShaderChunk[ "lights_phong_vertex" ],
        THREE.ShaderChunk[ "morphtarget_vertex" ],
        THREE.ShaderChunk[ "default_vertex" ],

    "}"

].join("\n");
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

        //THREE.ShaderChunk[ "linear_to_gamma_fragment" ],

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

    this.controls = new THREE.TrackballControls( this.camera, this.container[0] );
    
    this.controls.rotateSpeed = 5.0;
    this.controls.zoomSpeed = 20;
    this.controls.panSpeed = 2;

    this.controls.noZoom = false;
    this.controls.noPan = false;

    this.controls.staticMoving = true;
    this.controls.dynamicDampingFactor = 0.3;
    this.controls.target.set(0,0,0);
    
    this.light = new THREE.DirectionalLight( 0xffffff );
    this.light.position.set( 200, 200, 1000 ).normalize();
    this.camera.add( this.light );

    // renderer
    this.renderer = new THREE.WebGLRenderer( { antialias: true } );
    this.renderer.setClearColorHex( 0x0, 1 );
    this.renderer.setSize( window.innerWidth,window.innerHeight);

    var uniforms = THREE.UniformsUtils.merge( [
        THREE.UniformsLib[ "lights" ],
        {
            diffuse:    { type:'v3', value:new THREE.Vector3( 1,1,1 )},
            specular:   { type:'v3', value:new THREE.Vector3( 1,1,1 )},
            emissive:    { type:'v3', value:new THREE.Vector3( 0.05,0.05,0.05 )},
            shininess:  { type:'f', value:500},

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

    this.shader = new THREE.ShaderMaterial( { 
        vertexShader:vertexShader,
        fragmentShader:fragmentShader,
        uniforms: uniforms,
        attributes: { datamap:true },
        morphTargets:true, 
        morphNormals:true, 
        lights:true, 
        vertexColors:true,
    });
    this.shader.map = true;
    
    this.container.html( this.renderer.domElement );
    $(window).resize(this.resize.bind(this));

    $("#mix").bind("slide", function() { this.setMix($("#mix").slider("option", "value")); }.bind(this));
    $("#pivot").bind("slide", function() { this.setPivot($("#pivot").slider("option", "value")); }.bind(this));
    $("#roi_fillalpha").bind("slidechange", this._updateROIs.bind(this));
    $("#roi_linealpha").bind("slidechange", this._updateROIs.bind(this));
    $("#roi_linewidth").bind("slidechange", this._updateROIs.bind(this));
    $("#roi_shadowalpha").bind("slidechange", this._updateROIs.bind(this));
    $("#roi_fillcolor").miniColors("option", "close", this._updateROIs.bind(this));
    $("#roi_linecolor").miniColors("option", "close", this._updateROIs.bind(this));
    $("#roi_shadowcolor").miniColors("option", "close", this._updateROIs.bind(this));

    var blanktex = new THREE.DataTexture(new Uint8Array(16*16*4), 16, 16);
    blanktex.needsUpdate = true;
    var _this = this;
    $("#roishow").change(function() {
        if (this.checked) 
            _this._updateROIs();
        else
            _this.shader.uniforms.map.texture = blanktex;
    })
}
MRIview.prototype = { 
    draw: function () {
        requestAnimationFrame( this.draw.bind(this) );
        this.controls.update();
        this.renderer.render( this.scene, this.camera );
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
        
        var loader = new THREE.CTMLoader( );
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
                this._makeFlat(geometries[right], json.flatlims[right], polyfilt[name], right);
                this._makeMesh(geometries[right], name);

                this.polys.norm[name] =  geometries[right].attributes.index;
                this.polys.flat[name] = geometries[right].attributes.flatindex;
            }

            this.container.html(this.renderer.domElement);
            this.draw();

        }.bind(this), true, true );

    },
    resize: function() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.camera.aspect = window.innerWidth / (window.innerHeight);
        this.camera.updateProjectionMatrix();
    },
    setMix: function(val) {
        var num = this.meshes.left.geometry.morphTargets.length;
        var flat = num - 1;
        var n1 = Math.floor(val * num)-1;
        var n2 = Math.ceil(val * num)-1;

        for (var h in this.meshes) {
            var hemi = this.meshes[h];
            
            for (var i=0; i < num; i++)
                hemi.morphTargetInfluences[i] = 0;

            if ((this.lastn2 == flat) ^ (n2 == flat)) {
                this.setPoly(n2 == flat ? "flat" : "norm");
            }
            if (n2 == flat)
                this.setPivot(((val*num-.000001)%1)*180);
            else 
                this.setPivot(0);

            hemi.morphTargetInfluences[n2] = (val * num)%1;
            if (n1 >= 0)
                hemi.morphTargetInfluences[n1] = 1 - (val * num)%1;
        }
        this.lastn2 = n2;
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
    },
    setPoly: function(polyvar) {
        for (var name in this.meshes) {
            this.meshes[name].geometry.attributes.index = this.polys[polyvar][name];
        }
    },
    setData: function(data) {
        var datatex = new THREE.DataTexture(data, 256, data.length/4 / 256);
        datatex.needsUpdate = true;
        datatex.flipY = false;
        datatex.minFilter = THREE.NearestFilter;
        datatex.maxFilter = THREE.NearestFilter;

        this.shader.uniforms.data.texture = datatex;
        this.shader.uniforms.datasize.value.set(256, data.length/4/256);
    },

    setColormap: function(cmap) {
        cmap.needsUpdate = true;
        cmap.flipY = false;
        this.shader.uniforms.colormap.texture = cmap;
    },

    _makeFlat: function(geom, lim, polyfilt, right) {
        var side = right ? "right" : "left";
        geom.computeBoundingBox();
        geom.computeBoundingSphere();
        geom.dynamic = true;
        
        var uv = geom.attributes.uv.array;
        var flat = new Float32Array(uv.length / 2 * 3);
        var norms = new Float32Array(uv.length / 2 * 3);
        var yrange = geom.boundingBox.max.y - geom.boundingBox.min.y, ymin = geom.boundingBox.min.y;
        var zrange = geom.boundingBox.max.z - geom.boundingBox.min.z, zmin = geom.boundingBox.min.z;
        for (var i = 0, il = uv.length / 2; i < il; i++) {
            flat[i*3+1] = (uv[i*2] - lim[0]) / lim[1];
            if (!right) {
                flat[i*3] = geom.boundingBox.min.x / 3;
                flat[i*3+1] = 1 - flat[i*3+1]
            } else {
                flat[i*3] = geom.boundingBox.max.x / 3;
            }

            flat[i*3+1] = 1.2*(flat[i*3+1] * yrange + ymin);
            flat[i*3+2] = 1.2*(uv[i*2+1] *lim[2]* yrange + zmin);
            norms[i*3] = 2*right-1;
        }
        geom.morphTargets.push({ array:flat, stride:3 })
        geom.morphNormals.push( norms );

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
        mesh.position.y = -1.2*geom.boundingBox.min.y;
        this.meshes[name] = mesh;
        this.pivot[name] = {back:new THREE.Object3D(), front:new THREE.Object3D()};
        this.pivot[name].back.add(mesh);
        this.pivot[name].front.add(this.pivot[name].back);
        this.pivot[name].back.position.y = 1.2*(geom.boundingBox.min.y - geom.boundingBox.max.y);
        this.pivot[name].front.position.y = 1.2*geom.boundingBox.max.y;
        this.scene.add(this.pivot[name].front);
    }, 
    _loadROIs: function(svgdoc) {
        this.svgroi = svgdoc.getElementsByTagName("svg")[0];
        this.svgroi.id = "svgroi";
        document.getElementById("hiderois").appendChild(this.svgroi);
        this.rois = $(this.svgroi).find("path");
        //this.rois.attr("filter", "url(#dropshadow)");
        /*
        var svgns = "http://www.w3.org/2000/svg";
        var defs = document.createElementNS(svgns, "defs");
        var filter = document.createElementNS(svgns, "filter");
        filter.id = "dropshadow";
        filter.height = "130%";
        var fegb = document.createElementNS(svgns, "feGaussianBlur");
        fegb.setAttribute("in", "SourceAlpha");
        fegb.setAttribute("stdDeviation", "3");
        filter.appendChild(fegb);
        var femerge = document.createElementNS(svgns, "feMerge");
        var femn = document.createElementNS(svgns, "feMergeNode");
        femerge.appendChild(femn);
        var femns = document.createElementNS(svgns, "feMergeNode");
        femns.setAttribute("in", "SourceGraphic");
        femerge.appendChild(femns);
        filter.appendChild(femerge);
        defs.appendChild(filter);
        viewer.svgroi.insertBefore(defs, viewer.svgroi.firstChild);
        
        this.roidrop = fegb;
        */

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
            console.log(img);
            var tex = new THREE.Texture(canvas);
            tex.needsUpdate = true;
            tex.premultiplyAlpha = true;
            this.shader.uniforms.map.texture = tex;
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
                        renderCallback: texfunc,
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