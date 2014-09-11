var label_size = 22;
var dpi_ratio = window.devicePixelRatio || 1;
var roilabel_vshader = [
    "uniform float size;",

    "attribute vec2 idx;",

    "varying vec2 vidx;",

    "void main() {",
        "vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);",
        "gl_PointSize = size;",
        "vidx = idx;",
        "gl_Position = projectionMatrix * mvPosition;",
    "}",
].join("\n");

var roilabel_fshader = [
    "uniform float size;",
    "uniform float aspect;",

    "uniform vec2 scale;",
    "uniform vec2 texsize;",
    "uniform sampler2D text;",
    "uniform sampler2D depth;",

    "varying vec2 vidx;",

    "float unpack_depth(const in vec4 cdepth) {",
        "const vec4 bit_shift = vec4( 1.0 / ( 256.0 * 256.0 * 256.0 ), 1.0 / ( 256.0 * 256.0 ), 1.0 / 256.0, 1.0 );",
        "float depth = dot( cdepth, bit_shift );",
        "return depth;",
    "}",

    "float avg_depth( const in vec2 text, const in vec2 screen ) {",
        "const float w = 1.;",
        "const float d = 1. / ((w * 2. + 1.) * (w * 2. + 1.));",
        "vec2 center = (size * (vec2(0.5) - text) + screen) / scale;",
        "float avg = 0.;",
        "vec2 pos = vec2(0.);",
        "for (float i = -w; i <= w; i++) {",
            "for (float j = -w; j <= w; j++) {",
                "pos = center + vec2(i, j) / scale;",
                "avg += unpack_depth(texture2D(depth, pos));",
            "}",
        "}",

        "return avg * d;",
    "}",

    "float min_depth( const in vec2 text, const in vec2 screen ) {",
        "const float w = 1.;",
        "vec2 center = (size * (vec2(0.5) - text) + screen) / scale;",
        "float minum = 1000.;",
        "vec2 pos = vec2(0.);",
        "for (float i = -w; i <= w; i++) {",
            "for (float j = -w; j <= w; j++) {",
                "pos = center + vec2(i, j) / scale;",
                "minum = min(minum, unpack_depth(texture2D(depth, pos)));",
            "}",
        "}",

        "return minum;",
    "}",

    "void main() {",
        //"vec2 scoord = gl_FragCoord.xy / scale;",
        "vec2 p = gl_PointCoord;",
        "p.y = 1. - p.y;",
        "float d = min_depth(p, gl_FragCoord.xy);",
        "if ( gl_FragCoord.z >= d*1.0001 && d > 0. ) {",
            "discard;",
        "} else {",
            "vec2 c = gl_PointCoord;",
            "c.y = (c.y - 0.5) * aspect + 0.5;",
            "c.y = clamp(c.y, 0., 1.);",
            "vec2 tcoord = c*texsize+vidx;",

            //"gl_FragColor = texture2D(depth, scoord);",
            "gl_FragColor = texture2D(text, tcoord);",
            //"gl_FragColor = vec4(gl_PointCoord, 0., 1.);",
        "}",
    "}",
].join("\n");

function ROIpack(svgdoc, renderer, positions) {
    this.svgroi = svgdoc.getElementsByTagName("svg")[0];
    this.svgroi.id = "svgroi";
    this.layers = {};
    this.positions = positions;
    this.renderer = renderer;
    this.layer_label_visibility = {};

    for (var li=0; li<disp_layers.length; li++) {
        this.layers[disp_layers[li]] = $(this.svgroi).find("#"+disp_layers[li]).find("path");
        this.layers[disp_layers[li]].each(function() { this.removeAttribute("filter"); });
        this.layer_label_visibility[disp_layers[li]] = true;
    }
    $(this.svgroi).find("defs").remove();

    this.update_labels();

    var w = this.svgroi.getAttribute("width");
    var h = this.svgroi.getAttribute("height");
    this.aspect = w / h;
    this.svgroi.setAttribute("viewBox", "0 0 "+w+" "+h);

    var gl = renderer.context;
    var height = Math.min(4096, gl.getParameter(gl.MAX_TEXTURE_SIZE)) / this.aspect;
    this.setHeight(height);
    
    this.canvas = document.createElement("canvas");
}
ROIpack.prototype = {
    update_labels: function() {
        var names = {};
        for (var li=0; li<disp_layers.length; li++) {
            if(this.layer_label_visibility[disp_layers[li]]) {
                $(this.svgroi).find("#"+disp_layers[li]+"_labels").children().each(function() {
                    var name = $(this).text();
                    if (names[name] === undefined)
                        names[name] = [];
                    names[name].push($(this).data("ptidx"));
                });
            }
        }
        this.labels = new ROIlabels(names, this.positions);
        this.labels.update(this.renderer, 22);
    },
    resize: function(width, height) {
        this.labels.resize(width * dpi_ratio, height * dpi_ratio);
    },
    setHeight: function(height) {
        this.height = height;
        this.width = this.height * this.aspect;
        this.svgroi.setAttribute("width", this.width);
        this.svgroi.setAttribute("height", this.height);
        this._shadowtex = new ShadowTex(Math.ceil(this.width), Math.ceil(this.height), 4);
    }, 
    update: function(renderer) {
        var loaded = $.Deferred();

        for (var li=0; li<disp_layers.length; li++) {
            var layer = disp_layers[li];
            var fo = $("#"+layer+"_fillalpha").length > 0 ? $("#"+layer+"_fillalpha").slider("option", "value") : 0;
            var lo = $("#"+layer+"_linealpha").length > 0 ? $("#"+layer+"_linealpha").slider("option", "value") : 1;
            var fc = $("#"+layer+"_fillcolor").length > 0 ? $("#"+layer+"_fillcolor").attr("value") : "#000";
            var lc = $("#"+layer+"_linecolor").length > 0 ? $("#"+layer+"_linecolor").attr("value") : "#FFF";
            var lw = $("#"+layer+"_linewidth").length > 0 ? $("#"+layer+"_linewidth").slider("option", "value") : 3;
            var sw = $("#"+layer+"_shadowalpha").length > 0 ? $("#"+layer+"_shadowalpha").slider("option", "value") : 5 ;
            fo = "fill-opacity:" + fo;
            lo = "stroke-opacity:" + lo;
            fc = "fill:" + fc;
            lc = "stroke:" + lc;
            lw = "stroke-width:" + lw + "px";
            sw = parseInt(sw);

            this.layers[layer].attr("style", [fo, lo, fc, lc, lw].join(";"));
        }

        var svg_xml = (new XMLSerializer()).serializeToString(this.svgroi);

        // if (sw > 0) {
        //     var sc = $("#roi_shadowcolor").length > 0 ? $("#roi_shadowcolor").val() : "#000";
        //     sc = "stroke:" + sc;
        //     this.rois.attr("style", [sc, fc, fo, lo, lw].join(";"));
        //     this.sulci.attr("style", [fo, lo, fc, lc, lw].join(";"));
        //     this.miscdisp.attr("style", [fo, lo, fc, lc, lw].join(";"));

        //     var shadow_xml = (new XMLSerializer()).serializeToString(this.svgroi);

        //     canvg(this.canvas, shadow_xml, {
        //         ignoreMouse:true, 
        //         ignoreAnimation:true,
        //         ignoreClear:false,
        //         renderCallback: function() {
        //             this._shadowtex.setRadius(sw/4);
        //             var tex = this._shadowtex.blur(renderer, new THREE.Texture(this.canvas));
        //             canvg(this.canvas, svg_xml, {
        //                 ignoreMouse:true,
        //                 ignoreAnimation:true,
        //                 renderCallback: function() {
        //                     var tex = this._shadowtex.overlay(renderer, new THREE.Texture(this.canvas));
        //                     loaded.resolve(tex);
        //                 }.bind(this)
        //             });
        //         }.bind(this)
        //     });

        // } else {
        canvg(this.canvas, svg_xml, {
            ignoreMouse:true,
            ignoreAnimation:true,
            renderCallback:function() {
                var tex = new THREE.Texture(this.canvas);
                tex.needsUpdate = true;
                tex.premultiplyAlpha = true;
                loaded.resolve(tex);
            }.bind(this),
        });
        // }
        return loaded.promise();
    }, 

    saveSVG: function(png, posturl) {
        var svgdoc = this.svgroi.parentNode;
        var newsvg = svgdoc.implementation.createDocument(svgdoc.namespaceURI, null, null);
        var svg = newsvg.importNode(svgdoc.documentElement, true);
        newsvg.appendChild(svg);

        var img = newsvg.createElement("image");
        img.setAttribute("id", "flatdata");   
        img.setAttribute("x", "0");
        img.setAttribute("y", "0");
        img.setAttribute("height", this.height);
        img.setAttribute("width", this.width);
        img.setAttribute("xlink:href", png);
        var gi = newsvg.createElement("g");
        gi.setAttribute("id", 'dataimg');
        gi.setAttribute("style", "display:inline;");
        gi.setAttribute("inkscape:label", "dataimg")
        gi.setAttribute("inkscape:groupmode", "layer")
        gi.appendChild(img);
        $(svg).find("#roilayer").before(gi);

        var gt = newsvg.createElement("g");
        gt.setAttribute("id", 'roilabels');
        gt.setAttribute("style", "display:inline;");
        gt.setAttribute("inkscape:label", "roilabels")
        gt.setAttribute("inkscape:groupmode", "layer")
        $(svg).find("#roilayer").after(gt);

        var h = this.height, w = this.width;
        var luv = viewer.meshes.left.geometry.attributes.uv.array;
        var llen = luv.length / 2;
        var ruv = viewer.meshes.right.geometry.attributes.uv.array;
        this.labels.each(function() {
            if ($(this).data("ptidx") >= llen) {
                var uv = ruv;
                var ptidx = $(this).data("ptidx") - llen;
            } else {
                var uv = luv;
                var ptidx = $(this).data("ptidx");
            }
            var text = newsvg.createElement("text");
            text.setAttribute("x", uv[ptidx*2]*w);
            text.setAttribute("y", (1-uv[ptidx*2+1])*h);
            text.setAttribute('font-size', '24');
            text.setAttribute('font-family', 'helvetica');
            text.setAttribute("style", "text-anchor:middle;");
            text.appendChild(newsvg.createTextNode(this.innerText));
            gt.appendChild(text);
        })

        var svgxml = (new XMLSerializer()).serializeToString(newsvg);
        if (posturl == undefined) {
            var anchor = document.createElement("a"); 
            anchor.href = "data:image/svg+xml;utf8,"+svgxml;
            anchor.download = "flatmap.svg";
            anchor.click()
        } else {
            $.post(posturl, {svg: svgxml});
        }
    }
}

function ROIlabels(names, positions) {
    this.canvas = document.createElement("canvas");
    this.geometry =  {left:new THREE.Geometry(), right:new THREE.Geometry()};
    var uniforms = {
                text:   {type:'t', value:1},
                depth:  {type:'t', value:2},
                size:   {type:'f', value:10},
                mix:    {type:'f', value:0},
                aspect: {type:'f', value:0},
                texsize:{type:'v2', value:new THREE.Vector2()},
                scale:  {type:'v2', value:new THREE.Vector2()},
            };
    this.shader = {
        left: new THREE.ShaderMaterial({
            uniforms: THREE.UniformsUtils.clone(uniforms),
            attributes: { idx:{type:'v2', value:[]} }, 
            vertexShader: roilabel_vshader,
            fragmentShader: roilabel_fshader,
            blending: THREE.CustomBlending,
            depthTest: false,
            transparent: true,
            depthWrite: false,
        }),
        right: new THREE.ShaderMaterial({
            uniforms: THREE.UniformsUtils.clone(uniforms),
            attributes: { idx:{type:'v2', value:[]} }, 
            vertexShader: roilabel_vshader,
            fragmentShader: roilabel_fshader,
            blending: THREE.CustomBlending,
            depthTest: false,
            transparent: true,
            depthWrite: false,
        }),
    }

    var depthShader = THREE.ShaderLib["depthRGBA"];
    var depthUniforms = THREE.UniformsUtils.clone(depthShader.uniforms);
    this.depthmat =  new THREE.ShaderMaterial( { 
        fragmentShader: depthShader.fragmentShader, 
        vertexShader: depthShader.vertexShader, 
        uniforms: depthUniforms,
        blending: THREE.NoBlending,
        morphTargets: true 
    });

    this.geometry.left.dynamic = true;
    this.geometry.right.dynamic = true;
    this.geometry.left.attributes = {idx:{type:'v2', value:null}};
    this.geometry.right.attributes = {idx:{type:'v2', value:null}};

    this.particles = {
        left: new THREE.ParticleSystem(this.geometry.left, this.shader.left),
        right: new THREE.ParticleSystem(this.geometry.right, this.shader.right)
    };
    this.particles.left.dynamic = true;
    this.particles.right.dynamic = true;
    this.particles.left.sortParticles = true;
    this.particles.right.sortParticles = true;

    this.positions = {};
    var leftlen = positions.left[0].array.length / 3;
    for (var name in names) {
        var labels = names[name];
        if (this.positions[name] === undefined)
            this.positions[name] = [];

        for (var i = 0, il = labels.length; i < il; i++) {
            var pos = [labels[i] < leftlen ? this.geometry.left : this.geometry.right];
            var hemi = labels[i] < leftlen ? positions.left : positions.right;
            var idx = labels[i] < leftlen ? labels[i] : labels[i] - leftlen;
            for (var j = 0, jl = positions.left.length; j < jl; j++) {
                var pt = new THREE.Vector3();
                var arr = hemi[j].array;
                var k = idx * hemi[j].stride;
                pt.set(arr[k], arr[k+1], arr[k+2]);
                pos.push(pt);
            }
            this.positions[name].push(pos);
        }
    }
    this.mixlen = positions.left.length;
}

ROIlabels.prototype = {
    resize: function(width, height) {
        width *= dpi_ratio;
        height *= dpi_ratio;
        this.depth = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format:THREE.RGBAFormat,
            stencilBuffer:false,
        });
        this.shader.left.uniforms.depth.texture = this.depth;
        this.shader.right.uniforms.depth.texture = this.depth;
        this.shader.left.uniforms.scale.value.set( width, height);
        this.shader.right.uniforms.scale.value.set( width, height);
    },
    update: function(renderer, height) {
        height *= 2;
        height *= dpi_ratio;
        var w, width = 0, allnames = [];
        var ctx = this.canvas.getContext('2d');
        ctx.font = 'italic bold '+(height*0.5)+'px helvetica';
        for (var name in this.positions) {
            w = ctx.measureText(name).width;
            if (width < w)
                width = w;
            allnames.push(name);
        }
        
        var aspect = width / height;
        var nwide = Math.ceil(Math.sqrt(allnames.length / aspect));
        var ntall = Math.ceil(nwide * aspect);
        this.texPos = {};
        this.aspect = aspect;
        this.size = width;
        this.canvas.width = width * nwide;
        this.canvas.height = height * ntall;
        ctx.font = 'italic bold '+(height*0.5)+'px helvetica';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'black';

        var n = 0, name;
        for (var i = 0; i < ntall; i++) {
            for (var j = 0; j < nwide; j++) {
                if (n < allnames.length) {
                    name = allnames[n++];
                    ctx.fillText(name, (j+.5)*width, (i+.5)*height);
                    this.texPos[name] = new THREE.Vector2(j / nwide, i / ntall);
                }
            }
        }

        var shadow = new ShadowTex(this.canvas.width, this.canvas.height, 1.);
        var tex = new THREE.Texture(this.canvas);
        tex.flipY = false;
        tex = shadow.blur(renderer, tex);

        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.fillStyle = 'white';
        n = 0;
        for (var i = 0; i < ntall; i++) {
            for (var j = 0; j < nwide; j++) {
                if (n < allnames.length) {
                    ctx.fillText(allnames[n++], (j+.5)*width, (i+.5)*height);
                }
            }
        }

        tex = new THREE.Texture(this.canvas);
        tex.flipY = false;
        tex = shadow.overlay(renderer, tex);

        this.shader.left.uniforms.aspect.value = aspect;
        this.shader.right.uniforms.aspect.value = aspect;
        this.shader.left.uniforms.texsize.value.set(1/nwide, 1/ntall);
        this.shader.right.uniforms.texsize.value.set(1/nwide, 1/ntall);
        this.shader.left.attributes.idx.value = [];
        this.shader.right.attributes.idx.value = [];

        var hemilut = {};
        hemilut[this.geometry.left.id] = this.shader.left.attributes.idx.value;
        hemilut[this.geometry.right.id]= this.shader.right.attributes.idx.value;
        for (var name in this.positions) {
            for (var i = 0, il = this.positions[name].length; i < il; i++) {
                hemilut[this.positions[name][i][0].id].push(this.texPos[name]);
            }
        }
        this.setMix(0);

        this.shader.left.uniforms.size.value = width;
        this.shader.left.uniforms.text.texture = tex;
        this.shader.right.uniforms.size.value = width;
        this.shader.right.uniforms.text.texture = tex;
        this.resize($("#brain").width(), $("#brain").height());
    },

    setMix: function(val) {
        //Adjust the indicator particle to match the current mix state
        //Indicator particles are set off from the surface by the normal
        this.geometry.left.vertices = [];
        this.geometry.right.vertices = [];
        var mixlen = val * (this.mixlen - 1);
        var low = Math.floor(mixlen);
        var mix = mixlen % 1;

        var hemi, label, vec;
        for (var name in this.positions) {
            for (var i = 0; i < this.positions[name].length; i++) {
                label = this.positions[name][i];
                hemi = label[0];
                vec = label[low+1].clone().multiplyScalar(1 - mix);
                if (low+2 < label.length)
                    vec.addSelf(label[low+2].clone().multiplyScalar(mix));
                hemi.vertices.push(vec);
            }
        }

        this.geometry.left.verticesNeedUpdate = true;
        this.geometry.right.verticesNeedUpdate = true;
    }, 

    render: function(viewer, renderer) {
        var clearAlpha = renderer.getClearAlpha();
        var clearColor = renderer.getClearColor();
        renderer.setClearColorHex(0x0, 0);
        viewer.scene.overrideMaterial = this.depthmat;
        renderer.render(viewer.scene, viewer.camera, this.depth);
        renderer.setClearColor(clearColor, clearAlpha);

        viewer.scene.overrideMaterial = null;
        viewer.meshes.left.add(this.particles.left);
        viewer.meshes.right.add(this.particles.right);
        renderer.render(viewer.scene, viewer.camera);
        viewer.meshes.left.remove(this.particles.left);
        viewer.meshes.right.remove(this.particles.right);
    }
};