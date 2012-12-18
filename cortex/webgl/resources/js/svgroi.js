var label_size = 22;
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

    "uniform vec2 screen;",
    "uniform vec2 texsize;",
    "uniform sampler2D text;",
    "uniform sampler2D depth;",

    "varying vec2 vidx;",

    "float unpack_depth(vec4 cdepth) {",
        "const vec4 bit_shift = vec4( 1.0 / ( 256.0 * 256.0 * 256.0 ), 1.0 / ( 256.0 * 256.0 ), 1.0 / 256.0, 1.0 );",
        "return dot( rgba_depth, bit_shift );",
    "}",

    "void main() {",
        "vec2 pos = gl_FragCoord + ",
        "float d = unpack_depth(texture2D(depth, pos));",
        "if (d < gl_FragCoord.z) {",
            "discard;",
        "} else {",
            "vec2 c = gl_PointCoord;",
            "c.y = (c.y - 0.5) * aspect + 0.5;",
            "c.y = clamp(c.y, 0., 1.);",
            "vec2 tcoord = c*texsize+vidx;",

            "gl_FragColor = texture2D(text, tcoord);",
        "}",
    "}",
].join("\n");

function ROIpack(svgdoc, callback, viewer) {
    this.callback = callback;
    this.svgroi = svgdoc.getElementsByTagName("svg")[0];
    this.svgroi.id = "svgroi";
    this.rois = $(this.svgroi).find("path");
    this.rois.each(function() { this.removeAttribute("filter"); });

    $(this.svgroi).find("#roilabels").children().each(function() {
        var name = $(this).text();
        if (names[name] === undefined)
            names[name] = [];
        names[name].push(viewer.remap($(this).data("ptidx")));
    });
    $(this.svgroi).find("#roilabels").remove();
    $(this.svgroi).find("defs").remove();

    this.labels = new ROIlabels(names);
    this.labels.update(viewer, 22);

    var w = this.svgroi.getAttribute("width");
    var h = this.svgroi.getAttribute("height");
    this.aspect = w / h;
    this.svgroi.setAttribute("viewBox", "0 0 "+w+" "+h);

    var gl = viewer.renderer.context;
    var height = Math.min(4096, gl.getParameter(gl.MAX_TEXTURE_SIZE)) / this.aspect;
    this.setHeight(height);
    
    this.canvas = document.createElement("canvas");
}
ROIpack.prototype = {
    resize: function(width, height) {
        this.labels.resize(width, height);
    },
    setHeight: function(height) {
        this.height = height;
        this.width = this.height * this.aspect;
        this.svgroi.setAttribute("width", this.width);
        this.svgroi.setAttribute("height", this.height);
        this._shadowtex = new ShadowTex(Math.ceil(this.width), Math.ceil(this.height), 4);
    }, 
    update: function(renderer) {
        var fo = $("#roi_fillalpha").length > 0 ? $("#roi_fillalpha").slider("option", "value") : 0;
        var lo = $("#roi_linealpha").length > 0 ? $("#roi_linealpha").slider("option", "value") : 1;
        var fc = $("#roi_fillcolor").length > 0 ? $("#roi_fillcolor").attr("value") : "#000";
        var lc = $("#roi_linecolor").length > 0 ? $("#roi_linecolor").attr("value") : "#FFF";
        var lw = $("#roi_linewidth").length > 0 ? $("#roi_linewidth").slider("option", "value") : 3;
        var sw = $("#roi_shadowalpha").length > 0 ? $("#roi_shadowalpha").slider("option", "value") : 5 ;
        fo = "fill-opacity:" + fo;
        lo = "stroke-opacity:" + lo;
        fc = "fill:" + fc;
        lc = "stroke:" + lc;
        lw = "stroke-width:" + lw + "px";
        sw = parseInt(sw);

        this.rois.attr("style", [fo, lo, fc, lc, lw].join(";"));
        var svg_xml = (new XMLSerializer()).serializeToString(this.svgroi);

        if (sw > 0) {
            var sc = $("#roi_shadowcolor").length > 0 ? $("#roi_shadowcolor").val() : "#000";
            sc = "stroke:" + sc;
            this.rois.attr("style", [sc, fc, fo, lo, lw].join(";"));
            var shadow_xml = (new XMLSerializer()).serializeToString(this.svgroi);

            canvg(this.canvas, shadow_xml, {
                ignoreMouse:true, 
                ignoreAnimation:true,
                ignoreClear:false,
                renderCallback: function() {
                    this._shadowtex.setRadius(sw/4);
                    var tex = this._shadowtex.blur(renderer, new THREE.Texture(this.canvas));
                    canvg(this.canvas, svg_xml, {
                        ignoreMouse:true,
                        ignoreAnimation:true,
                        renderCallback: function() {
                            var tex = this._shadowtex.overlay(renderer, new THREE.Texture(this.canvas));
                            this.callback(tex);
                        }.bind(this)
                    });
                }.bind(this)
            });

        } else {
            canvg(this.canvas, svg_xml, {
                ignoreMouse:true,
                ignoreAnimation:true,
                renderCallback:function() {
                    var tex = new THREE.Texture(this.canvas);
                    tex.needsUpdate = true;
                    tex.premultiplyAlpha = true;
                    this.callback(tex);
                }.bind(this),
            });
        }
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

function ROIlabels(viewer, names) {
    this.names = names;
    this.canvas = document.createElement("canvas");
    this.geometry =  {left:new THREE.Geometry(), right:new THREE.Geometry()};
    this.shader = {
        left: new THREE.ShaderMaterial({
            uniforms: {
                text:   {type:'t', value:0},
                depth:   {type:'t', value:0},
                size:   {type:'f', value:10},
                mix:    {type:'f', value:0},
                aspect: {type:'f', value:0},
                texsize:{type:'v2', value:new THREE.Vector2()},
            },
            attributes: { 
                idx:    {type:'v2', value:[]},
            }, 
            vertexShader: roilabel_vshader,
            fragmentShader: roilabel_fshader,
            blending: THREE.CustomBlending,
            depthTest: false,
            transparent: true,
            depthWrite: false,
        }),
        right: new THREE.ShaderMaterial({
            uniforms: {
                text:   {type:'t', value:0},
                size:   {type:'f', value:10},
                mix:    {type:'f', value:0},
                aspect: {type:'f', value:0},
                texsize:{type:'v2', value:new THREE.Vector2()},
            },
            attributes: { 
                idx:    {type:'v2', value:[]},
            }, 
            vertexShader: roilabel_vshader,
            fragmentShader: roilabel_fshader,
            blending: THREE.CustomBlending,
            depthTest: false,
            transparent: true,
            depthWrite: false,
        }),
    }

    this.geometry.left.dynamic = true;
    this.geometry.right.dynamic = true;
    this.geometry.left.attributes = {idx:{type:'v2', value:null}};
    this.geometry.right.attributes = {idx:{type:'v2', value:null}};

    this.particles = {
        left: new THREE.ParticleSystem(geoms.left, this.shader.left),
        right: new THREE.ParticleSystem(geoms.right, this.shader.right)
    };
    this.particles.left.dynamic = true;
    this.particles.right.dynamic = true;
}

ROIlabels.prototype = {
    update: function(viewer, height) {
        height *= 2;
        var w, width = 0, allnames = [], names = this.names;
        var ctx = this.canvas.getContext('2d');
        ctx.font = 'italic bold '+(height*0.5)+'px helvetica';
        for (var name in names) {
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
        this.width = width;
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
        tex = shadow.blur(viewer.renderer, tex);

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
        tex = shadow.overlay(viewer.renderer, tex);

        this.shader.left.uniforms.aspect.value = aspect;
        this.shader.right.uniforms.aspect.value = aspect;
        this.shader.left.uniforms.texsize.value.set(1/nwide, 1/ntall);
        this.shader.right.uniforms.texsize.value.set(1/nwide, 1/ntall);
        this.shader.left.attributes.idx.value = [];
        this.shader.right.attributes.idx.value = [];
        var hemi, ptdat;
        for (var name in names) {
            for (var i = 0; i < names[name].length; i++) {
                ptdat = viewer.getVert(names[name][i]);
                hemi = this.shader[ptdat.name];
                hemi.attributes.idx.value.push(this.texPos[name]);
            }
        }
        this.setMix(viewer);

        this.shader.left.uniforms.size.value = width;
        this.shader.left.uniforms.text.texture = tex;
        this.shader.right.uniforms.size.value = width;
        this.shader.right.uniforms.text.texture = tex;
    },

    setMix: function(viewer) {
        //Adjust the indicator particle to match the current mix state
        //Indicator particles are set off from the surface by the normal
        this.geometry.left.vertices = [];
        this.geometry.right.vertices = [];

        var hemi, ptdat;
        for (var name in this.names) {
            for (var i = 0; i < this.names[name].length; i++) {
                ptdat = viewer.getVert(this.names[name][i]);
                hemi = this.geometry[ptdat.name];
                hemi.vertices.push(ptdat.norm);
            }
        }

        this.geometry.left.verticesNeedUpdate = true;
        this.geometry.right.verticesNeedUpdate = true;
    }, 

    render: function(viewer, renderer) {
        
        renderer.render(viewer.scene, viewer.camera, 
    }
};