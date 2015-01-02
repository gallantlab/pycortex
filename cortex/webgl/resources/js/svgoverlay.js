var svgoverlay = (function(module) {
    var svgns = "http://www.w3.org/2000/svg";
    var svgdoctype = document.implementation.createDocumentType("svg", 
            "-//W3C//DTD SVG 1.0//EN", "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd");
    var padding = 8; //Width of the padding around text

    var gl = document.createElement("canvas").getContext("experimental-webgl");
    var max_tex_size = gl.getParameter(gl.MAX_TEXTURE_SIZE); 

    var retina_scale = window.devicePixelRatio || 1;

    module.SVGOverlay = function(svgpath, posdata, surf, callback) {
        this.posdata = posdata;
        this.callback = callback;
        this.labels = {left:new THREE.Group(), right:new THREE.Group()};
        this.layers = {};

        $.get(svgpath, null, this.init.bind(this));

        var shader = Shaders.depth({
            morphs:surf.names.length,
            volume:surf.volume,
        });
        this.depthshade = new THREE.ShaderMaterial({
            vertexShader: shader.vertex, 
            fragmentShader: shader.fragment, 
            uniforms: {
                thickmix:surf.uniforms.thickmix,
                surfmix:surf.uniforms.surfmix,
            },
            attributes: shader.attrs,
            blending: THREE.CustomBlending,
            blendSrc: THREE.OneFactor,
            blendDst: THREE.ZeroFactor,
        });
        this.resize(surf.width, surf.height);
    }

    module.SVGOverlay.prototype.init = function(svgdoc) {
        this.doc = svgdoc
        this.svg = svgdoc.getElementsByTagName("svg")[0];
        this.svg.id = "svgroi";

        var w = this.svg.getAttribute("width");
        var h = this.svg.getAttribute("height");
        this.aspect = w / h;
        this.svg.setAttribute("viewBox", "0 0 "+w+" "+h);

        var layers = svgdoc.getElementsByClassName("display_layer")
        for (var i = 0; i < layers.length; i++) {
            var name = layers[i].getAttribute("inkscape:label");
            this[name] = new module.Overlay(layers[i], this.posdata);
            this.layers[name] = this[name];
            this.labels.left.add(this[name].labels.meshes.left);
            this.labels.right.add(this[name].labels.meshes.right);
        }
        console.log(this.layers);

        this.setHeight(Math.min(4096, max_tex_size) / this.aspect);
        this.update();
    },
    module.SVGOverlay.prototype.setHeight = function(height) {
        this.height = height;
        this.width = this.height * this.aspect;
        this.svg.setAttribute("width", this.width);
        this.svg.setAttribute("height", this.height);
    }, 
    module.SVGOverlay.prototype.update = function() {
        var loaded = $.Deferred();

        this.svg.toDataURL("image/png", {renderer:"native", callback:function(dataurl) {
            var img = new Image();
            img.src = dataurl;
            var tex = new THREE.Texture(img);
            tex.needsUpdate = true;
            tex.premultiplyAlpha = true;
            tex.flipY = true;
            this.callback(tex);
            loaded.resolve(tex);
        }.bind(this)});

        return loaded.promise();
    }, 
    module.SVGOverlay.prototype.prerender = function(renderer, scene, camera) {
        var needed = false;
        for (var name in this.layers) {
            needed = needed || this.layers[name].labels.meshes.left.visible;
        }
        if (needed) {
            this.labels.left.visible = false;
            this.labels.right.visible = false;
            //Render depth image
            var clearAlpha = renderer.getClearAlpha();
            var clearColor = renderer.getClearColor();
            renderer.setClearColor(0x0, 0);
            scene.overrideMaterial = this.depthshade;
            renderer.render(scene, camera, this.depth);
            scene.overrideMaterial = null;
            renderer.setClearColor(clearColor, clearAlpha);
            this.labels.left.visible = true;
            this.labels.right.visible = true;
        }
    }
    module.SVGOverlay.prototype.resize = function(width, height) {
        this.depth = new THREE.WebGLRenderTarget(width/2, height/2, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format:THREE.RGBAFormat,
            stencilBuffer:false,
        });

        for (var name in this.layers) {
            this.layers[name].labels.shader.uniforms.depth.value = this.depth;
        }
    }
    module.SVGOverlay.prototype.setMix = function(mix) {
        for (var name in this.layers) {
            this.layers[name].labels.setMix(mix);
        }
    }

    module.SVGOverlay.prototype.saveSVG = function(png, posturl) {
        var svgdoc = this.svg.parentNode;
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

    module.Overlay = function(layer, posdata) {
        this.name = layer.id;
        this.layer = layer;
        this.shapes = {};

        var shapes = layer.parentNode.getElementById(this.name+"_shapes").children;
        for (var i = 0 ; i < shapes.length; i++) {
            var name = shapes[i].getAttribute("inkscape:label");
            this.shapes[name] = shapes[i];
        }

        var labels = layer.parentNode.getElementById(this.name+"_labels");
        this.labels = new module.Labels(labels, posdata);
    }
    THREE.EventDispatcher.prototype.apply(module.Overlay.prototype);
    module.Overlay.prototype.set = function(options) {
        for (var i = 0; i < this.shapes.length; i++) {
            paths = this.shapes[i].getElementsByTagNameNS(svgns, "path");
            for (var j = 0; j < paths.length; j++) {
                for (var css in options)
                    paths[j].style[css] = options[css];
            }
        }
    }
    module.Overlay.prototype.show = function() {
        this.layer.style.display = "none";
        this.dispatchEvent({type:"update"});
    }
    module.Overlay.prototype.hide = function() {
        this.layer.style.display = "inline";
        this.dispatchEvent({type:"update"});
    }


    module.Labels = function(labels, posdata) {
        this.name = labels.parentNode.id;
        this.layer = labels;
        this.posdata = posdata;
        this.labels = {};

        //Generate the svg that will render the text for the particles
        this.doc = document.implementation.createDocument(svgns, "svg", svgdoctype);
        this.svg = this.doc.documentElement;
        var defs = labels.parentNode.parentNode.getElementsByTagNameNS(svgns, "defs");
        this.svg.appendChild(defs[0].cloneNode(true));
        this.svg.style.position = "absolute";

        var leftlen = posdata.left.positions[0].array.length / posdata.left.positions[0].itemSize;
        //Process all text nodes, deduplicate
        this.num = 0;
        this.indices = {left:{verts:[],panel:[]}, right:{verts:[],panel:[]}}
        var labels = labels.getElementsByTagNameNS(svgns, "text");
        for (var i = 0; i < labels.length; i++) {
            var name = labels[i].textContent;
            //This text hasn't been seen yet, generate a new element
            if (this.labels[name] === undefined) {
                this.labels[name] = {idx:this.num++, element:labels[i].cloneNode(true)};
                this.svg.appendChild(this.labels[name].element);
            }

            //store the vertex information that matches this label
            var idx = parseInt(labels[i].getAttribute("data-ptidx"));
            if (idx < leftlen) {
                this.indices.left.verts.push(this.posdata.left.map[idx]);
                this.indices.left.panel.push(this.labels[name].idx);
            } else {
                idx -= leftlen;
                this.indices.right.verts.push(this.posdata.right.map[idx]);
                this.indices.right.panel.push(this.labels[name].idx);
            }
        }

        this._make_object();
        this.update();
    }

    module.Labels.prototype._make_object = function() {
        this.geometry = {left:new THREE.BufferGeometry(), right:new THREE.BufferGeometry()};
        var position = new Float32Array(this.indices.left.verts.length*3);
        var offset = new Float32Array(this.indices.left.verts.length*2);
        this.geometry.left.addAttribute("position", new THREE.BufferAttribute(position, 3));
        this.geometry.left.addAttribute("offset", new THREE.BufferAttribute(offset, 2));

        var position = new Float32Array(this.indices.right.verts.length*3);
        var offset = new Float32Array(this.indices.right.verts.length*2);
        this.geometry.right.addAttribute("position", new THREE.BufferAttribute(position, 3));
        this.geometry.right.addAttribute("offset", new THREE.BufferAttribute(offset, 2));

        this.setMix(0);

        this.meshes = {};
        this.shader = new THREE.ShaderMaterial({
            vertexShader: module.label_shader.vertex,
            fragmentShader: module.label_shader.fragment,
            uniforms: {
                size:  { type:'f', value:0 },
                pane:  { type:'v2', value:new THREE.Vector2() },
                depth: { type:'t', value:null },
                text:  { type:'t', value:null },
                scale: { type:'v2', value:new THREE.Vector2(1,1)},
            },
            attributes: {
                offset:{type:'v2', value:null}
            },
            blending: THREE.NormalBlending,
            // blendSrc: THREE.OneFactor,
            // blendDst: THREE.OneMinusSrcAlphaFactor,
            depthTest:false,
            depthWrite:false,
            transparent:true,
        });
        this.meshes.left = new THREE.PointCloud(this.geometry.left, this.shader);
        this.meshes.right = new THREE.PointCloud(this.geometry.right, this.shader);
    }

    module.Labels.prototype.update = function() {
        //render the svg to compute the bounding boxes for all the text
        this.size = [0, 0];
        this.svg.setAttribute("width", "0");
        this.svg.setAttribute("height", "0");
        document.body.appendChild(this.svg);
        for (var name in this.labels) {
            var box = this.labels[name].element.getBBox();
            this.size[0] = Math.max(this.size[0], box.width);
            this.size[1] = Math.max(this.size[1], box.height);
        }
        document.body.removeChild(this.svg);
        this.shader.uniforms.size.value = this.size[0] * retina_scale;

        //Compute the number of panels that results in the "squarest" image
        var aspect = (this.size[0] + 2*padding) / (this.size[1] + 2*padding);
        var nwide = Math.ceil(Math.sqrt(this.num / aspect));
        var ntall = Math.ceil(nwide * aspect);
        var width = nwide * (this.size[0] + 2*padding);
        var height = ntall * (this.size[1] + 2*padding);

        //Resize and move the text elements to their final locations
        this.svg.setAttribute("width", Math.ceil(width * retina_scale));
        this.svg.setAttribute("height", Math.ceil(height * retina_scale));
        this.svg.setAttribute("viewBox", "0 0 "+width+" "+height);
        for (var name in this.labels) {
            var idx = this.labels[name].idx;
            var x = Math.floor(idx / ntall);
            var y = idx % ntall;
            this.labels[name].element.setAttribute("x", x*(this.size[0]+2*padding)+this.size[0]/2+padding);
            this.labels[name].element.setAttribute("y", y*(this.size[1]+2*padding)+this.size[1]  +padding);

            //set the texture offsets for all matching labels
            for (var i = 0, il = this.indices.left.panel.length; i < il; i++) {
                if (this.indices.left.panel[i] == idx) {
                    this.geometry.left.attributes.offset.array[i*2+0] = x / nwide;
                    this.geometry.left.attributes.offset.array[i*2+1] = y / ntall;
                }
            }
            for (var i = 0, il = this.indices.right.panel.length; i < il; i++) {
                if (this.indices.right.panel[i] == idx) {
                    this.geometry.right.attributes.offset.array[i*2+0] = x / nwide;
                    this.geometry.right.attributes.offset.array[i*2+1] = y / ntall;
                }
            }
        }

        this.svg.toDataURL("image/png", {renderer:"native", callback:function(dataurl) {
            var img = new Image();
            img.src = dataurl;
            var tex = new THREE.Texture(img);
            tex.needsUpdate = true;
            tex.premultiplyAlpha = true;
            tex.flipY = false;
            //delete old texture
            if (this.shader.uniforms.text.value && this.shader.uniforms.text.value.dispose)
                this.shader.uniforms.text.value.dispose();
            this.shader.uniforms.text.value = tex;
        }.bind(this)});

        this.shader.uniforms.pane.value.set(1 / nwide, 1 / ntall);
        this.shape = [nwide, ntall];
        this.layer.style.display = "none";
    }
    module.Labels.prototype.set = function(options) {
        for (var i = 0; i < this.labels.length; i++) {
            for (var css in options) {
                this.labels[i].style[css] = options[css]
            }
        }
        this.update();
    }
    module.Labels.prototype.show = function() {
        this.meshes.left.visible = true;
        this.meshes.right.visible = true;
    }
    module.Labels.prototype.hide = function() {
        this.meshes.left.visible = false;
        this.meshes.right.visible = false;
    }
    module.Labels.prototype.setMix = function(mix) {
        //Adjust the indicator particle to match the current mix state
        //Indicator particles are set off from the surface by the normal
        var position = this.geometry.right.attributes.position.array;
        for (var i = 0; i < this.indices.right.verts.length; i++) {
            var idx = this.indices.right.verts[i];
            var vert = mriview.get_position(this.posdata.right, mix, idx);
            position[i*3+0] = vert.pos.x + vert.norm.x;
            position[i*3+1] = vert.pos.y + vert.norm.y;
            position[i*3+2] = vert.pos.z + vert.norm.z;
        }
        this.geometry.right.attributes.position.needsUpdate = true;

        var position = this.geometry.left.attributes.position.array;
        for (var i = 0; i < this.indices.left.verts.length; i++) {
            var idx = this.indices.left.verts[i];
            var vert = mriview.get_position(this.posdata.left, mix, idx);
            position[i*3+0] = vert.pos.x + vert.norm.x;
            position[i*3+1] = vert.pos.y + vert.norm.y;
            position[i*3+2] = vert.pos.z + vert.norm.z;
        }
        this.geometry.left.attributes.position.needsUpdate = true;
    }

    module.label_shader = {
        vertex: [
    "uniform float size;",
    "uniform sampler2D depth;",
    "uniform vec2 scale;",

    "attribute vec2 offset;",

    "varying float alpha;",
    "varying vec2 vOffset;",

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
        "vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);",
        "gl_PointSize = size;",
        "vOffset = offset;",
        "alpha = 1.;",
        "gl_Position = projectionMatrix * mvPosition;",
    "}",
    ].join("\n"),

        fragment: [
    "varying float alpha;",
    "varying vec2 vOffset;",

    "uniform vec2 pane;",
    "uniform sampler2D text;",

    "void main() {",
        "float aspect = pane.x / pane.y;",
        "vec2 c = gl_PointCoord;",
        "c.y = (c.y - 0.5) * aspect + 0.5;",
        "c.y = clamp(c.y, 0., 1.);",

        "vec2 tcoord = c*pane + vOffset;",
        //"gl_FragColor = texture2D(depth, scoord);",
        "gl_FragColor = texture2D(text, tcoord);",
        "gl_FragColor.a *= alpha;",
        //"gl_FragColor = vec4(gl_PointCoord, 0., 1.);",
    "}",
    ].join("\n"),
    }
/*
    module.LabelRenderer = function() {
        this.layers = {};
        this.objects = {left:new THREE.Group(), right:new THREE.Group()};
        this.canvas = document.createElement("canvas");
        this.ctx = this.canvas.getContext('2d');
    }
    module.LabelRenderer.prototype.register = function(layer) {
        var geometry = {left:new THREE.Geometry(), right:new THREE.Geometry()};
        var name = layer.parentNode.id;
        var texts = layer.getElementsByTagNameNS("http://www.w3.org/2000/svg", "text");
        console.log(name, texts);
    }
    module.LabelRenderer.prototype.init = function(positions) {
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
                var pos = [labels[i] < leftlen ? this.geometry.letft : this.geometry.right];
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

    module.LabelRenderer.prototype.resize = function(width, height) {
        width *= dpi_ratio;
        height *= dpi_ratio;
        
        this.shader.left.uniforms.depth.texture = this.depth;
        this.shader.right.uniforms.depth.texture = this.depth;
        this.shader.left.uniforms.scale.value.set( width, height);
        this.shader.right.uniforms.scale.value.set( width, height);
    },
    module.LabelRenderer.prototype.update = function(renderer, height) {
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

    module.LabelRenderer.prototype.setMix = function(val) {
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

    module.LabelRenderer.prototype.render = function(viewer, renderer) {
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
    };

    module.renderer = new module.LabelRenderer();
    */
    return module;
}(svgoverlay || {}));
