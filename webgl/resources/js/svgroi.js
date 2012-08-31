function ROIpack(svgdoc, callback, viewer, height) {
    this.callback = callback;
    this.svgroi = svgdoc.getElementsByTagName("svg")[0];
    this.svgroi.id = "svgroi";
    //document.getElementById("hiderois").appendChild(this.svgroi);
    this.rois = $(this.svgroi).find("path");
    var geoms = {left:new THREE.Geometry(), right:new THREE.Geometry()};
    geoms.left.dynamic = true;
    geoms.right.dynamic = true;
    var verts = {};
    var labels = [];
    var colors = 1;
    $(this.svgroi).find("foreignObject p").each(function() {
        var name = $(this).text();
        var el = document.createElement("p");
        var pt = viewer.remap($(this).data("ptidx"));
        var color = new THREE.Color(colors++);
        var ptdat = viewer.getVert(pt);
        //var arr = viewer.meshes[ptdat.name].geometry.attributes.position.array;
        //console.log(ptdat.norm, arr[pt*3+0], arr[pt*3+1], arr[pt*3+2]);
        verts[pt] = ptdat.norm;
        geoms[ptdat.name].vertices.push(ptdat.norm);
        geoms[ptdat.name].colors.push(color);

        el.setAttribute("data-ptidx", pt);
        el.setAttribute("data-ptcolor", color.getHex());
        el.className = "roilabel";
        el.innerHTML = name;
        labels.push(el);
    });
    this.vertices = verts;
    this.particlemat = new THREE.ParticleBasicMaterial({
        size:2, 
        vertexColors:THREE.VertexColors, 
        depthTest:true,
        sizeAttenuation:false, 
        opacity:1.0
    });
    this.particles = {
        left: new THREE.ParticleSystem(geoms.left, this.particlemat),
        right: new THREE.ParticleSystem(geoms.right, this.particlemat)
    };
    this.particles.left.dynamic = true;
    this.particles.right.dynamic = true;
    viewer.pivot.left.back.add(this.particles.left);
    viewer.pivot.right.back.add(this.particles.right);

    this.labels = $(labels);
    this.svgroi.removeChild(this.svgroi.getElementsByTagName("foreignObject")[0]);
    $("#roilabels").append(this.labels);

    var w = this.svgroi.getAttribute("width");
    var h = this.svgroi.getAttribute("height");
    this.aspect = w / h;
    this.svgroi.setAttribute("viewBox", "0 0 "+w+" "+h);
    this.setHeight(height === undefined ? h : height);
}
ROIpack.prototype = {
    setHeight: function(height) {
        this.height = height;
        this.width = this.height * this.aspect;
        this.svgroi.setAttribute("width", this.width);
        this.svgroi.setAttribute("height", this.height);
        this._shadowtex = new ShadowTex(Math.ceil(this.width), Math.ceil(this.height), 4);
    }, 
    update: function(renderer) {
        var fo = "fill-opacity:"+$("#roi_fillalpha").slider("option", "value");
        var lo = "stroke-opacity:"+$("#roi_linealpha").slider("option", "value");
        var fc = "fill:"+$("#roi_fillcolor").attr("value");
        var lc = "stroke:"+$("#roi_linecolor").attr("value");
        var lw = "stroke-width:"+$("#roi_linewidth").slider("option", "value") + "px";
        var sw = parseInt($("#roi_shadowalpha").slider("option", "value"));

        this.rois.attr("style", [fo, lo, fc, lc, lw].join(";"));
        var svg_xml = (new XMLSerializer()).serializeToString(this.svgroi);

        var canvas = document.getElementById("canvasrois");

        if (sw > 0) {
            var sc = "stroke:"+$("#roi_shadowcolor").val();
            this.rois.attr("style", [sc, fc, fo, lo, lw].join(";"));
            var shadow_xml = (new XMLSerializer()).serializeToString(this.svgroi);

            canvg(canvas, shadow_xml, {
                ignoreMouse:true, 
                ignoreAnimation:true,
                ignoreClear:false,
                renderCallback: function() {
                    this._shadowtex.setRadius(sw/4);
                    var tex = this._shadowtex.blur(renderer, new THREE.Texture(canvas));
                    canvg(canvas, svg_xml, {
                        ignoreMouse:true,
                        ignoreAnimation:true,
                        renderCallback: function() {
                            var tex = this._shadowtex.overlay(renderer, new THREE.Texture(canvas));
                            this.callback(tex);
                        }.bind(this)
                    });
                }.bind(this)
            });

        } else {
            canvg(canvas, svg_xml, {
                ignoreMouse:true,
                ignoreAnimation:true,
                renderCallback:function() {
                    var tex = new THREE.Texture(canvas);
                    tex.needsUpdate = true;
                    tex.premultiplyAlpha = true;
                    this.callback(tex);
                }.bind(this),
            });
        }
    }, 
    move: function(viewer) {
        var pt, opacity, pixel, cull;
        var gl = viewer.renderer.context;
        var height = viewer.renderer.domElement.clientHeight;
        var pix = new Uint8Array(4);
        $(this.labels).each(function() {
            pt = viewer.getPos(this.getAttribute("data-ptidx"));

            gl.readPixels(pt.norm[0], height - pt.norm[1], 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pix);
            pixel = (pix[0] << 16) + (pix[1] << 8) + pix[2];
            cull = pixel == this.getAttribute("data-ptcolor");
            
            opacity = Math.max(-pt.dot, 0);
            opacity = viewer.flatmix + (1 - viewer.flatmix)*opacity;
            this.style.left = Math.round(pt.pos[0])+"px";
            this.style.top =  Math.round(pt.pos[1])+"px";
            this.style.opacity = cull * (1 - Math.exp(-opacity * 10));
        });
    },
    setMix: function(viewer) {
        var verts = this.vertices;
        $(this.labels).each(function() {
            var idx = this.getAttribute("data-ptidx");
            var vert = viewer.getVert(idx);
            verts[idx].copy(vert.norm);
        });
        this.particles.left.geometry.verticesNeedUpdate = true;
        this.particles.right.geometry.verticesNeedUpdate = true;
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
