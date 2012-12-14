function ROIpack(svgdoc, callback, viewer) {
    this.callback = callback;
    this.svgroi = svgdoc.getElementsByTagName("svg")[0];
    this.svgroi.id = "svgroi";
    //document.getElementById("hiderois").appendChild(this.svgroi);
    this.rois = $(this.svgroi).find("path");
    this.rois.each(function() { this.removeAttribute("filter"); });
    var geoms = {left:new THREE.Geometry(), right:new THREE.Geometry()};
    geoms.left.dynamic = true;
    geoms.right.dynamic = true;
    var verts = {};
    var labels = [];
    var colors = 1;
    $(this.svgroi).find("#roilabels text").each(function() {
        var name = $(this).text();
        var el = document.createElement("p");
        var pt = viewer.remap($(this).data("ptidx"));
        var color = new THREE.Color(colors++);
        var ptdat = viewer.getVert(pt);
        verts[pt] = ptdat.norm;
        geoms[ptdat.name].vertices.push(ptdat.norm);
        geoms[ptdat.name].colors.push(color);

        el.setAttribute("data-ptidx", pt);
        el.setAttribute("data-ptcolor", color.getHex());
        el.className = "roilabel";
        el.innerHTML = name;
        labels.push(el);
    });
    $(this.svgroi).find("#roilabels").remove();
    $(this.svgroi).find("defs").remove();
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
    $("#roilabels").append(this.labels);

    var w = this.svgroi.getAttribute("width");
    var h = this.svgroi.getAttribute("height");
    this.aspect = w / h;
    this.svgroi.setAttribute("viewBox", "0 0 "+w+" "+h);

    var gl = viewer.renderer.context;
    var height = Math.min(4096, gl.getParameter(gl.MAX_TEXTURE_SIZE)) / this.aspect;
    this.setHeight(height);
    
    this._updatemove = true;
    this._updatemix = true;
    this.setLabels = function() {
        if (this._updatemove) {
            this.move(viewer);
            this._updatemove = false;
        }
        if (this._updatemix) {
            this.setMix(viewer);
            this._updatemix = false;
        }
    }.bind(this);
    this.canvas = document.createElement("canvas");
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
    move: function(viewer) {
        //Move each label to the 
        var pt, opacity, pixel, idx, cull = 1;
        var gl = viewer.renderer.context;
        var width = viewer.renderer.domElement.clientWidth;
        var height = viewer.renderer.domElement.clientHeight;
        var pix = viewer.pixbuf;
        var labelcull = $("#labelcull").attr("checked") == "checked";

        if (labelcull)
            gl.readPixels(0,0,width,height,gl.RGBA, gl.UNSIGNED_BYTE, pix);
        this.labels.each(function() {
            pt = viewer.getPos(this.getAttribute("data-ptidx"));

            //Check if the indicator particle is visible
            if (labelcull) {
                idx = Math.floor(height - pt.norm[1])*width*4+Math.floor(pt.norm[0])*4;
                pixel = (pix[idx+0] << 16) + (pix[idx+1]<<8) + pix[idx+2];
                cull = pixel == this.getAttribute("data-ptcolor");
            }

            opacity = cull*Math.max(-pt.dot, 0);
            opacity = viewer.flatmix + (1 - viewer.flatmix)*opacity;
            opacity = 1 - Math.exp(-opacity * 10);
            
            this.style.cssText = [
                "left:", Math.round(pt.norm[0]), "px;",
                "top:", Math.round(pt.norm[1]), "px;",
                "opacity:", opacity,
            ].join("");
        });
    },
    setMix: function(viewer) {
        //Adjust the indicator particle to match the current mix state
        //Indicator particles are set off from the surface by the normal
        var verts = this.vertices;
        $(this.labels).each(function() {
            var idx = this.getAttribute("data-ptidx");
            var vert = viewer.getVert(idx);
            verts[idx].copy(vert.norm);
        });
        this.particles.left.geometry.verticesNeedUpdate = true;
        this.particles.right.geometry.verticesNeedUpdate = true;
        this.move(viewer);
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
