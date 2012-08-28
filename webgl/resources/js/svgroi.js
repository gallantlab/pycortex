function ROIpack(svgdoc, callback, height) {
    this.callback = callback;
    this.svgroi = svgdoc.getElementsByTagName("svg")[0];
    this.svgroi.id = "svgroi";
    //document.getElementById("hiderois").appendChild(this.svgroi);
    this.rois = $(this.svgroi).find("path");
    var labels = [];
    $(this.svgroi).find("foreignObject p").each(function() {
        var name = $(this).text();
        var el = document.createElement("p");
        el.setAttribute("data-ptidx", $(this).data("ptidx"));
        el.className = "roilabel";
        el.innerHTML = name;
        labels.push(el);
    });
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
                    //this.callback(tex);
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
        $(this.labels).each(function() {
            var posdot = viewer.getPos(this.getAttribute("data-ptidx"));
            var opacity = Math.max(-posdot[1], 0);
            opacity = viewer.flatmix + (1 - viewer.flatmix)*opacity;
            this.style.left = Math.round(posdot[0][0])+"px";
            this.style.top =  Math.round(posdot[0][1])+"px";
            this.style.opacity = 1 - Math.exp(-opacity * 10);
        })
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