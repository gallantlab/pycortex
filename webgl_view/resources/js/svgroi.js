function ROIpack(svgdoc, callback) {
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
    $("#main").children().first().before(this.labels);
    this._shadowtex = new ShadowTex(
        Math.ceil(this.svgroi.width.baseVal.value), 
        Math.ceil(this.svgroi.height.baseVal.value), 
        4
    );
}
ROIpack.prototype = {
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
                    //tex.premultiplyAlpha = true;
                    this.callback(tex);
                }.bind(this),
            });
        }
    }, 
    move: function(viewer) {
        $(this.labels).each(function() {
            var posdot = viewer.get_pos($(this).data("ptidx"));
            var opacity = Math.max(-posdot[1], 0);
            opacity = viewer.flatmix + (1 - viewer.flatmix)*opacity;
            $(this).css({
                left: Math.round(posdot[0][0]),
                top: Math.round(posdot[0][1]),
                opacity: 1 - Math.exp(-opacity * 20)
            })
        })
    },
}