function Graph() {
    this.nodes = {};
    this.selcategory = null;
    this.voxdir = null;
    this.subject = null;
    this.catmapdir = null;
    this.lastdataname = "";
    this.showinglabels = true;
}
Graph.prototype.setrgbdata = function(data) {
    // Changes what is shown on the graph to the given [data].
    // [data] should be an associative array of either
    // 1. Hex color string and float size pairs
    // 2. Floating point values
    this.hideselcategory();

    // Right now let's just assume the former
    for (var name in data) {
        // console.log(name);
        var ndata = data[name];

        if (name in this.nodes) {
            // console.log("Animating "+name)
            // $(this.nodes[name].obj.nodemarker).attr("fill", ndata[0]);
            // $(this.nodes[name].obj.nodemarker).animate({svgFill: ndata[0], svgR: Math.sqrt(ndata[1]*60)}, 2000, "linear");
            $(this.nodes[name]).attr("fill", ndata[0]).attr("r", Math.sqrt(ndata[1]*60));
        }
    }
}
Graph.prototype.setlindata = function(data) {
    // Changes what is shown on the graph to the given [data].
    // [data] should be an associative array of values

    // Right now let's just assume the former
    for (var name in data) {
        // console.log(name);
        var ndata = data[name];
        if (name in this.nodes) {
            // console.log("Animating "+name)
            var nodecolor = ndata>0 ? "red" : "blue";
            // $(this.nodes[name].obj.nodemarker).animate({svgFill: nodecolor, svgR: Math.sqrt(Math.abs(ndata))}, 2000, "linear");
            $(this.nodes[name]).attr("fill", nodecolor).attr("r", 2*Math.sqrt(Math.abs(ndata)));
            // $(this.nodes[name].obj.nodemarker).attr("fill", ndata[0]);
            // $(this.nodes[name]).animate({svgR: Math.sqrt(ndata[1]*60)}, 5000, "linear");
        }
    }
}
Graph.prototype.showvoxel = function(voxind, vertind) {
    // Shows the data for voxel [voxind] on the graph.
    // Automatically loads from a set directory..
    var gr = this;
    var voxcorr = this.viewer.datasets.ModelPerformance.textures[0][vertind].toFixed(3); //Get the reordered index value
    // console.log("Showing STRF for voxel "+voxind);
    $("#wngraph-title").html("Voxel "+this.subject+"-"+voxind+", <span title='Correlation between model prediction and actual responses. r over 0.1 is significant, r over 0.3 is good. Only trust voxels where r is over 0.1'><i>r</i>&nbsp;=&nbsp;"+voxcorr+"</span>");
    var data = NParray.fromURL(this.voxdir+this.subject+"-"+voxind+".bin", 
        function (npa) {
            for (var i=0; i<nodenames.length; i++) {
                var val = npa.data[i];
                var nodecolor = val>0 ? "red" : "blue";
                $(gr.nodes[nodenames[i]]).attr("fill", nodecolor).attr("r", 2*Math.sqrt(Math.abs(val)));
            }
    });
    // Setup link
    var voxurl = window.location.href.split("?")[0]+"?vox="+voxind+"&vert="+vertind;
    $("#linkurl").val(voxurl);
    $("#linkurl").hide();
    $("#viewlink").show();
}
Graph.prototype.showcategory = function(catname) {
    // If there was a previously selected node, un-select it
    this.hideselcategory();
    this.selcategory = catname;
    var node = this.nodes[catname];
    
    // Expand the marker of the selected node, add to its edge
    $(node).attr("stroke", "#fff");
    //node.origsw = $(node).attr("stroke-width");
    //node.origr = $(node).attr("r");
    $(node).animate({svgStrokeWidth: "+=4", svgR: "+=20"}, 500, "swing");
    $(node).mouseout();
    
    for (var i=0; i<nodenames.length; i++) {
        if (nodenames[i]==catname) {
            // console.log("Showing map for category "+nodenames[i]+", from url "+this.catmapdir+this.subject+"-"+i+".bin");
            if (this.lastdataname != "") {
                this.viewer.rmData(this.lastdataname);
            }
            this.lastdataname = 'Category: '+catname.split(".")[0].replace(/_/g, " ");
            NParray.fromURL(this.catmapdir+this.subject+"-"+i+".bin", 
                function (data) {
                    var catdata = Object();
                    var dset = new Dataset(data);
                    dset.cmap = "RdBu_r";
                    dset.min = -3;
                    dset.max = 3;
                    catdata['Category: '+catname.split(".")[0].replace(/_/g, " ")] = dset;
                    this.viewer.addData(catdata);
                });
        }
    }
}
Graph.prototype.hideselcategory = function() {
    if (this.selcategory != null) {
	var oldsel = this.nodes[this.selcategory];
	$(oldsel).animate({svgStrokeWidth:oldsel.origsw, svgR:oldsel.origr}, 100, "swing");
    }
}
Graph.prototype.togglelabels = function() {
    if (this.showinglabels) {
        // Hide them!
        $("#wngraph > g#biglabels").animate({svgOpacity: 0.0}, 500, "swing");
        $("#wnlabelbutton").text("Show Labels");
        this.showinglabels = false;
    }
    else {
        // Show them!
        $("#wngraph > g#biglabels").animate({svgOpacity: 1.0}, 500, "swing");
        $("#wnlabelbutton").text("Hide Labels");
        this.showinglabels = true;
    }
}
Graph.prototype.showrgb = function() {
    this.setrgbdata(colordata);
    $("#wngraph-title").html("Semantic Space");
    $("#viewlink").hide();
    $("#linkurl").hide();
    this.viewer.picker.axes[0].obj.parent.remove(this.viewer.picker.axes[0].obj);
    this.viewer.picker.axes = [];
    this.viewer.controls.dispatchEvent({type:"change"});
}

function setupGraph() {
    var svgnode = $("#wngraph")[0];
    var graph = new Graph();
    $("#wngraph circle").each(function() {
        graph.nodes[this.id] = this;
        $(this).attr("stroke-width", "0");
        /// Add mouse events
        $(this).click(function(e){
	    graph.showcategory(this.id);
	});
        $(this).hoverIntent({
            over: function(e){
		var exp = 10;
		if (graph.selcategory != this.id) {
		    exp = 10;
		}
		else {
		    exp = 0;
		}
                var lbgpad = 3;
		var lbgoffset = 10;
		var labeloffsetsign = 1;
		var lbgoffx = -lbgoffset - lbgpad;
		var lbgoffw = 2*lbgpad + lbgoffset;

                // Move this node in front
                $("#svgnodes")[0].removeChild(this);
                $("#svglabels")[0].appendChild(this);

                var svglabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
		if ($(this).attr("cx")>400) {
		    // The label will go on the left
		    svglabel.setAttribute("text-anchor", "end");
		    labeloffsetsign = -1;
		    lbgoffx = -lbgpad;
		    lbgoffw = 2*lbgpad + lbgoffset;
		}

                svglabel.setAttribute("x", $(this).attr("cx"));
                svglabel.setAttribute("y", Math.round($(this).attr("cy")));
                svglabel.setAttribute("fill", "#fff");
                svglabel.setAttribute("font-weight", "bold");
                svglabel.setAttribute("font-family", "arial");
                svglabel.setAttribute("font-size", "24px");
                svglabel.setAttribute("visibility", "hidden");
                svglabel.setAttribute("alignment-baseline", "middle");
                svglabel.textContent = this.id.split(".")[0].replace(/_/g, " ");
                $("#svglabels")[0].appendChild(svglabel);

                var label = $(svglabel);
                this.label = label;
                // Make the label visible, position it correctly
                label.attr("visibility", "visible");
                label.fadeIn(100);
                label.attr("x", parseFloat($(this).attr("cx")) + labeloffsetsign*(parseFloat($(this).attr("r")) + parseFloat($(this).attr("stroke-width")) + exp + lbgpad));

                // Create background for label
                var labelbg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                var lbb = label[0].getBBox();
                labelbg.setAttribute("x", Math.round(lbb.x+lbgoffx));
                labelbg.setAttribute("y", Math.round(lbb.y-lbgpad+2));
                labelbg.setAttribute("width", Math.round(lbb.width+lbgoffw));
                labelbg.setAttribute("height", Math.round(lbb.height+2*lbgpad-4));
                labelbg.setAttribute("fill", "#000");
                labelbg.setAttribute("fill-opacity", 0.8);
                labelbg.setAttribute("stroke", "#fff");
                labelbg.setAttribute("stroke-width", 2);
                labelbg.setAttribute("stroke-opacity", 1.0);
                labelbg.setAttribute("rx", 5);
                labelbg.setAttribute("ry", 5);
                $("#svgnodes")[0].appendChild(labelbg);
                this.labelbg = labelbg;

                // Expand the marker, add to its edge
		if (graph.selcategory != this.id) {
                    $(this).attr("stroke", "#fff");
                    this.origsw = $(this).attr("stroke-width");
                    this.origr = $(this).attr("r");
                    $(this).animate({svgStrokeWidth: "+=2", svgR: "+="+exp}, 100, "swing");
		}

                // Show the node definition
                $("#wngraph-definition").html("<span id='wngraph-def-synset'>"+this.id + "</span>: " + wndefs[this.id]);
                 
            },
            timeout: 300,
            sensitivity: 100,
            interval: 20,
            out: function(e){
                this.label.fadeOut(100);
                $(this.labelbg).fadeOut(100);
                //$(this).animate({svgStrokeWidth: "-=2", svgR: "-="+exp}, 100, "swing");
		if (graph.selcategory != this.id) {
                    $(this).animate({svgStrokeWidth: "-=2", svgR: this.origr}, 100, "swing");
		    if (this.origsw==0) {
			$(this).attr("stroke", "none");
                    }
		}
                $("#svgnodes")[0].removeChild(this.labelbg);
                $("#svglabels")[0].removeChild(this);
                $("#svgnodes").append(this);
            }
        });
    });
    return graph;
}

var dtypeNames = {
    "uint32":Uint32Array,
    "uint16":Uint16Array,
    "uint8":Uint8Array,
    "int32":Int32Array,
    "int16":Int16Array,
    "int8":Int8Array,
    "float32":Float32Array,
}
var dtypeMap = [Uint32Array, Uint16Array, Uint8Array, Int32Array, Int16Array, Int8Array, Float32Array]
var loadCount = 0;
function NParray(data, dtype, shape) {
    this.data = data;
    this.shape = shape;
    this.dtype = dtype;
    this.size = this.data.length;
}
NParray.fromJSON = function(json) {
    var size = json.shape.length ? json.shape[0] : 0;
    for (var i = 1, il = json.shape.length; i < il; i++) {
        size *= json.shape[i];
    }
    var pad = 0;
    if (json.pad) {
        size += json.pad;
        pad = json.pad;
    }

    var data = new (dtypeNames[json.dtype])(size);
    var charview = new Uint8Array(data.buffer);
    var bytes = charview.length / data.length;
    
    var str = atob(json.data.slice(0,-1));
    var start = pad*bytes, stop  = size*bytes - start;
    for ( var i = start, il = Math.min(stop, str.length); i < il; i++ ) {
        charview[i] = str.charCodeAt(i - start);
    }

    return new NParray(data, json.dtype, json.shape);
}
NParray.fromURL = function(url, callback) {
    loadCount++;
    $("#dataload").show();
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';
    xhr.onload = function(e) {
        if (this.readyState == 4 && this.status == 200) {
            loadCount--;
            var data = new Uint32Array(this.response, 0, 2);
            var dtype = dtypeMap[data[0]];
            var ndim = data[1];
            var shape = new Uint32Array(this.response, 8, ndim);
            var array = new dtype(this.response, (ndim+2)*4);
            callback(new NParray(array, dtype, shape));
            if (loadCount == 0)
                $("#dataload").hide();
        }
    };
    xhr.send()
}
NParray.prototype.view = function() {
    if (arguments.length == 1 && this.shape.length > 1) {
        var shape;
        var slice = arguments[0];
        if (this.shape.slice)
            shape = this.shape.slice(1);
        else
            shape = this.shape.subarray(1);
        var size = shape[0];
        for (var i = 1, il = shape.length; i < il; i++) {
            size *= shape[i];
        }
        return new NParray(this.data.subarray(slice*size, (slice+1)*size), this.dtype, shape);
    } else {
        throw "Can only slice in first dimension for now"
    }
}
NParray.prototype.minmax = function() {
    var min = Math.pow(2, 32);
    var max = -min;

    for (var i = 0, il = this.data.length; i < il; i++) {
        if (this.data[i] < min) {
            min = this.data[i];
        }
        if (this.data[i] > max) {
            max = this.data[i];
        }
    }

    return [min, max];
}
