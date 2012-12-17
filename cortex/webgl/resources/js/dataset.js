function Dataset(nparray) {
    if (nparray instanceof NParray) {
        this.init(nparray);
    }
}
Dataset.fromJSON = function(json) {
    var ds;
    if (json.data instanceof NParray) {
        ds = new Dataset(json.data);
    } else if (typeof(json.data) == "string") {
        ds = new Dataset(null);
        NParray.fromURL(json.data, function(array) {
            ds.init(array);
        });
    }
    if (json.stim !== undefined)
        ds.addStim(json.stim, json.delay);
    ds.min = json.min;
    ds.max = json.max;
    ds.cmap = json.cmap;
    ds.rate = json.rate === undefined ? 1 : json.rate;
    return ds;
}
Dataset.maketex = function(array, shape, raw, slice) {
    //Creates paired textures on top of each other
    var tex, data, form;
    var arr = slice === undefined ? array : array.view(slice);
    var size = array.shape[array.shape.length-1];
    var len = shape[0] * shape[1] * (raw ? size : 1);
    
    data = new array.data.constructor(len*2);
    data.subarray(0, len).set(arr.data);
    if (slice+1 < array.shape[0]) {
        arr = array.view(slice+1);
        data.subarray(len).set(arr.data);
    }
    
    if (raw) {
        form = size == 4 ? THREE.RGBAFormat : THREE.RGBFormat;
        tex = new THREE.DataTexture(data, shape[0], shape[1]*2, form, THREE.UnsignedByteType);
    } else {
        tex = new THREE.DataTexture(data, shape[0], shape[1]*2, THREE.LuminanceFormat, THREE.FloatType);
    }
    tex.needsUpdate = true;
    tex.flipY = false;
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    return tex;
}
Dataset.prototype = { 
    init: function(nparray) {
        this.array = nparray;
        this.raw = nparray.data instanceof Uint8Array && nparray.shape.length > 1;
        this.textures = [];

        if ((this.raw && nparray.shape.length > 2) || (!this.raw && nparray.shape.length > 1)) {
            //Movie
            this.datasize = new THREE.Vector2(256, Math.ceil(nparray.shape[1] / 256));
            for (var i = 0; i < nparray.shape[0]; i++) {
                this.textures.push(Dataset.maketex(nparray, [this.datasize.x, this.datasize.y], this.raw, i));
            }
        } else {
            //Single frame
            this.datasize = new THREE.Vector2(256, Math.ceil(nparray.shape[0] / 256));
            this.textures.push(Dataset.maketex(nparray, [this.datasize.x, this.datasize.y], this.raw));
        }
        var minmax = nparray.minmax();
        this.lmin = minmax[0];
        this.lmax = minmax[1];
        if (this.min === undefined)
            this.min = this.lmin;
        if (this.max === undefined)
            this.max = this.lmax;

        if (this._func !== undefined) {
            this._func();
            delete this._func;
        }
    },
    addStim: function(url, delay) {
        this.delay = delay;
        this.stim = document.createElement("video");
        this.stim.id = "stim_movie";
        this.stim.setAttribute("preload", "");
        this.stim.setAttribute("loop", "loop");
        var src = document.createElement("source");
        var ext = url.match(/^(.*)\.(\w{3,4})$/);
        if (ext.length == 3) {
            if (ext[2] == 'ogv') {
                src.setAttribute('type', 'video/ogg; codecs="theora, vorbis"');
            } else if (ext[2] == 'webm') {
                src.setAttribute('type', 'video/webm; codecs="vp8, vorbis"');
            } else if (ext[2] == 'mp4') {
                src.setAttribute('type', 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"');
            }
        }
        src.setAttribute("src", url);
        this.stim.appendChild(src);

        allStims.push(this.stim);

        this.stim.seekTo = function(time) {
            if (this.seekable.length > 0 && 
                this.seekable.end(0) >= time) {
                this.currentTime = time;
                $("#pluginload").hide()
                if (typeof(this.callback) == "function")
                    this.callback();
            } else {
                this.seekTarget = time;
                $("#pluginload").show()
            }
        }
        $(this.stim).bind("progress", function() {
            if (this.seekTarget != null && 
                this.seekable.length > 0 && 
                this.seekable.end(0) >= this.seekTarget &&
                this.parentNode != null) {
                var func = function() {
                    try {
                        this.currentTime = this.seekTarget;
                        this.seekTarget = null;
                        $("#pluginload").hide()
                        if (typeof(this.callback) == "function")
                            this.callback();
                    } catch (e) {
                        console.log(e);
                        setTimeout(func, 5);
                    }
                }.bind(this);
                func();
            }
        });
        this.stim.seekTo(delay);
    },
    set: function(viewer, dim) {
        if (dim === undefined)
            dim = 0;
        var func = function() {
            viewer._setShader(this.raw, this.datasize);
            if (this.textures.length > 1) {
                viewer.setFrame(0);
                $("#moviecontrols").show();
                $("#bottombar").addClass("bbar_controls");
                $("#movieprogress div").slider("option", {min:0, max:this.textures.length});
            } else {
                viewer.shader.uniforms['ds_' + (dim ? 'y':'x')].texture = this.textures[0];
                $("#moviecontrols").hide();
                $("#bottombar").removeClass("bbar_controls");
            }

            if (this.cmap !== undefined) {
                viewer.setColormap(this.cmap);
            }

            $(dim == 0 ? "#vrange" : "#vrange2").slider("option", {min: this.lmin, max:this.lmax});
            viewer.setVminmax(this.min, this.max, dim);

            if (dim > 0)
                return;

            if (this.stim === undefined) {
                viewer.rmPlugin();
            } else {
                for (var i = 0; i < allStims.length; i++)
                    $(allStims[i]).unbind("timeupdate");
                
                var delay = this.delay;
                viewer.addPlugin(this.stim);
                //this.stim.seekTo(viewer.frame + delay);
                $(this.stim).bind("timeupdate", function() {
                    var now = new Date().getTime() / 1000;
                    var sec = now - viewer._startplay / 1000;
                    console.log("Time diff: "+(this.currentTime - delay - sec));
                    viewer._startplay = (now - this.currentTime + delay)*1000;
                })
            }
        }.bind(this);
        if (this.array !== undefined)
            func();
        else
            this._func = func;
    },
}