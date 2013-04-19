var filtertypes = { nearest: THREE.NearestFilter, trilinear: THREE.LinearFilter };
function Dataset(json) {
    this.loaded = $.Deferred();
    this.data = json.data;
    this.min = json.min;
    this.max = json.max;
    this.cmap = json.cmap;
    this.stim = json.stim;
    this.rate = json.rate === undefined ? 1 : json.rate;
    this.raw = json.raw === undefined ? false : json.raw;
    this.frames = json.frames === undefined ? 1 : json.frames;
    this.delay = json.delay === undefined ? 0 : json.delay;
    this.filter = json.filter == undefined ? "nearest" : json.filter;
    this.textures = [];

    var loadtex = function(idx) {
        var img = new Image();
        img.addEventListener("load", function() {
            var tex = new THREE.Texture(img);
            tex.minFilter = filtertypes[this.filter];
            tex.magFilter = filtertypes[this.filter];
            tex.premultiplyAlpha = false;
            tex.flipY = true;
            tex.needsUpdate = true;
            if (!this.raw) {
                tex.type = THREE.FloatType;
                tex.format = THREE.LuminanceFormat;
            }
            this.textures.push(tex);

            if (this.textures.length < this.frames) {
                this.loaded.progress(textures.length);
                loadtex(textures.length);
            } else {
                this.loaded.resolve();
            }
        }.bind(this));
        var fname = this.frames > 1 ? "_"+idx : "";
        img.src = this.data + fname + ".png";
    }.bind(this);

    loadtex();
}
Dataset.fromJSON = function(json) {
    return new Dataset(json);
}

Dataset.prototype = { 
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
    },
    set: function(viewer, dim) {
        if (dim === undefined)
            dim = 0;

        this.loaded.done(function() {
            viewer._setShader(this.raw);
            if (this.textures.length > 1) {
                viewer.setFrame(0);
                $("#moviecontrols").show();
                $("#bottombar").addClass("bbar_controls");
                $("#movieprogress div").slider("option", {min:0, max:this.textures.length-1});
            } else {
                //viewer._setVerts(this.textures[0], dim);
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
                if (delay >= 0)
                    this.stim.seekTo(viewer.frame - delay);
                else
                    viewer.setFrame(-delay);
                $(this.stim).bind("timeupdate", function() {
                    var now = new Date().getTime() / 1000;
                    var sec = now - viewer._startplay / 1000;
                    console.log("Time diff: "+(this.currentTime + delay - sec));
                    viewer._startplay = (now - this.currentTime - delay)*1000;
                })
            }
        }.bind(this));
    },
}