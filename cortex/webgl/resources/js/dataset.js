var filtertypes = { 
    nearest: THREE.NearestFilter, 
    trilinear: THREE.LinearFilter, 
    nearlin: THREE.LinearFilter, 
    debug: THREE.NearestFilter 
};

function Dataset(json) {
    this.loaded = $.Deferred().done(function() { $("#dataload").hide(); });

    this.data = json.data;
    this.mosaic = json.mosaic;
    this.frames = json.data.length;
    this.xfm = json.xfm;
    this.min = json.min;
    this.max = json.max;
    this.lmin = json.lmin;
    this.lmax = json.lmax;
    this.cmap = json.cmap;
    this.stim = json.stim;
    this.rate = json.rate === undefined ? 1 : json.rate;
    this.raw = json.raw === undefined ? false : json.raw;
    this.delay = json.delay === undefined ? 0 : json.delay;
    this.filter = json.filter == undefined ? "nearest" : json.filter;
    this.textures = [];

    var loadtex = function(idx) {
        var img = new Image();
        img.addEventListener("load", function() {
            var tex;
            if (this.raw) {
                tex = new THREE.Texture(img);
                tex.premultiplyAlpha = true;
            } else {
                var canvas = document.createElement("canvas");
                var ctx = canvas.getContext('2d');
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                var im = ctx.getImageData(0, 0, img.width, img.height).data;
                var arr = new Float32Array(im.buffer);
                tex = new THREE.DataTexture(arr, img.width, img.height, THREE.LuminanceFormat, THREE.FloatType);
                tex.premultiplyAlpha = false;
            }
            tex.minFilter = filtertypes[this.filter];
            tex.magFilter = filtertypes[this.filter];
            tex.needsUpdate = true;
            tex.flipY = false;
            this.shape = [((img.width-1) / this.mosaic[0])-1, ((img.height-1) / this.mosaic[1])-1];
            this.textures.push(tex);

            if (this.textures.length < this.frames) {
                this.loaded.progress(this.textures.length);
                loadtex(this.textures.length);
            } else {
                this.loaded.resolve();
            }
        }.bind(this));
        img.src = this.data[this.textures.length];
    }.bind(this);

    loadtex(0);
    $("#dataload").show();
}
Dataset.fromJSON = function(json) {
    return new Dataset(json);
}

Dataset.prototype = { 
    set: function(uniforms, dim, frame, init) {
        if (dim === undefined)
            dim = 0;

        if (init) {
            uniforms.mosaic.value[dim].set(this.mosaic[0], this.mosaic[1]);
            uniforms.shape.value[dim].set(this.shape[0], this.shape[1]);
            uniforms.volxfm.value[dim].set.apply(uniforms.volxfm.value[dim], this.xfm);
        }

        if (uniforms.data.texture[dim] !== this.textures[frame]) {
            uniforms.data.texture[dim] = this.textures[frame];
            uniforms.data.texture[dim].needsUpdate = true;
            if (this.frames > 1) {
                uniforms.data.texture[dim+2] = this.textures[frame+1];
                uniforms.data.texture[dim+2].needsUpdate = true;
            }
        }
    },
}

// if (this.stim === undefined) {
//     viewer.rmPlugin();
// } else {
//     for (var i = 0; i < allStims.length; i++)
//         $(allStims[i]).unbind("timeupdate");
    
//     var delay = this.delay;
//     viewer.addPlugin(this.stim);
//     if (delay >= 0)
//         this.stim.seekTo(viewer.frame - delay);
//     else
//         viewer.setFrame(-delay);
//     $(this.stim).bind("timeupdate", function() {
//         var now = new Date().getTime() / 1000;
//         var sec = now - viewer._startplay / 1000;
//         console.log("Time diff: "+(this.currentTime + delay - sec));
//         viewer._startplay = (now - this.currentTime - delay)*1000;
//     })
// }

    // addStim: function(url, delay) {
    //     this.delay = delay;
    //     this.stim = document.createElement("video");
    //     this.stim.id = "stim_movie";
    //     this.stim.setAttribute("preload", "");
    //     this.stim.setAttribute("loop", "loop");
    //     var src = document.createElement("source");
    //     var ext = url.match(/^(.*)\.(\w{3,4})$/);
    //     if (ext.length == 3) {
    //         if (ext[2] == 'ogv') {
    //             src.setAttribute('type', 'video/ogg; codecs="theora, vorbis"');
    //         } else if (ext[2] == 'webm') {
    //             src.setAttribute('type', 'video/webm; codecs="vp8, vorbis"');
    //         } else if (ext[2] == 'mp4') {
    //             src.setAttribute('type', 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"');
    //         }
    //     }
    //     src.setAttribute("src", url);
    //     this.stim.appendChild(src);

    //     allStims.push(this.stim);

    //     this.stim.seekTo = function(time) {
    //         if (this.seekable.length > 0 && 
    //             this.seekable.end(0) >= time) {
    //             this.currentTime = time;
    //             $("#pluginload").hide()
    //             if (typeof(this.callback) == "function")
    //                 this.callback();
    //         } else {
    //             this.seekTarget = time;
    //             $("#pluginload").show()
    //         }
    //     }
    //     $(this.stim).bind("progress", function() {
    //         if (this.seekTarget != null && 
    //             this.seekable.length > 0 && 
    //             this.seekable.end(0) >= this.seekTarget &&
    //             this.parentNode != null) {
    //             var func = function() {
    //                 try {
    //                     this.currentTime = this.seekTarget;
    //                     this.seekTarget = null;
    //                     $("#pluginload").hide()
    //                     if (typeof(this.callback) == "function")
    //                         this.callback();
    //                 } catch (e) {
    //                     console.log(e);
    //                     setTimeout(func, 5);
    //                 }
    //             }.bind(this);
    //             func();
    //         }
    //     });
    // },