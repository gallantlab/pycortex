var dataset = (function(module) {
    module.filtertypes = { 
        nearest: THREE.NearestFilter, 
        trilinear: THREE.LinearFilter, 
        nearlin: THREE.LinearFilter, 
        debug: THREE.NearestFilter 
    };

    module.samplers = {
        trilinear: "trilinear",
        nearest: "nearest",
        nearlin: "nearest",
        debug: "debug",
    };

    module.brains = {}; //Singleton containing all BrainData objects

    module.makeFrom = function(dvx, dvy) {
        //Generate a new DataView given two existing ones
        var json = {};
        json.name = dvx.name + " vs. "+ dvy.name;
        json.data = [[dvx.data[0].name, dvy.data[0].name]];
        json.description = "2D colormap for "+dvx.name+" and "+dvy.name;
        json.cmap = [viewopts.default_2Dcmap];
        json.vmin = [[dvx.vmin, dvy.vmin]];
        json.vmax = [[dvx.vmax, dvy.vmax]];
        json.attrs = dvx.attrs;
        json.state = dvx.state;
        json.xfm = [[dvx.xfm, dvy.xfm]];
        return new module.DataView(json);
    };

    module.DataView = function(json) {
        this.data = [];
        //Do not handle muliviews for now!
        for (var i = 0; i < json.data.length; i++) {
            if (json.data[i] instanceof Array) {
                this.data.push(module.brains[json.data[i][0]]);
                this.data.push(module.brains[json.data[i][1]]);
            } else {
                this.data.push(module.brains[json.data[i]]);
            }
        }
        this.name = json.name;
        this.description = json.desc;

        //Still no multiviews!
        this.xfm = json.xfm[0];
        this.cmap = json.cmap[0];
        this.vmin = json.vmin[0];
        this.vmax = json.vmax[0];

        this.attrs = json.attrs;
        this.state = json.state;
        this.loaded = $.Deferred().done(function() { $("#dataload").hide(); });
        this.rate = json.attrs.rate === undefined ? 1 : json.attrs.rate;
        this.delay = json.attrs.delay === undefined ? 0 : json.attrs.delay;
        this.filter = json.attrs.filter === undefined ? "nearest" : json.attrs.filter;
        if (json.attrs.stim !== undefined)
            this.stim = "stim/"+json.attrs.stim;

        this.frames = this.data[0].frames
        this.length = this.frames / this.rate;

        this.uniforms = {
            data:       { type:'tv', value:0, texture: [this.blanktex, this.blanktex, this.blanktex, this.blanktex]},
            colormap:   { type:'t',  value:4, texture: this.blanktex },
            mosaic:     { type:'v2v', value:[new THREE.Vector2(6, 6), new THREE.Vector2(6, 6)]},
            dshape:     { type:'v2v', value:[new THREE.Vector2(100, 100), new THREE.Vector2(100, 100)]},
            volxfm:     { type:'m4v', value:[new THREE.Matrix4(), new THREE.Matrix4()] },
            framemix:   { type:'f',  value:0},

            vmin:       { type:'fv1',value:[0,0]},
            vmax:       { type:'fv1',value:[1,1]},

            dataAlpha:  { type:'f', value:1.0},
            voxlineColor:{type:'v3', value:new THREE.Vector3( 0,0,0 )},
            voxlineWidth:{type:'f', value:viewopts.voxline_width},
        }
    }
    module.DataView.prototype.get_shader = function(uniforms, shaderfunc) {
        if (this.loaded.state() == "pending")
            $("#dataload").show();

        if (this.shader !== undefined)
            this.shader.dispose();
        if (this.fastshader !== undefined)
            this.fastshader.dispose();

        var rois = uniforms.map.texture !== null;
        var sampler = module.samplers[this.filter];
        var shaders = Shaders.main(sampler, this.data[0].raw, this.data.length > 1, viewopts.voxlines, uniforms.nsamples.value, rois);
        this.shader = new THREE.ShaderMaterial({ 
            vertexShader:"#define SUBJ_SURF\n"+shaders.vertex,
            fragmentShader:"#define SUBJ_SURF\n"+shaders.fragment,
            uniforms: uniforms,
            attributes: { wm:true, auxdat:true },
            morphTargets:true, 
            morphNormals:true, 
            lights:true, 
            blending:THREE.CustomBlending,
        });
        this.shader.map = true;
        this.shader.metal = true;

        if (uniforms.nsamples.value > 1) {
            shaders = Shaders.main(sampler, this.data[0].raw, this.data.length > 1, viewopts.voxlines, 1, rois);
            this.fastshader = new THREE.ShaderMaterial({
                vertexShader:"#define SUBJ_SURF\n"+shaders.vertex,
                fragmentShader:"#define SUBJ_SURF\n"+shaders.fragment,
                uniforms: uniforms,
                attributes: { wm:true, auxdat:true },
                morphTargets:true, 
                morphNormals:true, 
                lights:true, 
                blending:THREE.CustomBlending,
            });
            this.fastshader.map = true;
            this.fastshader.metal = true;
        } else {
            this.fastshader = null;
        }

        var allready = [];
        for (var i = 0; i < this.data.length; i++) {
            allready.push(false);
        }

        var deferred = this.data.length == 1 ? 
            $.when(this.data[0].loaded) : 
            $.when(this.data[0].loaded, this.data[1].loaded);
        deferred.done(function() {
            this.loaded.resolve();
            for (var i = 0; i < this.data.length; i++) {
                this.data[i].init(uniforms, i);
                this.data[i].setFilter(this.filter);
                this.data[i].set(uniforms, i, 0);
            }
        }.bind(this)).progress(function() {
            for (var i = 0; i < this.data.length; i++) {
                if (this.data[i].textures.length > this.delay && !allready[i]) {
                    this.data[i].setFilter(this.filter);
                    this.data[i].set(uniforms, i, 0);
                    allready[i] = true;

                    var test = true;
                    for (var i = 0; i < allready.length; i++)
                        test = test && allready[i];
                    if (test)
                        this.loaded.resolve();
                }
            }
        }.bind(this));

        return this.shader;
    }
    module.DataView.prototype.set = function(uniforms, time) {
        var xfm;
        var frame = ((time + this.delay) * this.rate).mod(this.frames);
        var fframe = Math.floor(frame);
        uniforms.framemix.value = frame - fframe;
        for (var i = 0; i < this.data.length; i++) {
            this.data[i].set(uniforms, i, fframe);
            xfm = uniforms.volxfm.value[i];
            xfm.set.apply(xfm, this.xfm.length != 16 ? this.xfm[i] : this.xfm);
        }
    };
    module.DataView.prototype.setFilter = function(interp) {
        this.filter = interp;
        for (var i = 0; i < this.data.length; i++)
            this.data[i].setFilter(interp);
    }

    module.BrainData = function(json, images) {
        this.loaded = $.Deferred();
        this.subject = json.subject;
        this.movie = images[json.name].length > 1;
        this.raw = json.raw;
        this.min = json.min;
        this.max = json.max;
        this.mosaic = json.mosaic;
        this.name = json.name;

        this.data = images[json.name];
        this.frames = images[json.name].length;

        this.textures = [];
        var loadmosaic = function(idx) {
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
                tex.minFilter = module.filtertypes['nearest'];
                tex.magfilter = module.filtertypes['nearest'];
                tex.needsUpdate = true;
                tex.flipY = false;
                this.shape = [((img.width-1) / this.mosaic[0])-1, ((img.height-1) / this.mosaic[1])-1];
                this.textures.push(tex);

                if (this.textures.length < this.frames) {
                    this.loaded.notify(this.textures.length);
                    loadmosaic(this.textures.length);
                } else {
                    this.loaded.resolve();
                }
            }.bind(this));
            img.src = this.data[this.textures.length];
        }.bind(this);

        loadmosaic(0);
    };
    module.BrainData.prototype.setFilter = function(interp) {
        //this.filter = interp;
        for (var i = 0, il = this.textures.length; i < il; i++) {
            this.textures[i].minFilter = module.filtertypes[interp];
            this.textures[i].magFilter = module.filtertypes[interp];
            this.textures[i].needsUpdate = true;
        }
    };
    module.BrainData.prototype.init = function(uniforms, dim) {
        uniforms.mosaic.value[dim].set(this.mosaic[0], this.mosaic[1]);
        uniforms.dshape.value[dim].set(this.shape[0], this.shape[1]);
    };
    module.BrainData.prototype.set = function(uniforms, dim, fframe) {
        if (uniforms.data.texture[dim*2] !== this.textures[fframe]) {
            uniforms.data.texture[dim*2] = this.textures[fframe];
            if (this.frames > 1) {
                uniforms.data.texture[dim*2+1] = this.textures[(fframe+1).mod(this.frames)];
            } else {
                uniforms.data.texture[dim*2+1] = null;
            }
        }
    }
    module.fromJSON = function(dataset) {
        for (var name in dataset.data) {
            module.brains[name] = new module.BrainData(dataset.data[name], dataset.images);
        }
        var dataviews = [];
        for (var i = 0; i < dataset.views.length; i++) {
            dataviews.push(new module.DataView(dataset.views[i]));
        }
        return dataviews
    }

    return module;
}(dataset || {}));