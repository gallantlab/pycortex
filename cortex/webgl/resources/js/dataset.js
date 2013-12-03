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
        json.cmap = viewopts.default_2Dcmap;
        json.vmin = [[dvx.vmin[0][0], dvy.vmin[0][0]]];
        json.vmax = [[dvx.vmax[0][0], dvy.vmax[0][0]]];
        json.attrs = dvx.attrs;
        json.state = dvx.state;
        return new module.DataView(json);
    };

    module.DataView = function(json) {
        this.data = [];
        //Only handle 2D case for now -- muliviews are difficult to handle in this framework
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
        this.cmap = json.cmap;
        this.vmin = json.vmin;
        this.vmax = json.vmax;
        this.attrs = json.attrs;
        this.state = json.state;
        this.loaded = $.Deferred().done(function() { $("#dataload").hide(); });
        this.rate = json.attrs.rate === undefined ? 1 : json.attrs.rate;
        this.delay = json.attrs.delay === undefined ? 0 : json.attrs.delay;
        this.filter = json.attrs.filter === undefined ? "nearest" : json.attrs.filter;
        if (json.attrs.stim !== undefined)
            this.stim = "stim/"+json.attrs.stim;

        if (!(json.vmin instanceof Array))
            this.vmin = [[json.vmin]]
        if (!(json.vmax instanceof Array))
            this.vmax = [[json.vmax]]

        this.frames = this.data[0].frames
        this.length = this.frames / this.rate;
    }
    module.DataView.prototype.init = function(uniforms, meshes, rois, frames) {
        if (this.loaded.state() == "pending")
            $("#dataload").show();
        frames = frames || 0;

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

        meshes.left.material = this.shader;
        meshes.right.material = this.shader;

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
    }
    module.DataView.prototype.set = function(uniforms, time) {
        var frame = ((time + this.delay) * this.rate).mod(this.frames);
        var fframe = Math.floor(frame);
        uniforms.framemix.value = frame - fframe;
        for (var i = 0; i < this.data.length; i++) {
            this.data[i].set(uniforms, i, fframe);
        }
    };
    module.DataView.prototype.setFilter = function(interp) {
        this.filter = interp;
        for (var i = 0; i < this.data.length; i++)
            this.data[i].setFilter(interp);
    }

    module.BrainData = function(json, images) {
        this.loaded = $.Deferred();
        this.xfm = json.xfm;
        this.subject = json.subject;
        this.movie = json.movie;
        this.raw = json.raw;
        this.min = json.min;
        this.max = json.max;
        this.mosaic = json.mosaic;
        this.name = json.data;

        this.data = images[json.data];
        this.frames = images[json.data].length;

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
        module.brains[json.data] = this;
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
        var xfm = uniforms.volxfm.value[dim];
        xfm.set.apply(xfm, this.xfm);
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
            new module.BrainData(dataset.data[name], dataset.images);
        }
        var dataviews = [];
        for (var i = 0; i < dataset.views.length; i++) {
            dataviews.push(new module.DataView(dataset.views[i]));
        }
        return dataviews
    }

    module.render = function(viewer, res) {
        //Function is not finished yet
        var scene = new THREE.Scene(), 
            camera = new THREE.OrthographicCamera(-100, 100, -100, 100, -100, 100),
            shaders = Shaders.data(),
            shader = new THREE.ShaderMaterial( {
                vertexShader: "#define SUBJ_SURF\n"+shaders.vertex,
                fragmentShader: "#define SUBJ_SURF\n"+shaders.fragment,
                uniforms: {
                    volxfm: { type:'m4', value: new THREE.Matrix4() },
                    data:   { type: 't', value: 0, texture: null },
                },
                attributes: { flatpos: true, wm:true, auxdat:true, },
            }),
            targets = {
                left: new THREE.WebGLRenderTarget(res, res, {
                    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
                    stencilBuffer:false,
                    generateMipmaps:true,
                }),
                right: new THREE.WebGLRenderTarget(res, res, {
                    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
                    stencilBuffer:false,
                    generateMipmaps:true,
                }),
            }
        var hemi, geom, target, mesh, name, attr;
        var names = ["position", "wm", "auxdat", "index"];
        var limits = {top:-1000, bottom:1000};
        var xfm = shader.uniforms.volxfm.value;
        xfm.set.apply(xfm, this.xfm);
        scene.add(camera);
        camera.up.set(0, 0, 1);
        camera.position.set(0,-100,0);
        camera.lookAt(scene.position);

        for (hemi in targets) {
            target = targets[hemi];
            geom = new THREE.BufferGeometry();
            for (var i = 0; i < names.length; i++) {
                name = names[i];
                attr = viewer.meshes[hemi].geometry.attributes[name];
                if (attr !== undefined)
                    geom.attributes[name] = attr;
            }
            geom.offsets = viewer.meshes[hemi].geometry.offsets;
            var mt = viewer.meshes[hemi].geometry.morphTargets;
            geom.attributes.flatpos = mt[mt.length-1];

            var min = [1000, 1000, 1000], max=[0,0,0];
            for (var i = 0, il = geom.attributes.flatpos.array.length; i < il; i+=3) {
                for (var j = 0; j < 3; j++) {
                    min[j] = Math.min(min[j], geom.attributes.flatpos.array[i+j]);
                    max[j] = Math.max(max[j], geom.attributes.flatpos.array[i+j]);
                }
            }
            limits[hemi] = max[1] - min[1];
            limits.top = Math.max(max[2], limits.top);
            limits.bottom = Math.min(min[2], limits.bottom);

            mesh = new THREE.Mesh(geom, shader);
            mesh.position.y = -min[1];
            mesh.doubleSided = true;
            mesh.updateMatrix();
            obj = new THREE.Object3D();
            obj.rotation.z = hemi == "left" ? Math.PI / 2. : -Math.PI / 2.;
            obj.add(mesh);
            scene.add(obj);
        }

        var aspect = (limits.right - limits.left) / (limits.top - limits.bottom);

        
        camera.left = -limits.left;
        camera.right = limits.right;
        camera.top = limits.top;
        camera.bottom = limits.bottom;

        for (var frame = 0; frame < this.textures.length; frame++) {

        }

        camera.updateProjectionMatrix();
        viewer.renderer.render(scene, camera);
        return {scene:scene, camera:camera};
    };

    return module;
}(dataset || {}));