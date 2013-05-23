var filtertypes = { 
    nearest: THREE.NearestFilter, 
    trilinear: THREE.LinearFilter, 
    nearlin: THREE.LinearFilter, 
    debug: THREE.NearestFilter 
};

var samplers = {
    trilinear: "trilinear",
    nearest: "nearest",
    nearlin: "nearest",
    debug: "debug",
}

function Dataset(json) {
    this.loaded = $.Deferred().done(function() { $("#dataload").hide(); });
    this.subject = json.subject;
    this.data = json.data;
    this.mosaic = json.mosaic;
    this.frames = json.data.length;
    this.xfm = json.xfm;
    this.min = json.vmin;
    this.max = json.vmax;
    this.lmin = json.lmin;
    this.lmax = json.lmax;
    this.cmap = json.cmap;
    this.stim = json.stim;
    this.rate = json.rate === undefined ? 1 : json.rate;
    this.raw = json.raw === undefined ? false : json.raw;
    this.delay = json.delay === undefined ? 0 : json.delay;
    this.filter = json.filter == undefined ? "nearest" : json.filter;
    this.textures = [];
    this.datatex = [];
    this.length = this.frames / this.rate;
    
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
            tex.minFilter = filtertypes[this.filter];
            tex.magFilter = filtertypes[this.filter];
            tex.needsUpdate = true;
            tex.flipY = false;
            this.shape = [((img.width-1) / this.mosaic[0])-1, ((img.height-1) / this.mosaic[1])-1];
            this.textures.push(tex);

            if (this.textures.length < this.frames) {
                this.loaded.notify(this.textures.length);
                loadtex(this.textures.length);
            } else {
                this.loaded.resolve();
            }
        }.bind(this));
        img.src = this.data[this.textures.length];
    }.bind(this);
    
    var loadnpy = function(idx) {
        
    }

    loadtex(0);
    $("#dataload").show();
}
Dataset.fromJSON = function(json) {
    return new Dataset(json);
}

Dataset.prototype = {
    init: function(uniforms, meshes, dim, ndim) {
        if (dim == 0) {
            var sampler = samplers[this.filter];
            var shaders = Shaders.main(sampler, this.raw, ndim > 1, viewopts.voxlines, uniforms.nsamples.value);
            this.shader = new THREE.ShaderMaterial({ 
                vertexShader:shaders.vertex,
                fragmentShader:shaders.fragment,
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
                shaders = Shaders.main(sampler, this.raw, ndim > 1, viewopts.voxlines, 1);
                this.fastshader = new THREE.ShaderMaterial({
                    vertexShader:shaders.vertex,
                    fragmentShader:shaders.fragment,
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
        }

        var xfm = uniforms.volxfm.value[dim];
        xfm.set.apply(xfm, this.xfm);
        uniforms.mosaic.value[dim].set(this.mosaic[0], this.mosaic[1]);
        uniforms.dshape.value[dim].set(this.shape[0], this.shape[1]);
    },
    set: function(uniforms, dim, time) {
        var frame = ((time - this.delay) / this.rate).mod(this.frames);
        var fframe = Math.floor(frame);
        uniforms.framemix.value = frame - fframe;
        if (uniforms.data.texture[dim*2] !== this.textures[fframe]) {
            uniforms.data.texture[dim*2] = this.textures[fframe];
            if (this.frames > 1) {
                uniforms.data.texture[dim*2+1] = this.textures[(fframe+1).mod(this.frames)];
            } else {
                uniforms.data.texture[dim*2+1] = mriview.blanktex;
            }
        }
    },
    setFilter: function(interp) {
        this.filter = interp;
        for (var i = 0, il = this.textures.length; i < il; i++) {
            this.textures[i].minFilter = filtertypes[interp];
            this.textures[i].magFilter = filtertypes[interp];
            this.textures[i].needsUpdate = true;
        }
    },


    render: function(viewer, res) {
        var scene = new THREE.Scene(), 
            camera = new THREE.OrthographicCamera(-100, 100, -100, 100, -100, 100),
            shaders = Shaders.data(),
            shader = new THREE.ShaderMaterial( {
                vertexShader: shaders.vertex,
                fragmentShader: shaders.fragment,
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
    }
}
