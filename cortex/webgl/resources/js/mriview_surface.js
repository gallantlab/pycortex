var mriview = (function(module) {
    var flatscale = 0.25;

    module.Surface = function(ctminfo) {
        this.loaded = $.Deferred();

        this.meshes = [];
        this.pivots = {};
        this.hemis = {};
        this.volume = 0;
        this._pivot = 0;
        //this.rotation = [ 0, 0, 200 ]; //azimuth, altitude, radius

        this.object = new THREE.Object3D();
        this.uniforms = THREE.UniformsUtils.merge( [
            THREE.UniformsLib[ "lights" ],
            {
                diffuse:    { type:'v3', value:new THREE.Vector3( .8,.8,.8 )},
                specular:   { type:'v3', value:new THREE.Vector3( 1,1,1 )},
                emissive:   { type:'v3', value:new THREE.Vector3( .2,.2,.2 )},
                shininess:  { type:'f',  value:250},
                specularStrength:{ type:'f',  value:1},

                thickmix:   { type:'f',  value:0.5},
                surfmix:    { type:'f',  value:0},

                //hatch:      { type:'t',  value:0, texture: module.makeHatch() },
                //hatchrep:   { type:'v2', value:new THREE.Vector2(108, 40) },
                hatchAlpha: { type:'f', value:1.},
                hatchColor: { type:'v3', value:new THREE.Vector3( 0,0,0 )},
                overlay:    { type:'t', value:this.blanktex },
                curvAlpha:  { type:'f', value:1.},
                curvScale:  { type:'f', value:.5},
                curvLim:    { type:'f', value:.2},

                screen:     { type:'t', value:this.volumebuf},
                screen_size:{ type:'v2', value:new THREE.Vector2(100, 100)},
            }
        ]);
    
        this._update = {};

        var loader = new THREE.CTMLoader(false);
        loader.loadParts( ctminfo, function( geometries, materials, json ) {
            geometries[0].computeBoundingBox();
            geometries[1].computeBoundingBox();

            this.flatlims = json.flatlims;
            this.flatoff = [
                Math.max(
                    Math.abs(geometries[0].boundingBox.min.x),
                    Math.abs(geometries[1].boundingBox.max.x)
                ) / 3, Math.min(
                    geometries[0].boundingBox.min.y, 
                    geometries[1].boundingBox.min.y
                )];

            this.names = json.names;
            var gb0 = geometries[0].boundingBox, gb1 = geometries[1].boundingBox;
            var center = [
                ((gb1.max.x - gb0.min.x) / 2) + gb0.min.x,
                (Math.max(gb0.max.y, gb1.max.y) - Math.min(gb0.min.y, gb1.min.y)) / 2 + Math.min(gb0.min.y, gb1.min.y),
                (Math.max(gb0.max.z, gb1.max.z) - Math.min(gb0.min.z, gb1.min.z)) / 2 + Math.min(gb0.min.z, gb1.min.z),
            ];
            this.center = center;
            this.object.position.set(0, -center[1], -center[2]);

            var names = {left:0, right:1};
            var posdata = {left:[], right:[]};
            for (var name in names) {
                var hemi = geometries[names[name]];
                posdata[name].push(hemi.attributes.position);

                //Put attributes in the correct locations for the shader
                if (hemi.attributes['wm'] !== undefined) {
                    this.volume = 1;
                    hemi.attributes['position2'] = hemi.attributes['wm'];
                    hemi.attributes['position2'].stride = 4;
                    hemi.attributes['position2'].itemSize = 3;
                    hemi.attributes['normal2'] = module.computeNormal(hemi.attributes['wm'], hemi.attributes.index, hemi.offsets);
                    delete hemi.attributes['wm'];
                }
                for (var i = 0; i < json.names.length; i++ ) {
                    hemi.attributes['mixSurfs'+i] = hemi.attributes[json.names[i]];
                    hemi.attributes['mixSurfs'+i].stride = 4;
                    hemi.attributes['mixSurfs'+i].itemSize = 3;
                    hemi.attributes['mixNorms'+i] = module.computeNormal(hemi.attributes[json.names[i]], hemi.attributes.index, hemi.offsets);
                    posdata[name].push(hemi.attributes['mixSurfs'+i]);
                    delete hemi.attributes[json.names[i]];
                }

                if (this.flatlims !== undefined) {
                    var flats = this._makeFlat(hemi.attributes.uv.array, json.flatlims, names[name]);
                    hemi.attributes['mixSurfs'+json.names.length] = {itemSize:3, array:flats.pos};
                    hemi.attributes['mixNorms'+json.names.length] = {itemSize:3, array:flats.norms};
                    posdata[name].push(hemi.attributes['mixSurfs'+json.names.length]);
                }

                hemi.dynamic = true;
                var pivots = {back:new THREE.Object3D(), front:new THREE.Object3D()};
                pivots.front.add(pivots.back);
                pivots.back.position.y = hemi.boundingBox.min.y - hemi.boundingBox.max.y;
                pivots.front.position.y = hemi.boundingBox.max.y - hemi.boundingBox.min.y + this.flatoff[1];
                this.pivots[name] = pivots;
                this.hemis[name] = hemi;
                this.object.add(pivots.front);
            }
            this.setHalo(1);

            //Add anatomical and flat names
            this.names.unshift("anatomicals");
            if (this.flatlims !== undefined) {
                this.names.push("flat");
            }
            this.loaded.resolve();

        }.bind(this), {useWorker:true});
    };
    THREE.EventDispatcher.prototype.apply(module.Surface.prototype);
    module.Surface.prototype.resize = function(width, height) {
        this.volumebuf = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format:THREE.RGBAFormat,
            stencilBuffer:false,
        });
        this.uniforms.screen.value = this.volumebuf;
        this.uniforms.screen_size.value.set(width, height);
    };
    module.Surface.prototype.init = function(dataview) {
        this.preshaders = [];
        this.shaders = [];
        this.loaded.done(function() {
            if (this._update.func !== undefined)
                this._update.data.removeEventListener("update", this._update.func);
            this._update.func = function() {
                this.init(dataview);
            }.bind(this);
            this._update.data = dataview;
            dataview.addEventListener("update", this._update.func);

            if (this.meshes.length > 1 && !this.points) {
                for (var i = 0; i < this.meshes.length; i++) {
                    var shaders = dataview.getShader(Shaders.surface, this.uniforms, {
                        morphs:this.names.length, volume:1, rois: false, halo: true });
                    for (var j = 0; j < shaders.length; j++) {
                        shaders[j].transparent = i != 0;
                        shaders[j].depthTest = true;
                        shaders[j].depthWrite = i == 0;
                        shaders[j].uniforms.thickmix = {type:'f', value: 1 - i / (this.meshes.length-1)};
                        shaders[j].blending = THREE.CustomBlending;
                        shaders[j].blendSrc = THREE.OneFactor;
                        shaders[j].blendDst = THREE.OneFactor;
                    }
                    this.preshaders.push(shaders);
                }
                var shaders = dataview.getShader(Shaders.surface, this.uniforms, {
                    morphs:this.names.length, volume:1, rois:false, halo:false });
                for (var j = 0; j < shaders.length; j++) {
                    shaders[j].uniforms.thickmix = {type:'f', value:1};
                    shaders[j].uniforms.dataAlpha = {type:'f', value:0};
                }
                this.shaders.push(shaders);

                this.quadshade = dataview.getShader(Shaders.cmap_quad, this.uniforms);
                this.quadshade.transparent = true;
                this.quadshade.blending = THREE.CustomBlending
                this.quadshade.blendSrc = THREE.OneFactor
                this.quadshade.blendDst = THREE.OneMinusSrcAlphaFactor
                this.quadshade.depthWrite = false;
                this.prerender = this._prerender.bind(this);
            } else {
                var shaders = dataview.getShader(Shaders.surface, this.uniforms, {
                            morphs:this.names.length, 
                            volume:this.volume, 
                            rois:  false,
                            halo: false,
                        });
                this.shaders.push(shaders);
                if (this.prerender !== undefined)
                    delete this.prerender;
            }
            if (this.points !== undefined) {
                for (var j = 0; j < this.shaders[0].length; j++) {
                    this.shaders[0][j].uniforms.dataAlpha = {type:'f', value:0};
                    this.shaders[0][j].uniforms.thickmix = {type:'f', value:1};
                }
                this.uniforms.curvAlpha.value = 1;
                var shaders = dataview.getShader(Shaders.halopoint, this.uniforms, {
                        morphs:this.names.length, volume:1});
                var inverse = {type:'m4', value:null};
                for (var j = 0; j < shaders.length; j++) {
                    shaders[j].transparent = true;
                    shaders[j].uniforms.inverse = inverse;
                    shaders[j].blending = THREE.CustomBlending;
                    shaders[j].blendSrc = THREE.OneFactor;
                    shaders[j].blendDst = THREE.OneMinusSrcAlphaFactor;
                }
                this.shaders.push(shaders);
            }
        }.bind(this));
    };
    module.Surface.prototype.prerender = function(idx, renderer, scene, camera) {
        this.dispatchEvent({type:"prerender", idx:idx, renderer:renderer, scene:scene, camera:camera});
    }
    var oldcolor, black = new THREE.Color(0,0,0);
    module.Surface.prototype._prerender_halosurf = function(idx, renderer, scene, camera) {
        camera.add(scene.fsquad);
        scene.fsquad.material = this.quadshade[idx];
        scene.fsquad.visible = false;
        for (var i = 0; i < this.meshes.length; i++) {
            this.meshes[i].left.material = this.preshaders[i][idx];
            this.meshes[i].right.material = this.preshaders[i][idx];
            this.meshes[i].left.visible = true;
            this.meshes[i].right.visible = true;
        }
        oldcolor = renderer.getClearColor()
        renderer.setClearColor(black, 0);
        renderer.render(scene, camera, this.volumebuf);
        renderer.setClearColor(oldcolor, 1);
        for (var i = 1; i < this.meshes.length; i++) {
            this.meshes[i].left.visible = false;
            this.meshes[i].right.visible = false;
        }
        scene.fsquad.visible = true;
    };

    var __vec = new THREE.Vector3();
    var __mat = new THREE.Matrix4();
    module.Surface.prototype._prerender_halosprite = function(idx, renderer, scene, camera) {
        var sortArray, pts, sprites, ptvec, dist;
        for (var name in {left:null, right:null}) {
            sprites = this.sprites[name];
            pts = sprites.geometry.attributes.position.array;

            __mat.copy(camera.projectionMatrix);
            __mat.multiply(sprites.matrixWorld);
            if (sprites.geometry.__sortArray === undefined) {
                sprites.geometry.__sortArray = [];
                for (var i = 0, il = sprites.geometry.ptvecs.length; i < il; i++) {
                    sprites.geometry.__sortArray.push([]);
                }
            }
            sortArray = sprites.geometry.__sortArray;

            for (var i = 0, il = sprites.geometry.ptvecs.length; i < il; i++) {
                __vec.copy(sprites.geometry.ptvecs[i][0]);
                __vec.applyProjection(__mat);
                sortArray[i] = [pt.z, i];
            }
            sortArray.sort(function ( a, b ) {return b[ 0 ] - a[ 0 ]});
            for (var i = 0, il = sortArray.length; i < il; i++) {
                ptvec = sprites.geometry.ptvecs[sortArray[i][1]];
                __vec.copy(ptvec[1]);
                __vec.applyProject(__mat);
                pts[i*3*4 + 0] = ptvec[0].x - __vec;
            }
        }
    };

    module.Surface.prototype.apply = function(idx) {
        for (var i = 0; i < this.shaders.length; i++) {
            this.meshes[i].left.material = this.shaders[i][idx];
            this.meshes[i].right.material = this.shaders[i][idx];
        }
    };

    module.Surface.prototype.setHalo = function(layers) {
        var lmesh, rmesh;
        layers = Math.max(layers, 1);
        for (var i = 0; i < this.meshes.length; i++) {
            this.pivots.left.back.remove(this.meshes[i].left);
            this.pivots.right.back.remove(this.meshes[i].right);
        }
        this.meshes = [];
        for (var i = 0; i < layers; i++) {
            lmesh = this._makeMesh(this.hemis.left);
            rmesh = this._makeMesh(this.hemis.right);
            this.meshes.push({left:lmesh, right:rmesh});
            this.pivots.left.back.add(lmesh);
            this.pivots.right.back.add(rmesh);
        }
        if (this._update.func)
            this._update.func();
    };

    module.Surface.prototype.setPointHalo = function() {
        var lparticles = new THREE.ParticleSystem(this.hemis.left, null);
        var rparticles = new THREE.ParticleSystem(this.hemis.right, null);
        lparticles.sortParticles = true;
        rparticles.sortParticles = true;
        lparticles.position.y = -this.flatoff[1];
        rparticles.position.y = -this.flatoff[1];
        this.points = true;
        this.meshes.push({left:lparticles, right:rparticles});
        this.pivots.left.back.add(lparticles);
        this.pivots.right.back.add(rparticles);

        if (this._update.func)
            this._update.func();
    };
    function gen_sprites(hemi) {
        var npts = hemi.attributes.position.numItems / 3;
        var pia = hemi.attributes.position.array;
        var wm = hemi.attributes.position2.array;

        var geom = new THREE.BufferGeometry();
        geom.dynamic = true;
        geom.addAttribute("position", Float32Array, npts * 3 * 4, 3);
        geom.addAttribute("surfnorm", Float32Array, npts * 3, 3);
        geom.addAttribute("spriteidx", Float32Array, npts, 1);
        geom.addAttribute("index", Uint16Array, npts*2*3, 3);
        geom.ptvecs = [];
        var size = geom.attributes.size.array;
        var idx = geom.attributes.index.array;

        var vec1 = new THREE.Vector3(),
            vec2 = new THREE.Vector3();
        for (var i = 0; i < npts; i++) {
            vec1.set(pia[i*3], pia[i*3+1], pia[i*3+2]);
            vec2.set( wm[i*4],  wm[i*4+1],  wm[i*4+2]);
            geom.ptvecs.push([vec2.clone(), vec1.sub(vec2).clone()]);
        }

        for (var i = 0, il = Math.ceil(npts / 32768); i < il; i++) {
            for (var j = 0; j < 32768; j++) {
                idx[i*32768+j*6+0] = j*2+0;
                idx[i*32768+j*6+1] = j*2+1;
                idx[i*32768+j*6+2] = j*2+2;
                idx[i*32768+j*6+3] = j*2+1;
                idx[i*32768+j*6+4] = j*2+3;
                idx[i*32768+j*6+5] = j*2+2;
            }
            geom.offsets.push({start:i*32768, count:32768, index:i*32768});
        }
        if (this._update.func)
            this._update.func();

        this.addEventListener("prerender", this._prerender_halosprite.bind(this));
        return geom;
    }
    module.Surface.prototype.setSpriteHalo = function() {
        var lgeom = gen_sprites(this.hemis.left);
        var rgeom = gen_sprites(this.hemis.right);
        var shader = new THREE.MeshPhongMaterial({color:"#FF0000"});
        var lmesh = new THREE.Mesh(lgeom, shader);
        var rmesh = new THREE.Mesh(rgeom, shader);
        this.sprites = {left:lgeom, right:rgeom};
        this.pivots.left.back.add(lmesh);
        this.pivots.right.back.add(rmesh);
    }

    module.Surface.prototype.rotate = function(x, y) {
        //Rotate the surface given the X and Y mouse displacement
        this.rotation[0] += x;
        this.rotation[1] += y;
        this.rotation[0] = this.rotation[0] % 360;
        this.rotation[1] = -90 < this.rotation[1] ? (this.rotation[1] < 90 ? this.rotation[1] : 90) : -90;
        this.object.rotation.set(0,0,0, 'ZYX');
        this.object.rotateX(this.rotation[1] * Math.PI / 180);
        this.object.rotateZ(this.rotation[0] * Math.PI / 180);
    }

    module.Surface.prototype.setMix = function(mix) {
        this.uniforms.surfmix.value = mix;
        var smix = mix * (this.names.length - 1);
        var factor = 1 - Math.abs(smix - (this.names.length-1));
        var clipped = 0 <= factor ? (factor <= 1 ? factor : 1) : 0;
        this.uniforms.specularStrength.value = 1-clipped;
        this.setPivot( 180 * clipped);
    };
    module.Surface.prototype.setPivot = function (val) {
        this._pivot = val;
        var names = {left:1, right:-1}
        if (val > 0) {
            for (var name in names) {
                this.pivots[name].front.rotation.z = 0;
                this.pivots[name].back.rotation.z = val*Math.PI/180 * names[name]/ 2;
            }
        } else {
            for (var name in names) {
                this.pivots[name].back.rotation.z = 0;
                this.pivots[name].front.rotation.z = val*Math.PI/180 * names[name] / 2;
            }
        }
    };
    module.Surface.prototype.setShift = function(val) {
        this.pivots.left.front.position.x = -val;
        this.pivots.right.front.position.x = val;
    };

    module.Surface.prototype._makeMesh = function(geom, shader) {
        //Creates the mesh object given the geometry and shader
        var mesh = new THREE.Mesh(geom, shader);
        mesh.position.y = -this.flatoff[1];
        return mesh;
    };

    module.Surface.prototype._makeFlat = function(uv, flatlims, right) {
        var fmin = flatlims[0], fmax = flatlims[1];
        var flat = new Float32Array(uv.length / 2 * 3);
        var norms = new Float32Array(uv.length / 2 * 3);
        for (var i = 0, il = uv.length / 2; i < il; i++) {
            if (right) {
                flat[i*3+1] = flatscale*uv[i*2] + this.flatoff[1];
                norms[i*3] = 1;
            } else {
                flat[i*3+1] = flatscale*-uv[i*2] + this.flatoff[1];
                norms[i*3] = -1;
            }
            flat[i*3+2] = flatscale*uv[i*2+1];
            uv[i*2]   = (uv[i*2]   + fmin[0]) / fmax[0];
            uv[i*2+1] = (uv[i*2+1] + fmin[1]) / fmax[1];
        }

        return {pos:flat, norms:norms};
    }

    return module;
}(mriview || {}));
