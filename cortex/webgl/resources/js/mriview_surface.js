var mriview = (function(module) {
    var flatscale = 0.25;

    module.Surface = function(viewer, ctminfo) {
        this.viewer = viewer;
        this.loaded = $.Deferred();
        this.object = new THREE.Object3D();
        this.meshes = {};
        this.pivot = {};

        this.uniforms = THREE.UniformsUtils.merge( [
            THREE.UniformsLib[ "lights" ],
            {
                diffuse:    { type:'v3', value:new THREE.Vector3( .8,.8,.8 )},
                specular:   { type:'v3', value:new THREE.Vector3( 1,1,1 )},
                emissive:   { type:'v3', value:new THREE.Vector3( .2,.2,.2 )},
                shininess:  { type:'f',  value:200},

                thickmix:   { type:'f',  value:0.5},
                mix:        { type:'f', value:0},

                //hatch:      { type:'t',  value:0, texture: module.makeHatch() },
                //hatchrep:   { type:'v2', value:new THREE.Vector2(108, 40) },
                hatchAlpha: { type:'f', value:1.},
                hatchColor: { type:'v3', value:new THREE.Vector3( 0,0,0 )},

                overlay:    { type:'t',  value:0, texture: this.blanktex },
                hide_mwall: { type:'i', value:0},

                curvAlpha:  { type:'f', value:1.},
                curvScale:  { type:'f', value:.5},
                curvLim:    { type:'f', value:.2},
            }
        ]);

        var loader = new THREE.CTMLoader(false);
        loader.loadParts( ctminfo, function( geometries, materials, header, json ) {
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
            this.center = [
                ((gb1.max.x - gb0.min.x) / 2) + gb0.min.x,
                (Math.max(gb0.max.y, gb1.max.y) - Math.min(gb0.min.y, gb1.min.y)) / 2 + Math.min(gb0.min.y, gb1.min.y),
                (Math.max(gb0.max.z, gb1.max.z) - Math.min(gb0.min.z, gb1.min.z)) / 2 + Math.min(gb0.min.z, gb1.min.z),
            ];

            var names = {left:0, right:1};
            var posdata = {left:[], right:[]};
            for (var name in names) {
                var hemi = geometries[names[name]];

                if (this.flatlims !== undefined) {
                    var flats = this._makeFlat(hemi.attributes.uv.array, json.flatlims, names[name]);
                    hemi.morphTargets.push({itemSize:3, stride:3, array:flats.pos});
                    hemi.morphNormals.push(flats.norms);
                    posdata[name].push(hemi.attributes.position);
                    for (var i = 0, il = hemi.morphTargets.length; i < il; i++) {
                        posdata[name].push(hemi.morphTargets[i]);
                    }
                }

                hemi.reorderVertices();
                hemi.dynamic = true;

                var meshpiv = this._makeMesh(hemi, null);
                this.meshes[name] = meshpiv.mesh;
                this.pivot[name] = meshpiv.pivots;
                this.object.add(meshpiv.pivots.front);
            }

            this.loaded.resolve();

        }.bind(this), true, true);
    };
    module.Surface.prototype.apply = function(dataview) {
        var shader = dataview.getShader(Shaders.surface, this.uniforms, {morphs:, volume:, rois:});
        this.meshes.left.material = shader;
        this.meshes.right.material = shader;
    };

    module.Surface.prototype.setMix = function(mix) {
        this.uniforms.mix.value = mix;
    };
    module.Surface.prototype.setPivot = function (val) {
        this._pivot = val;
        var names = {left:1, right:-1}
        if (val > 0) {
            for (var name in names) {
                this.pivot[name].front.rotation.z = 0;
                this.pivot[name].back.rotation.z = val*Math.PI/180 * names[name]/ 2;
            }
        } else {
            for (var name in names) {
                this.pivot[name].back.rotation.z = 0;
                this.pivot[name].front.rotation.z = val*Math.PI/180 * names[name] / 2;
            }
        }
    };
    module.Surface.prototype.setShift = function(val) {
        this.pivot.left.front.position.x = -val;
        this.pivot.right.front.position.x = val;
    };

    module.Surface.prototype._makeMesh = function(geom, shader) {
        //Creates the pivots and the mesh object given the geometry and shader
        var mesh = new THREE.Mesh(geom, shader);
        mesh.doubleSided = true;
        mesh.position.y = -this.flatoff[1];
        mesh.updateMatrix();
        var pivots = {back:new THREE.Object3D(), front:new THREE.Object3D()};
        pivots.back.add(mesh);
        pivots.front.add(pivots.back);
        pivots.back.position.y = geom.boundingBox.min.y - geom.boundingBox.max.y;
        pivots.front.position.y = geom.boundingBox.max.y - geom.boundingBox.min.y + this.flatoff[1];

        return {mesh:mesh, pivots:pivots};
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
