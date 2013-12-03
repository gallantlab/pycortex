var sliceplane = (function(module) {

    module.Plane = function(viewer, dir) {
        this.viewer = viewer;
        this.dir = dir;

        this.uniforms = THREE.UniformsUtils.merge( [
            THREE.UniformsLib[ "lights" ],
            {
                diffuse:    { type:'v3', value:new THREE.Vector3( .4,.4,.4 )},
                specular:   { type:'v3', value:new THREE.Vector3( 0, 0, 0)},
                emissive:   { type:'v3', value:new THREE.Vector3( .6,.6,.6 )},
                shininess:  { type:'f',  value:1},

                framemix:   { type:'f',  value:0},
                
                colormap:   { type:'t',  value:0, texture: null },
                vmin:       { type:'fv1',value:[0,0]},
                vmax:       { type:'fv1',value:[1,1]},

                voxlineColor:{type:'v3', value:new THREE.Vector3( 0,0,0 )},
                voxlineWidth:{type:'f', value:viewopts.voxline_width},
            }
        ]);
        this.uniforms.data = viewer.uniforms.data;
        this.uniforms.dataAlpha = viewer.uniforms.dataAlpha;
        this.uniforms.mosaic = viewer.uniforms.mosaic;
        this.uniforms.dshape = viewer.uniforms.dshape;
        this.uniforms.volxfm = viewer.uniforms.volxfm;
        this.uniforms.colormap = viewer.uniforms.colormap;
        //this.uniforms.colormap.texture = viewer.uniforms.colormap.texture;
        this.uniforms.vmin.value = viewer.uniforms.vmin.value;
        this.uniforms.vmax.value = viewer.uniforms.vmax.value;

        this.geometry = new THREE.PlaneGeometry();
        this.geometry.dynamic = true;
        this._makeShader(viewer.active);
        this.mesh = new THREE.Mesh(this.geometry, this.shader);
        this.mesh.doubleSided = true;
        viewer.scene.add(this.mesh);

        this.viewer.addEventListener("setData", this.update.bind(this));
        this.update();
    }
    module.Plane.prototype._makeShader = function(active) {
        var raw = active === undefined ? false : active.data[0].raw;
        var twod = active == undefined ? false : active.data.length > 1;
        var shaders = Shaders.main("nearest", raw, twod, viewopts.voxlines, 0, false);
        this.shader = new THREE.ShaderMaterial({ 
            vertexShader:shaders.vertex,
            fragmentShader:shaders.fragment,
            uniforms: this.uniforms,
            lights:true, 
            blending:THREE.CustomBlending,
            transparent:true,
        });
    }
    module.Plane.prototype.update = function(slice) {
        //var brain = this.viewer.active.data[0];
        var dshape = this.viewer.uniforms.dshape.value[0];
        var mosaic = this.viewer.uniforms.mosaic.value[0];
        var shape = [dshape.x, dshape.y, mosaic.x*mosaic.y];

        if (slice === undefined || slice.type !== undefined)
            slice = Math.round(shape[this.dir] / 2);

        this.shape = shape;
        this.slice = slice;

        var xfm = this.viewer.uniforms.volxfm.value[0];
        var imat = new THREE.Matrix4()
        imat.getInverse(xfm);

        if (this.dir == 0) {
            imat.multiplyVector3(this.geometry.vertices[0].set(slice,0,0));
            imat.multiplyVector3(this.geometry.vertices[1].set(slice,0, shape[2]));
            imat.multiplyVector3(this.geometry.vertices[2].set(slice,shape[1],0));
            imat.multiplyVector3(this.geometry.vertices[3].set(slice,shape[1],shape[2]));
        } else if (this.dir == 1) {
            imat.multiplyVector3(this.geometry.vertices[0].set(0,slice,0));
            imat.multiplyVector3(this.geometry.vertices[1].set(0,slice,shape[2]));
            imat.multiplyVector3(this.geometry.vertices[2].set(shape[0],slice,0));
            imat.multiplyVector3(this.geometry.vertices[3].set(shape[0],slice,shape[2]));
        } else if (this.dir == 2) {
            imat.multiplyVector3(this.geometry.vertices[0].set(0,0,slice));
            imat.multiplyVector3(this.geometry.vertices[1].set(0,shape[1],slice));
            imat.multiplyVector3(this.geometry.vertices[2].set(shape[0],0,slice));
            imat.multiplyVector3(this.geometry.vertices[3].set(shape[0],shape[1],slice));
        }

        this.geometry.computeFaceNormals();
        this.geometry.computeVertexNormals();
        this.geometry.verticesNeedUpdate = true;
        this.geometry.normalsNeedUpdate = true;

        this._makeShader(this.viewer.active);
        this.mesh.material = this.shader;
        this.viewer.schedule();
    }
    module.Plane.prototype.next = function() {
        this.update((this.slice+1).mod(this.shape[this.dir]));
    }
    module.Plane.prototype.prev = function() {
        this.update((this.slice-1).mod(this.shape[this.dir]));
    }

    module.MIP = function(viewer) {
        
    }
    
    return module;
}(sliceplane || {}));