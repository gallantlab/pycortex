var sliceplane = (function(module) {

    module.Plane = function(viewer, dir) {
        this.viewer = viewer;
        this.dir = dir;
        this._visible = false;
        this._angle = 0;
    }

    module.Plane.prototype.setData = function(active) {
	if (this.viewer.active.vertex) {
	    return;
	}
        this.scene = viewer.views[0].scene;

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
        this.uniforms.data = active.uniforms.data;
        this.uniforms.dataAlpha = active.uniforms.dataAlpha;
        this.uniforms.mosaic = active.uniforms.mosaic;
        this.uniforms.dshape = active.uniforms.dshape;
        this.uniforms.volxfm = active.uniforms.volxfm;
        this.uniforms.colormap = active.cmap[0];
        //this.uniforms.colormap.texture = viewer.uniforms.colormap.texture;
        this.uniforms.vmin.value = active.vmin[0].value;
        this.uniforms.vmax.value = active.vmax[0].value;

        if (this.mesh !== undefined) {
            this.scene.remove(this.object);
            this.geometry.dispose();
            this.shader.dispose();
        }

        this.object = new THREE.Group();

        this.geometry = new THREE.PlaneGeometry();
        this.geometry.dynamic = true;
        this._makeShader(active);

        this.mesh = new THREE.Mesh(this.geometry, this.shader);
        this.mesh.doubleSided = true;

        this.object.add(this.mesh);
        //this.scene.add(this.mesh);
        this.scene.add(this.object);
        //viewer.surfs[0].surf.pivots.left.back.add(this.mesh);

        // var wiremat = new THREE.LineBasicMaterial({color:0xffffff, linewidth:2});
        // this.wireframe = new THREE.Line(this.geometry, wiremat);
        // this.scene.add(this.wireframe);

        this.viewer.addEventListener("setData", this.update.bind(this));
        this.update();
    }

    module.Plane.prototype._makeShader = function(active) {
        var raw = active === undefined ? false : active.data[0].raw;
        var twod = active == undefined ? false : active.data.length > 1;
        // var shaders = Shaders.main("nearest", raw, twod, viewopts.voxlines, 0, false);
        // var shader_func = Shaders.surface_pixel;
        var opts = {sampler: "nearest",
                    raw: raw,
                    twod: twod,
                    voxline: false,
                    viewspace: true};
        var shaders = Shaders.main(opts);
        //var shadecode = Shaders.surface_pixel(opts);
        this.shader = new THREE.ShaderMaterial({ 
            vertexShader:shaders.vertex,
            fragmentShader:shaders.fragment,
            uniforms: this.uniforms,
            lights:true, 
            blending:THREE.CustomBlending,
            transparent:true,
            side:THREE.DoubleSide,
            // depthWrite:false
        });
    }
    module.Plane.prototype.update = function(slice) {
	if (this.viewer.active.vertex) {
	    return;
	}
        //var brain = this.viewer.active.data[0];
        var dshape = this.viewer.active.uniforms.dshape.value[0];
        var mosaic = this.viewer.active.uniforms.mosaic.value[0];
        var shape = [dshape.x, dshape.y, this.viewer.active.data[0].numslices];

        if (slice === undefined || slice.type !== undefined)
            slice = Math.round(shape[this.dir] / 2);

        this.shape = shape;
        this.slice = slice;

        var xfm = this.viewer.active.uniforms.volxfm.value[0];
        var imat = new THREE.Matrix4()
        imat.getInverse(xfm);

        if (this.dir == 0) {
            imat.multiplyVector3(this.geometry.vertices[0].set(slice,-0.5,-0.5));
            imat.multiplyVector3(this.geometry.vertices[1].set(slice,-0.5, shape[2]-0.5));
            imat.multiplyVector3(this.geometry.vertices[2].set(slice,shape[1]-0.5,-0.5));
            imat.multiplyVector3(this.geometry.vertices[3].set(slice,shape[1]-0.5,shape[2]-0.5));
        } else if (this.dir == 1) {
            imat.multiplyVector3(this.geometry.vertices[0].set(-0.5,slice,-0.5));
            imat.multiplyVector3(this.geometry.vertices[1].set(-0.5,slice,shape[2]-0.5));
            imat.multiplyVector3(this.geometry.vertices[2].set(shape[0]-0.5,slice,-0.5));
            imat.multiplyVector3(this.geometry.vertices[3].set(shape[0]-0.5,slice,shape[2]-0.5));
        } else if (this.dir == 2) {
            imat.multiplyVector3(this.geometry.vertices[0].set(-0.5,-0.5,slice));
            imat.multiplyVector3(this.geometry.vertices[1].set(-0.5,shape[1]-0.5,slice));
            imat.multiplyVector3(this.geometry.vertices[2].set(shape[0]-0.5,-0.5,slice));
            imat.multiplyVector3(this.geometry.vertices[3].set(shape[0]-0.5,shape[1]-0.5,slice));
        }

        this.geometry.computeBoundingSphere();
        var center = this.geometry.boundingSphere.center;
        this.object.position.copy(center);

        var trans = new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z);
        this.geometry.applyMatrix(trans);

        this.geometry.computeFaceNormals();
        this.geometry.computeVertexNormals();
        this.geometry.verticesNeedUpdate = true;
        this.geometry.normalsNeedUpdate = true;

        // this._makeShader(this.viewer.active);
        // this.mesh.material = this.shader;

        this.setVisible(this._visible);
        this.viewer.schedule();
    }
    module.Plane.prototype.next = function() {

        this.setSmoothSlice((Math.round(this.slice+1)).mod(this.shape[this.dir]) / this.shape[this.dir]);
    }
    module.Plane.prototype.prev = function() {
        this.setSmoothSlice((Math.round(this.slice-1)).mod(this.shape[this.dir]) / this.shape[this.dir]);
    }
    module.Plane.prototype.setSmoothSlice = function(slice_fraction) {
        if (slice_fraction === undefined)
            if (this.slice === undefined)
                return 0.5;
            else
                return this.slice / this.shape[this.dir];

        this.update(slice_fraction * this.shape[this.dir]);
    }
    module.Plane.prototype.setAngle = function(angle) {
        if (angle === undefined)
            return this._angle;

        var axis;
        var xfm = this.viewer.active.uniforms.volxfm.value[0];
        var imat = new THREE.Matrix4();
        imat.getInverse(xfm);
        var rot_imat = (new THREE.Matrix4).extractRotation(imat);

        if (this.dir == 0) {
            axis = (new THREE.Vector3(0,1,0)).applyMatrix4(rot_imat);
        } else if (this.dir == 1) {
            axis = (new THREE.Vector3(1,0,0)).applyMatrix4(rot_imat);
        } else if (this.dir == 2) {
            axis = (new THREE.Vector3(0,1,0)).applyMatrix4(rot_imat);
        }

        axis.normalize();

        this.mesh.rotation.set(0,0,0);
        this.mesh.rotateOnAxis(axis, angle / 180 * Math.PI);
    }
    module.Plane.prototype.setVisible = function(val) {
        if (val === undefined)
            return this._visible;

        this._visible = val;

        if (this.mesh !== undefined)
            this.mesh.visible = this._visible;
    }

    module.MIP = function(viewer) {
        
    }
    
    return module;
}(sliceplane || {}));
