function FacePick(viewer, left, right) {
    this.viewer = viewer;

    var worker = new Worker("resources/js/facepick_worker.js");
    worker.addEventListener("message", function(e) {
        var dist = function (a, b) {
            return Math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]));
        }
        kdt = new kdTree([], dist, [0, 1, 2]);
        kdt.root = e.data.kdt;
        this[e.data.name] = kdt;
    }.bind(this));
    worker.postMessage({pos:left, name:"lkdt"});
    worker.postMessage({pos:right, name:"rkdt"});

    this.axes = [];

    var lbound = this.viewer.meshes.left.geometry.boundingBox;
    var rbound = this.viewer.meshes.right.geometry.boundingBox;
    var min = new THREE.Vector3(
        Math.min(lbound.min.x, rbound.min.x), 
        Math.min(lbound.min.y, rbound.min.y),
        Math.min(lbound.min.z, rbound.min.z));
    var max = new THREE.Vector3(
        Math.max(lbound.max.x, rbound.max.x), 
        Math.max(lbound.max.y, rbound.max.y),
        Math.max(lbound.max.z, rbound.max.z));

    this.bounds = {min:min, max:max};
    this.uniforms = {
        min:    { type:'v3', value:this.bounds.min},
        max:    { type:'v3', value:this.bounds.max},
        hide_mwall: this.viewer.uniforms.hide_mwall,
    };

    var shaders = Shaders.pick();
    this.shade_x = new THREE.ShaderMaterial({
        vertexShader: shaders.vertex,
        fragmentShader: shaders.fragment[0],
        uniforms: this.uniforms,
        attributes: { auxdat:true, },
        morphTargets:true, 
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.ZeroFactor,
    });
    this.shade_y = new THREE.ShaderMaterial({
        vertexShader: shaders.vertex,
        fragmentShader: shaders.fragment[1],
        uniforms: this.uniforms,
        attributes: { auxdat:true, },
        morphTargets:true,
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.ZeroFactor,
    });
    this.shade_z = new THREE.ShaderMaterial({
        vertexShader: shaders.vertex,
        fragmentShader: shaders.fragment[2],
        uniforms: this.uniforms,
        attributes: { auxdat:true, },
        morphTargets:true,
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.ZeroFactor,
    });
    this.resize($("#brain").width(), $("#brain").height());
}
FacePick.prototype = {
    resize: function(w, h) {
        this._valid = false;
        this.height = h;
        this.x = new THREE.WebGLRenderTarget(w, h, {
            minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
            stencilBuffer:false,
            generateMipmaps:false,
        });
        this.y = new THREE.WebGLRenderTarget(w, h, {
            minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
            stencilBuffer:false,
            generateMipmaps:false,
        });
        this.z = new THREE.WebGLRenderTarget(w, h, {
            minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
            stencilBuffer:false,
            generateMipmaps:false,
        });
    },

    draw: function(debug) {
        var renderer = this.viewer.renderer;
        var clearAlpha = renderer.getClearAlpha();
        var clearColor = renderer.getClearColor();
        renderer.setClearColorHex(0x0, 0);

        for (var i = 0; i < this.axes.length; i++) {
            this.axes[i].obj.visible = false;
        }

        this.viewer.scene.overrideMaterial = this.shade_x;
        if (debug)
            renderer.render(this.viewer.scene, this.viewer.camera);
        renderer.render(this.viewer.scene, this.viewer.camera, this.x);
        this.viewer.scene.overrideMaterial = this.shade_y;
        renderer.render(this.viewer.scene, this.viewer.camera, this.y);
        this.viewer.scene.overrideMaterial = this.shade_z;
        renderer.render(this.viewer.scene, this.viewer.camera, this.z);
        this.viewer.scene.overrideMaterial = null;

        for (var i = 0; i < this.axes.length; i++) {
            this.axes[i].obj.visible = true;
        }

        renderer.setClearColor(clearColor, clearAlpha);
        this._valid = true;
    },

    _pick: function(x, y) {
        if (!this._valid)
            this.draw();
        var gl = this.viewer.renderer.context;
        var unpack = function(buf) {
            return buf[0]/(256*256*256*256) + buf[1]/(256*256*256) + buf[2]/(256*256) + buf[3]/256;
        }
        var world = function(x, y, z) {
            var coord = new THREE.Vector3(unpack(x), unpack(y), unpack(z));
            if (coord.x == 0 && coord.y == 0 && coord.z == 0)
                return;

            var range = this.bounds.max.clone().subSelf(this.bounds.min);
            return coord.multiplySelf(range).addSelf(this.bounds.min);
        }.bind(this);
        var xbuf = new Uint8Array(4);
        var ybuf = new Uint8Array(4);
        var zbuf = new Uint8Array(4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.x.__webglFramebuffer);
        gl.readPixels(x, this.height - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, xbuf);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.y.__webglFramebuffer);
        gl.readPixels(x, this.height - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, ybuf);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.z.__webglFramebuffer);
        gl.readPixels(x, this.height - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, zbuf);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        var pos = world(xbuf, ybuf, zbuf);
        if (pos && this.lkdt && this.rkdt) {
            var left = this.lkdt.nearest([pos.x, pos.y, pos.z], 1)[0];
            var right = this.rkdt.nearest([pos.x, pos.y, pos.z], 1)[0];
            if (left[1] < right[1])
                return {hemi:"left", ptidx:left[0][3], dist:left[1], pos:pos};
            else
                return {hemi:"right", ptidx:right[0][3], dist:right[1], pos:pos};
        }
    },

    pick: function(x, y, keep) {
        // DISABLE MULTI-CURSORS to make linking to voxels easy
        var p = this._pick(x, y);
        if (p) {
            var vec = this.viewer.uniforms.volxfm.value[0].multiplyVector3(p.pos.clone());
            console.log("Picked vertex "+p.ptidx+" in "+p.hemi+" hemisphere, distance="+p.dist+", voxel=["+vec.x+","+vec.y+","+vec.z+"]");
            this.addMarker(p.hemi, p.ptidx, keep);
            this.viewer.figure.notify("pick", this, [vec]);
            if (this.callback !== undefined)
                this.callback(vec, p.hemi, p.ptidx);
        } else {
            for (var i = 0; i < this.axes.length; i++) {
                this.axes[i].obj.parent.remove(this.axes[i].obj);
            }
            this.axes = [];
            this.viewer.schedule();
        }
    },
    
    dblpick: function(x, y, keep) {
        var speed = 0.6;
        var p = this._pick(x, y);
        if (p) {
            var geom = this.viewer.meshes[p.hemi].geometry;
            var vpos = geom.attributes.position.array;
            var mbb = geom.boundingBox;
            var nx = (mbb.max.x + mbb.min.x)/2, ny = (mbb.max.y + mbb.min.y)/2, nz = (mbb.max.z + mbb.min.z)/2;
            var px = vpos[p.ptidx*3] - nx, py = vpos[p.ptidx*3+1] - ny, pz = vpos[p.ptidx*3+2] - nz;
            var newaz = (Math.atan2(-px, py)) * 180.0 / Math.PI;
            newaz = newaz > 0 ? newaz : 360 + newaz;
            //console.log("New az.: " + newaz);
            var newel = Math.acos(pz / Math.sqrt(Math.pow(px,2) + Math.pow(py,2) + Math.pow(pz,2))) * 180.0 / Math.PI;
            //console.log("New el.: " + newel);
            var states = ["mix", "radius", "target", "azimuth", "altitude", "pivot"];
            var viewer = this.viewer;
            this._undblpickanim = states.map(function(s) { return {idx:speed, state:s, value:viewer.getState(s)} });
            var front = newaz < 90 || newaz > 270;
            var newpivot;
            if (p.hemi == "left") {
                if (newaz > 180) {
                    if (front) {
                        newpivot = 40;
                    } else {
                        newpivot = -40;
                    }
                } else {
                    newpivot = 0;
                }
            } else {
                if (newaz < 180) {
                    if (front) {
                        newpivot = 40;
                    } else {
                        newpivot = -40;
                    }
                } else {
                    newpivot = 0;
                }
            }
            
            viewer.animate([{idx:speed, state:"mix", value:0}, 
                {idx:speed, state:"radius", value:200}, 
                {idx:speed, state:"target", value:[nx,ny,nz]}, 
                {idx:speed, state:"azimuth", value:newaz}, 
                {idx:speed, state:"altitude", value:newel},
                {idx:speed, state:"pivot", value:newpivot}
                   ]);
        }
    },

    _getPos: function(hemi, idx) {
        var geom = this.viewer.meshes[hemi].geometry;
        var influ = this.viewer.meshes[hemi].morphTargetInfluences;

        var sum = 0;
        var pos = new THREE.Vector3(), norm = new THREE.Vector3();
        for (var i = 0; i < influ.length; i++) {
            var stride = geom.morphTargets[i].stride;
            sum += influ[i];
            pos.addSelf((new THREE.Vector3(
                geom.morphTargets[i].array[stride*idx+0],
                geom.morphTargets[i].array[stride*idx+1],
                geom.morphTargets[i].array[stride*idx+2])).multiplyScalar(influ[i]));
            norm.addSelf((new THREE.Vector3(
                geom.morphNormals[i][idx*3+0],
                geom.morphNormals[i][idx*3+1],
                geom.morphNormals[i][idx*3+2])).multiplyScalar(influ[i]));
        }
        var fid = new THREE.Vector3(
            geom.attributes.position.array[idx*3],
            geom.attributes.position.array[idx*3+1],
            geom.attributes.position.array[idx*3+2]);
        if (geom.attributes.wm) {
            var mix = this.viewer.uniforms.thickmix.value;
            var stride = geom.attributes.wm.stride;
            fid.multiplyScalar(1-mix).addSelf(new THREE.Vector3(
                geom.attributes.wm.array[idx*stride],
                geom.attributes.wm.array[idx*stride+1],
                geom.attributes.wm.array[idx*stride+2]).multiplyScalar(mix)
            );
        }

        pos.addSelf(fid.clone().multiplyScalar(1-sum));
        norm.addSelf((new THREE.Vector3(
            geom.attributes.normal.array[idx*3],
            geom.attributes.normal.array[idx*3+1],
            geom.attributes.normal.array[idx*3+2])).multiplyScalar(1-sum));
        return {pos:pos, norm:norm, fid:fid, name:hemi};
    },

    undblpick: function() {
        if (this._undblpickanim)
            this.viewer.animate(this._undblpickanim);
    },

    setMix: function(mixevt) {
        var vert, ax;
        for (var i = 0, il = this.axes.length; i < il; i++) {
            ax = this.axes[i];
            vert = this._getPos(ax.hemi, ax.idx);
            //ax.obj.position = vert.pos;
	    ax.obj.position = addFlatNorm(vert.pos, vert.norm, mixevt.mix);
            // Rescale axes for flat view
            ax.obj.scale.x = 1.000-mixevt.flat;
            ax.obj.children[1].visible = mixevt.mix == 0;
        }
        this._valid = false;
    },

    addMarker: function(hemi, ptidx, keep) {
        var vert = this._getPos(hemi, ptidx);
        for (var i = 0; i < this.axes.length; i++) {
            if (keep === true) {
                this.axes[i].obj.material.color.setRGB(180,180,180);
            } else {
                this.axes[i].obj.parent.remove(this.axes[i].obj);
            }
        }
        if (keep !== true)
            this.axes = [];
	
	
	// Create axes
        var axes = makeAxes(500, 0xffffff);
	
	// Create voxel box
        var xfm = this.viewer.uniforms.volxfm.value[0];
        var inv = new THREE.Matrix4().getInverse(xfm);
        var vox = xfm.multiplyVector3(vert.fid.clone());
        var voxpos = new THREE.Vector3(Math.round(vox.x), Math.round(vox.y), Math.round(vox.z));
        axes.vox.applyMatrix(inv)
        axes.vox.position = inv.multiplyVector3(voxpos).subSelf(vert.fid);
        var mat = axes.vox.matrix.elements;
        for (var i = 0; i < 16; i++) {
            if (Math.abs(mat[i]) < 1e-8)
                mat[i] = 0;
        }
	axes.vox.visible = this.viewer.getState("mix") == 0;

        var marker = new THREE.Object3D();
        marker.add(axes.axes);
        marker.add(axes.vox);
        marker.position = addFlatNorm(vert.pos, vert.norm, this.viewer.getState("mix"));
        marker.scale.x = 1.000-this.viewer.flatmix;

        this.axes.push({idx:ptidx, obj:marker, hemi:hemi});
        this.viewer.meshes[vert.name].add(marker);
	//this.setMix({mix:this.viewer.getState("mix"), flat:this.viewer.getState("flat")});
        this.viewer.schedule();
    }
}

function addFlatNorm(pos, norm, flatmix) {
    var posfrac = flatmix - 0.01;
    var normfrac = flatmix;
    //return pos.clone().multiplyScalar(posfrac).addSelf(norm.clone().multiplyScalar(normfrac));
    return pos.clone().addSelf(norm.clone().multiplyScalar(normfrac));
}

function makeAxes(length, color) {
    function v(x,y,z){ 
        return new THREE.Vector3(x,y,z); 
    }
    var lineGeo = new THREE.Geometry();
    lineGeo.vertices.push(
        v(-length, 0, 0), v(length, 0, 0),
        v(0, -length, 0), v(0, length, 0),
        v(0, 0, -length), v(0, 0, length)
    );
    var lineMat = new THREE.LineBasicMaterial({ color: color, linewidth: 2});
    var axes = new THREE.Line(lineGeo, lineMat);
    axes.type = THREE.Lines;

    var vox = new THREE.CubeGeometry(1, 1, 1);
    var voxMat = new THREE.MeshLambertMaterial({color:0xff0000, transparent:true, opacity:0.75});
    var voxmesh = new THREE.Mesh(vox, voxMat);
    return {axes:axes, vox:voxmesh};
}