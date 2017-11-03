function PickPosition(surf, posdata) {
    var left = surf.hemis.left, right = surf.hemis.right;
    var worker = new Worker("resources/js/facepick_worker.js");
    worker.addEventListener("message", function(e) {
        var dist = function (a, b) {
            return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]);
        }
        kdt = new kdTree([], dist, [0, 1, 2]);
        kdt.root = e.data.kdt;
        this[e.data.name] = kdt;
    }.bind(this));
    var lmsg = {pos:left.attributes.position.array, name:"lkdt"};
    var rmsg = {pos:right.attributes.position.array, name:"rkdt"};
    if (left.attributes.wm !== undefined) {
        lmsg.wm = left.attributes.wm.array;
        rmsg.wm = right.attributes.wm.array;
    }
    worker.postMessage(lmsg);
    worker.postMessage(rmsg);
    
    surf.addEventListener("mix", this.setMix.bind(this));

    this.xfm = new THREE.Matrix4();
    this.scene = new THREE.Scene();
    this.scene.children.push(surf.object);

    this.posdata = posdata;
    this.revIndex = {left:surf.hemis.left.reverseIndexMap, right:surf.hemis.right.reverseIndexMap};

    this.axes = [];

    var lbound = left.boundingBox;
    var rbound = right.boundingBox;
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
        thickmix:surf.uniforms.thickmix,
        surfmix:surf.uniforms.surfmix,
    };

    var shaders = Shaders.pick({morphs:surf.names.length, volume:surf.volume});
    this.shade_x = new THREE.ShaderMaterial({
        vertexShader: shaders.vertex,
        fragmentShader: shaders.fragment[0],
        uniforms: this.uniforms,
        attributes: shaders.attrs,
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.ZeroFactor,
    });
    this.shade_y = new THREE.ShaderMaterial({
        vertexShader: shaders.vertex,
        fragmentShader: shaders.fragment[1],
        uniforms: this.uniforms,
        attributes: shaders.attrs,
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.ZeroFactor,
    });
    this.shade_z = new THREE.ShaderMaterial({
        vertexShader: shaders.vertex,
        fragmentShader: shaders.fragment[2],
        uniforms: this.uniforms,
        attributes: shaders.attrs,
        blending: THREE.CustomBlending,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.ZeroFactor,
    });

    this.markers = {left:new THREE.Group(), right:new THREE.Group()};
}
PickPosition.prototype = {
    //Set the transform for any voxels
    apply: function(dataview) {
        if (dataview)
            this.xfm.set.apply(this.xfm, dataview.xfm);

        for (var i = 0; i < this.axes.length; i++) {
            var ax = this.axes[i];
            var vert = mriview.get_position(this.posdata[ax.hemi], 
                this.uniforms.surfmix.value, 
                this.uniforms.thickmix.value, 
                ax.idx);

            var inv = (new THREE.Matrix4()).getInverse(this.xfm);
            var vox = vert.base.clone().applyMatrix4(this.xfm);
            var voxpos = new THREE.Vector3(Math.round(vox.x), Math.round(vox.y), Math.round(vox.z));
	    ax.vox.applyMatrix(ax.lastxfm);
            ax.vox.applyMatrix(inv);
	    ax.lastxfm = this.xfm;
            ax.vox.position.copy(voxpos.applyMatrix4(inv).sub(vert.base));

            ax.vox.visible = this.uniforms.surfmix.value == 0;
            ax.group.position.copy(addFlatNorm(vert.pos, vert.norm, this.uniforms.surfmix.value));
            //console.log(marker.position);
            ax.group.scale.x = 1.000-this.uniforms.surfmix.value;
        }
    },
    resize: function(w, h) {
        if (this.x) {
            this.x.dispose();
            this.y.dispose();
            this.z.dispose();
        }
        
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

    draw: function(renderer, camera, debug) {
        var clearAlpha = renderer.getClearAlpha();
        var clearColor = renderer.getClearColor().clone();
        renderer.setClearColor(0x0, 0);

        this.markers.left.visible = false;
        this.markers.right.visible = false;

        this.scene.overrideMaterial = this.shade_x;
        if (debug)
            renderer.render(this.scene, camera);
        renderer.render(this.scene, camera, this.x);
        this.scene.overrideMaterial = this.shade_y;
        renderer.render(this.scene, camera, this.y);
        this.scene.overrideMaterial = this.shade_z;
        renderer.render(this.scene, camera, this.z);

        this.markers.left.visible = true;
        this.markers.right.visible = true;

        renderer.setClearColor(clearColor, clearAlpha);
    },

    _pick: function(x, y, gl) {
        var unpack = function(buf) {
            return buf[0]/(256*256*256*256) + buf[1]/(256*256*256) + buf[2]/(256*256) + buf[3]/256;
        }
        var world = function(x, y, z) {
            var coord = new THREE.Vector3(unpack(x), unpack(y), unpack(z));
            if (coord.x == 0 && coord.y == 0 && coord.z == 0)
                return;

            var range = this.bounds.max.clone().sub(this.bounds.min);
            return coord.multiply(range).add(this.bounds.min);
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

    get_vertex_index: function(renderer, scene, x, y, surf) {
        if (typeof this.x === "undefined") {
            return -1
        }

        this.draw(renderer, scene);
        let p = this._pick(x, y, renderer.context);
        if (p) {
            let point = mriview.get_position(
                this.posdata[p.hemi],
                this.uniforms.surfmix.value,
                this.uniforms.thickmix.value,
                p.ptidx,
            );
            let vertex = this.revIndex[p.hemi][p.ptidx];

            let vec_float = point.base.clone().applyMatrix4(this.xfm);
            let voxel = new THREE.Vector3(
                Math.round(vec_float.x), 
                Math.round(vec_float.y),
                Math.round(vec_float.z),
            )

            return {'vertex': vertex, 'voxel': voxel}
        } else {
            return -1
        }
    },

    pick: function(renderer, scene, x, y, keep) {
        // DISABLE MULTI-CURSORS to make linking to voxels easy
        this.draw(renderer, scene);
        var p = this._pick(x, y, renderer.context);
    	if (p) {
    	    //var vec = p.pos.clone().applyMatrix4(this.xfm); // this uses "actual" picked coordinates, but is not very accurate (and doesn't match display!)
    	    var vert = mriview.get_position(this.posdata[p.hemi],
    					    this.uniforms.surfmix.value,
    					    this.uniforms.thickmix.value,
    					    p.ptidx);
    	    var vec_float = vert.base.clone().applyMatrix4(this.xfm);
    	    var vec = new THREE.Vector3(Math.round(vec_float.x), 
    					Math.round(vec_float.y),
    					Math.round(vec_float.z));
                var idx = this.revIndex[p.hemi][p.ptidx];
                console.log("Picked vertex "+idx+" (orig "+p.ptidx+") in "+p.hemi+" hemisphere, voxel=["+vec.x+","+vec.y+","+vec.z+"]");
    	    this.process_pick(vec, p.hemi, p.ptidx, keep);
            return {'vertex': idx, 'voxel': vec}
    	}
    	else {
    	    this.process_nonpick();
            return -1
    	}
    },
    
    process_pick: function(vec, hemi, ptidx, keep) {
        this.addMarker(hemi, ptidx, keep);
        if (this.callback !== undefined)
            this.callback(vec, hemi, ptidx);
    },

    process_nonpick: function() {
        for (var i = 0; i < this.axes.length; i++) {
            this.markers[this.axes[i].hemi].remove(this.axes[i].group);
        }
        this.axes = [];
	
    	if (this.callback_nonpick !== undefined)
    	    this.callback_nonpick();
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

    undblpick: function() {
        if (this._undblpickanim)
            this.viewer.animate(this._undblpickanim);
    },

    setMix: function(mixevt) {
        var vert, ax;
        for (var i = 0, il = this.axes.length; i < il; i++) {
            ax = this.axes[i];
            vert = mriview.get_position(this.posdata[ax.hemi], mixevt.mix, this.uniforms.thickmix.value, ax.idx);
            ax.group.position.copy(vert.pos);
	        // ax.obj.position = addFlatNorm(vert.pos, vert.norm, mixevt.mix);
            // Rescale axes for flat view
            ax.group.scale.x = 1.000-mixevt.flat;
            ax.group.children[1].visible = mixevt.mix == 0;
        }
    },

    addMarker: function(hemi, ptidx, keep) {
        for (var i = 0; i < this.axes.length; i++) {
            if (keep === true) {
                this.axes[i].obj.material.color.setRGB(180,180,180);
            } else {
                this.markers[this.axes[i].hemi].remove(this.axes[i].group);
                delete this.axes[i];
            }
        }

        if (!keep)
            this.axes = [];
	
	    // Create axes
        var axes = makeAxes(500, 0xffffff);
        var group = new THREE.Group();
        group.add(axes.axes);
        group.add(axes.vox);
        this.markers[hemi].add(group);
        this.axes.push({idx:ptidx, group:group, vox:axes.vox, axes:axes.axes, hemi:hemi, lastxfm:(new THREE.Matrix4())});
        this.apply();
    }
}

function addFlatNorm(pos, norm, flatmix) {
    var posfrac = flatmix - 0.01;
    var normfrac = flatmix;
    //return pos.clone().multiplyScalar(posfrac).addSelf(norm.clone().multiplyScalar(normfrac));
    return pos.clone().add(norm.clone().multiplyScalar(normfrac));
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
    /*
    camera = viewer.camera;
    hNear = 2 * Math.tan(camera.fov / 2) * camera.near; // height
    wNear = hNear * camera.aspect; // width
    var ntr = new THREE.Vector3( wNear / 2, hNear / 2, -camera.near );
    */
    
    //lineGeo.vertices.push(v(0, 0, 0), v(0, 0, 10))
    var lineMat = new THREE.LineBasicMaterial({ color: color, linewidth: 2});
    var axes = new THREE.Line(lineGeo, lineMat, THREE.LinePieces);
    axes.name = "marker_axes"

    var voxlines = new THREE.Geometry();
    voxlines.vertices.push(
	v(0.5, 0.5, 0.5), v(0.5, 0.5, -0.5),
	v(0.5, 0.5, 0.5), v(0.5, -0.5, 0.5),
	v(0.5, 0.5, -0.5), v(0.5, -0.5, -0.5),
	v(0.5, -0.5, -0.5), v(0.5, -0.5, 0.5),

	v(0.5, 0.5, 0.5), v(-0.5, 0.5, 0.5),
	v(0.5, 0.5, -0.5), v(-0.5, 0.5, -0.5),
	v(0.5, -0.5, -0.5), v(-0.5, -0.5, -0.5),
	v(0.5, -0.5, 0.5), v(-0.5, -0.5, 0.5),

	v(-0.5, 0.5, 0.5), v(-0.5, 0.5, -0.5),
	v(-0.5, 0.5, 0.5), v(-0.5, -0.5, 0.5),
	v(-0.5, 0.5, -0.5), v(-0.5, -0.5, -0.5),
	v(-0.5, -0.5, -0.5), v(-0.5, -0.5, 0.5))

    /*var vox = new THREE.BoxGeometry(1, 1, 1);
    var voxMat = new THREE.MeshLambertMaterial({color:0xffffff, 
						transparent:false, 
					        wireframe:true,
					        wireframeLinewidth:2});
    var voxmesh = new THREE.Mesh(vox, voxMat);
    //var voxmesh = new THREE.Mesh(vox, lineMat);
    voxmesh.name = "marker_voxel";*/
    var voxmesh = new THREE.Line(voxlines, lineMat, THREE.LinePieces);
    return {axes:axes, vox:voxmesh};
}
