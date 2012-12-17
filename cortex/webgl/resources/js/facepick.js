var flatVertShade = [
    "attribute vec3 idx;",
    "attribute vec4 auxdat;",
    "varying vec3 vColor;",
    THREE.ShaderChunk[ "morphtarget_pars_vertex" ],
    "void main() {",
        "vColor = (2.*idx+1.) / 512.;",
        THREE.ShaderChunk[ "morphtarget_vertex" ],
        THREE.ShaderChunk[ "default_vertex" ],
    "}",
].join("\n");

var flatFragShade = [
    "varying vec3 vColor;",
    "void main() {",
        "gl_FragColor = vec4(vColor, 1.);",
    "}"
].join("\n");

function FacePick(viewer, callback) {
    this.viewer = viewer;
    this.pivot = {};
    this.meshes = {};
    this.idxrange = {};
    if (typeof(callback) == "function") {
        this.callback = callback;
    } else {
        this.callback = function(voxidx) {
            console.log(voxidx);
        }
    }

    this.scene = new THREE.Scene();
    this.shader = new THREE.ShaderMaterial({
        vertexShader: flatVertShade,
        fragmentShader: flatFragShade,
        attributes: { idx: true, auxdat:true, },
        morphTargets:true
    });

    this.camera = new THREE.PerspectiveCamera( 60, $("#brain").width() / $("#brain").height(), 0.1, 5000 )
    this.camera.position = this.viewer.camera.position;
    this.camera.up.set(0,0,1);
    this.scene.add(this.camera);
    this.renderbuf = new THREE.WebGLRenderTarget($("#brain").width(), $("#brain").height(), {
        minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter
    })
    this.height = $("#brain").height();
    this.axes = [];

    var nface, nfaces = 0;
    for (var name in {left:"", right:""}) {
        var hemi = this.viewer.meshes[name];
        var morphs = [];
        for (var i = 0, il = hemi.geometry.morphTargets.length; i < il; i++) {
            morphs.push(hemi.geometry.morphTargets[i].array);
        }
        nface = hemi.geometry.attributes.index.array.length / 3;
        this.idxrange[name] = [nfaces, nfaces + nface];

        var worker = new Worker("resources/js/facepick_worker.js");
        worker.onmessage = this.handleworker.bind(this);
        worker.postMessage({
            func:    "genFlatGeom",
            name:    name,
            ppts:    hemi.geometry.attributes.position.array,
            ppolys:  hemi.geometry.attributes.index.array,
            aux:     hemi.geometry.attributes.auxdat.array,
            morphs:  morphs,
            offsets: hemi.geometry.offsets,
            reordered: hemi.geometry.indexMap !== undefined,
            faceoff: nfaces,
        });
        nfaces += nface;
    }
    
    this._valid = false;
}
FacePick.prototype = {
    resize: function(w, h) {
        this._valid = false;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.height = h;
        delete this.renderbuf;
        this.renderbuf = new THREE.WebGLRenderTarget(w, h, {
            minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter
        })
    },

    draw: function(debug) {
        console.log("redrawing facepick");
        this.camera.lookAt(this.viewer.controls.target);
        for (var name in this.meshes) {
            this.pivot[name].front.rotation.z = this.viewer.pivot[name].front.rotation.z;
            this.pivot[name].back.rotation.z = this.viewer.pivot[name].back.rotation.z;
            this.meshes[name].morphTargetInfluences = this.viewer.meshes[name].morphTargetInfluences;
        }
        if (debug === true)
            this.viewer.renderer.render(this.scene, this.camera);
        
        this.viewer.renderer.render(this.scene, this.camera, this.renderbuf);
        this._valid = true;
    },

    handleworker: function(event) {
        var msg = event.data;
        var i, il, geom = new THREE.BufferGeometry();
        geom.attributes.position = {itemSize:3, array:msg.pts, stride:3};
        geom.attributes.index = {itemSize:1, array:msg.polys, stride:1};
        geom.attributes.idx = {itemSize:3, array:msg.color, stride:3};
        geom.morphTargets = [];
        for (i = 0, il = msg.morphs.length; i < il; i++) {
            geom.morphTargets.push({itemSize:3, array:msg.morphs[i], stride:3});
        }
        geom.offsets = []
        for (i = 0, il = msg.polys.length; i < il; i += 65535) {
            geom.offsets.push({start:i, index:i, count:Math.min(65535, il - i)});
        }
        geom.computeBoundingBox();
        var meshpiv = this.viewer._makeMesh(geom, this.shader);
        meshpiv.mesh.doubleSided = false;
        this.meshes[msg.name] = meshpiv.mesh;
        this.pivot[msg.name] = meshpiv.pivots;
        this.scene.add(meshpiv.pivots.front);
    }, 

    pick: function(x, y, keep) {
        if (!this._valid)
            this.draw();
        var gl = this.viewer.renderer.context;
        var pix = new Uint8Array(4);
        var leftlen = this.viewer.meshes.left.geometry.attributes.position.array.length / 3;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.renderbuf.__webglFramebuffer);
        gl.readPixels(x, this.height - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pix);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        var faceidx = (pix[0] << 16) + (pix[1] << 8) + pix[2];
        if (faceidx == 0) {
            for (var i = 0; i < this.axes.length; i++)
                this.axes[i].obj.parent.remove(this.axes[i].obj);
            this.axes = [];
            return;
        }

        //adjust for clicking on black area
        faceidx -= 1;

        //Find which hemisphere it's in
        for (var hemi in this.idxrange) {
            var lims = this.idxrange[hemi];
            if (lims[0] <= faceidx && faceidx < lims[1]) {
                faceidx -= lims[0];
                var geom =  this.viewer.meshes[hemi].geometry;
                var polys = geom.attributes.index.array;
                var map = geom.attributes.datamap.array;

                //Find which offset
                for (var o = 0, ol = geom.offsets.length; o < ol; o++) {
                    var start = geom.offsets[o].start;
                    var index = geom.offsets[o].index;
                    var count = geom.offsets[o].count;

                    if (start <= faceidx*3 && faceidx*3 < (start+count)) {
                        //Pick only the first point of triangle
                        var ptidx = index + polys[faceidx*3];
                        var dataidx = map[ptidx*2] + (map[ptidx*2+1] << 8);
                        ptidx += hemi == "right" ? leftlen : 0;

                        this.addMarker(ptidx, keep);
                        this.callback(dataidx);
                        return;
                    }
                }
            }
        }

    },

    
    dblpick: function(x, y, keep) {
        //console.log("DOUBLE PICK");
        var speed = 0.6;

        if (!this._valid)
            this.draw();
        var gl = this.viewer.renderer.context;
        var pix = new Uint8Array(4);
        var leftlen = this.viewer.meshes.left.geometry.attributes.position.array.length / 3;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.renderbuf.__webglFramebuffer);
        gl.readPixels(x, this.height - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pix);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        var faceidx = (pix[0] << 16) + (pix[1] << 8) + pix[2];
        if (faceidx == 0) {
            for (var i = 0; i < this.axes.length; i++)
                this.axes[i].obj.parent.remove(this.axes[i].obj);
            this.axes = [];
            return;
        }

        //adjust for clicking on black area
        faceidx -= 1;

        //Find which hemisphere it's in
        for (var hemi in this.idxrange) {
            var lims = this.idxrange[hemi];
            if (lims[0] <= faceidx && faceidx < lims[1]) {
                faceidx -= lims[0];
                var geom =  this.viewer.meshes[hemi].geometry;
                var polys = geom.attributes.index.array;
                var map = geom.attributes.datamap.array;

                //Find which offset
                for (var o = 0, ol = geom.offsets.length; o < ol; o++) {
                    var start = geom.offsets[o].start;
                    var index = geom.offsets[o].index;
                    var count = geom.offsets[o].count;

                    if (start <= faceidx*3 && faceidx*3 < (start+count)) {
                        //Pick only the first point of triangle
                        var ptidx = index + polys[faceidx*3];
                        var dataidx = map[ptidx*2] + (map[ptidx*2+1] << 8);
                        //ptidx += hemi == "right" ? leftlen : 0;
                        var vpos = geom.attributes.position.array;
                        var mbb = viewer.meshes[hemi].geometry.boundingBox;
			var nx = (mbb.max.x + mbb.min.x)/2, ny = (mbb.max.y + mbb.min.y)/2, nz = (mbb.max.z + mbb.min.z)/2;
                        var px = vpos[ptidx*3] - nx, py = vpos[ptidx*3+1] - ny, pz = vpos[ptidx*3+2] - nz;
                        var PI = 3.14159;
                        var newaz = (Math.atan2(-px, py)) * 180.0 / PI;
                        newaz = newaz > 0 ? newaz : 360+newaz;
                        //console.log("New az.: " + newaz);
			var newel = Math.acos(pz / Math.sqrt(Math.pow(px,2) + Math.pow(py,2) + Math.pow(pz,2))) * 180.0 / PI;
                        //console.log("New el.: " + newel);
                        //console.log("X: "+vpos[ptidx*3]+", Y: "+vpos[ptidx*3+1]+", Z: "+vpos[ptidx*3+2]);
                        this._last_mix = viewer.getState("mix");
                        this._last_radius = viewer.getState("radius");
                        this._last_target = viewer.getState("target");
                        this._last_azimuth = viewer.getState("azimuth");
                        this._last_altitude = viewer.getState("altitude");
                        viewer.animate([{idx:speed, state:"mix", value:0}, 
					{idx:speed, state:"radius", value:200}, 
					{idx:speed, state:"target", value:[nx,ny,nz]}, 
					{idx:speed, state:"azimuth", value:newaz}, 
					{idx:speed, state:"altitude", value:newel}]);
                        return;
                    }
                }
            }
        }

    },

    undblpick: function() {
        var speed = 0.6;
        viewer.animate([{idx:speed, state:"mix", value:this._last_mix}, 
			{idx:speed, state:"radius", value:this._last_radius}, 
			{idx:speed, state:"target", value:this._last_target}, 
			{idx:speed, state:"azimuth", value:this._last_azimuth}, 
			{idx:speed, state:"altitude", value:this._last_altitude}]);
    },

    setMix: function(mixevt) {
        // console.log(mixevt);
        var pos, ax;
        for (var i = 0, il = this.axes.length; i < il; i++) {
            ax = this.axes[i];
            pos = this.viewer.getVert(ax.idx).norm;
            ax.obj.position.copy(pos);
            // Rescale axes for flat view
            ax.obj.scale.x = 1.0001-mixevt.flat;
        }
    },

    addMarker: function(ptidx, keep) {
        var vert = this.viewer.getVert(ptidx);

        for (var i = 0; i < this.axes.length; i++) {
            if (keep === true) {
                this.axes[i].obj.material.color.setRGB(180,180,180);

                this.axes[i].obj.geometry.vertices
            } else {
                this.axes[i].obj.parent.remove(this.axes[i].obj);
            }
        }
        if (keep !== true)
            this.axes = [];

        var axes = makeAxes(50, 0xffffff);
        axes.position.copy(vert.norm);
        axes.scale.x = 1.0001-this.viewer.flatmix;
        this.axes.push({idx:ptidx, obj:axes});
        this.viewer.pivot[vert.name].back.add(axes);
        this.viewer.controls.dispatchEvent({type:"change"});
    }
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
    return axes;
}