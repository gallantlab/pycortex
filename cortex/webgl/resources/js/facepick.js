function makePickShaders() {
    var vertShade = [
        "attribute vec3 idx;",
        "attribute vec4 auxdat;",
        "varying vec3 vPos;",
        THREE.ShaderChunk[ "morphtarget_pars_vertex" ],
        "void main() {",
            "vPos = position;",
            THREE.ShaderChunk[ "morphtarget_vertex" ],
        "}",
    ].join("\n");

    var fragShades = [];
    var dims = ['x', 'y', 'z'];
    for (var i = 0; i < 3; i++){
        var dim = dims[i];
        var shade = [
        "vec4 pack_float( const in float depth ) {",
            "const vec4 bit_shift = vec4( 256.0 * 256.0 * 256.0, 256.0 * 256.0, 256.0, 1.0 );",
            "const vec4 bit_mask  = vec4( 0.0, 1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0 );",
            "vec4 res = fract( depth * bit_shift );",
            "res -= res.xxyz * bit_mask;",
            "return res;",
        "}",
        "varying vec3 vPos;",
        "void main() {",
            "gl_FragColor = pack_float(vPos."+dim+");",
        "}"
        ].join("\n");
        fragShades.push(shade);
    }

    return {vertex:vertShade, fragment:fragShades};
}


function FacePick(viewer, callback) {
    this.viewer = viewer;

    var shaders = makePickShaders();
    this.shade_x = new THREE.ShaderMaterial({
        vertexShader: shaders.vertex,
        fragmentShader: shaders.fragment[0],
        attributes: { idx: true, auxdat:true, },
        morphTargets:true
    });
    this.shade_x = new THREE.ShaderMaterial({
        vertexShader: shaders.vertex,
        fragmentShader: shaders.fragment[1],
        attributes: { idx: true, auxdat:true, },
        morphTargets:true
    });
    this.shade_x = new THREE.ShaderMaterial({
        vertexShader: shaders.vertex,
        fragmentShader: shaders.fragment[2],
        attributes: { idx: true, auxdat:true, },
        morphTargets:true
    });
    this.resize($("#brain").width(), $("#brain").height());
}
FacePick.prototype = {
    resize: function(w, h) {
        this._valid = false;
        this.height = h;
        this.x = new THREE.WebGLRenderTarget(w, h, {
            minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter
        });
        this.y = new THREE.WebGLRenderTarget(w, h, {
            minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter
        });
        this.z = new THREE.WebGLRenderTarget(w, h, {
            minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter
        });
    },

    draw: function(debug) {
        this.viewer.scene.overrideMaterial = this.shade_x;
        this.viewer.renderer.render(this.viewer.scene, this.viewer.camera, this.x);
        this.viewer.scene.overrideMaterial = this.shade_y;
        this.viewer.renderer.render(this.viewer.scene, this.viewer.camera, this.y);
        this.viewer.scene.overrideMaterial = this.shade_z;
        this.viewer.renderer.render(this.viewer.scene, this.viewer.camera, this.z);
        this.viewer.scene.overrideMaterial = null;
        this._valid = true;
    },

    _pick: function(x, y) {
        if (!this._valid)
            this.draw();
        var gl = this.viewer.renderer.context;

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
        console.log(xbuf[0] / (256*256*256) + xbuf[1] / (256*256) + xbuf[2] / (256));
        
        // //adjust for clicking on black area
        // faceidx -= 1;

        // //Find which hemisphere it's in
        // for (var hemi in this.idxrange) {
        //     var lims = this.idxrange[hemi];
        //     if (lims[0] <= faceidx && faceidx < lims[1]) {
        //         faceidx -= lims[0];
        //         var geom =  this.viewer.meshes[hemi].geometry;
        //         var polys = geom.attributes.index.array;
        //         var map = geom.attributes.auxdat.array;

        //         //Find which offset
        //         for (var o = 0, ol = geom.offsets.length; o < ol; o++) {
        //             var start = geom.offsets[o].start;
        //             var index = geom.offsets[o].index;
        //             var count = geom.offsets[o].count;

        //             if (start <= faceidx*3 && faceidx*3 < (start+count)) {
        //                 //Pick the closest point in triangle to the click location
        //                 var pts = [index + polys[faceidx*3], 
        //                        index + polys[faceidx*3+1], 
        //                        index + polys[faceidx*3+2]];
        //                 var mousepos = new THREE.Vector2(x, y);
        //                 var ptdists = pts.map(function(pt) {
        //                     var spt = this.viewer.getPos(pt);
        //                     return mousepos.distanceTo(new THREE.Vector2(spt.pos[0], spt.pos[1]));
        //                 }.bind(this));
                        
        //                 var closest;
        //                 if (ptdists[0] < ptdists[1] && ptdists[0] < ptdists[2]) {
        //                     closest = pts[0];
        //                 } else if (ptdists[1] < ptdists[0] && ptdists[1] < ptdists[2]) {
        //                     closest = pts[1];
        //                 } else {
        //                     closest = pts[2];
        //                 }
                        
        //                 var dataidx = map[closest*4+3];
        //                 closest += hemi == "right" ? leftlen : 0;
        //                 return {ptidx: closest, dataidx: dataidx, hemi: hemi};
        //             }
        //         }
        //     }
        // }
    },

    pick: function(x, y, keep) {
        // DISABLE MULTI-CURSORS to make linking to voxels easy
        var p = this._pick(x, y);
        if (p) {
            console.log("Picked vertex "+p.ptidx+", voxel "+p.dataidx);
            this.addMarker(p.ptidx, keep);
            this.callback(p.dataidx, p.ptidx);
        }
    },
    
    dblpick: function(x, y, keep) {
        var speed = 0.6;
        var p = this._pick(x, y);
        if (p) {
            var leftlen = this.viewer.meshes.left.geometry.attributes.position.array.length / 3;
            p.ptidx -= p.hemi == "right" ? leftlen : 0;
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
            this._undblpickanim = states.map(function(s) { return {idx:speed, state:s, value:this.viewer.getState(s)} });
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
        viewer.animate(this._undblpickanim);
    },

    setMix: function(mixevt) {
        var pos, ax;
        for (var i = 0, il = this.axes.length; i < il; i++) {
            ax = this.axes[i];
            vert = this.viewer.getVert(ax.idx);
            ax.obj.position = blendPosNorm(vert.pos, vert.norm, mixevt.flat);
            // Rescale axes for flat view
            ax.obj.scale.x = 1.000-mixevt.flat;
        }
    },

    addMarker: function(ptidx, keep) {
        var vert = this.viewer.getVert(ptidx);

        for (var i = 0; i < this.axes.length; i++) {
            if (keep === true) {
                this.axes[i].obj.material.color.setRGB(180,180,180);
            } else {
                this.axes[i].obj.parent.remove(this.axes[i].obj);
            }
        }
        if (keep !== true)
            this.axes = [];

        var axes = makeAxes(50, 0xffffff);
        axes.position = blendPosNorm(vert.pos, vert.norm, this.viewer.flatmix);
        axes.scale.x = 1.0001-this.viewer.flatmix;
        this.axes.push({idx:ptidx, obj:axes});
        this.viewer.pivot[vert.name].back.add(axes);
        this.viewer.schedule();
    }
}

function blendPosNorm(pos, norm, flatmix) {
    var posfrac = flatmix - 0.01;
    var normfrac = 1 - posfrac;
    return pos.clone().multiplyScalar(posfrac).addSelf(norm.clone().multiplyScalar(normfrac));
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