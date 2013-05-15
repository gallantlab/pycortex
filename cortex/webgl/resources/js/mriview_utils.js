Number.prototype.mod = function(n) {
    return ((this%n)+n)%n;
}

var mriview = (function(module) {
    module.getTexture = function(gl, renderbuf) {
        var canvas = document.createElement("canvas");
        canvas.width = renderbuf.width;
        canvas.height = renderbuf.height;
        var ctx = canvas.getContext("2d");
        var img = ctx.createImageData(renderbuf.width, renderbuf.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, renderbuf.__webglFramebuffer);
        gl.readPixels(0, 0, renderbuf.width, renderbuf.height, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(img.data.buffer));
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        ctx.putImageData(img, 0,0);
        ctx.scale(1, -1);
        img = new Image();
        img.src = canvas.toDataURL();
        return img;
    }

    module.makeFlat = function(uv, flatlims, flatoff, right) {
        var fmin = flatlims[0], fmax = flatlims[1];
        var flat = new Float32Array(uv.length / 2 * 3);
        var norms = new Float32Array(uv.length / 2 * 3);
        for (var i = 0, il = uv.length / 2; i < il; i++) {
            if (right) {
                flat[i*3+1] = module.flatscale*uv[i*2] + flatoff[1];
                norms[i*3] = 1;
            } else {
                flat[i*3+1] = module.flatscale*-uv[i*2] + flatoff[1];
                norms[i*3] = -1;
            }
            flat[i*3+2] = module.flatscale*uv[i*2+1];
            uv[i*2]   = (uv[i*2]   + fmin[0]) / fmax[0];
            uv[i*2+1] = (uv[i*2+1] + fmin[1]) / fmax[1];
        }

        return {pos:flat, norms:norms};
    }

    module.makeHatch = function(size, linewidth, spacing) {
        //Creates a cross-hatching pattern in canvas for the dropout shading
        if (spacing === undefined)
            spacing = 32;
        if (size === undefined)
            size = 128;
        var canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        var ctx = canvas.getContext("2d");
        ctx.lineWidth = linewidth ? linewidth: 8;
        ctx.strokeStyle = "white";
        for (var i = 0, il = size*2; i < il; i+=spacing) {
            ctx.beginPath();
            ctx.moveTo(i-size, 0);
            ctx.lineTo(i, size);
            ctx.stroke();
            ctx.moveTo(i, 0)
            ctx.lineTo(i-size, size);
            ctx.stroke();
        }

        var tex = new THREE.Texture(canvas);
        tex.needsUpdate = true;
        tex.premultiplyAlpha = true;
        tex.wrapS = THREE.RepeatWrapping;
        tex.wrapT = THREE.RepeatWrapping;
        tex.magFilter = THREE.LinearFilter;
        tex.minFilter = THREE.LinearMipMapLinearFilter;
        return tex;
    };

    module.splitverts = function(geom, left_off) {
        var o, ol, i, il, j, jl, k, n = 0, stride, mpts, faceidx, attr, idx, pos;
        var npolys = geom.attributes.index.array.length;
        var newgeom = new THREE.BufferGeometry();
        for (var name in geom.attributes) {
            if (name != "index") {
                attr = geom.attributes[name];
                newgeom.attributes[name] = {itemSize:attr.itemSize, stride:attr.itemSize};
                newgeom.attributes[name].array = new Float32Array(npolys*attr.itemSize);
            }
        }
        newgeom.attributes.index = {itemSize:1, array:new Uint16Array(npolys), stride:1};
        newgeom.attributes.face = {itemSize:1, array:new Float32Array(npolys), stride:1};
        newgeom.attributes.bary = {itemSize:3, array:new Float32Array(npolys*3), stride:3};
        newgeom.attributes.vert = {itemSize:3, array:new Float32Array(npolys*3), stride:3};
        newgeom.morphTargets = [];
        newgeom.morphNormals = [];
        for (i = 0, il = geom.morphTargets.length; i < il; i++) {
            newgeom.morphTargets.push({itemSize:3, array:new Float32Array(npolys*3), stride:3});
            newgeom.morphNormals.push(new Float32Array(npolys*3));
        }
        newgeom.offsets = []
        for (i = 0, il = npolys; i < il; i += 65535) {
            newgeom.offsets.push({start:i, index:i, count:Math.min(65535, il - i)});
        }

        for (o = 0, ol = geom.offsets.length; o < ol; o++) {
            var start = geom.offsets[o].start;
            var count = geom.offsets[o].count;
            var index = geom.offsets[o].index;

            //For each face
            for (j = start, jl = start+count; j < jl; j+=3) {
                //For each vertex
                for (k = 0; k < 3; k++) {
                    idx = index + geom.attributes.index.array[j+k];

                    for (var name in geom.attributes) {
                        if (name != "index") {
                            attr = geom.attributes[name];
                            for (i = 0, il = attr.itemSize; i < il; i++)
                                newgeom.attributes[name].array[n*il+i] = attr.array[idx*attr.stride+i];
                        }
                    }

                    for (i = 0, il = geom.morphTargets.length; i < il; i++) {
                        stride = geom.morphTargets[i].stride;
                        newgeom.morphTargets[i].array[n*3+0] = geom.morphTargets[i].array[idx*stride+0];
                        newgeom.morphTargets[i].array[n*3+1] = geom.morphTargets[i].array[idx*stride+1];
                        newgeom.morphTargets[i].array[n*3+2] = geom.morphTargets[i].array[idx*stride+2];
                        newgeom.morphNormals[i][n*3+0] = geom.morphNormals[i][idx*3+0];
                        newgeom.morphNormals[i][n*3+1] = geom.morphNormals[i][idx*3+1];
                        newgeom.morphNormals[i][n*3+2] = geom.morphNormals[i][idx*3+2];
                    }

                    newgeom.attributes.vert.array[n*3+0] = left_off + index + geom.attributes.index.array[j+0];
                    newgeom.attributes.vert.array[n*3+1] = left_off + index + geom.attributes.index.array[j+1];
                    newgeom.attributes.vert.array[n*3+2] = left_off + index + geom.attributes.index.array[j+2];

                    newgeom.attributes.face.array[n] = j / 3;
                    newgeom.attributes.bary.array[n*3+k] = 1;
                    newgeom.attributes.index.array[n] = n % 65535;
                    n++;
                }
            }
        }
        newgeom.computeBoundingBox();
        return newgeom;
    }

    return module;
}(mriview || {}));
