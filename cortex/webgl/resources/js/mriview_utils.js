Number.prototype.mod = function(n) {
    return ((this%n)+n)%n;
}

var mriview = (function(module) {
    module.MultiView = function(parent, nrows, ncols, dataviews) {
        jsplot.GridFigure.call(this, parent, nrows, ncols);

        var subjects = {}, snames = [], view, subj;
        for (var i = 0; i < dataviews.length; i++) {
            view = dataviews[i];
            subj = view.data[0].subject;
            if (subjects[subj] === undefined) {
                subjects[subj] = true;
                snames.push(subj);
            }
        }

        var viewer;
        for (var i = 0; i < snames.length; i++) {
            viewer = this.add(mriview.Viewer, i);
            viewer.load(window.subjects[snames[i]]);
            this[snames[i]] = viewer;
        }

        this.subjects = snames;
        this.addData(dataviews);
    }
    module.MultiView.prototype = Object.create(jsplot.GridFigure.prototype);
    module.MultiView.prototype.constructor = module.MultiView;
    module.MultiView.prototype.addData = function(dataviews) {
        var data = {}, subj, view;
        for (var i = 0; i < dataviews.length; i++) {
            view = dataviews[i];
            subj = view.data[0].subject;
            if (data[subj] === undefined) {
                data[subj] = [];
            }
            data[subj].push(view);
        }

        for (subj in data) {
            this[subj].addData(data[subj]);
        }
    }

    module.MultiView.prototype.saveIMG = function(width, height, post) {
        var canvas = document.createElement("canvas");
        if (width === null || width === undefined)
            width = this[this.subjects[0]].canvas.width();
        if (height === null || height === undefined)
            height = this[this.subjects[0]].canvas.height();
        canvas.width = width * this.ncols;
        canvas.height = height * this.nrows;
        var ctx = canvas.getContext("2d");
        var subj, off;

        for (var i = 0; i < this.subjects.length; i++) {
            subj = this[this.subjects[i]];
            off = subj.canvas.offset();
            ctx.drawImage(subj.getImage(width, height), off.left, off.top);
        }

        var png = canvas.toDataURL();
        if (post !== undefined)
            $.post(post, {png:png});
        else
            window.location.href = png.replace("image/png", "application/octet-stream");
    }

    //Create a function multiplexer, to pass calls through to all viewers
    function multiview_prototype(prot, names) {
        for (var i = 0; i < names.length; i++) {
            prot[names[i]] = function(name) {
                return function() {
                    var viewer, outputs = [];
                    for (var j = 0; j < this.subjects.length; j++) {
                        viewer = this[this.subjects[j]]
                        outputs.push(viewer[name].apply(viewer, arguments));
                    }
                    return outputs;
                }
            }(names[i]);
        }
    }
    multiview_prototype(module.MultiView.prototype, ['getState', 'setState', 'setColormap', 'nextData']);

    var glcanvas = document.createElement("canvas");
    var glctx = glcanvas.getContext("2d");
    module.getTexture = function(gl, renderbuf) {
        glcanvas.width = renderbuf.width;
        glcanvas.height = renderbuf.height;
        var img = glctx.createImageData(renderbuf.width, renderbuf.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, renderbuf.__webglFramebuffer);
        gl.readPixels(0, 0, renderbuf.width, renderbuf.height, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(img.data.buffer));
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        glctx.putImageData(img, 0, 0);

        // //This ridiculousness is necessary to flip the image...
        var canvas = document.createElement("canvas");
        var ctx = canvas.getContext("2d");
        canvas.width = renderbuf.width;
        canvas.height = renderbuf.height;
        ctx.translate(0, renderbuf.height);
        ctx.scale(1,-1);
        ctx.drawImage(glcanvas, 0,0);
        return canvas;
    }

    module._cull_flatmap_vertices = function(indices, auxdat, offsets) {
        var culled = new Uint16Array(indices.length), vA, vB, vC;
        var newOffsets = [];
        for ( j = 0, jl = offsets.length; j < jl; ++ j ) {

            var start = offsets[ j ].start;
            var count = offsets[ j ].count;
            var index = offsets[ j ].index;
            var nvert = 0;

            for ( i = start, il = start + count; i < il; i += 3 ) {
                vA = index + indices[ i ];
                vB = index + indices[ i + 1 ];
                vC = index + indices[ i + 2 ];

                if (!auxdat[vA*4] && !auxdat[vB*4] && !auxdat[vC*4]) {
                    culled[start + nvert*3 + 0] = indices[i];
                    culled[start + nvert*3 + 1] = indices[i+1];
                    culled[start + nvert*3 + 2] = indices[i+2];
                    nvert++;
                }

            }
            newOffsets.push({start:start, index:index, count:nvert*3});
        }

        return {indices:culled, offsets:newOffsets};
    }

    //Returns the world position of a vertex given the mix state
    module.get_position = function(posdata, surfmix, thickmix, idx) {
        var positions = posdata.positions;
        var normals = posdata.normals;

        var pos = new THREE.Vector3(
            positions[0].array[idx*3+0],
            positions[0].array[idx*3+1],
            positions[0].array[idx*3+2]
        )

        var norm = new THREE.Vector3(
            normals[0].array[idx*3+0],
            normals[0].array[idx*3+1],
            normals[0].array[idx*3+2]
        )

        if (posdata.wm) {
            var stride = posdata.wm.itemSize;
            pos.multiplyScalar(1 - thickmix).add(
                (new THREE.Vector3(
                    posdata.wm.array[idx*stride+0],
                    posdata.wm.array[idx*stride+1],
                    posdata.wm.array[idx*stride+2]
                )).multiplyScalar(thickmix)
            );
            norm.multiplyScalar(1 - thickmix).add(
                (new THREE.Vector3(
                    posdata.wmnorm.array[idx*3+0],
                    posdata.wmnorm.array[idx*3+1],
                    posdata.wmnorm.array[idx*3+2]
                )).multiplyScalar(thickmix)
            );
        }
        var base = pos.clone();

        var mix = surfmix * (positions.length-1);
        //add the thickness factor
        if (posdata.wm) {
            var stride = posdata.wm.itemSize;
            var dist = Math.pow(positions[0].array[idx*3+0] - posdata.wm.array[idx*stride+0], 2);
            dist += Math.pow(positions[0].array[idx*3+1] - posdata.wm.array[idx*stride+1], 2);
            dist += Math.pow(positions[0].array[idx*3+2] - posdata.wm.array[idx*stride+2], 2);

            var offset = Math.max(0, Math.min(1, mix)) * .62 * (1 - thickmix) * Math.sqrt(dist);
            pos.add(norm.clone().normalize().multiplyScalar(offset));
        }
        var factor = Math.max(0, Math.min(1, 1 - mix));
        pos.multiplyScalar(factor);
        norm.multiplyScalar(factor);
        for (var i = 1; i < positions.length; i++) {
            stride = positions[i].itemSize;
            factor = Math.max(0, Math.min(1, 1-Math.abs(mix-i) ));
            pos.add((new THREE.Vector3(
                positions[i].array[idx*stride+0],
                positions[i].array[idx*stride+1],
                positions[i].array[idx*stride+2]
            )).multiplyScalar(factor));
            norm.add((new THREE.Vector3(
                normals[i].array[idx*3+0],
                normals[i].array[idx*3+1],
                normals[i].array[idx*3+2]
            )).multiplyScalar(factor));
        }

        return {pos:pos, norm:norm, base:base};
    }

    module.computeNormal = function(vertices, index, offsets) {
        var i, il;
        var j, jl;

        var indices = index.array;
        var positions = vertices.array;
        var stride = vertices.itemSize;
        var normals = new Float32Array( vertices.array.length / vertices.itemSize * 3 );

        var vA, vB, vC, x, y, z,
            pA = new THREE.Vector3(),
            pB = new THREE.Vector3(),
            pC = new THREE.Vector3(),

            cb = new THREE.Vector3(),
            ab = new THREE.Vector3();

        for ( j = 0, jl = offsets.length; j < jl; ++ j ) {

            var start = offsets[ j ].start;
            var count = offsets[ j ].count;
            var index = offsets[ j ].index;

            for ( i = start, il = start + count; i < il; i += 3 ) {

                vA = index + indices[ i ];
                vB = index + indices[ i + 1 ];
                vC = index + indices[ i + 2 ];

                x = positions[ vA * stride ];
                y = positions[ vA * stride + 1 ];
                z = positions[ vA * stride + 2 ];
                pA.set( x, y, z );

                x = positions[ vB * stride ];
                y = positions[ vB * stride + 1 ];
                z = positions[ vB * stride + 2 ];
                pB.set( x, y, z );

                x = positions[ vC * stride ];
                y = positions[ vC * stride + 1 ];
                z = positions[ vC * stride + 2 ];
                pC.set( x, y, z );

                cb.subVectors( pC, pB );
                ab.subVectors( pA, pB );
                cb.cross( ab );
                // cb.normalize();

                normals[ vA * 3 ]     += cb.x;
                normals[ vA * 3 + 1 ] += cb.y;
                normals[ vA * 3 + 2 ] += cb.z;

                normals[ vB * 3 ]     += cb.x;
                normals[ vB * 3 + 1 ] += cb.y;
                normals[ vB * 3 + 2 ] += cb.z;

                normals[ vC * 3 ]     += cb.x;
                normals[ vC * 3 + 1 ] += cb.y;
                normals[ vC * 3 + 2 ] += cb.z;

            }

        }

        var x, y, z, n;

        for ( var i = 0, il = normals.length; i < il; i += 3 ) {

            x = normals[ i ];
            y = normals[ i + 1 ];
            z = normals[ i + 2 ];

            n = 1.0 / Math.sqrt( x * x + y * y + z * z );

            normals[ i ]     *= n;
            normals[ i + 1 ] *= n;
            normals[ i + 2 ] *= n;

            if (isNaN(normals[i])) {
                normals[i] = 0;
                normals[i+1] = 0;
                normals[i+2] = 0;
            }

        }
        var attr = new THREE.BufferAttribute(normals, 3);
        attr.needsUpdate = true;
        return attr;
    }

    module.computeAreas = function(vertices, index, offsets) {
        var i, il;
        var j, jl;

        var indices = index.array;
        var positions = vertices.array;
        var stride = vertices.itemSize;

        var areas = new Float32Array( vertices.array.length / vertices.itemSize );

        var vA, vB, vC, x, y, z, area,
            pA = new THREE.Vector3(),
            pB = new THREE.Vector3(),
            pC = new THREE.Vector3(),

            cb = new THREE.Vector3(),
            ab = new THREE.Vector3(),

            tri = new THREE.Triangle();

        for ( j = 0, jl = offsets.length; j < jl; ++ j ) {

            var start = offsets[ j ].start;
            var count = offsets[ j ].count;
            var index = offsets[ j ].index;

            for ( i = start, il = start + count; i < il; i += 3 ) {

                vA = index + indices[ i ];
                vB = index + indices[ i + 1 ];
                vC = index + indices[ i + 2 ];

                x = positions[ vA * stride ];
                y = positions[ vA * stride + 1 ];
                z = positions[ vA * stride + 2 ];
                pA.set( x, y, z );

                x = positions[ vB * stride ];
                y = positions[ vB * stride + 1 ];
                z = positions[ vB * stride + 2 ];
                pB.set( x, y, z );

                x = positions[ vC * stride ];
                y = positions[ vC * stride + 1 ];
                z = positions[ vC * stride + 2 ];
                pC.set( x, y, z );

                tri.set( pA, pB, pC );
                varea = tri.area() / 3;


                areas[vA] += varea;
                areas[vB] += varea;
                areas[vC] += varea;

            }
        }

        return areas;
    }

    module.smoothVertexData = function(vertices, index, offsets, data, factor) {
        var i, il;
        var j, jl;

        var indices = index.array;
        var positions = vertices.array;
        var stride = vertices.itemSize;

        var smoothdata = new Float32Array( data.length ),
            counts = new Uint16Array( data.length );

        var vA, vB, vC, x, y, z, area,
            pA = new THREE.Vector3(),
            pB = new THREE.Vector3(),
            pC = new THREE.Vector3(),

            cb = new THREE.Vector3(),
            ab = new THREE.Vector3(),

            tri = new THREE.Triangle();

        for ( j = 0, jl = offsets.length; j < jl; ++ j ) {

            var start = offsets[ j ].start;
            var count = offsets[ j ].count;
            var index = offsets[ j ].index;

            for ( i = start, il = start + count; i < il; i += 3 ) {

                vA = index + indices[ i ];
                vB = index + indices[ i + 1 ];
                vC = index + indices[ i + 2 ];

                // x = positions[ vA * stride ];
                // y = positions[ vA * stride + 1 ];
                // z = positions[ vA * stride + 2 ];
                // pA.set( x, y, z );

                // x = positions[ vB * stride ];
                // y = positions[ vB * stride + 1 ];
                // z = positions[ vB * stride + 2 ];
                // pB.set( x, y, z );

                // x = positions[ vC * stride ];
                // y = positions[ vC * stride + 1 ];
                // z = positions[ vC * stride + 2 ];
                // pC.set( x, y, z );

                // tri.set( pA, pB, pC );
                // varea = tri.area() / 3;

                smoothdata[vA] += data[vB] + data[vC];
                smoothdata[vB] += data[vA] + data[vC];
                smoothdata[vC] += data[vA] + data[vB];

                counts[vA] += 2;
                counts[vB] += 2;
                counts[vC] += 2;

            }
        }

        for ( var k = 0; k < data.length; k ++ ) {
            smoothdata[k] = smoothdata[k] / counts[k] * factor + (data[k] * (1.0 - factor));
        }

        return smoothdata;
    }

    module.iterativelySmoothVertexData = function(vertices, index, offsets, data, factor, iters) {
        var smoothed = data;
        for ( var i = 0; i < iters; i ++ ) {
            smoothed = module.smoothVertexData(vertices, index, offsets, smoothed, factor);
        }
        return smoothed;
    }

    module.computeVertexPrismVolume = function(areas1, areas2, dists) {
        var num = areas1.length;

        var volumes = new Float32Array(num);

        for ( var i = 0; i < num; i ++ ) {
            volumes[i] = dists[i] / 3 * ( areas1[i] + areas2[i] + Math.sqrt(areas1[i] * areas2[i]));
        }

        return volumes;
    }

    module.computeFlatVolumeHeight = function(areas, volumes) {
        var num = areas.length;
        var flatheights = new Float32Array(num);

        for ( var i = 0; i < num; i ++ ) {
            flatheights[i] = volumes[i] / areas[i];
        }

        var attr = new THREE.BufferAttribute(flatheights, 1);
        attr.needsUpdate = true;
        return attr;
    }

    module.computeDist = function(verts1, verts2) {
        var stride1 = verts1.itemSize;
        var stride2 = verts2.itemSize;
        var num = verts1.length / stride1;
        var dists = new Float32Array(num);

        var tv1 = new THREE.Vector3(), tv2 = new THREE.Vector3();

        for ( var i = 0; i < num; i ++ ) {
            tv1.set(verts1.array[i * stride1], 
                    verts1.array[i * stride1 + 1], 
                    verts1.array[i * stride1 + 2]);
            tv2.set(verts2.array[i * stride2], 
                    verts2.array[i * stride2 + 1], 
                    verts2.array[i * stride2 + 2]);
            dists[i] = tv1.distanceTo(tv2);
        }
        return dists;
        // var attr = new THREE.BufferAttribute(dists, 1);
        // attr.needsUpdate = true;
        // return attr;
    }

    module.offsetVerts = function(verts, offsets, axis, factor) {
        var num = verts.length / verts.itemSize;
        var newpos = new Float32Array(verts.array);

        for ( var i = 0; i < num; i ++ ) {
            newpos[i*verts.itemSize + axis] += factor * offsets.array[i];
        }

        var attr = new THREE.BufferAttribute(newpos, verts.itemSize);
        attr.needsUpdate = true;
        return attr;
    }

    //Generates a hatch texture in canvas
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

    //Still possibly useful? splits the faces into independent vertices
    module.splitverts = function(geom) {
        var o, ol, i, il, j, jl, k, n = 0, stride, mpts, faceidx, attr, idx, pos;
        var npolys = geom.attributes.index.array.length;

        var newgeom = new THREE.BufferGeometry();
        for (var name in geom.attributes) {
            var type = geom.attributes[name].array.constructor;
            var size = geom.attributes[name].itemSize;
            newgeom.addAttribute(name, new THREE.BufferAttribute(new type(npolys*size), size));
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
                                newgeom.attributes[name].array[n*il+i] = attr.array[idx*attr.itemSize+i];
                        }
                    }

                    // newgeom.attributes.vert.array[n*3+0] = left_off + index + geom.attributes.index.array[j+0];
                    // newgeom.attributes.vert.array[n*3+1] = left_off + index + geom.attributes.index.array[j+1];
                    // newgeom.attributes.vert.array[n*3+2] = left_off + index + geom.attributes.index.array[j+2];
                    // newgeom.attributes.bary.array[n*3+k] = 1;

                    newgeom.attributes.face.array[n] = Math.floor(j / 3);
                    newgeom.attributes.index.array[n] = n % 65535;
                    n++;
                }
            }
        }
        return newgeom;
    }

    return module;
}(mriview || {}));
