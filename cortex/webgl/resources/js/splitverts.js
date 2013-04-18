function splitverts(geom) {
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
    newgeom.attributes.pos1 = {itemSize:3, array:new Float32Array(npolys*3), stride:3};
    newgeom.attributes.pos2 = {itemSize:3, array:new Float32Array(npolys*3), stride:3};
    newgeom.attributes.pos3 = {itemSize:3, array:new Float32Array(npolys*3), stride:3};
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

                for (i = 0; i < 3; i++) {
                    pos = newgeom.attributes["pos"+(i+1)].array;
                    idx = index + geom.attributes.index.array[j+i];
                    pos[n*3+0] = geom.attributes.position.array[idx*3+0];
                    pos[n*3+1] = geom.attributes.position.array[idx*3+1];
                    pos[n*3+2] = geom.attributes.position.array[idx*3+2];
                }
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

function makeFlat(uv, flatlims, flatoff, right) {
    var fmin = flatlims[0], fmax = flatlims[1];
    var flat = new Float32Array(uv.length / 2 * 3);
    var norms = new Float32Array(uv.length / 2 * 3);
    for (var i = 0, il = uv.length / 2; i < il; i++) {
        if (right) {
            flat[i*3+1] = flatscale*uv[i*2] + flatoff[1];
            norms[i*3] = 1;
        } else {
            flat[i*3+1] = flatscale*-uv[i*2] + flatoff[1];
            norms[i*3] = -1;
        }
        flat[i*3+2] = flatscale*uv[i*2+1];
        uv[i*2]   = (uv[i*2]   + fmin[0]) / fmax[0];
        uv[i*2+1] = (uv[i*2+1] + fmin[1]) / fmax[1];
    }

    return [flat, norms];
}