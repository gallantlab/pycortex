function genFlatGeom(msg) {
    var pts = new Float32Array(msg.ppolys.length*3);
    var color = new Float32Array(msg.ppolys.length*3);
    var polys = new Uint16Array(msg.ppolys.length);
    var morphs = [];
    var o, ol, i, il, j, jl, k, n = 0, stride, mpts, faceidx;
    var idx, idy, idz;

    for (i = 0, il = msg.morphs.length; i < il; i++) {
        mpts = new Float32Array(msg.ppolys.length*3);
        morphs.push(mpts);
    }

    for (o = 0, ol = msg.offsets.length; o < ol; o++) {
        var start = msg.offsets[o].start;
        var count = msg.offsets[o].count;
        var index = msg.offsets[o].index;

        //For each face
        for (j = start, jl = start+count; j < jl; j+=3) {
            idx = index + msg.ppolys[j];
            idy = index + msg.ppolys[j+1];
            idz = index + msg.ppolys[j+2];
            //If any vertex is not cut
            if (msg.aux[idx*4+2] == 0 || msg.aux[idy*4+2] == 0 || msg.aux[idz*4+2] == 0) {
                faceidx = (j / 3) + msg.faceoff + 1;
                //For each vertex
                for (k = 0; k < 3; k++) {
                    idx = index + msg.ppolys[j+k];

                    pts[n*3+0] = msg.ppts[idx*3+0];
                    pts[n*3+1] = msg.ppts[idx*3+1];
                    pts[n*3+2] = msg.ppts[idx*3+2];

                    color[n*3+0] = faceidx >> 16;
                    color[n*3+1] = (faceidx >> 8) % 256;
                    color[n*3+2] = faceidx % 256;

                    polys[n] = n % 65535;

                    for (i = 0, il = msg.morphs.length; i < il; i++) {
                        //complete hack; flat is stride 3, all others stride 4
                        stride = msg.reordered ? 3 : (i == il-1 ? 3 : 4);
                        morphs[i][n*3+0] = msg.morphs[i][idx*stride+0];
                        morphs[i][n*3+1] = msg.morphs[i][idx*stride+1];
                        morphs[i][n*3+2] = msg.morphs[i][idx*stride+2];
                    }

                    n++;
                }

            }
        }

    }

    return {name:msg.name, pts:pts, polys:polys.subarray(0, n), color:color, morphs:morphs};
}

self.onmessage = function( event ) {    
    if (event.data.func == "genFlatGeom") {
        response = genFlatGeom(event.data);
    }

    self.postMessage( response );
    self.close();
}