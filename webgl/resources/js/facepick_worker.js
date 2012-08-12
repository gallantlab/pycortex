function genFlatGeom(msg) {
    var pts = new Float32Array(msg.ppolys.length*3);
    var polys = new Uint16Array(msg.ppolys.length);
    var color = new Float32Array(msg.ppolys.length);
    var morphs = [];
    var i, il, j, jl, k, stride, mpts;

    for (j = 0, jl = polys.length; j < jl; j++) {
        pts[j*3+0] = msg.ppts[msg.ppolys[j]*3+0];
        pts[j*3+1] = msg.ppts[msg.ppolys[j]*3+1];
        pts[j*3+2] = msg.ppts[msg.ppolys[j]*3+2];
        color[j] = Math.floor(j / 3) + msg.faceoff + 1;
        polys[j] = j % 65535;
    }

    for (i = 0, il = msg.morphs.length; i < il; i++) {
        stride = i == il-1 ? 3 : 4; //complete hack; flat is stride 3, all others stride 4
        mpts = new Float32Array(msg.ppolys.length*3);

        for (j = 0, jl = polys.length; j < jl; j++) {
            for (k = 0; k < 3; k++) {
                mpts[j*3+k] = msg.morphs[i][msg.ppolys[j]*stride+k];
            }
        }
        morphs.push(mpts);
    }

    return {name:msg.name, pts:pts, polys:polys, color:color, morphs:morphs};
}

self.onmessage = function( event ) {    
    if (event.data.func == "genFlatGeom") {
        response = genFlatGeom(event.data);
    }

    self.postMessage( response );
    self.close();
}