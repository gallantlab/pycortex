/**
 * @author jamesgao / james@jamesgao.com
 */

THREE.BinSurfLoader = function (context, showStatus) {
    this.context = context;
    THREE.Loader.call( this, showStatus );
};

THREE.BinSurfLoader.prototype = new THREE.Loader();
THREE.BinSurfLoader.prototype.constructor = THREE.BinSurfLoader;

THREE.BinSurfLoader.prototype.load = function (subject, callback) {
    //Load the minimal data
    this.callback = callback;
    this.subject = subject;
    this.loaded = {fiducial:false, flat:false};
    this.request(subject, "fiducial", true, this.parse.bind(this));
    this.request(subject, "flat", true, this.parse.bind(this));
};
THREE.BinSurfLoader.prototype.request = function(subject, type, polys, callback) {
    var _this = this;
    var url = "/surfaces/"+subject+"/"+type+"/";
    if (polys)
        url += "?polys=True"

    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        if ( xhr.readyState == 4 ) {
            if ( xhr.status == 200 || xhr.status == 0 ) {
                callback(type, this.response);
            } else {
                console.error( 'THREE.BinSurfLoader: Couldn\'t load ' + url + ' (' + xhr.status + ')' );
            }
        }
    };
    xhr.open( "GET", url, true );
    xhr.responseType = 'arraybuffer';
    xhr.send( null );
}
THREE.BinSurfLoader.prototype.parse = function(surftype, data) {
    this.loaded[surftype] = this.parseSurf(data);
    var loaded = true;
    for (var type in this.loaded) {
        if (typeof(this.loaded[type]) == "boolean")
            loaded &= this.loaded[type];
    }
    if (loaded) {
        console.log("Download complete");
        var fid = this.loaded.fiducial;
        fid.left.norms = this._computeNorms(fid.left.pts, fid.left.polys);
        console.log("Computed left norms");
        fid.right.norms = this._computeNorms(fid.right.pts, fid.right.polys);
        console.log("Computed right norms");

        var left = new THREE.SurfGeometry(this.context, {fiducial:this.loaded.fiducial.left, flat:this.loaded.flat.left});
        var right = new THREE.SurfGeometry(this.context, {fiducial:this.loaded.fiducial.right, flat:this.loaded.flat.right});
        this.callback(left, right);
    }
}

THREE.BinSurfLoader.prototype.parseSurf = function ( data ) {
    var pos = 0, header, pts, polys, norms, ptbuf;
    var heminames = ['left', 'right'];
    var geoms = {};

    for (var hemi = 0; hemi < 2; hemi++) {
        header = { 
            compressed: new Uint32Array(data.slice(pos), 0, 1)[0] == 1,
            lengths: new Uint32Array(data.slice(pos+4), 0, 2),
            min: new Float32Array(data.slice(pos+12), 0, 3),
            max: new Float32Array(data.slice(pos+24), 0, 3),
        }
        pos += 36;
        if (header.compressed) {
            ptbuf = new Uint16Array(data.slice(pos), 0, header.lengths[0]*3);
            pts = new Float32Array(header.lengths[0]*3);
            for (var i = 0, il = header.lengths[0]; i < il; i++) {
                for (var j = 0; j < 3; j++) {
                    pts[i*3+j] = (ptbuf[i*3+j]/65535)*header.max[j] + header.min[j];
                }
            }
            pos += header.lengths[0]*3*2;
        } else {
            pts = new Float32Array(data.slice(pos), 0, header.lengths[0]*3);
            pos += header.lengths[0]*3*4;
        }
        polys = new Uint32Array(data.slice(pos), 0, header.lengths[1]*3);
        pos += header.lengths[1]*3*4;
        geoms[heminames[hemi]] = {pts:pts, polys:polys};
    }
    return geoms;
}

THREE.BinSurfLoader.prototype._computeNorms = function(ptdat, polydat) {
    var ptface = {}; //Stores the faces that a point belongs to, for averaging
    var crosspt = [[0,0,0],[0,0,0]];
    var facenorms = new Float32Array(polydat.length*3);
    for (var i=0, il=polydat.length / 3; i < il; i++) {
        for (var j=0; j < 3; j++) {
            if (!ptface[polydat[i*3+j]]) 
                ptface[polydat[i*3+j]] = new Array();
            ptface[polydat[i*3+j]].push(i);

            //subtract the first point for cross product
            crosspt[0][j] = ptdat[polydat[i*3+1]*3+j] - ptdat[polydat[i*3]*3+j];
            crosspt[1][j] = ptdat[polydat[i*3+2]*3+j] - ptdat[polydat[i*3]*3+j];
        }
        //Compute cross product
        facenorms[i*3+0] = crosspt[0][1]*crosspt[1][2] - crosspt[0][2]*crosspt[1][1];
        facenorms[i*3+1] = crosspt[0][2]*crosspt[1][0] - crosspt[0][0]*crosspt[1][2];
        facenorms[i*3+2] = crosspt[0][0]*crosspt[1][1] - crosspt[0][1]*crosspt[1][0];

        //Normalize
        var norm = 0;
        for (var j=0; j < 3; j++)
            norm += Math.pow(facenorms[i*3+j],2);
        norm = Math.sqrt(norm);
        for (var j=0; j < 3; j++)
            facenorms[i*3+j] /= norm;
    }
    var ptnorms = new Float32Array(ptdat.length*3);
    for (var i=0, il=ptdat.length; i < il; i++) {
        if (ptface[i]) {
            for (var j=0; j < 3; j++) {
                ptnorms[i*3+j] = 0;
                for (var k=0, kl=ptface[i].length; k < kl; k++)
                    ptnorms[i*3+j] += facenorms[ptface[i][k]*3+j]
                ptnorms[i*3+j] /= ptface[i].length;
            }
        }
    }
    return ptnorms;
}