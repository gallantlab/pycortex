/**
 * @author mrdoob / http://mrdoob.com/
 */

THREE.BinSurfLoader = function () {};

THREE.BinSurfLoader.prototype = new THREE.Loader();
THREE.BinSurfLoader.prototype.constructor = THREE.BinSurfLoader;

THREE.BinSurfLoader.prototype.load = function ( url, post, callback ) {
    var that = this;
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        if ( xhr.readyState == 4 ) {
            if ( xhr.status == 200 || xhr.status == 0 ) {
                callback( that.parse( this.response ) );
            } else {
                console.error( 'THREE.BinSurfLoader: Couldn\'t load ' + url + ' (' + xhr.status + ')' );
            }
        }
    };
    if (post) {
        var params = "types="+post+"&";
        xhr.open( "POST", url, true );
        xhr.responseType = 'arraybuffer';
        xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        xhr.send( params );
    } else {
        xhr.open( "GET", url, true );
        xhr.responseType = 'arraybuffer';
        xhr.send( null );
    }
};

THREE.BinSurfLoader.prototype.parse = function ( data ) {
    var idata = new Uint32Array(data);
    var fdata = new Float32Array(data);
    var numsurfs = idata[0];
    var ptlen = idata[1];
    var attrib = {};

    var geometry = new THREE.Geometry();

    var pts = fdata.subarray(2,ptlen+2);
    for (var i=0; i<ptlen; i+=3) {
        geometry.vertices.push(new THREE.Vector3(pts[i], pts[i+1], pts[i+2]));
    }
    console.log("Got points");

    var ptdat, verts;
    for (var i = 1; i < numsurfs-1; i++) {
        verts = [];
        ptdat = fdata.subarray(i*ptlen+2, (i+1)*ptlen+2);
        for (var i=0; i < ptdat.length; i+=3)
            verts.push(new THREE.Vector3(ptdat[i], ptdat[i+1], ptdat[i+2]));
        geometry.morphTargets.push({name:"surf"+(i-1), vertices:verts});
    }
    console.log("Got morphs");

    var flat = fdata.subarray((numsurfs-1)*ptlen+2, numsurfs*ptlen+2);
    var min = [1000,1000], max = [0,0], verts = [], pt;
    for (var i=0; i < flat.length; i+=3) {
        pt = new THREE.Vector3(flat[i], flat[i+1], flat[i+2]);
        verts.push(pt);
        if (min[0] < pt.x) min[0] = pt.x;
        else if (max[0] > pt.x) max[0] = pt.x;
        if (min[1] < pt.z) min[1] = pt.z;
        else if (max[1] > pt.z) max[0] = pt.z;
    }
    geometry.morphTargets.push({name:"flat", vertices:verts});
    console.log("Got flats");

    var polys = idata.subarray(numsurfs*ptlen+2);
    for (var i=0; i < polys.length; i+=3) {
        geometry.faces.push(new THREE.Face3(polys[i], polys[i+1], polys[i+2]));
    }
    console.log("Got faces");

    
    //Generate the UV coordinates
    var u, v, face, tcoords, n, names = ["a","b","c"];
    for (var i=0; i < geometry.faces.length; i++) {
        face = geometry.faces[i];
        tcoords = [];
        for (var j=0; j < 3; j++) {
            n = names[j];
            u = (verts[face[n]].x - min[0]) / (max[0] - min[0]);
            v = (verts[face[n]].z - min[1]) / (max[1] - min[1]);
            tcoords.push(new THREE.UV(u, v));
        }
        geometry.faceVertexUvs.push(tcoords);
    }
    console.log("Got UVs");

    
    geometry.computeCentroids();
    geometry.computeFaceNormals();
    geometry.computeVertexNormals();
    geometry.computeMorphNormals();
    geometry.computeBoundingSphere();
    //geometry.mergeVertices();
    console.log("finished model!");
    return geometry;
}