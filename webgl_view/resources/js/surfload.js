/**
 * @author mrdoob / http://mrdoob.com/
 */

THREE.BinSurfLoader = function () {};

THREE.BinSurfLoader.prototype = new THREE.Loader();
THREE.BinSurfLoader.prototype.constructor = THREE.BinSurfLoader;

THREE.BinSurfLoader.prototype.load = function ( url, callback ) {
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
    xhr.open( "GET", url, true );
    xhr.responseType = 'arraybuffer'
    xhr.send( null );
};

function makeVector(data) {
    var output = [];
    for (var i=0; i < data.length; i+=3) {
        output.push(new THREE.Vector3(data[i], data[i+1], data[i+2]))
    }
    return output
}

THREE.BinSurfLoader.prototype.parse = function ( data ) {
    var idata = new Uint32Array(data);
    var fdata = new Float32Array(data);
    var numsurfs = idata[0];
    var ptlen = idata[1];
    var attrib = {};
    var pts = fdata.subarray(2,ptlen+2);

    for (var i = 1; i < numsurfs-1; i++) {
        attrib['pts'+i] = {type:'v3'}
        attrib['pts'+i]['value'] = makeVector(fdata.subarray(i*ptlen+2, (i+1)*ptlen+2));
    }
    var flat = fdata.subarray((numsurfs-1)*ptlen+2, numsurfs*ptlen+2);
    attrib['flat'] = {type:'v3', value:makeVector(flat)};

    var geometry = new THREE.Geometry();

    for (var i=0; i<ptlen; i+=3) {
        geometry.vertices.push(new THREE.Vector3(pts[i], pts[i+1], pts[i+2]))
    }

    var polys = idata.subarray(numsurfs*ptlen+2);

    for (var i=0; i < polys.length; i+=3) {
        geometry.faces.push(new THREE.Face3(polys[i], polys[i+1], polys[i+2]))
    }

    verts = geometry.vertices;
    faces = geometry.faces;

    geometry.computeCentroids();
    geometry.computeFaceNormals();
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    
    return {geometry:geometry, attributes:attrib};
}