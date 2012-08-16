/**
 * @author jamesgao / james@jamesgao.com
 */

THREE.SurfGeometry = function (gl, geoms) {
    this.vertexPositionArray = geoms.fiducial.pts;
    this.vertexIndexArray = geoms.fiducial.polys;
    this.vertexNormalArray = geoms.fiducial.norms;

    this.vertexPositionBuffer = gl.createBuffer();
    gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexPositionBuffer );
    gl.bufferData( gl.ARRAY_BUFFER, this.vertexPositionArray, gl.STATIC_DRAW );
    this.vertexPositionBuffer.itemSize = 3;
    this.vertexPositionBuffer.numItems = this.vertexPositionArray.length / 3;

    this.vertexIndexBuffer = gl.createBuffer();
    gl.bindBuffer( gl.ELEMENT_ARRAY_BUFFER, this.vertexIndexBuffer );
    gl.bufferData( gl.ELEMENT_ARRAY_BUFFER, this.vertexIndexArray, gl.STATIC_DRAW );
    this.vertexIndexBuffer.itemSize = 1;
    this.vertexIndexBuffer.numItems = this.vertexIndexArray.length;

    if (this.vertexNormalArray) {
        this.vertexNormalBuffer = gl.createBuffer();
        gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexNormalBuffer );
        gl.bufferData( gl.ARRAY_BUFFER, this.vertexNormalArray, gl.STATIC_DRAW );
        this.vertexNormalBuffer.itemSize = 3;
        this.vertexNormalBuffer.numItems = this.vertexNormalArray.length / 3;
    }
};

THREE.SurfGeometry.prototype = new THREE.BufferGeometry();
THREE.SurfGeometry.prototype.constructor = THREE.SurfGeometry;
