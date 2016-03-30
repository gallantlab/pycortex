importScripts( "kdTree-min.js" );

var num = 0;
self.onmessage = function( event ) {
    var pts = [], x, y, z;
    if (event.data.wm !== undefined) {
        for (var i = 0, il = event.data.pos.length / 3; i < il; i++) {
            x = event.data.pos[i*3+0] * 0.5 + event.data.wm[i*4+0]*0.5;
            y = event.data.pos[i*3+1] * 0.5 + event.data.wm[i*4+1]*0.5;
            z = event.data.pos[i*3+2] * 0.5 + event.data.wm[i*4+2]*0.5;
            pts.push([x, y, z, i]);
        }
    } else {
        for (var i = 0, il = event.data.pos.length; i < il; i+= 3) {
            pts.push([event.data.pos[i], event.data.pos[i+1], event.data.pos[i+2], i/3]);
        }
    }

    var dist = function (a, b) {
        return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]);
    }
    var kdt = new kdTree(pts, dist, [0, 1, 2]);

    self.postMessage( {kdt:kdt.root, name:event.data.name} );
    if (++num > 1) //close after two hemispheres
        self.close();
}
