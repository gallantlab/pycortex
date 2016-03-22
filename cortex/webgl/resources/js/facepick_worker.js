importScripts( "kdTree-min.js" );

var num = 0;
self.onmessage = function( event ) {
    var pts = [];
    var mnipts = [];
    for (var i = 0, il = event.data.pos.length; i < il; i+= 3)
        pts.push([event.data.pos[i], event.data.pos[i+1], event.data.pos[i+2], i/3]);
    for (var i = 0, il = event.data.mni.length; i < il; i+= 4)
        mnipts.push([event.data.mni[i], event.data.mni[i+1], event.data.mni[i+2], i/4])
    var dist = function (a, b) {
        return Math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]));
    }
    var kdt = new kdTree(pts, dist, [0, 1, 2]);
    var mnikdt = new kdTree(mnipts, dist, [0, 1, 2]);

    self.postMessage( {kdt:kdt.root, name:event.data.name, mnikdt:mnikdt.root} );
    if (++num > 1)
        self.close();
}
