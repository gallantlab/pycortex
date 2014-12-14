importScripts( "lzma.js", "ctm.js" );

self.onmessage = function( event ) {

	var files = [];

	for ( var i = 0; i < event.data.offsets.length; i ++ ) {

		var stream = new CTM.Stream( event.data.data );
		stream.offset = event.data.offsets[ i ];

		var file = new CTM.File( stream );

	}

	self.postMessage( files );
	self.close();

}
