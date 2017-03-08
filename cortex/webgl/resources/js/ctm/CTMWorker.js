importScripts( "lzma.js", "ctm.js" );

self.onmessage = function( event ) {

	var files = [];

	for ( var i = 0; i < event.data.groups.length; i ++ ) {

		var stream = new CTM.Stream( event.data.data );
		stream.offset = event.data.groups[ i ];

		self.postMessage( new CTM.File( stream ) );

	}

	self.close();

}
