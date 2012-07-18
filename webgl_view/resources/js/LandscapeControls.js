/**
 * @author Eberhard Graether / http://egraether.com/
 */

THREE.LandscapeControls = function ( camera, domElement ) {

    THREE.EventTarget.call( this );

    var _this = this;
    STATE = { NONE : -1, ROTATE : 0, PAN : 1, ZOOM : 2 };
    var statefunc = { 
        0: this.rotate.bind(this),
        1: this.pan.bind(this),
        2: this.zoom.bind(this),
    }

    this.keystate = null;
    this.camera = camera;
    this.domElement = ( domElement !== undefined ) ? domElement : document;

    // API

    this.enabled = true;

    this.rotateSpeed = .8;
    this.zoomSpeed = .002;
    this.panSpeed = 0.3;

    // internals

    this.target = new THREE.Vector3();
    this.azimuth = 45;
    this.altitude = 45;
    this.radius = 200;

    var lastPosition = new THREE.Vector3();

    var _keyPressed = false,
    _state = STATE.NONE,

    _eye = new THREE.Vector3(),

    _start = new THREE.Vector3(),
    _end = new THREE.Vector3();

    // events

    var changeEvent = { type: 'change' };

    this.update = function () {

        _eye.copy( this.camera.position ).subSelf( this.target );
        var mouseChange = _end.clone().subSelf(_start);

        if (mouseChange.length() > 0 && statefunc[_state]) {
            statefunc[_state](mouseChange, _eye);
        }

        _start = _end;
        
        if (mouseChange.length() > 0) {
            this.setCamera(_eye);
            this.dispatchEvent( changeEvent );
        }

    }.bind(this);

    // listeners

    function keydown( event ) {

        if ( ! this.enabled ) return;

        if (event.keyCode == 17) {
            this.keystate = STATE.ZOOM;
        } else if (event.keyCode == 16) {
            this.keystate = STATE.PAN;
        } else {
            this.keystate = null;
        }

    };

    function keyup( event ) {

        if ( ! this.enabled ) return;

        this.keystate = null;

    };

    function mousedown( event ) {

        event.preventDefault();
        event.stopPropagation();

        if ( _state === STATE.NONE ) {

            _state = this.keystate ? this.keystate : event.button;
            _start = _end = this.getMouse(event);

        }

    };

    function mousemove( event ) {

        if ( ! this.enabled ) return;

        if ( _state === STATE.NONE ) {
            return;
        } else {
            _end = this.getMouse(event);
        }

    };

    function mouseup( event ) {

        if ( ! this.enabled ) return;

        event.preventDefault();
        event.stopPropagation();

        _state = STATE.NONE;

    };

    this.domElement.addEventListener( 'contextmenu', function ( event ) { event.preventDefault(); }, false );

    this.domElement.addEventListener( 'mousemove', mousemove.bind(this), false );
    this.domElement.addEventListener( 'mousedown', mousedown.bind(this), false );
    this.domElement.addEventListener( 'mouseup', mouseup.bind(this), false );

    window.addEventListener( 'keydown', keydown.bind(this), false );
    window.addEventListener( 'keyup', keyup.bind(this), false );

    this.setCamera();

};

THREE.LandscapeControls.prototype = {
    handleEvent: function ( event ) {

        if ( typeof this[ event.type ] == 'function' ) {

            this[ event.type ]( event );

        }

    },

    getMouse: function ( event ) {
        return new THREE.Vector2( event.clientX, event.clientY);
    },

    setCamera: function( _eye ) {
        this.altitude = this.altitude > 179.9999 ? 179.9999 : this.altitude;
        this.altitude = this.altitude < 0.0001 ? 0.0001 : this.altitude;

        var altrad = this.altitude*Math.PI / 180;
        var azirad = this.azimuth*Math.PI / 180;

        if (!_eye)
            _eye = new THREE.Vector3();

        _eye.set(
            this.radius*Math.sin(altrad)*Math.cos(azirad),
            this.radius*Math.sin(altrad)*Math.sin(azirad),
            this.radius*Math.cos(altrad)
        );

        this.camera.position.add( this.target, _eye );
        this.camera.lookAt( this.target );
    },

    rotate: function ( mouseChange, eye ) {
        this.azimuth -= this.rotateSpeed*mouseChange.x;
        this.altitude -= this.rotateSpeed*mouseChange.y;
    }, 

    pan: function( mouseChange, eye ) {
        //mouseChange.multiplyScalar( eye.length() * this.panSpeed );

        var pan = eye.clone().crossSelf( this.camera.up ).setLength( this.panSpeed*mouseChange.x );
        pan.addSelf( this.camera.up.clone().setLength( this.panSpeed*mouseChange.y ) );

        this.camera.position.addSelf( pan );
        this.target.addSelf( pan );
    },

    zoom: function( mouseChange, eye ) {
        var factor = 1.0 + mouseChange.y*this.zoomSpeed;
        this.radius *= factor;
    }


}