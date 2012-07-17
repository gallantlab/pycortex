/**
 * @author Eberhard Graether / http://egraether.com/
 */

THREE.LandscapeControls = function ( camera, domElement ) {

    THREE.EventTarget.call( this );

    var _this = this,
    STATE = { NONE : -1, ROTATE : 0, ZOOM : 2, PAN : 1 };

    this.camera = camera;
    this.domElement = ( domElement !== undefined ) ? domElement : document;

    // API

    this.enabled = true;

    this.screen = { width: window.innerWidth, height: window.innerHeight, offsetLeft: 0, offsetTop: 0 };

    this.rotateSpeed = 1.0;
    this.zoomSpeed = 1.0;
    this.panSpeed = 0.3;

    this.noRotate = false;
    this.noZoom = false;
    this.noPan = false;

    this.keys = [ 65 /*A*/, 83 /*S*/, 68 /*D*/ ];

    // internals

    this.target = new THREE.Vector3();

    var lastPosition = new THREE.Vector3();

    var _keyPressed = false,
    _state = STATE.NONE,

    _eye = new THREE.Vector3(),

    _rotateStart = new THREE.Vector3(),
    _rotateEnd = new THREE.Vector3(),

    _zoomStart = new THREE.Vector2(),
    _zoomEnd = new THREE.Vector2(),

    _panStart = new THREE.Vector2(),
    _panEnd = new THREE.Vector2();

    // events

    var changeEvent = { type: 'change' };


    // methods

    this.handleEvent = function ( event ) {

        if ( typeof this[ event.type ] == 'function' ) {

            this[ event.type ]( event );

        }

    };

    this.getMouseOnScreen = function ( clientX, clientY ) {

        return new THREE.Vector2(
            ( clientX - _this.screen.offsetLeft ) / _this.radius * 0.5,
            ( clientY - _this.screen.offsetTop ) / _this.radius * 0.5
        );

    };

    this.rotateCamera = function () {

        var mouseChange = _rotateEnd.clone().subSelf( _rotateStart );

        if ( mouseChange.lengthSq() ) {

            

        }

        var angle = Math.acos( _rotateStart.dot( _rotateEnd ) / _rotateStart.length() / _rotateEnd.length() );

        if ( angle ) {

            var axis = ( new THREE.Vector3() ).cross( _rotateStart, _rotateEnd ).normalize(),
                quaternion = new THREE.Quaternion();

            angle *= _this.rotateSpeed;

            quaternion.setFromAxisAngle( axis, -angle );

            quaternion.multiplyVector3( _eye );
            quaternion.multiplyVector3( _this.camera.up );

            quaternion.multiplyVector3( _rotateEnd );

            _rotateStart = _rotateEnd;

        }

    };

    this.zoomCamera = function () {

        var factor = 1.0 + ( _zoomEnd.y - _zoomStart.y ) * _this.zoomSpeed;

        if ( factor !== 1.0 && factor > 0.0 ) {

            _eye.multiplyScalar( factor );

            _zoomStart = _zoomEnd;

        }

    };

    this.panCamera = function () {

        var mouseChange = _panEnd.clone().subSelf( _panStart );

        if ( mouseChange.lengthSq() ) {

            mouseChange.multiplyScalar( _eye.length() * _this.panSpeed );

            var pan = _eye.clone().crossSelf( _this.camera.up ).setLength( mouseChange.x );
            pan.addSelf( _this.camera.up.clone().setLength( mouseChange.y ) );

            _this.camera.position.addSelf( pan );
            _this.target.addSelf( pan );
            
            _panStart = _panEnd;

        }

    };

    this.update = function () {

        _eye.copy( _this.camera.position ).subSelf( _this.target );

        if ( !_this.noRotate ) {

            _this.rotateCamera();

        }

        if ( !_this.noZoom ) {

            _this.zoomCamera();

        }

        if ( !_this.noPan ) {

            _this.panCamera();

        }

        _this.camera.position.add( _this.target, _eye );

        _this.checkDistances();

        _this.camera.lookAt( _this.target );

        if ( lastPosition.distanceTo( _this.camera.position ) > 0 ) {

            _this.dispatchEvent( changeEvent );

            lastPosition.copy( _this.camera.position );

        }

    };

    // listeners

    function keydown( event ) {

        if ( ! _this.enabled ) return;

        if ( _state !== STATE.NONE ) {

            return;

        } else if ( event.keyCode === _this.keys[ STATE.ROTATE ] && !_this.noRotate ) {

            _state = STATE.ROTATE;

        } else if ( event.keyCode === _this.keys[ STATE.ZOOM ] && !_this.noZoom ) {

            _state = STATE.ZOOM;

        } else if ( event.keyCode === _this.keys[ STATE.PAN ] && !_this.noPan ) {

            _state = STATE.PAN;

        }

        if ( _state !== STATE.NONE ) {

            _keyPressed = true;

        }

    };

    function keyup( event ) {

        if ( ! _this.enabled ) return;

        if ( _state !== STATE.NONE ) {

            _state = STATE.NONE;

        }

    };

    function mousedown( event ) {

        event.preventDefault();
        event.stopPropagation();

        if ( _state === STATE.NONE ) {

            _state = event.button;

            if ( _state === STATE.ROTATE && !this.noRotate ) {

                _rotateStart = _rotateEnd = _this.getMouseOnScreen( event.clientX, event.clientY );

            } else if ( _state === STATE.ZOOM && !this.noZoom ) {

                _zoomStart = _zoomEnd = _this.getMouseOnScreen( event.clientX, event.clientY );

            } else if ( !this.noPan ) {

                _panStart = _panEnd = _this.getMouseOnScreen( event.clientX, event.clientY );

            }

        }.bind(this)

    };

    function mousemove( event ) {

        if ( ! _this.enabled ) return;

        if ( _keyPressed ) {

            _rotateStart = _rotateEnd = _this.getMouseOnScreen( event.clientX, event.clientY );
            _zoomStart = _zoomEnd = _this.getMouseOnScreen( event.clientX, event.clientY );
            _panStart = _panEnd = _this.getMouseOnScreen( event.clientX, event.clientY );

            _keyPressed = false;

        }

        if ( _state === STATE.NONE ) {

            return;

        } else if ( _state === STATE.ROTATE && !_this.noRotate ) {

            _rotateEnd = _this.getMouseOnScreen( event.clientX, event.clientY );

        } else if ( _state === STATE.ZOOM && !_this.noZoom ) {

            _zoomEnd = _this.getMouseOnScreen( event.clientX, event.clientY );

        } else if ( _state === STATE.PAN && !_this.noPan ) {

            _panEnd = _this.getMouseOnScreen( event.clientX, event.clientY );

        }

    };

    function mouseup( event ) {

        if ( ! _this.enabled ) return;

        event.preventDefault();
        event.stopPropagation();

        _state = STATE.NONE;

    };

    this.domElement.addEventListener( 'contextmenu', function ( event ) { event.preventDefault(); }, false );

    this.domElement.addEventListener( 'mousemove', mousemove, false );
    this.domElement.addEventListener( 'mousedown', mousedown, false );
    this.domElement.addEventListener( 'mouseup', mouseup, false );

    window.addEventListener( 'keydown', keydown, false );
    window.addEventListener( 'keyup', keyup, false );

};