/**
 * @author Eberhard Graether / http://egraether.com/
 */

THREE.LandscapeControls = function ( camera ) {

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
    this.domElement = document.getElementById("braincover");

    // API
    this.enabled = true;
    
    // Constants
    this.rotateSpeed = .4;
    this.zoomSpeed = .002;
    this.maxRadius = 300; // makes sure axes & flatmat don't merge in depth buffer
    this.minRadius = function(mix){return 101*mix}; // limits zoom for flatmap, which disappears at r=100
    this.panSpeed = 0.3;
    this.clickTimeout = 200; // milliseconds
    

    // internals
    this.target = new THREE.Vector3();
    this.azimuth = 45;
    this.altitude = 45;
    this.radius = 200;

    var _state = STATE.NONE,
        _start = new THREE.Vector3(),
        _end = new THREE.Vector3();
        _touch = false;

    var _mousedowntime = 0;
    var _clicktime = 0; // Time of last click (mouseup event)
    var _indblpick = 0; // In double-click and hold?
    var _picktimer = false; // timer that runs pick event

    // events

    var changeEvent = { type: 'change' };

    this.update = function (flatmix) {
        var mousechange;

        if (_touch) {
            _state = 0;
            mouseChange = _end;
        } else 
            mouseChange = _end.clone().subSelf(_start);

        if (mouseChange.length() > 0 && statefunc[_state]) {
            if (statefunc[_state])
                statefunc[_state](mouseChange);
        }

        this.flatmix = flatmix;
        this.setCamera();

        _start = _end;
    };

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
            if (event.button == 0) {
                _mousedowntime = new Date().getTime();
            }
            // Run double-click event if time since last click is short enough
            if ( _mousedowntime - _clicktime < this.clickTimeout && event.button == 0 ) {
                if (_picktimer) clearTimeout(_picktimer);
                var mouse2D = this.getMouse(event).clone();
                this.dispatchEvent({ type:"dblpick", x:mouse2D.x, y:mouse2D.y, keep:this.keystate == STATE.ZOOM });
                _indblpick = 1;
            }
        }
    };

    function mouseup( event ) {
        if ( ! this.enabled ) return;

        event.preventDefault();
        event.stopPropagation();

        _state = STATE.NONE;
        if (event.button == 0) {
            _clicktime = new Date().getTime();
        }

        // Run picker if time since mousedown is short enough
        if ( _clicktime - _mousedowntime < this.clickTimeout && event.button == 0) {
            var mouse2D = this.getMouse(event).clone();
            _picktimer = setTimeout(function(){
                this.dispatchEvent({ type:"pick", x:mouse2D.x, y:mouse2D.y, keep:this.keystate == STATE.ZOOM});
            }.bind(this), this.clickTimeout);
        }

        if ( event.button == 0 && _indblpick == 1 ) {
            this.dispatchEvent({ type:"undblpick" });
            _indblpick = 0;
        }
    };

    function mousemove( event ) {
        if ( ! this.enabled ) return;

        if ( _state === STATE.NONE ) {
            return;
        } else {
            _end = this.getMouse(event);
        }
        this.dispatchEvent( changeEvent );
    };

    function touchmove( event ) {
        $(this.domElement).append("touch");
        _touch = true;
        _end = new THREE.Vector2(event.touches[0].clientX, event.touches[1].clientY);
        this.dispatchEvent( changeEvent );
    };

    function mousewheel( event ) {
        if ( ! this.enabled ) return;

        event.preventDefault();
        event.stopPropagation();

        this.wheelzoom( event );
        this.setCamera();
        this.dispatchEvent( changeEvent );

    };

    this.domElement.addEventListener( 'touchmove', touchmove.bind(this));
    this.domElement.addEventListener( 'touchend', function() { _touch = false; }, false);

    this.domElement.addEventListener( 'contextmenu', function ( event ) { event.preventDefault(); }, false );

    this.domElement.addEventListener( 'mousemove', mousemove.bind(this), false );
    this.domElement.addEventListener( 'mousedown', mousedown.bind(this), false );
    this.domElement.addEventListener( 'mouseup', mouseup.bind(this), false );
    this.domElement.addEventListener( 'mousewheel', mousewheel.bind(this), false);
    this.domElement.addEventListener( 'mouseout', mouseup.bind(this), false );

    window.addEventListener( 'keydown', keydown.bind(this), false );
    window.addEventListener( 'keyup', keyup.bind(this), false );

    this.resize($("#brain").width(), $("#brain").height() );
    this.flatmix = 0;
    this.setCamera();
};

THREE.LandscapeControls.prototype = {
    _limitview: function( ) {
        var flatmix = this.flatmix;
        var azlim = flatmix * 180;
        var altlim = flatmix * 90;
        
        this.altitude = this.altitude > 179.9999-altlim ? 179.9999-altlim : this.altitude;
        this.altitude = this.altitude < 0.0001+altlim ? 0.0001+altlim : this.altitude;

        if (azlim > this.azimuth || this.azimuth > (360 - azlim)) {
            var d1 = azlim - this.azimuth;
            var d2 = 360 - azlim - this.azimuth;
            this.azimuth = Math.abs(d1) > Math.abs(d2) ? 360-azlim : azlim;
        }

        var rad = this.radius, target = this.target.clone();
        if ($("#zlockwhole").length > 0) {
            if ($("#zlockwhole")[0].checked) {
                rad  = this.flatsize / 2 / this.camera.aspect;
                rad /= Math.tan(this.camera.fov / 2 * Math.PI / 180);
                rad -= this.flatoff;
                rad = flatmix * rad + (1 - flatmix) * this.radius;
            } else if (!$("#zlocknone")[0].checked) {
                rad  = this.flatsize / 4 / this.camera.aspect;
                rad /= Math.tan(this.camera.fov / 2 * Math.PI / 180);
                rad -= this.flatoff;
                rad = flatmix * rad + (1 - flatmix) * this.radius;
                if ($("#zlockleft")[0].checked) {
                    target.x = flatmix * (-this.flatsize / 4) + (1 - flatmix) * target.x;
                } else if ($("#zlockright")[0].checked) {
                    target.x = flatmix * ( this.flatsize / 4) + (1 - flatmix) * target.x;
                }
            }
        }

        this._radius = rad;
        this._target = target;
    },

    resize: function(w, h) {
        this.domElement.style.width = w+"px";
        this.domElement.style.height = h+"px";
    },

    getMouse: function ( event ) {
        return new THREE.Vector2( event.clientX, event.clientY);
    },

    setCamera: function(az, alt, rad) {
        var changed = false;
        if (az !== undefined) {
            this.azimuth = ((az % 360)+360)%360;
            changed = true;
        }
        if (alt !== undefined) {
            this.altitude = alt;
            changed = true;
        }
        if (rad !== undefined) {
            this.radius = rad;
            changed = true;
        }

        this._limitview();

        var altrad = this.altitude*Math.PI / 180;
        var azirad = (this.azimuth+90)*Math.PI / 180;

        var eye = new THREE.Vector3(
            this._radius*Math.sin(altrad)*Math.cos(azirad),
            this._radius*Math.sin(altrad)*Math.sin(azirad),
            this._radius*Math.cos(altrad)
        );

        this.camera.position.add( this._target, eye );
        this.camera.lookAt( this._target );
        if (changed)
            this.dispatchEvent( {type:'change'} );
    },

    rotate: function ( mouseChange ) {
        var azdiff = -this.rotateSpeed*mouseChange.x;
        var az = this.azimuth + azdiff;
        if ( this.azlim > 0 ) {
            if ( this.azlim > az || az > (360-this.azlim)) {
                this.azimuth = azdiff < 0 ? this.azlim : 360-this.azlim;
            } else {
                this.azimuth = az - 360*Math.floor(az / 360);
            }
        } else {
            this.azimuth = az - 360*Math.floor(az / 360);
        }


        this.altitude -= this.rotateSpeed*mouseChange.y;

    // Add panning depending on flatmix
    var panMouseChange = new Object;
    panMouseChange.x = mouseChange.x * Math.pow(this.flatmix, 2);
    panMouseChange.y = mouseChange.y * Math.pow(this.flatmix, 2);
    this.pan(panMouseChange);
    }, 

    pan: function( mouseChange ) {
        var eye = this.camera.position.clone().subSelf(this.target);
        var right = eye.clone().crossSelf( this.camera.up );
        var up = right.clone().crossSelf(eye);
        var pan = (new THREE.Vector3()).add(
            right.setLength( this.panSpeed*mouseChange.x ), 
            up.setLength( this.panSpeed*mouseChange.y ));
        this.camera.position.addSelf( pan );
        this.target.addSelf( pan );
    },

    zoom: function( mouseChange ) {
        var factor = 1.0 + mouseChange.y*this.zoomSpeed;
        this.radius *= factor;
	if (this.radius>this.maxRadius) { this.radius = this.maxRadius; }
	if (this.radius<this.minRadius(this.flatmix)) { this.radius = this.minRadius(this.flatmix); }
    },
    
    wheelzoom: function( wheelEvent ) {
        var factor = 1.0 + this.zoomSpeed * -1 * wheelEvent.wheelDelta/10.0;
        this.radius *= factor;
    }

}
