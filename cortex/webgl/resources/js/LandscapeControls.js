/**
 * @author Eberhard Graether / http://egraether.com/
 */

THREE.LandscapeControls = function ( cover, camera ) {
    var _this = this;
    STATE = { NONE : -1, ROTATE : 0, PAN : 1, ZOOM : 2 };
    var statefunc = { 
        0: this.rotate.bind(this),
        1: this.pan.bind(this),
        2: this.zoom.bind(this),
    }

    this.keystate = null;
    this.camera = camera;
    this.domElement = cover;

    // API
    this.enabled = true;
    
    // Constants
    this.rotateSpeed = .4;
    this.zoomSpeed = .002;
    this.maxRadius = 400; // makes sure axes & flatmat don't merge in depth buffer
    this.minRadius = function(mix){return 101*mix}; // limits zoom for flatmap, which disappears at r=100
    this.panSpeed = 0.3;
    this.clickTimeout = 200; // milliseconds
    this.friction = 0.05; // velocity lost per milisecond    

    // internals
    this.target = new THREE.Vector3();
    this.azimuth = 45;
    this.altitude = 75;
    this.radius = 250;
    this.azlim = 0;

    var _state = STATE.NONE,
        _start = new THREE.Vector3(),
        _end = new THREE.Vector3();
        _touch = false;

    var _mousedowntime = 0;
    var _clicktime = 0; // Time of last click (mouseup event)
    var _indblpick = false; // In double-click and hold?
    var _picktimer = false; // timer that runs pick event
    this._momentumtimer = false; // time that glide has been going on post mouse-release
    var _nomove_timer;

    // events

    var changeEvent = { type: 'change' };

    this.update = function () {
        var mousechange = _end.clone().sub(_start);

        if (statefunc[_state]) {
            if (statefunc[_state])
                statefunc[_state](mousechange);
        }

        var altlim = this.mix * 90;
        this.altitude = this.altitude > 179.9999-altlim ? 179.9999-altlim : this.altitude;
        this.altitude = this.altitude < 0.0001+altlim ? 0.0001+altlim : this.altitude;

        var azlim = this.mix * 180;
        if (azlim > this.azimuth || this.azimuth > (360 - azlim)) {
            var d1 = azlim - this.azimuth;
            var d2 = 360 - azlim - this.azimuth;
            this.azimuth = Math.abs(d1) > Math.abs(d2) ? 360-azlim : azlim;
        }

        //this._zoom(1.0) // Establish zoom (?)
        var altrad = this.altitude*Math.PI / 180;
        var azirad = (this.azimuth+90)*Math.PI / 180;

        var eye = new THREE.Vector3(
            this.radius*Math.sin(altrad)*Math.cos(azirad),
            this.radius*Math.sin(altrad)*Math.sin(azirad),
            this.radius*Math.cos(altrad)
        );

        camera.position.addVectors( this.target, eye );
        camera.lookAt( this.target );

        _start = _end;
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

        if ( _state === STATE.NONE) {
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
                _indblpick = true;
            } else {
                this.dispatchEvent({ type:"mousedown" });
            }
        }
    };

    function mouseup( event ) {
        if ( ! this.enabled ) return;
        this.dispatchEvent( changeEvent );

        this._momentumtimer = new Date().getTime();

        event.preventDefault();
        event.stopPropagation();

        _state = STATE.NONE;
        if (event.button == 0) {
            _clicktime = new Date().getTime();
        }

        // Run picker if time since mousedown is short enough
        if ( _clicktime - _mousedowntime < this.clickTimeout && event.button == 0) {
            var mouse2D = this.getMouse(event).clone();
            this.dispatchEvent({ type: "mouseup" });
            this.dispatchEvent({ type:"pick", x:mouse2D.x, y:mouse2D.y, keep:this.keystate == STATE.ZOOM});
        } else if ( event.button == 0 && _indblpick == true ) {
            this.dispatchEvent({ type:"undblpick" });
            _indblpick = false;
        } else {
            this.dispatchEvent({ type: "mouseup" });
        }
    };

    function mousemove( event ) {
        if ( ! this.enabled ) return;

        if ( _state === STATE.NONE ) {
            return;
        } else {
            _end = this.getMouse(event);
        }

        var nomove_evt = function() {
            if ( _state === STATE.NONE ) {
                return;
            } else {
                _end = _start.clone();
            }
        };
        clearTimeout(_nomove_timer);
        _nomove_timer = setTimeout(nomove_evt.bind(this), 100);
        this.dispatchEvent( changeEvent );
    };

    function touchmove( event ) {
        _touch = true;
        _end = new THREE.Vector2(event.touches[0].clientX, event.touches[1].clientY);
        this.dispatchEvent( changeEvent );
    };

    function mousewheel( event ) {
        if ( ! this.enabled ) return;

        event.preventDefault();
        event.stopPropagation();

        this.wheelzoom( event );
        this.dispatchEvent( changeEvent );

    };

    //code from http://vetruvet.blogspot.com/2010/12/converting-single-touch-events-to-mouse.html
    var touchToMouse=function(b){if(!(b.touches.length>1)){var a=b.changedTouches[0],c="";switch(b.type){case "touchstart":c="mousedown";break;case "touchmove":c="mousemove";break;case "touchend":c="mouseup";break;default:return}var d=document.createEvent("MouseEvent");d.initMouseEvent(c,true,true,window,1,a.screenX,a.screenY,a.clientX,a.clientY,false,false,false,false,0,null);a.target.dispatchEvent(d);b.preventDefault()}};
    this.domElement.addEventListener( 'touchstart', touchToMouse );
    this.domElement.addEventListener( 'touchmove', touchToMouse );
    this.domElement.addEventListener( 'touchend', touchToMouse );

    this.domElement.addEventListener( 'contextmenu', function ( event ) { event.preventDefault(); }, false );

    this.domElement.addEventListener( 'mousemove', mousemove.bind(this), false );
    this.domElement.addEventListener( 'mousedown', mousedown.bind(this), false );
    this.domElement.addEventListener( 'mouseup', mouseup.bind(this), false );
    this.domElement.addEventListener( 'mousewheel', mousewheel.bind(this), false);
    this.domElement.addEventListener( 'mouseout', mouseup.bind(this), false );

    window.addEventListener( 'keydown', keydown.bind(this), false );
    window.addEventListener( 'keyup', keyup.bind(this), false );

    this.mix = 0;
    this.update();
};
THREE.LandscapeControls.prototype = {
    setMix: function(mix) {
        this.mix = mix;
    },

    getMouse: function ( event ) {
        var off = $(event.target).offset();
        return new THREE.Vector2( event.clientX - off.left, event.clientY - off.top);
    },

    rotate: function ( mouseChange ) {
        this.azvel_init = -this.rotateSpeed * mouseChange.x;
        this.azimuth += this.azvel_init;

        this.altvel_init = -this.rotateSpeed * mouseChange.y;
        this.altitude += this.altvel_init;

        // Add panning depending on flatmix
        if ( this.mix > 0 ) {
            var panMouseChange = new Object;
            panMouseChange.x = mouseChange.x * Math.pow(this.mix, 2);
            panMouseChange.y = mouseChange.y * Math.pow(this.mix, 2);
            this.pan(panMouseChange);
        } else {
            this.azimuth = this.azimuth % 360;
            this.azimuth = this.azimuth < 0 ? this.azimuth + 360 : this.azimuth;
        }
    }, 

    pan: function( mouseChange ) {
        this.panxvel_init = this.panSpeed * mouseChange.x;
        this.panyvel_init = this.panSpeed * mouseChange.y;
        this.setpan( this.panSpeed * mouseChange.x, this.panSpeed * mouseChange.y );
    },

    setpan: function( x, y ) {
        var eye = this.camera.position.clone().sub(this.target);
        var right = eye.clone().cross( this.camera.up );
        var up = right.clone().cross(eye);
        var pan = (new THREE.Vector3()).addVectors(
            right.setLength( x ), 
            up.setLength( y ));
        this.camera.position.add( pan );
        this.target.add( pan );
    },
 
    zoom: function( mouseChange ) {
        this._zoom(1.0 + mouseChange.y*this.zoomSpeed);
    },
    
    wheelzoom: function( wheelEvent ) {
        this._zoom(1.0 + this.zoomSpeed * -1 * wheelEvent.wheelDelta/10.0);
    },

    _zoom: function( factor ) {
        this.radius *= factor;
        // if (this.camera.inPerspectiveMode) {
            // Perspective mode, zoom by changing radius from camera to object
            if (this.radius > this.maxRadius) { 
                this.radius = this.maxRadius; 
            }
            if (this.radius < this.minRadius(this.flatmix)) { 
                this.radius = this.minRadius(this.flatmix); 
            }
        // } else {
        //     // Orthographic mode, zoom by changing frustrum limits
        //     var aspect = this.camera.aspect
        //     var height = 2.0*this.radius*Math.tan(this.camera.fov/2.0)
        //     var width = aspect*height
        //     this.camera.cameraO.top = height/2.0
        //     this.camera.cameraO.bottom = -height/2.0
        //     this.camera.cameraO.right = width/2.0
        //     this.camera.cameraO.left = -width/2.0
        //     this.camera.cameraO.updateProjectionMatrix();
        // }
    },

    resetvel: function() {
        this.altvel_init = 0;
        this.azvel_init = 0;
        this.panxvel_init = 0;
        this.panyvel_init = 0;
    },
 
    // anim: function() {
    //     var now = new Date().getTime();
    //     if ( Math.abs(this.azvel_init) > 0 || Math.abs(this.altvel_init) > 0 ) {
    //         // console.log("Animating", this.azvel, this.altvel);
    //         //this.azvel *= 1 - this.friction;
    //         this.azvel = Math.pow( 1 - this.friction, (now - this._momentumtimer) / 1) * this.azvel_init;
    //         this.azimuth += this.azvel;
    //         this.azimuth = ((this.azimuth % 360) + 360) % 360;

    //         //this.altvel *= 1 - this.friction;
    //         this.altvel = Math.pow( 1 - this.friction, (now - this._momentumtimer) / 1) * this.altvel_init;
    //         this.altitude += this.altvel;
    //         // this.altitude = Math.max(Math.min(this.altitude, 180), 0.01);

    //         // console.log("Animating", this.azvel, this.altvel);
    //     }
    //     else {
    //         this.azvel = 0;
    //         this.altvel = 0;
    //     }

    //     if ( Math.abs(this.panxvel_init) > 0 || Math.abs(this.panyvel_init) > 0 ) {
    //         // console.log("Animating", this.panxvel, this.panyvel);
    //         this.panxvel = Math.pow( 1 - this.friction, (now - this._momentumtimer) / 1) * this.panxvel_init;
    //         this.panyvel = Math.pow( 1 - this.friction, (now - this._momentumtimer) / 1) * this.panyvel_init;
    //         //this.panxvel *= 1 - this.friction;
    //         //this.panyvel *= 1 - this.friction;

    //         this.setpan(this.panxvel, this.panyvel);
    //     }
    //     else {
    //         this.panxvel = 0;
    //         this.panyvel = 0;
    //     }
     
    //     this.dispatchEvent( {type:'change'} );

    //     // console.log("velocities: ", this.altvel, this.azvel, this.panxvel, this.panyvel);
    //     if (Math.abs(this.altvel)<0.01 && Math.abs(this.azvel)<0.01 && Math.abs(this.panxvel)<0.01 && Math.abs(this.panyvel)<0.01) {
    //         this.unschedule();
    //     } else {
    //         this.schedule();
    //     }

    // }

}

THREE.EventDispatcher.prototype.apply(THREE.LandscapeControls.prototype);