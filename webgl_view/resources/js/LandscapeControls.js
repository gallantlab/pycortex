/**
 * @author Eberhard Graether / http://egraether.com/
 */

THREE.LandscapeControls = function ( camera, domElement, scene ) {

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
    this.scene = scene;

    // API

    this.enabled = true;

    this.rotateSpeed = .4;
    this.zoomSpeed = .002;
    this.panSpeed = 0.3;

    // Picker
    this.projector = new THREE.Projector();

    // internals

    this.target = new THREE.Vector3();
    this.azimuth = 45;
    this.altitude = 45;
    this.radius = 200;

    var _state = STATE.NONE,
        _start = new THREE.Vector3(),
        _end = new THREE.Vector3();

    // events

    var changeEvent = { type: 'change' };

    this.update = function (flatmix) {
        var mouseChange = _end.clone().subSelf(_start);
        this.azlim = flatmix * 180;
        this.altlim = flatmix * 90;
        //this.altlim = 0;

        if (mouseChange.length() > 0 && statefunc[_state]) {
            statefunc[_state](mouseChange);
        }

        _start = _end;
        
        if (mouseChange.length() > 0) {
            this.setCamera();
        }
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

    function mouseup( event ) {

        if ( ! this.enabled ) return;

        event.preventDefault();
        event.stopPropagation();

        _state = STATE.NONE;

    };

    function click( event ) {
        var mouse2D = this.getMouse(event).clone();
        var mouse3D = new THREE.Vector3(0, 10000, 0.5);
        mouse3D.x = (mouse2D.x / window.innerWidth) * 2 - 1;
        mouse3D.y = -(mouse2D.y / window.innerHeight) * 2 + 1;

        //console.log("picker: "+mouse3D.x+", "+mouse3D.y+", "+mouse3D.z);
        var ray = this.projector.pickingRay(mouse3D, this.camera);
        //console.log("picker: "+ray.origin.x+", "+ray.origin.y+", "+ray.origin.z);

        // create actual ray for debugging..
        //var linemat = new THREE.LineBasicMaterial({color: 0xffffff});
        //var linegeom = new THREE.Geometry();
        //var st = ray.direction.clone(), end = ray.direction.clone();
        //st.multiplyScalar(1000).addSelf(ray.origin);
        //end.multiplyScalar(-100).addSelf(ray.origin);
        //linegeom.vertices.push(st);
        //linegeom.vertices.push(end);
        //var line = new THREE.Line(linegeom, linemat);
        //this.scene.add(line);

        // this is kind of a hack
        var meshes = new Array([this.scene.children[1].children[0].children[0], this.scene.children[2].children[0].children[0]]);
        var intersects = ray.intersectObjects(meshes);
        this.intersects = intersects;
        this.ray = ray;

    };

    this.domElement.addEventListener( 'contextmenu', function ( event ) { event.preventDefault(); }, false );

    this.domElement.addEventListener( 'mousemove', mousemove.bind(this), false );
    this.domElement.addEventListener( 'mousedown', mousedown.bind(this), false );
    this.domElement.addEventListener( 'mouseup', mouseup.bind(this), false );
    this.domElement.addEventListener( 'click', click.bind(this), false );

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

    set: function(azimuth, altitude, radius) {
        if (azimuth !== undefined) this.azimuth = azimuth;
        if (altitude !== undefined) this.altitude = altitude;
        if (radius !== undefined) this.radius = radius;
        this.setCamera();
    },

    setCamera: function() {
        this.altitude = this.altitude > 179.9999-this.altlim ? 179.9999-this.altlim : this.altitude;
        this.altitude = this.altitude < 0.0001+this.altlim ? 0.0001+this.altlim : this.altitude;

        if (this.azlim > this.azimuth || this.azimuth > (360 - this.azlim)) {
            var d1 = this.azlim - this.azimuth;
            var d2 = 360 - this.azlim - this.azimuth;
            this.azimuth = Math.abs(d1) > Math.abs(d2) ? 360-this.azlim : this.azlim;
        }

        var altrad = this.altitude*Math.PI / 180;
        var azirad = (this.azimuth+90)*Math.PI / 180;

        var eye = new THREE.Vector3(
            this.radius*Math.sin(altrad)*Math.cos(azirad),
            this.radius*Math.sin(altrad)*Math.sin(azirad),
            this.radius*Math.cos(altrad)
        );

        this.camera.position.add( this.target, eye );
        this.camera.lookAt( this.target );
    },

    rotate: function ( mouseChange ) {
        var azdiff = -this.rotateSpeed*mouseChange.x;
        var az = this.azimuth + azdiff;
        if ( this.azlim > az || az > (360-this.azlim)) {
            this.azimuth = azdiff < 0 ? this.azlim : 360-this.azlim;
        } else {
            this.azimuth = az - 360*Math.floor(az / 360);
        }

        this.altitude -= this.rotateSpeed*mouseChange.y;
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
    }


}
