var jsplot = (function (module) {
	var STATE = { NONE : -1, ROTATE : 0, PAN : 1, ZOOM : 2 };
	module.LandscapeControls = function() {
		this.target = new THREE.Vector3();
		this._foldedtarget = new THREE.Vector3();
		this._flattarget = new THREE.Vector3();
		this._flattarget.y = -60;
		
		this.azimuth = 45;
		this._foldedazimuth = 45;
		this._flatazimuth = 180;
		
		this.altitude = 75;
		this._foldedaltitude = 75;
		this._flataltitude = 0.1;
		
		this.radius = 400;

		this.mix = 0;

		this.rotateSpeed = 0.4;
		this.panSpeed = 0.3;
		this.zoomSpeed = 0.002;
		this.clickTimeout = 200; // milliseconds

		this.friction = .9;

		this._start = new THREE.Vector2();
		this._end = new THREE.Vector2();

		this._momentum = {change:[0,0]};
		this._state = STATE.NONE;

		this.twodbutton = $(document.createElement('button'));
		this.twodbutton.attr('id', 'twodbutton');
		this.twodbutton.html('2d');
		// this.twodbutton.click(this.set2d.bind(this));
	}
	THREE.EventDispatcher.prototype.apply(module.LandscapeControls.prototype);

	module.LandscapeControls.prototype._position = function() {
		var altrad = this.altitude*Math.PI / 180;
		var azirad = (this.azimuth+90)*Math.PI / 180;

		return new THREE.Vector3(
			this.radius*Math.sin(altrad)*Math.cos(azirad),
			this.radius*Math.sin(altrad)*Math.sin(azirad),
			this.radius*Math.cos(altrad)
		);
	}

	module.LandscapeControls.prototype.set2d = function() {
		this.setAzimuth(180);
		this.setAltitude(0.1);
	}

	module.LandscapeControls.prototype.update = function(camera) {
		var func;
		if (this._state != STATE.NONE) {
			if (this._state == STATE.ROTATE)
				func = this.rotate
			else if (this._state == STATE.PAN && this.mix == 1)
				func = this.rotate
			else if (this._state == STATE.PAN)
				func = this.pan
			else if (this._state == STATE.ZOOM)
				func = this.zoom

			var mousechange = this._end.clone().sub(this._start);
			func.call(this, mousechange.x, mousechange.y);
		}

		if (Math.abs(this._momentum.change[0]) > .05) {
			this._momentum.change[0] *= this.friction;
			this._momentum.change[1] *= this.friction;
		//	console.log(this._momentum.change);
			this._momentum.func.apply(this, this._momentum.change);
			setTimeout(function() {
				this.dispatchEvent( { type: "change" } );
			}.bind(this), 0);
		}

		camera.position.addVectors( this.target, this._position() );
		camera.lookAt( this.target );
		this._start = this._end;
	}

	module.LandscapeControls.prototype.rotate = function(x, y) {
		// in FLAT mode (mix = 1), PAN and ROTATE are reversed
		var mix;
		if (this._state != STATE.PAN) {
			mix = Math.pow(this.mix, 2);
			this.pan(x * mix, y * mix);
		} else {
			mix = 0;
		}
		
		var rx = x  * (1 - mix), ry = y * (1 - mix);
		this.setAzimuth(this.azimuth - this.rotateSpeed * rx);		
		this.setAltitude(this.altitude - this.rotateSpeed * ry);

		this._momentum.change = [x, y];
		this._momentum.func = this.rotate;
	}

	var _upvec = new THREE.Vector3(0,0,1);
	module.LandscapeControls.prototype.pan = function(x, y) {
		var eye = this._position();

		var right = eye.clone().cross( _upvec );
		var up = right.clone().cross(eye);
		var pan = right.setLength( this.panSpeed * x ).add(
			up.setLength( this.panSpeed * y ));
		this.setTarget((new THREE.Vector3).copy(this.target).add(pan).toArray());
		// this.target.add( pan );
	}

	module.LandscapeControls.prototype.zoom = function(x, y) {
		this.setRadius(this.radius * (1 + this.zoomSpeed * y));
	}

	module.LandscapeControls.prototype.setMix = function(mix) {
		this.mix = mix;
		if (mix > 0 && mix < 1 || true) { // hacky, I'm leaving this for now..
			this.azimuth = (1 - this.mix) * this._foldedazimuth + this.mix * this._flatazimuth;
			this.altitude = (1 - this.mix) * this._foldedaltitude + this.mix * this._flataltitude;
			this.target.set(this._foldedtarget.x * (1-mix) + mix*this._flattarget.x,
			                this._foldedtarget.y * (1-mix) + mix*this._flattarget.y,
			                this._foldedtarget.z * (1-mix) + mix*this._flattarget.z);
		} else {
			this.setAzimuth(this.azimuth);
			this.setAltitude(this.altitude);
			this.setTarget(this.target.toArray());
		}

		this.update2Dbutton();
	}

	module.LandscapeControls.prototype.setAzimuth = function(az) {
		if (az === undefined)
			return this.azimuth;

		az = az < 0 ? az + 360 : az % 360;
		
		if ( this.mix == 1.0 )
			this._flatazimuth = az;
		else
			this._foldedazimuth = az;
		this.azimuth = az;
		this.update2Dbutton();
	}
	module.LandscapeControls.prototype.setAltitude = function(alt) {
		if (alt === undefined)
			return this.altitude;

		if ( this.mix == 1.0 ) {
			this._flataltitude = Math.min(Math.max(alt, 0.1), 75);
			this.altitude = this._flataltitude
		}
		else {
			this._foldedaltitude = Math.min(Math.max(alt, 0.1), 179.9);
			this.altitude = this._foldedaltitude;
		}
		this.update2Dbutton();
	}

	module.LandscapeControls.prototype.setRadius = function(rad) {
		if (rad === undefined)
			return this.radius;

		this.radius = Math.max(Math.min(rad, 600), 10);
	}

	module.LandscapeControls.prototype.setTarget = function(xyz) {
		if (!(xyz instanceof Array))
			return [this.target.x, this.target.y, this.target.z];

		if (this.mix < 1) {
			this._foldedtarget.set(xyz[0], xyz[1], xyz[2]);
			this.target.set(xyz[0], xyz[1], xyz[2]);
		} else{
			this._flattarget.set(xyz[0], xyz[1], 0);
			this.target.set(xyz[0], xyz[1], 0);
		}
	}

	module.LandscapeControls.prototype.update2Dbutton = function() {
		if ( this.mix == 1.0 ){
			if ( this.altitude > 0.1  ||  this.azimuth != 180 )
				this.enable2Dbutton();
			else
				this.disable2Dbutton();
		}
		else
			this.disable2Dbutton();
			
	}
	module.LandscapeControls.prototype.enable2Dbutton = function() {
		// this.twodbutton.prop('disabled', false);
		this.twodbutton.show();
	}
	module.LandscapeControls.prototype.disable2Dbutton = function() {
		// this.twodbutton.prop('disabled', true);
		this.twodbutton.hide();
	}

	module.LandscapeControls.prototype.bind = function(object) {
		var _mousedowntime = 0;
		var _clicktime = 0; // Time of last click (mouseup event)
		var _indblpick = false; // In double-click and hold?
		var _picktimer = false; // timer that runs pick event
		//this._momentumtimer = false; // time that glide has been going on post mouse-release
		var _nomove_timer;

		var keystate = null;
		var changeEvent = { type: 'change' };

		function getMouse ( event ) {
			var off = $(event.target).offset();
			return new THREE.Vector2( event.clientX - off.left, event.clientY - off.top);
		};

		// listeners
		function keydown( event ) {
			if (event.keyCode == 17) { // ctrl
				keystate = STATE.ZOOM;
			} else if (event.keyCode == 16) { // shift
				keystate = STATE.PAN;
			} else {
				keystate = null;
			}

		};

		function keyup( event ) {
			keystate = null;
		};

		function blur ( event ) {
			keystate = null
		}

		function mousedown( event ) {
			event.preventDefault();
			event.stopPropagation();

			if ( this._state === STATE.NONE ) {
				this._state = keystate !== null ? keystate : event.button;
				this._start = this._end = getMouse(event);
				if (event.button == 0) {
					_mousedowntime = new Date().getTime();
				}

				// Run double-click event if time since last click is short enough
				if ( _mousedowntime - _clicktime < this.clickTimeout && event.button == 0 ) {
					if (_picktimer) clearTimeout(_picktimer);
					var mouse2D = getMouse(event).clone();
					this.dispatchEvent({ type:"dblpick", x:mouse2D.x, y:mouse2D.y, keep:keystate == STATE.ZOOM });
					_indblpick = true;
				} else {
					this.dispatchEvent({ type:"mousedown" });
				}
			}
		};

		function mouseup( event ) {
			this._momentumtimer = new Date().getTime();

			event.preventDefault();
			event.stopPropagation();

			this._state = STATE.NONE;
			if (event.button == 0) {
				_clicktime = new Date().getTime();
			}

			// Run picker if time since mousedown is short enough
			if ( _clicktime - _mousedowntime < this.clickTimeout && event.button == 0) {
				var mouse2D = getMouse(event).clone();
				this.dispatchEvent({ type: "mouseup" });
				this.dispatchEvent({ type:"pick", x:mouse2D.x, y:mouse2D.y, keep:keystate == STATE.ZOOM});
			} else if ( event.button == 0 && _indblpick == true ) {
				this.dispatchEvent({ type:"undblpick" });
				_indblpick = false;
			} else {
				this.dispatchEvent({ type: "mouseup" });
			}
			this.dispatchEvent(changeEvent);
		};

		function mousemove( event ) {
			if ( this._state !== STATE.NONE ) {
				this._end = getMouse(event);
				this.dispatchEvent( changeEvent );
			}
		};


		function mousewheel( event ) {
			if (!event.altKey) {
				delta = event.deltaY

				// normalize across browsers
				if(navigator.userAgent.toLowerCase().indexOf('firefox') > -1){
					delta = delta * 18
				}

			    this.setRadius(this.radius + this.zoomSpeed * delta * 110.0);
			    this.dispatchEvent( changeEvent );
			}
		};

		//code from http://vetruvet.blogspot.com/2010/12/converting-single-touch-events-to-mouse.html
		var touchToMouse=function(b){if(!(b.touches.length>1)){var a=b.changedTouches[0],c="";switch(b.type){case "touchstart":c="mousedown";break;case "touchmove":c="mousemove";break;case "touchend":c="mouseup";break;default:return}var d=document.createEvent("MouseEvent");d.initMouseEvent(c,true,true,window,1,a.screenX,a.screenY,a.clientX,a.clientY,false,false,false,false,0,null);a.target.dispatchEvent(d);b.preventDefault()}};
		object.addEventListener( 'touchstart', touchToMouse );
		object.addEventListener( 'touchmove', touchToMouse );
		object.addEventListener( 'touchend', touchToMouse );

		object.addEventListener( 'contextmenu', function ( event ) { event.preventDefault(); }, false );

		object.addEventListener( 'mousemove', mousemove.bind(this), false );
		object.addEventListener( 'mousedown', mousedown.bind(this), false );
		object.addEventListener( 'mouseup', mouseup.bind(this), false );
		object.addEventListener( 'wheel', mousewheel.bind(this), false);
		object.addEventListener( 'mouseout', mouseup.bind(this), false );

		window.addEventListener( 'keydown', keydown.bind(this), false );
		window.addEventListener( 'keyup', keyup.bind(this), false );
		window.addEventListener( 'blur', blur.bind(this), false );
	}

	return module;
}(jsplot || {}));
