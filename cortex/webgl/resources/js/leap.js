var _last_strength = [];
var _last_mix = null;
var _last_gesture = null;
var _last_valid = new Date();
var _target = null;
var _wave_times = [];

var frustration_length = 8000;

var leapcontrol = Leap.loop({enableGestures: true}, function(frame) {
	if (!(frame.valid))
		return;

	frame.gestures.forEach(function(gesture) {
		if (gesture.type == "circle" && viewer.controls._state == -1) {
			if (gesture.pointableIds.length == 1) {
				if (gesture.state == "start") {
					_last_gesture = gesture.id;
				} else if (_last_gesture == gesture.id && gesture.progress >= 1) {
					var clockwise = false;
					var pointableID = gesture.pointableIds[0];
					var direction = frame.pointable(pointableID).direction;
					var dotProduct = Leap.vec3.dot(direction, gesture.normal);

					if (dotProduct  >  0) clockwise = true;

					if (demo.state == "movement" && !clockwise) {
						demo.advance();
						_last_valid = new Date();
					} else if (demo.state == "movement" && clockwise) {
						demo.nextData(true);
						_last_valid = new Date();
					} else if (clockwise) {
						demo.advance();
						_last_valid = new Date();
					}
					_last_gesture = null;
				}
			}
		}
		_idle_time = new Date();
	});

	if (frame.hands.length > 0) {
		//update the hand feedback indicator
		var normdist = (frame.hands[0].palmPosition[1] - 20) / 500;
		demo.setFeedback(normdist);

		//pop up a helper when it's been idle for a while
		var now = new Date();
		demo.update(now);

		//Check if it should rotate
		var rotate = frame.hands[0].pinchStrength > .95;
		rotate = frame.hands.length > 1 ? rotate ^ frame.hands[1].pinchStrength > .95 : rotate;

		if (rotate) {
			var extended = 0;
			for (var i = 0, il = frame.hands[0].fingers.length; i < il; i++) {
				extended += frame.hands[0].fingers[i].extended;
			}
			if (extended > 1) {
				var pos = frame.hands[0].palmPosition;
				if (viewer.controls._state != 0) {
					viewer.controls._state = 0;
					viewer.controls._start = new THREE.Vector2(pos[0]*1.5, pos[2]*1.5-pos[1]*1.5);
				}
				viewer.controls._end = new THREE.Vector2(pos[0]*1.5, pos[2]*1.5-pos[1]*1.5);
				viewer.controls.dispatchEvent( { type: "change" } );
				_last_valid = new Date();
			}
			_last_strength = [];
		} else if (frame.hands.length == 2) {
			var lpalm = frame.hands[0].palmNormal;
			var rpalm = frame.hands[1].palmNormal;
			var strength = Math.min(frame.hands[0].grabStrength, frame.hands[1].grabStrength);

			if (Math.min(lpalm[0], rpalm[0]) < -.8 && 
				Math.max(lpalm[0], rpalm[0]) > .8) {
				if (strength < .1) {
					//zoom
					var diff = Math.abs(frame.hands[0].palmPosition[0] - frame.hands[1].palmPosition[0])*1.5;
					if (viewer.controls._state != 2) {
						viewer.controls._state = 2;
						viewer.controls._start = new THREE.Vector2(0, -diff);
					}
					viewer.controls._end = new THREE.Vector2(0, -diff);
					viewer.controls.dispatchEvent({type:"change"});
					_last_valid = new Date();
				} else {
					viewer.controls._state = -1;
				}
			} else {
				viewer.controls._state = -1;

				_last_strength.push(strength);
				if (_last_strength.length > 10) {
					_last_strength.shift();

					var dir = 0;
					for (var i = 1; i < _last_strength.length; i++) {
						dir += _last_strength[i] - _last_strength[i-1];
					}
					
					if (dir > .2 && viewer.setMix() > .5) {
						viewer.reset_view();
						_last_valid = new Date();
						_last_strength = [];
					} else if (dir < -.2 && viewer.setMix() < .5) {
						viewer.animate([{state:'mix', value:1, idx:2}]);
						_last_valid = new Date();
						_last_strength = [];
					}
				}
			}			

		} else {
			viewer.controls._state = -1;
			_last_strength = [];
		}

		if (demo.state == "gesture") {
			if (now - _last_valid < .5)
				demo.advance();
		} else if (now - _last_valid > frustration_length) {
			//_last_valid = new Date();
			demo.advance();
		} 

	} else {
		_last_valid = new Date();
		demo.setFeedback(-1);
	}
});

