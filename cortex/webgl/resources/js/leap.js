var _last_pos_ = null;
var _last_strength = [];
var _zoom = null;

Leap.loop(function(frame) {
	if (!(frame.valid))
		return;

	if (frame.hands.length == 1) {
		_last_strength = [];
		_zoom = null;
		if (frame.hands[0].grabStrength < .5 && 
			Math.abs(frame.hands[0].palmNormal[0]) < .8) {
			var extended = 0;
			for (var i = 0; i < frame.hands[0].fingers.length; i++)
				if (frame.hands[0].fingers[i].extended) extended+=1;

			var pos = frame.hands[0].palmPosition;
			if (_last_pos !== null) {
				var x = (pos[0] - _last_pos[0])*1.5;
				var y = (pos[2] - _last_pos[2])*1.5;
				if (extended == 5) {
					viewer.controls.rotate(x, y);
				} else if (extended == 3) {
					viewer.controls.pan(x, y);
				}
				viewer.schedule();
			}
			_last_pos = pos;
		} else {
			_last_pos = null;
		}
	} else if (frame.hands.length == 2) {
		var lpalm = frame.hands[0].palmNormal;
		var rpalm = frame.hands[1].palmNormal;
		var strength = Math.min(frame.hands[0].grabStrength, frame.hands[1].grabStrength);

		if (strength < .1 && 
			Math.min(lpalm[0], rpalm[0]) < -.8 && 
			Math.max(lpalm[0], rpalm[0]) > .8) {
			var lpos = new THREE.Vector3(), rpos = new THREE.Vector3();
			lpos.set.apply(lpos, frame.hands[0].palmPosition);
			rpos.set.apply(rpos, frame.hands[1].palmPosition);
			lpos.sub(rpos);

			if (_zoom === null) {
				_zoom = lpos.length();
			} else {
				viewer.controls.setRadius(500 - lpos.length());
				viewer.schedule();
			}
		} else {
			_zoom = null;
			_last_strength.push(strength);
			if (_last_strength.length > 21) {
				_last_strength.shift();

				var dir = 0;
				for (var i = 1; i < _last_strength.length; i++) {
					dir += _last_strength[i] - _last_strength[i-1];
				}
				
				if (dir > .2 && viewer.setMix() > .5) {
					viewer.animate([{state:'mix', value:0, idx:2}, {state:'camera.target', value:[0,0,0], idx:2}]);
					_last_strength = [];
				} else if (dir < -.2 && viewer.setMix() < .5) {
					viewer.animate([{state:'mix', value:1, idx:2}]);
					_last_strength = [];
				}
			}
		}

		_last_pos = null;
	} else {
		_last_pos = null;
		_last_strength = [];
		_zoom = null;
	}
});