var _last_pos = null;
var _last_strength = [];
var _zoom = null;
var _wave_times = [];

var explain = 0;
var frame_pause = 10000;
var _last_gesture = null;
var _explain_timer = null;

Leap.loop({enableGestures: true}, function(frame) {
	if (!(frame.valid))
		return;

	frame.gestures.forEach(function(gesture) {
		if (gesture.type == "swipe") {
			//console.log(gesture.state);
			if (gesture.pointableIds.length == 1) {
				if (gesture.state == "start") {
					_last_gesture = {id:gesture.id, num:0};
				} else if (_last_gesture.id == gesture.id) {
					var advance_frame = function() {
						if (explain < 4) {
							explain += 1;
							nextFrame();
							_explain_timer = setTimeout(advance_frame, frame_pause);
						} else {
							explain = -1;
							nextFrame();
							$("#display_cover").css("opacity", 0);
							$("#canvas1").css("left", "-600px");
							$(figure.object).css("opacity", 1);
							viewer.playpause();
						}
					}
					if (_last_gesture.num == 2) {
						if (explain == -1 && gesture.direction[0] > 0) {
							explain = 0;
							viewer.playpause();
							setTimeout(function() {
								$(figure.object).css("opacity", 0);
							}, 1000);
							$("#display_cover").css("opacity", 1);
							$("#canvas1").css("left", "50%");
							_explain_timer = setTimeout(advance_frame, frame_pause);
						} else if (explain == -1 && gesture.direction[0] < 0) {
							//viewer.nextData();
						} else if (explain < 4) {
							explain += 1;
							nextFrame();
							clearTimeout(_explain_timer);
							_explain_timer = setTimeout(advance_frame, frame_pause);
						} else {
							explain = -1;
							nextFrame();
							$("#display_cover").css("opacity", 0);
							$("#canvas1").css("left", "-600px");
							$(figure.object).css("opacity", 1);
							viewer.playpause();
							clearTimeout(_explain_timer);
						}
					}
					_last_gesture.num += 1;
				}

				_wave_times = [];
			} else {
				_wave_times.push(new Date());
			}
		}
	});

	if (explain > -1)
		return;

	if (frame.hands.length == 1) {
		_last_strength = [];
		_zoom = null;
		var _extended = [];
		if (frame.hands[0].grabStrength < .5 && 
			Math.abs(frame.hands[0].palmNormal[0]) < .8) {
			var extended = 0;
			for (var i = 0; i < frame.hands[0].fingers.length; i++)
				if (frame.hands[0].fingers[i].extended) extended+=1;

			//PAN only when 3 fingers are obviously held out
			var pan_mode = false;
			_extended.push(extended);
			if (_extended.length > 21) {
				_extended.shift();
				pan_mode = _extended[0] == 3;
				for (var i = 1; i < _extended.length; i++)
					pan_mode |= _extended == 3;
			}

			var pos = frame.hands[0].palmPosition;
			if (_last_pos !== null) {
				var x = (pos[0] - _last_pos[0])*1.5;
				var y = (pos[2] - _last_pos[2])*1.5;
				var z = (pos[2] - _last_pos[2]);
				if (extended == 5) {
					viewer.controls.rotate(x, y);
					//viewer.controls.zoom(0, -z);
				} else if (pan_mode) {
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