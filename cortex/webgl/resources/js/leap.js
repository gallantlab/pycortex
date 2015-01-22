var frame_pause = 10000;
var idle_length = 10000; //length of time before new interaction gives helper
var gesture_info = {
	reps: 3,
	names: ["rotate", "flatten", "scale"],
	speed: [300, 1000, 300],
	imgs: [],
}

var _last_pos = null;
var _last_strength = [];
var _zoom = null;
var _wave_times = [];

var explain = 0;
var _last_gesture = null;
var _explain_timer = null;
var _idle_time = new Date();
var _session_start = null;

var current_dataset = "Localizer";

function Gesture(info) {
	this.active = false;
	this.info = info;
	this.imgs = [];
	for (var i = 0; i < info.names.length; i++) {
		var im1 = new Image();
		im1.src = "resources/explo_demo/"+info.names[i]+"_1.svg";
		var im2 = new Image();
		im2.src = "resources/explo_demo/"+info.names[i]+"_2.svg";
		this.imgs.push([im1, im2]);
	}
}
Gesture.prototype.show = function() {
	this.idx = 0;
	this.frame = 0;
	this.active = true;
	this.update();
	this.promise = $.Deferred();
	$("#gestures").show().css("opacity", 1);
	return this.promise;
}
Gesture.prototype.update = function() {
	if (this.active) {
		$("#gesture_name").text(this.info.names[this.idx]);
		$("#gesture_1").attr("src", this.imgs[this.idx][0].src);
		$("#gesture_2").attr("src", this.imgs[this.idx][1].src);

		if (this.frame % 2 == 0) {
			$("#gesture_1").fadeIn({duration:this.info.speed[this.idx], easing:"easeInOutQuint"});
			$("#gesture_2").fadeOut({duration:this.info.speed[this.idx], easing:"easeInOutQuint", complete:this.update.bind(this)});
		} else {
			$("#gesture_2").fadeIn({duration:this.info.speed[this.idx], easing:"easeInOutQuint"});
			$("#gesture_1").fadeOut({duration:this.info.speed[this.idx], easing:"easeInOutQuint", complete:this.update.bind(this)});
		}

		if (this.frame / 2 > this.info.reps[this.idx]) {
			this.frame = 0;
			this.idx += 1;
			if (this.idx >= this.info.names.length) {
				this.stop();
				this.promise.resolve();
			}
		} else
			this.frame += 1;
	}
}
Gesture.prototype.stop = function() {
	this.active = false;
	$("#gestures").fadeOut();
}

var gesture_anim = new Gesture({reps:[2, 1, 2], names:["rotate", "flatten", "zoom"], speed:[1000, 2000, 1000]});

var advance_frame = function() {
	if (explain < 4) {
		explain += 1;
		nextFrame();
		clearTimeout(_explain_timer);
		_explain_timer = setTimeout(advance_frame, frame_pause);
	} else if (explain == 4) {
		explain += 1;
		$("#intro").css("left", "-600px");
		gesture_anim.show().done(advance_frame);
		clearTimeout(_explain_timer);
		//_explain_timer = setTimeout(advance_frame, frame_pause);
	} else {
		gesture_anim.stop();
		explain = -1;
		nextFrame();
		viewer.reset_view();
		$("#display_cover").css("opacity", 0);
		$("#swipe_left").css("opacity", 0);
		$("#swipe_left_text").text("for "+current_dataset);
		clearTimeout(_explain_timer);
		setTimeout(viewer.playpause.bind(viewer), 1000);
	}
}

$(document).ready(function() {
	$("#display_cover").on("click", advance_frame);
});

Leap.loop({enableGestures: true}, function(frame) {
	if (!(frame.valid))
		return;

	frame.gestures.forEach(function(gesture) {
		if (gesture.type == "swipe") {
			if (gesture.pointableIds.length == 1) {
				if (gesture.state == "start") {
					_last_gesture = {id:gesture.id, num:0};
				} else if (_last_gesture.id == gesture.id) {
					if (_last_gesture.num == 4) {
						if (explain == -1 && gesture.direction[0] > 0) {
							viewer.playpause();
							$("#swipe_left").css("opacity", 1);
							$("#swipe_left_text").text("to continue");
							if ((new Date()) - _session_start > idle_length) {
								explain = 4;
								$("#display_cover").css("opacity", 1);
								gesture_anim.show().done(advance_frame);
							} else {
								explain = 0;
								$("#display_cover").css("opacity", 1);
								$("#intro").css("left", "50%");
								_explain_timer = setTimeout(advance_frame, frame_pause);
							}
						} else if (explain == -1 && gesture.direction[0] < 0) {
							//viewer.nextData();
							current_dataset = current_dataset == "Decoding" ? "Localizer" : "Decoding";
							$("#swipe_left_text").text("for "+current_dataset);
						} else {
							advance_frame();
						}
					}
					_last_gesture.num += 1;
				}

				_wave_times = [];
			} else {
				_wave_times.push(new Date());
			}
		}
		_idle_time = new Date();
	});

	if (explain > -1)
		return;

	//pop up a helper when it's been idle for a while
	if (frame.hands.length > 0) {
		var now = new Date();
		if (now - _idle_time > idle_length) {
			_session_start = now;
			$("#display_cover").css("opacity", .5);
			$("#swipe_right").css("opacity", 1);
			$("#swipe_left").css("opacity", 1);

			setTimeout(function() {
				$("#display_cover").css("opacity", 0);
				$("#swipe_right").css("opacity", 0);
				$("#swipe_left").css("opacity", 0);
			}, 8000);
		}
		_idle_time = now;
	}

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