var frame_pause = 10000;
var idle_length = 10000; //length of time before new interaction gives helper
var gesture_info = {
	reps: 3,
	names: ["rotate", "flatten", "scale"],
	speed: [300, 1000, 300],
	imgs: [],
}

var _last_pos = null;
var _last_mix = null;
var _last_gesture = null;
var _target = null;
var _wave_times = [];

var explain = 0;
var _explain_timer = null;
var _idle_time = new Date();
var _session_start = null;
var _helper_timer = null;

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
	$("#gestures").animate({opacity:1});
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
	$("#gestures").animate({opacity:0});
}

var gesture_anim = new Gesture({reps:[2, 1, 2], names:["rotate", "flatten", "zoom"], speed:[1000, 2000, 1000]});

var _advancing = false;
var advance_frame = function() {
	if (!_advancing) {
		_advancing = true;
		if (explain < 4) {
			explain += 1;
			nextFrame();
			clearTimeout(_explain_timer);
			_explain_timer = setTimeout(advance_frame, frame_pause);
		} else if (explain == 4) {
			explain += 1;
			$("#intro").css("left", "200%");
			nextFrame();
			gesture_anim.show().done(function() {
				_advancing = false;
				advance_frame();
			});
			clearTimeout(_explain_timer);
			//_explain_timer = setTimeout(advance_frame, frame_pause);
		} else {
			gesture_anim.stop();
			explain = -1;
			viewer.reset_view();
			$("#swipe_left").fadeOut();
			$("#display_cover").fadeOut(400, function() {
				$("#display_cover").hide();
				$("#swipe_left_text").text("for "+current_dataset);
			});
			clearTimeout(_explain_timer);
			setTimeout(viewer.playpause.bind(viewer), 1000);
			_advancing = false
		}
	}
}

$(document).ready(function() {
	$("#display_cover").on("click", advance_frame);
});

Leap.loop({enableGestures: true}, function(frame) {
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

					if (explain == -1 && !clockwise) {
						viewer.playpause();
						clearTimeout(_helper_timer);
						$("#swipe_right").fadeOut();
						$("#swipe_left").fadeIn();
						$("#swipe_left_text").text("to continue");
						if ((new Date()) - _session_start > idle_length) {
							explain = 5;
							$("#display_cover").show().fadeIn();
							gesture_anim.show().done(advance_frame);
						} else {
							explain = 0;
							$("#display_cover").show().fadeIn();
							$("#intro").css("left", "50%");
							_explain_timer = setTimeout(advance_frame, frame_pause);
						}
					} else if (explain == -1 && clockwise) {
						viewer.nextData();
						viewer.playpause();
						current_dataset = current_dataset == "Decoding" ? "Localizer" : "Decoding";
						if (current_dataset == "Decoding") {
							$("#decode_note").hide();
						} else {
							$("#decode_note").show();
						}
						$("#swipe_left_text").text("for "+current_dataset);
					} else if (clockwise) {
						advance_frame();
					}
					_last_gesture = null;
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

	if (frame.hands.length > 0) {
		//pop up a helper when it's been idle for a while
		var now = new Date();
		if (now - _idle_time > idle_length) {
			_session_start = now;
			$("#display_cover").css("background", "rgba(0,0,0,.4)").fadeIn();
			$("#swipe_left").show().css("opacity", 1);
			$("#swipe_right").show().css("opacity", 1);

			_helper_timer = setTimeout(function() {
				$("#display_cover").fadeOut(400, function() {
					$("#display_cover").css("background", "rgba(0,0,0,.8)").hide();
				});
			}, 8000);
		}
		_idle_time = now;

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
					viewer.controls._start = new THREE.Vector2(pos[0]*1.5, pos[2]*1.5);
				}
				viewer.controls._end = new THREE.Vector2(pos[0]*1.5, pos[2]*1.5);
				viewer.controls.dispatchEvent( { type: "change" } );
			}
		} else if (frame.hands.length == 2) {
			var lpalm = frame.hands[0].palmNormal;
			var rpalm = frame.hands[1].palmNormal;
			var strength = Math.min(frame.hands[0].grabStrength, frame.hands[1].grabStrength);

			if (Math.min(lpalm[0], rpalm[0]) < -.8 && 
				Math.max(lpalm[0], rpalm[0]) > .8) {
				if (strength == 1) {
					if (_last_mix === null || now - _last_mix[0] > 3000) {
						_last_mix = [now, 0];
					} else if (_last_mix[1] == 1) {
						viewer.reset_view();
						_last_mix = null;
					}
				} else if (strength < .1) {
					//zoom
					var diff = Math.abs(frame.hands[0].palmPosition[0] - frame.hands[1].palmPosition[0])*1.5;
					if (viewer.controls._state != 2) {
						viewer.controls._state = 2;
						viewer.controls._start = new THREE.Vector2(0, -diff);
					}
					viewer.controls._end = new THREE.Vector2(0, -diff);
					viewer.controls.dispatchEvent({type:"change"});
				}
			} else if (lpalm[1] < -.8 && rpalm[1] < -.8) {
				if (_last_mix === null || now - _last_mix[0] > 3000) {
					_last_mix = [now, 1];
					//console.log("start flat");
				} else if (_last_mix[1] == 0) {
					viewer.animate([{state:'mix', idx:2, value:1}]);
					_last_mix = null;
				}
			} else {
				viewer.controls._state = -1;
			}
		} else {
			viewer.controls._state = -1;
		}
	}

});