var idle_length = 10000; //ms between updates that's considered idle
var helper_length = 5000;
var reset_length = 30000;
var data_length = 30000 * 3;

function Demo() {
	this.state = "movement"; //intro, gesture, movement
	this._leap = false;

	this.idleTime = new Date();
	this._explain_timer = null;
	this._advancing = false;

	this._reset_timer = null;
	this._next_timer = null;
	this.update();
	this.dataset = "Categories";
	setInterval(this.nextData.bind(this), data_length);
}
Demo.prototype.setLeap = function() {
	//Activates this mode when it detects the leap motion
	console.log('Leap control activated');
	this.advance();
	this._leap = true;
	$("#leap_feedback").show();
}
Demo.prototype.advance = function() {
	if (!this._advancing) {
		this._advancing = true;
		if (this.state == "movement") {
			this.state = "gesture";
			if (viewer.state == "play")
				viewer.playpause();

			$("#display_cover").fadeIn();
			gesture_anim.show().done(function() {
				this._advancing = false;
				$("#display_cover").fadeOut();
				viewer.playpause();
			}.bind(this));
		} else {
			this.state = "movement";
			gesture_anim.stop();
			viewer.reset_view();
			$("#swipe_left").fadeOut();
			$("#display_cover").fadeOut(400, function() {
				$("#display_cover").hide();
			});
			setTimeout(function() {
				if (viewer.state != "play") 
					viewer.playpause();
			}, 1000);
			this._advancing = false;
		}
	}
}
// Demo.prototype.intro = function() {
// 	viewer.playpause();
// 	clearTimeout(_helper_timer);
// 	$("#swipe_right").fadeOut();
// 	$("#swipe_left").fadeIn();
// 	$("#swipe_left_text").text("to continue");
// 	if (!this._new_session) {
// 		explain = 5;
// 		$("#display_cover").show().fadeIn();
// 		gesture_anim.show().done(advance_frame);
// 	} else {
// 		explain = 0;
// 		this._new_session = false;
// 		$("#display_cover").show().fadeIn();
// 		$("#intro").css("left", "50%");
// 		this._explain_timer = setTimeout(advance_frame, frame_pause);
// 	}
// }
Demo.prototype.nextData = function() {
	viewer.nextData();
	viewer.playpause();
	this.dataset = this.dataset == "Decoding" ? "Categories" : "Decoding";
	if (this.dataset == "Decoding") {
		$("#decode_note").hide();
	} else {
		$("#decode_note").show();
	}
	$("#swipe_left_text").text("for "+this.dataset);
}
Demo.prototype.update = function(now) {
	if (!now)
		var now = new Date();

	if (now - this.idleTime > idle_length) {
		this.newSession();
	}

	this.idleTime = now;
	clearTimeout(this._reset_timer);
	this._reset_timer = setTimeout(function() {
		viewer.reset_view();
		if (viewer.state != "play")
			viewer.playpause();
	}, reset_length);
}
Demo.prototype.newSession = function() {
	if (this._leap) {
		// $("#display_cover").css("background", "rgba(0,0,0,.4)").fadeIn();
		// $("#swipe_left").show().css("opacity", 1);
		// $("#swipe_right").show().css("opacity", 1);

		// this._helper_timer = setTimeout(function() {
		// 	$("#display_cover").fadeOut(400, function() {
		// 		$("#display_cover").css("background", "rgba(0,0,0,.8)").hide();

		// 	});
		// }, helper_length);
		$("#leap_feedback").animate({boxShadow:'0 0 50px white'});
		$("#just_right").show();
		setTimeout(function() {
			$("#leap_feedback").animate({boxShadow:'0 0 0px white'});
			$("#just_right").hide();
		}, 6000);
	}
}
Demo.prototype.setFeedback = function(val) {
	var range = $("#leap_feedback svg").width();
	var pos = Math.min(Math.max(val, 0), 1) * range;
	$("#dist_indicator").attr("cx", pos).show();
	if (val <= .2) {
		$("#just_right").css("opacity", 0);
		$("#too_close").css("opacity", 1);
		$("#too_far").css("opacity", 0);
	} else if (val >= .8) {
		$("#just_right").css("opacity", 0);
		$("#too_close").css("opacity", 0);
		$("#too_far").css("opacity", 1);
	} else {
		$("#just_right").css("opacity", 1);
		$("#too_close").css("opacity", 0);
		$("#too_far").css("opacity", 0);
	}
}


var gesture_info = {
	reps: 3,
	names: ["rotate", "flatten", "scale"],
	speed: [300, 1000, 300],
	imgs: [],
}

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
var demo = new Demo();
