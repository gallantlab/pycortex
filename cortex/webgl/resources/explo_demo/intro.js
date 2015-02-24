var intro_frame = 0;
$("#intro_frame").load("resources/explo_demo/intro.svg");
var explain_fade = function(frame) {
  for (var i = 0; i < 5; i++) {
    $("#explanation_"+i).css("opacity", frame==i);
  }
  $("#explanation_"+frame).fadeOut();
}
var explain_dots = function(frame) {
  $("#explanation_"+frame).fadeIn();
  for (var i = 0; i < 5; i++) {
    $("#framedot_"+i).css("fill", frame == i ? "steelblue" : "gray");
  }
}

function nextFrame() {
  if (intro_frame == 0) {
    intro_frame = 1;
    explain_fade(0);
    $("#human_brain").fadeOut();
    $("#dots").fadeOut(400, function() {
        explain_dots(1);
      $("#model").fadeIn();
      $("#dots").fadeIn(400, function() {
        _advancing = false;
      });
    });
  } else if (intro_frame == 1) {
    intro_frame = 2;
    explain_fade(1);
    $("#enc_arrows").fadeOut();
    $("#presented_movie").fadeOut();
    $("#fmri_responses").fadeOut();
    $("#dots").fadeOut(400, function() {
      $("#model").animate({svgTransform:"translate(0,270)"}, {duration:800, complete:function() {
        explain_dots(2);
        $("#dots").fadeIn(400, function() {
          _advancing = false;
        });
      }});
    });
  } else if (intro_frame == 2) {
    intro_frame = 3;
    explain_fade(2);
    $("#dots").fadeOut(400, function() {
      explain_dots(3);
      $("#new_movie").fadeIn();
      $("#human_brain").fadeIn();
      $("#new_responses").fadeIn();
      $("#enc_arrows").fadeIn();
      $("#dots").fadeIn(400, function() {
        _advancing = false;
      });
    });
  } else if (intro_frame == 3) {
    intro_frame = 4;
    explain_fade(3);
    $("#dots").fadeOut(400, function() {
      explain_dots(4);
      $("#dec_arrows").fadeIn();
      $("#decoded_movie").fadeIn();
      $("#dots").fadeIn(400, function() {
        _advancing = false;
      });
    });
  } else if (intro_frame == 4) {
    intro_frame = 0;
    $("#model").fadeOut().animate({svgTransform:"translate(0,0)"});
    $("#enc_arrows").fadeOut();
    $("#dec_arrows").fadeOut();
    $("#decoded_movie").fadeOut();
    $("#new_movie").fadeOut();
    $("#new_responses").fadeOut();
    $("#human_brain").fadeOut();
    explain_fade(4);
    $("#explanation_4").fadeOut();
    $("#dots").fadeOut(400, function() {
      $("#fmri_responses").fadeIn();
      $("#human_brain").fadeIn();
      explain_dots(0);
      $("#enc_arrows").fadeIn();
      $("#presented_movie").fadeIn();
      $("#dots").fadeIn(400, function() {
        _advancing = false;
      });
    })
  }
};