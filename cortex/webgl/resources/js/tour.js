/* tour.js — optional guided-tour stepper for pycortex webgl viewers.
 *
 * Enabled by cortex.webgl.make_static(..., tour=True). Ships a generic stub
 * tour (no dataset-specific text); mriview.js constructs it on boot when
 * viewopts.tour is set and binds the 'o' key to Tour.toggle().
 *
 * Adapted from the tour used in the gallantlab semantic viewers (itself after
 * the NYT visualizer by Gregor Aisch & Amanda Cox). Cleaned for reuse: it
 * self-injects its markup, per-step `view`/`call` hooks are optional, and its
 * keyboard show/hide acts on the whole box (the old showhide left a stray strip).
 */
(function () {
    "use strict";

    // self-contained tourbox markup (no template HTML needed)
    var TOURBOX_HTML =
        '<div class="tour-stepper">' +
        '<div class="tour-steps"></div>' +
        '<div class="tour-hideshow">(hide tour)</div><br/>' +
        '<div class="tour-button tour-back">Back</div>' +
        '<div class="tour-button tour-next active">Next</div>' +
        '</div>' +
        '<div class="tour-title"></div>' +
        '<div class="tour-content"></div>';

    // generic stub steps — nothing dataset-specific (f/i/r/h/o are real core keys)
    var STEPS = [
        {title: "Welcome",
         content: "<p>This is a short guided tour of the viewer. Use <b>Back</b> / <b>Next</b> or the dots above to step through it. Press <b>o</b> to hide or show this box.</p>"},
        {title: "Navigating",
         content: "<p>Drag to rotate the brain, scroll to zoom, and <b>Shift</b>+drag to pan.</p>"},
        {title: "The surface",
         content: "<p>Press <b>f</b> to flatten the cortex, <b>i</b> to inflate it, and <b>r</b> to reset the view.</p>"},
        {title: "Explore",
         content: "<p>Press <b>h</b> at any time to see every keyboard shortcut. That's it — go explore!</p>"}
    ];

    // Tour(viewer, content?) — content defaults to the generic stub STEPS.
    var Tour = function (viewer, content) {
        this.viewer = viewer;
        this.content = content || STEPS;
        this.hidden = false;
        this.object = document.createElement("div");
        this.object.id = "tourbox";
        this.object.innerHTML = TOURBOX_HTML;
        document.body.appendChild(this.object);
        this.setup();
    };

    Tour.prototype.setup = function () {
        var self = this;
        $(this.object).find(".tour-hideshow").click(this.showhide.bind(this));
        this.content.forEach(function (step, i) {
            var el = document.createElement("div");
            $(el).addClass("tour-step").attr("title", step.title)
                 .click(function () { self.goto_step(i); });
            $(self.object).find(".tour-steps").append(el);
        });
        this.goto_step(0);
    };

    Tour.prototype.goto_step = function (idx) {
        this.current_step = parseInt(idx, 10);
        this.update_buttons();
        var steps = $(this.object).find(".tour-step");
        steps.removeClass("active");
        $(steps[this.current_step]).addClass("active");
        var cont = this.content[this.current_step];
        $(this.object).find(".tour-title").html(cont.title);
        $(this.object).find(".tour-content").html(cont.content);
        // per-step camera move / callback are optional (the stub has neither)
        if (cont.view && this.viewer && this.viewer.animate) this.viewer.animate(cont.view);
        if (typeof cont.call === "function") cont.call(this.viewer);
    };

    Tour.prototype.update_buttons = function () {
        var self = this;
        var last_idx = this.current_step - 1;
        var next_idx = this.current_step + 1;
        var back = $(this.object).find(".tour-back");
        var next = $(this.object).find(".tour-next");
        back.unbind("click");
        if (this.current_step === 0) {
            back.removeClass("active");
        } else {
            back.addClass("active").click(function () { self.goto_step(last_idx); });
        }
        if (this.current_step === this.content.length - 1) next_idx = 0;
        next.unbind("click").click(function () { self.goto_step(next_idx); });
    };

    // on-screen link: collapse to / expand from the stepper (keeps the affordance)
    Tour.prototype.showhide = function () {
        var box = $(this.object);
        if (this.hidden) {
            box.find(".tour-button, .tour-title, .tour-content").show();
            box.find(".tour-hideshow").html("(hide tour)");
            this.hidden = false;
        } else {
            box.find(".tour-button, .tour-title, .tour-content").hide();
            box.find(".tour-hideshow").html("(show tour)");
            this.hidden = true;
        }
    };

    // keyboard 'o': fully show/hide the whole box (nothing left behind — the fix)
    Tour.prototype.toggle = function () {
        var el = this.object;
        el.style.display = (el.style.display === "none" ? "" : "none");
    };

    window.Tour = Tour;
})();
