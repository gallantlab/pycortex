// Multi-viewer helpers: yoke controller and toggle UI for show_multi pages.
//
// YokeController listens to each viewer's LandscapeControls "change" event
// (fired only on user input — see movement.js), and when enabled broadcasts
// camera + surface-mix state to the others. A `_propagating` flag breaks the
// feedback loop: setAzimuth/setAltitude/etc. don't dispatch "change" themselves,
// but defensive guarding still costs nothing.

var multiview = (function (module) {
    module.YokeController = function (viewers) {
        this.viewers = viewers;
        this.enabled = false;
        this._propagating = false;
        this._listeners = [];

        var self = this;
        viewers.forEach(function (v, idx) {
            var listener = function () {
                if (!self.enabled || self._propagating) return;
                self._broadcast(idx);
            };
            v.controls.addEventListener("change", listener);
            self._listeners.push([v, listener]);

            // Surface mix is a separate state path: the fold/flatten slider
            // updates `controls.mix` via setMix and also fires "mix" via
            // the surface event. We capture the post-setMix change event above
            // because LandscapeControls dispatches "change" from its mouseup
            // and bind handlers — sufficient for camera AND mix updates that
            // come from user interaction with this viewer's UI.
        });
    };

    module.YokeController.prototype.setEnabled = function (on) {
        this.enabled = !!on;
    };

    module.YokeController.prototype._broadcast = function (sourceIdx) {
        var src = this.viewers[sourceIdx];
        var state = {
            azimuth: src.controls.azimuth,
            altitude: src.controls.altitude,
            radius: src.controls.radius,
            target: src.controls.target.toArray(),
            mix: src.controls.mix,
        };
        this._propagating = true;
        try {
            for (var i = 0; i < this.viewers.length; i++) {
                if (i === sourceIdx) continue;
                var v = this.viewers[i];
                // setMix recomputes az/alt/target from the per-viewer
                // _foldedazimuth/_flatazimuth interpolation, so apply mix
                // FIRST and then overwrite az/alt/target with the source's
                // exact values to keep all panels in lockstep.
                if (typeof v.controls.setMix === "function" && state.mix != null) {
                    v.controls.setMix(state.mix);
                }
                v.controls.setAzimuth(state.azimuth);
                v.controls.setAltitude(state.altitude);
                v.controls.setRadius(state.radius);
                v.controls.setTarget(state.target);
                v.schedule();
            }
        } finally {
            this._propagating = false;
        }
    };

    module.YokeController.prototype.destroy = function () {
        this._listeners.forEach(function (pair) {
            pair[0].controls.removeEventListener("change", pair[1]);
        });
        this._listeners = [];
    };

    // Append a yoke-toggle button to the figure container. Caller passes the
    // figure DOM element and the YokeController; clicking flips `enabled`.
    module.attachYokeToggle = function (containerEl, yoke, initialOn) {
        var btn = document.createElement("button");
        btn.id = "yoke-toggle";
        btn.textContent = initialOn ? "Yoke: ON" : "Yoke: OFF";
        btn.style.cssText = [
            "position:absolute",
            "top:8px",
            "right:120px",
            "z-index:1000",
            "padding:4px 10px",
            "background:rgba(40,40,40,0.85)",
            "color:#fff",
            "border:1px solid #888",
            "border-radius:4px",
            "cursor:pointer",
            "font-family:sans-serif",
            "font-size:12px",
        ].join(";");
        yoke.setEnabled(!!initialOn);
        btn.addEventListener("click", function () {
            yoke.setEnabled(!yoke.enabled);
            btn.textContent = yoke.enabled ? "Yoke: ON" : "Yoke: OFF";
        });
        containerEl.appendChild(btn);
        return btn;
    };

    return module;
})(multiview || {});
