//monkeypatch a remove folder command
dat.GUI.prototype.removeFolder = function(name) {
    this.__folders[name].close();
    this.__folders[name].domElement.parentNode.parentNode.removeChild(this.__folders[name].domElement.parentNode);
    this.__folders[name] = undefined;
    this.onResize();
}

var jsplot = (function (module) {
    module.Color = function(data) {
        this.color = Color.colors[data];
    }
    module.Color.prototype.toRGBA = function() {
        return "rgba("+(this.color[0]*255)+", "+(this.color[1]*255)+", "+(this.color[2]*255)+", "+(this.color[3]*255)+")";
    }
    module.Color.colors = {
        'k': [0,0,0,1],
        'r': [1,0,0,1],
        'g': [0,1,0,1],
        'b': [0,0,1,1],
    }

    module.construct = function(cls, args) {
        function F() {
            return cls.apply(this, args);
        }
        F.prototype = cls.prototype;
        return new F();
    }

    module.Figure = function(parent) {
        this._notifying = false;
        this.axes = [];
        this.ax = null;
        this._registrations = {};

        this.object = document.createElement("div");
        this.object.className = 'jsplot_figure';
        this.parent = parent;
        if (parent === undefined) {
            $(document.body).append(this.object);
            this.init();
            window.addEventListener('resize', this.resize.bind(this));
        } else {
            parent.addEventListener('resize', this.resize.bind(this));
        }

        this.gui = new dat.GUI({autoPlace:false});
        this.gui.close();
        this.ui_element = document.createElement("div");
        this.ui_element.id = "figure_ui";
        this.object.appendChild(this.ui_element);
        this.ui_element.appendChild(this.gui.domElement);
    }
    THREE.EventDispatcher.prototype.apply(module.Figure.prototype);

    module.Figure.prototype.init = function() {}
    module.Figure.prototype.register = function(eventType, self, func) {
        if (this.parent && this.parent instanceof module.Figure) {
            this.parent.register(eventType, self, func);
        } else {
            if (!(this._registrations[eventType] instanceof Array))
                this._registrations[eventType] = [];

            var register = function(evt) { 
                if (evt.self != self)
                    func.apply(self, evt.args);
            }.bind(this);
            this._registrations[eventType].push([self, register]);

            this.addEventListener(eventType, register);
        }
    }
    module.Figure.prototype.unregister = function(eventType, self) {
        var objects = this._registrations[eventType];
        for (var i = 0; i < objects.length; i++) {
            if (objects[i][0] === self)
                this.removeEventListener(eventType,objects[i][1]);
        }
    }
    module.Figure.prototype.notify = function(eventType, self, arguments) {
        if (this.parent && this.parent instanceof module.Figure) {
            this.parent.notify(eventType, self, arguments);
        } else {
            if (!this._notifying) { 
                this._notifying = true;
                this.dispatchEvent({type:eventType, self:self, args:arguments});
                this._notifying = false;
            }
        }
    }
    module.Figure.prototype.resize = function(width, height) {
        if (width !== undefined)
            $(this.object).width(width);
        if (height !== undefined)
            $(this.object).height(height);
        var w = $(this.object).width();
        var h = $(this.object).height();
        this.dispatchEvent({type:'resize', width:w, height:h});
    }
    module.Figure.prototype.close = function() {
        window.close();
    }
    module.Figure.prototype.add = function(axcls) {
        var args = Array.prototype.slice.call(arguments).slice(1);
        args.unshift(this);
        this.ax = module.construct(axcls, args);
        this.axes.push(this.ax);
        $(this.parent.object).append(this.ax.object);
        if (this.ax.ui !== undefined)
            this.ax.ui.init(this.gui);
    }

    var w2fig_layer = 0;
    module.W2Figure = function(parent) {
        this._resizing = true;
        module.Figure.call(this, parent);
        this.axes = {};
    }
    module.W2Figure.prototype = Object.create(module.Figure.prototype);
    module.W2Figure.prototype.constructor = module.W2Figure;
    module.W2Figure.prototype.init = function() {
        //var style = "border:1px solid #dfdfdf; padding:5px;";
        this.w2obj = $(this.object).w2layout({
            name: 'w2figure'+(w2fig_layer++),
            panels: [
                { type: 'top', resizable: true, hidden: true },
                { type: 'bottom', resizable: true, hidden: true },
                { type: 'left', resizable: true, hidden: true },
                { type: 'right', resizable: true, hidden: true },
                { type: 'main' },
            ],
        });
        this.w2obj.onResize = this.resize.bind(this);
        this._resizing = false;
    }
    module.W2Figure.prototype.resize = function() {
        if (!this._resizing) {
            this._resizing = true;
            this.w2obj.resize();
            module.Figure.prototype.resize.call(this);
            this._resizing = false;
        }
    }
    module.W2Figure.prototype.add = function(axcls, where, instant) {
        var args = Array.prototype.slice.call(arguments).slice(3);
        args.unshift(this);

        var axes = module.construct(axcls, args);
        this.w2obj.show(where, instant);
        this.w2obj.content(where, axes.object);
        if (axes instanceof module.Figure) {
            axes.init();
        }
        this.axes[where] = axes;
        this.ax = axes;

        if (this.ax.ui !== undefined)
            this.ax.ui.init(this.gui);

        return axes;
    }
    module.W2Figure.prototype.show = function(where, instant) {
        this.w2obj.show(where, instant);
    }
    module.W2Figure.prototype.hide = function(where, instant) {
        this.w2obj.hide(where, instant);
    }
    module.W2Figure.prototype.toggle = function(where, instant) {
        this.w2obj.toggle(where, instant);
    }
    module.W2Figure.prototype.setSize = function(where, size) {
        if (typeof(size) == "string" && size[size.length-1] == '%') {
            var prop = parseFloat(size.slice(0, size.length-1)) / 100;
            size = $(this.object).width() * prop;
        }
        this.w2obj.set(where, {size:size});
        this.w2obj.resize();
    }
    module.W2Figure.prototype.getSize = function(where) {
        return this.w2obj.get(where).size;
    }

    module.GridFigure = function(parent, nrows, ncols) {
        module.Figure.call(this, parent);

        this.nrows = nrows;
        this.ncols = ncols;

        this.axes = [];
        this.cells = [];
        var table = document.createElement("table");
        this.object.appendChild(table);
        for (var i = 0; i < nrows; i++) {
            var tr = document.createElement("tr");
            tr.style.height = (100 / nrows)+"%";
            table.appendChild(tr);
            for (var j = 0; j < ncols; j++) {
                var td = document.createElement('td');
                td.style.width = (100 / ncols)+'%';
                //td.style.height = "100%";
                tr.appendChild(td);
                this.cells.push(td);
                this.axes.push(null);
            }
        }
    }
    module.GridFigure.prototype = Object.create(module.Figure.prototype);
    module.GridFigure.prototype.constructor = module.GridFigure;
    module.GridFigure.prototype.add = function(axcls, where) {
        var args = Array.prototype.slice.call(arguments).slice(2);
        args.unshift(this);
        this.ax = module.construct(axcls, args);
        this.axes[where] = this.ax;
        this.cells[where].appendChild(this.ax.object);
        return this.ax;
    }


    module.Axes = function(figure) {
        this.figure = figure;
        if (this.object === undefined) {
            this.object = document.createElement("div");
            this.object.className = "jsplot_axes";

            this.figure.addEventListener("resize", this.resize.bind(this));
        }

        // color legend
        function formatState (state) {
            if (!state.id) { return state.text; }
            var $state = $('<span class="colorlegend-option"><img class="colorlegend-option-image" src="' + colormaps[state.text].image.currentSrc + '" class="img-flag" />' + state.text + '</span>');
            return $state;
        };
        $(document).ready(function() {
            var selector = $(".colorlegend-select").select2({
                templateResult: formatState
            });
            $("#colorlegend-colorbar").on('click', function() {
                selector.show();
                selector.select2('open');
            });
            $('#brain').on('click', function () { selector.select2("close"); })
        });

    }
    THREE.EventDispatcher.prototype.apply(module.Axes.prototype);
    module.Axes.prototype.resize = function() {}

    module.MovieAxes = function(figure, url) {
        module.Axes.call(this, figure);
        $(this.object).html($("#movieaxes_html").html());
        this._target = null;
        var types = { 
            ogv: 'video/ogg; codecs="theora, vorbis"', 
            webm: 'video/webm; codecs="vp8, vorbis"',
            mp4: 'video/mp4; codecs="h264, aac"'
        }
        var src = $(this.object).find("source");
        var ext = url.match(/^(.*)\.(\w{3,4})$/);
        src.attr("type", types[ext]);
        src.attr("src", url);

        this.loadmsg = $(this.object).find("div.movie_load");
        this.movie = $(this.object).find("video")[0];

        this._update_func = function() {
            this.figure.notify("playsync", this, [this.movie.currentTime]);
        }.bind(this);
        this._progress_func = function() {
            if (this._target != null && 
                this.movie.seekable.length > 0 && 
                this.movie.seekable.end(0) >= this._target &&
                this.movie.parentNode != null) {
                var func = function() {
                    try {
                        this.movie.currentTime = this._target;
                        this._target = null;
                        this.loadmsg.hide()
                    } catch (e) {
                        console.log(e);
                        setTimeout(func, 5);
                    }
                }.bind(this);
                func();
            }
        }.bind(this);
        this.movie.addEventListener("timeupdate", this._update_func);
        this.movie.addEventListener("progress", this._progress_func);
        this.figure.register("playtoggle", this, this.playtoggle.bind(this));
        this.figure.register("setFrame", this, this.setFrame.bind(this));
    }
    module.MovieAxes.prototype = Object.create(module.Axes.prototype);
    module.MovieAxes.prototype.constructor = module.MovieAxes;
    module.MovieAxes.prototype.destroy = function() {
        this.figure.unregister("playtoggle", this);
        this.figure.unregister("setFrame", this);
        this.movie.removeEventListener("timeupdate", this._update_func);
    }
    module.MovieAxes.prototype.setFrame = function(time) {
        if (this.movie.seekable.length > 0 && 
            this.movie.seekable.end(0) >= time) {
            this.movie.currentTime = time;
            this.loadmsg.hide()
        } else {
            this._target = time;
            this.loadmsg.show()
        }
    }
    module.MovieAxes.prototype.playtoggle = function(state) {
        if (!this.movie.paused && state == "pause")
            this.movie.pause();
        else
            this.movie.play();
        this.figure.notify("playtoggle", this, [this.movie.paused?"pause":"play"]);
    }

    // Timeseries panel: overlays the picked voxel's timecourse for every
    // checked dataset channel fetched on demand from the /timeseries handler. 
    module.TimeseriesAxes = function(figure, viewer) {
        module.Axes.call(this, figure);
        this.viewer = viewer || null;
        this.object.style.backgroundColor = this.style.bg;
        this.object.style.width = "100%";
        this.object.style.height = "100%";
        this.object.style.display = "flex";
        this.object.style.flexDirection = "column";

        // control strip: one group per trace
        this.controls = document.createElement("div");
        var cs = this.controls.style;
        cs.display = "flex";
        cs.flexWrap = "wrap";
        cs.alignItems = "center";
        cs.gap = "14px";
        cs.padding = "3px 10px";
        cs.minHeight = "20px";
        cs.font = this.style.font;
        cs.backgroundColor = this.style.bg;
        cs.flex = "0 0 auto";
        this.object.appendChild(this.controls);

        // display mode: raw values or per-trace z-scores
        this.mode = "raw";
        var sel = document.createElement("select");
        sel.style.background = this.style.bg;
        sel.style.color = this.style.text;
        sel.style.border = "1px solid " + this.style.spine;
        sel.style.borderRadius = "3px";
        sel.style.font = this.style.font;
        ["raw", "z-scored"].forEach(function(mname) {
            var o = document.createElement("option");
            o.value = mname;
            o.textContent = mname;
            sel.appendChild(o);
        });
        sel.addEventListener("change", function() {
            this.mode = sel.value === "z-scored" ? "z" : "raw";
            this.draw();
        }.bind(this));
        this.controls.appendChild(sel);

        this.canvas = document.createElement("canvas");
        this.canvas.style.display = "block";
        this.canvas.style.width = "100%";
        this.canvas.style.flex = "1 1 auto";
        this.canvas.style.minHeight = "0";
        this.object.appendChild(this.canvas);

        // name -> {type: 'data'|'ref', resp, ref, channels: [{on, color}]}
        this.traces = {};
        this.order = [];
        this.label = "";
        this.message = "Click a voxel to plot its timeseries";
        this.frame = null;  // playhead position, in data frames
        this._xmap = null;  // plot x-geometry of the last draw, for seeking

        // click a timepoint -> seek the brain to that volume
        this.canvas.addEventListener("click", function(evt) {
            if (!this._xmap || !this.viewer)
                return;
            var rect = this.canvas.getBoundingClientRect();
            var fx = (evt.clientX - rect.left - this._xmap.x0) / this._xmap.w;
            if (fx < 0 || fx > 1)
                return;
            this.viewer.seekFrame(Math.round(fx * (this._xmap.n - 1)));
        }.bind(this));

        // one control group per movie dataset (3D views have no
        // timecourse); the active one starts checked, or the first movie
        // if the active view is a plain 3D volume
        if (viewer && viewer.dataviews) {
            var names = Object.keys(viewer.dataviews);
            var movies = names.filter(function(nm) {
                return viewer.dataviews[nm].frames > 1;
            });
            var def = (viewer.active && viewer.active.frames > 1)
                ? viewer.active.name : movies[0];
            for (var i = 0; i < movies.length; i++) {
                var nch = viewer.dataviews[movies[i]].data[0].raw ? 3 : 1;
                this.addTrace(movies[i], "data", movies[i] === def, nch);
            }
        }
        setTimeout(this.resize.bind(this), 0);
    }
    module.TimeseriesAxes.prototype = Object.create(module.Axes.prototype);
    module.TimeseriesAxes.prototype.constructor = module.TimeseriesAxes;
    module.TimeseriesAxes.prototype.style = {
        bg: "#0D1117", text: "#E8ECF5", muted: "#9AA3B5",
        spine: "#3A4250", play: "#FFB454",
        font: "11px sans-serif",
        dataColors: ["#6FA8FF", "#FF6B6B", "#5DD97C", "#FFB454", "#B48EAD", "#66D9E8"],
        refColors: ["#C8A96E", "#B48EAD", "#8FBCBB", "#D08770"],
        // per-channel defaults for RGB datasets: the first gets true
        // R/G/B, later ones get shifted triads so overlays stay readable
        rgbTriads: [["#FF6B6B", "#5DD97C", "#6FA8FF"],
                    ["#FFA94D", "#3BC9DB", "#B197FC"],
                    ["#F783AC", "#A9E34B", "#748FFC"]],
    };
    module.TimeseriesAxes.prototype._anyOn = function(t) {
        for (var i = 0; i < t.channels.length; i++)
            if (t.channels[i].on)
                return true;
        return false;
    }
    module.TimeseriesAxes.prototype.addTrace = function(name, type, on, nchan) {
        if (this.traces[name])
            return this.traces[name];
        var S = this.style;
        nchan = nchan || 1;
        var nScalar = 0, nRGB = 0, nRef = 0;
        for (var i = 0; i < this.order.length; i++) {
            var o = this.traces[this.order[i]];
            if (o.type === "ref") nRef++;
            else if (o.channels.length === 3) nRGB++;
            else nScalar++;
        }
        var colors;
        if (type === "ref")
            colors = [S.refColors[nRef % S.refColors.length]];
        else if (nchan === 3)
            colors = S.rgbTriads[nRGB % S.rgbTriads.length];
        else
            colors = [S.dataColors[nScalar % S.dataColors.length]];

        var t = {type: type, resp: null, ref: null, channels: []};
        for (var c = 0; c < nchan; c++)
            t.channels.push({on: !!on, color: colors[c]});
        this.traces[name] = t;
        this.order.push(name);

        var group = document.createElement("span");
        group.style.display = "flex";
        group.style.alignItems = "center";
        group.style.gap = "5px";
        group.style.font = S.font;
        group.style.color = type === "ref" ? S.muted : S.text;
        var txt = document.createElement("span");
        txt.textContent = name;
        group.appendChild(txt);

        var chanNames = nchan === 3 ? ["R", "G", "B"] : [""];
        var self = this;
        t.channels.forEach(function(ch, ci) {
            var pair = document.createElement("label");
            pair.style.display = "flex";
            pair.style.alignItems = "center";
            pair.style.gap = "2px";
            pair.style.cursor = "pointer";
            var cb = document.createElement("input");
            cb.type = "checkbox";
            cb.checked = ch.on;
            cb.style.margin = "0";
            cb.addEventListener("change", function() {
                ch.on = cb.checked;
                if (ch.on && t.type === "data" && !t.resp)
                    self.refetch();
                self.draw();
            });
            var col = document.createElement("input");
            col.type = "color";
            col.value = ch.color;
            col.style.width = "15px";
            col.style.height = "15px";
            col.style.padding = "0";
            col.style.border = "none";
            col.style.background = "none";
            col.style.cursor = "pointer";
            col.addEventListener("input", function() {
                ch.color = col.value;
                self.draw();
            });
            pair.appendChild(cb);
            pair.appendChild(col);
            if (chanNames[ci])
                pair.appendChild(document.createTextNode(chanNames[ci]));
            group.appendChild(pair);
        });
        this.controls.appendChild(group);
        return t;
    }
    module.TimeseriesAxes.prototype.refetch = function() {
        if (this.viewer && this.viewer._tsCoords)
            this.viewer.fetchTimeseries(this.viewer._tsCoords);
    }
    module.TimeseriesAxes.prototype.resize = function() {
        var dpr = window.devicePixelRatio || 1;
        this.canvas.width = this.canvas.clientWidth * dpr;
        this.canvas.height = this.canvas.clientHeight * dpr;
        this.draw();
    }
    module.TimeseriesAxes.prototype.setMessage = function(msg) {
        this.message = msg;
        this.draw();
    }
    module.TimeseriesAxes.prototype.update = function(name, resp, label) {
        var t = this.addTrace(name, "data", true, resp.data.length);
        t.resp = resp;
        this.label = label;
        // register reference traces (design-matrix regressors); they start
        // unchecked so QC stays uncluttered by default
        if (resp.refs) {
            for (var rname in resp.refs) {
                var rt = this.addTrace(rname, "ref", false, 1);
                rt.ref = resp.refs[rname];
            }
        }
        this.draw();
    }
    module.TimeseriesAxes.prototype.setFrame = function(frame) {
        this.frame = frame;
        this.draw();
    }
    module.TimeseriesAxes.prototype._zscore = function(series) {
        var i, m = 0;
        for (i = 0; i < series.length; i++)
            m += series[i];
        m /= series.length;
        var sd = 0;
        for (i = 0; i < series.length; i++)
            sd += (series[i] - m) * (series[i] - m);
        sd = Math.sqrt(sd / series.length) || 1;
        var out = new Array(series.length);
        for (i = 0; i < series.length; i++)
            out[i] = (series[i] - m) / sd;
        return out;
    }
    // matplotlib-style "nice" tick locations: steps of 1/2/5 x 10^k
    module.TimeseriesAxes.prototype._ticks = function(lo, hi, target) {
        var span = hi - lo;
        if (!(span > 0))
            return [lo];
        var step = Math.pow(10, Math.floor(Math.log(span / target) / Math.LN10));
        var err = target / (span / step);
        if (err <= 0.15) step *= 10;
        else if (err <= 0.35) step *= 5;
        else if (err <= 0.75) step *= 2;
        var out = [];
        for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step)
            out.push(Math.abs(v) < 1e-12 ? 0 : parseFloat(v.toPrecision(6)));
        return out;
    }
    module.TimeseriesAxes.prototype.draw = function() {
        var S = this.style;
        var dpr = window.devicePixelRatio || 1;
        var W = this.canvas.width / dpr, H = this.canvas.height / dpr;
        if (W < 10 || H < 10)
            return;
        var ctx = this.canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = S.bg;
        ctx.fillRect(0, 0, W, H);
        ctx.font = S.font;
        this._xmap = null;

        var dataAct = [], refAct = [];
        for (var i = 0; i < this.order.length; i++) {
            var name = this.order[i], t = this.traces[name];
            if (!this._anyOn(t))
                continue;
            if (t.type === "data" && t.resp)
                dataAct.push({name: name, t: t});
            else if (t.type === "ref" && t.ref)
                refAct.push({name: name, t: t});
        }
        if (dataAct.length === 0) {
            ctx.fillStyle = S.muted;
            ctx.textAlign = "center";
            ctx.fillText(this.message, W / 2, H / 2);
            return;
        }

        var lead = dataAct[0].t.resp;
        var n = lead.data[0].length;
        var zmode = this.mode === "z";
        var pad = {l: 52, r: 14, t: 20, b: 28};

        // one labeled color strip per checked RGB dataset, stacked
        var strips = [];
        for (var di = 0; di < dataAct.length; di++)
            if (dataAct[di].t.resp.data.length === 3)
                strips.push(dataAct[di]);
        var stripH = 12, stripGap = 2;
        var y0 = pad.t + (strips.length ? strips.length * (stripH + stripGap) + 3 : 0);
        var plotH = H - y0 - pad.b;
        var x0 = pad.l, w = W - pad.l - pad.r;

        // assemble the visible channel lines, applying the mode transform
        var lines = [];   // {series, color, lw, alpha, own}
        for (var di = 0; di < dataAct.length; di++) {
            var t = dataAct[di].t;
            for (var c = 0; c < t.channels.length; c++) {
                if (!t.channels[c].on)
                    continue;
                lines.push({series: zmode ? this._zscore(t.resp.data[c])
                                          : t.resp.data[c],
                            color: t.channels[c].color,
                            lw: 1.5, alpha: 1, own: false});
            }
        }
        for (var ri = 0; ri < refAct.length; ri++)
            lines.push({series: refAct[ri].t.ref,
                        color: refAct[ri].t.channels[0].color,
                        lw: 1.1, alpha: 0.7, own: true});

        // shared y-range across all data channels: raw mode shows true
        // values, z mode shows z units (reference traces stay min-max
        // scaled — their units are arbitrary)
        var mn = Infinity, mx = -Infinity;
        for (var li = 0; li < lines.length; li++) {
            if (lines[li].own)
                continue;
            mn = Math.min(mn, Math.min.apply(null, lines[li].series));
            mx = Math.max(mx, Math.max.apply(null, lines[li].series));
        }
        if (mn === mx) { mn -= 1; mx += 1; }
        var shared = [mn, mx];

        ctx.fillStyle = S.text;
        ctx.textAlign = "left";
        ctx.fillText(this.label, x0, 13);

        // strips: always blend all three channels — they show the color
        // actually painted on the brain, independent of line visibility
        for (var si = 0; si < strips.length; si++) {
            var sr = strips[si].t.resp;
            var sy = pad.t + si * (stripH + stripGap);
            var sn = sr.data[0].length;
            var segW = w / sn;
            for (var i = 0; i < sn; i++) {
                ctx.fillStyle = "rgb(" + Math.round(sr.data[0][i] * 255) + "," +
                    Math.round(sr.data[1][i] * 255) + "," +
                    Math.round(sr.data[2][i] * 255) + ")";
                ctx.fillRect(x0 + i * segW, sy, segW + 1, stripH);
            }
            ctx.fillStyle = S.muted;
            ctx.textAlign = "right";
            ctx.fillText(strips[si].name, x0 - 6, sy + stripH - 2);
        }

        ctx.strokeStyle = S.spine;
        ctx.strokeRect(x0 + .5, y0 + .5, w, plotH);

        // x axis is the volume index — QC thinks in frames, not seconds
        var xticks = this._ticks(0, n - 1, 6);
        ctx.fillStyle = S.muted;
        ctx.textAlign = "center";
        ctx.strokeStyle = S.spine;
        for (var xi = 0; xi < xticks.length; xi++) {
            var tx = x0 + (xticks[xi] / (n - 1)) * w;
            ctx.beginPath();
            ctx.moveTo(tx, y0 + plotH);
            ctx.lineTo(tx, y0 + plotH + 4);
            ctx.stroke();
            ctx.fillText(xticks[xi], tx, y0 + plotH + 14);
        }
        ctx.fillText("volume", x0 + w / 2, H - 3);

        // y ticks in real units (raw) or z units (z-scored), plus a rotated
        // axis label naming the mode
        var yticks = this._ticks(shared[0], shared[1], 4);
        ctx.textAlign = "right";
        for (var yi = 0; yi < yticks.length; yi++) {
            var ty = y0 + (1 - (yticks[yi] - shared[0]) / (shared[1] - shared[0])) * plotH;
            ctx.beginPath();
            ctx.moveTo(x0 - 4, ty);
            ctx.lineTo(x0, ty);
            ctx.stroke();
            ctx.fillText(yticks[yi], x0 - 6, ty + 3.5);
        }
        ctx.save();
        ctx.translate(11, y0 + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillText(zmode ? "z-scored" : "raw", 0, 0);
        ctx.restore();

        for (var li = 0; li < lines.length; li++) {
            var L = lines[li];
            var py;
            if (L.own) {
                var lmn = Math.min.apply(null, L.series);
                var lmx = Math.max.apply(null, L.series);
                if (lmn === lmx) { lmn -= 1; lmx += 1; }
                py = (function(a, b) {
                    return function(v) { return y0 + (1 - (v - a) / (b - a)) * plotH; };
                })(lmn, lmx);
            } else {
                py = (function(a, b) {
                    return function(v) { return y0 + (1 - (v - a) / (b - a)) * plotH; };
                })(shared[0], shared[1]);
            }
            ctx.strokeStyle = L.color;
            ctx.globalAlpha = L.alpha;
            ctx.lineWidth = L.lw;
            ctx.lineJoin = "round";
            ctx.beginPath();
            for (var i = 0; i < L.series.length; i++) {
                var lx = x0 + (i / (L.series.length - 1)) * w;
                i ? ctx.lineTo(lx, py(L.series[i])) : ctx.moveTo(lx, py(L.series[i]));
            }
            ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.lineWidth = 1;
        }

        if (this.frame !== null) {
            var fx = x0 + (Math.min(this.frame, n - 1) / (n - 1)) * w;
            ctx.strokeStyle = S.play;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(fx, pad.t);
            ctx.lineTo(fx, H - pad.b);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        this._xmap = {x0: x0, w: w, n: n};
    }

    module.ImageAxes = function(figure) {
        module.Axes.call(this, figure);
    }
    module.ImageAxes.prototype = Object.create(module.Axes.prototype);
    module.ImageAxes.prototype.constructor = module.ImageAxes;
    module.ImageAxes.prototype.set = function(url) {
        $(this.object).fadeTo(0);
        var img = new Image();
        img.onload = function() {
            $(this.object).html(img);
            $(this.object).fadeTo(1);
        }.bind(this);
        img.src = url;
    }

    return module;
}(jsplot || {}));