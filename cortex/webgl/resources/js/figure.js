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