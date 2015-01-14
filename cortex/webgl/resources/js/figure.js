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

    module.Menu = function(gui) {
        this._gui = gui;
        this._desc = {};
        this._folders = {};
        this._controls = {};
        this._hidden = false; //the initial state of this folder
        this._bubble_evt = function() {
            this.dispatchEvent({type:"update"});
        }.bind(this);
    }
    THREE.EventDispatcher.prototype.apply(module.Menu.prototype);
    module.Menu.prototype.init = function(gui) {
        this._gui = gui;
        for (var name in this._desc) {
            if (this._folders[name] === undefined && this._desc[name] instanceof module.Menu) {
                var hidden = this._desc[name]._hidden;
                this._folders[name] = this._addFolder(gui, name, hidden);
                this[name].init(this._folders[name]);
            } else if (this._controls[name] === undefined) {
                this._controls[name] = this._add(gui, name, this._desc[name]);
            }
        }
    }
    module.Menu.prototype.set = function(name, value) {
        var namespace = name.split('.');
        var n = namespace.shift();
        if (namespace.length > 0) {
            return this[n].set(namespace.join('.'), value);
        } else if (this[n] === undefined) {
            var action = this._desc[n].action;
            action[0][action[1]] = value;
        } else {
            this[n] = value;
        }
        this._controls[n].updateDisplay();
        this.dispatchEvent({type:"update"});
    }
    module.Menu.prototype.get = function(name) {
        var namespace = name.split('.');
        var n = namespace.shift();
        if (namespace.length > 0) {
            return this[n].get(namespace.join('.'));
        } else if (this[n] === undefined) {
            var action = this._desc[n].action;
            return action[0][action[1]];
        } else {
            return this[n];
        }
    }
    module.Menu.prototype.addFolder = function(name, hidden, menu) {
        if (menu === undefined)
            menu = new module.Menu();
        menu._hidden = hidden;
        this._desc[name] = menu;
        //cause update events to bubble
        menu.addEventListener("update", this._bubble_evt);

        if (this._gui !== undefined) {
            this._folders[name] = this._addFolder(this._gui, name, hidden);
            menu.init(this._folders[name]);
        }

        return menu;
    }
    module.Menu.prototype.add = function(desc) {
        for (var name in desc) {
            var args = desc[name];
            this._desc[name] = args;

            if (this._gui !== undefined) {
                this._controls[name] = this._add(this._gui, name, args);
            }
        }
        return this;
    }
    module.Menu.prototype.remove = function(name) {
        if (name === undefined) {
            for (var n in this._desc) {
                this._remove(n);
            }
        } else if (name instanceof module.Menu){
            for (var n in this._desc) {
                if (this._desc[n] === name)
                    this._remove(n);
            }
        } else {
            this._remove(name);
        }
    }

    module.Menu.prototype._addFolder = function(gui, name, hidden) {
        var folder = gui.addFolder(name);
        if (!hidden)
            folder.open();
        this[name] = this._desc[name];
        return folder;
    }
    module.Menu.prototype._add = function(gui, name, desc) {
        if (desc.action instanceof Function) {
            //A button that runs a function (IE Reset)
        } else if ( desc.action instanceof Array) {
            var obj = desc.action[0][desc.action[1]];
            if (obj instanceof Function) {;
                //copy the args to avoid changing the desc
                var parent = desc.action[0];
                var method = desc.action[1];
                var newargs = [this, name];
                for (var i = 2; i < desc.action.length; i++)
                    newargs.push(desc.action[i]);
                this[name] = parent[method]();

                var ctrl = gui.add.apply(gui, newargs);
                ctrl.onChange(function(name) {
                    parent[method](this[name]);
                    this.dispatchEvent({type:"update"});
                }.bind(this, name));
                
                //replace the function so that calling it will update the control
                ctrl._oldfunc = parent[method]; //store original so we can replace
                var func = parent[method].bind(parent);
                parent[method] = function(name, val) {
                    func(val);
                    this[name] = val;
                    ctrl.updateDisplay();
                }.bind(this, name)
            } else {
                var ctrl = gui.add.apply(gui, desc.action).name(name);
                ctrl.onChange(function() {
                    this.dispatchEvent({type:"update"});
                }.bind(this));
            }
        }
        return ctrl;
    }
    module.Menu.prototype._remove = function(name) {
        if (this._desc[name] instanceof module.Menu) {
            this[name].remove();
            this[name].removeEventListener(this._bubble_evt);
            delete this._folders[name];
            if (this._gui !== undefined)
                this._gui.removeFolder(name);
        } else {
            if (this._controls[name]._oldfunc !== undefined) {
                var args = this._desc[name].action;
                var parent = args[0], func = args[1];
                parent[func] = this._controls[name]._oldfunc;
            }
            if (this._gui !== undefined)
                this._gui.remove(this._controls[name]);
            delete this._controls[name];
        }
    }


    module.Figure = function(parent) {
        this._notifying = false;
        this.axes = [];
        this.ax = null;

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
            this.addEventListener(eventType, function(evt) { 
                if (evt.self != self)
                    func.apply(self, evt.args);
            }.bind(this));
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

        this.movie.addEventListener("progress", function() {
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
        }.bind(this));

        this.movie.addEventListener("timeupdate", function() {
            this.figure.notify("playsync", this, [this.movie.currentTime]);
        }.bind(this));
        this.figure.register("playtoggle", this, this.playtoggle.bind(this));
        this.figure.register("setFrame", this, this.setFrame.bind(this));
    }
    module.MovieAxes.prototype = Object.create(module.Axes.prototype);
    module.MovieAxes.prototype.constructor = module.MovieAxes;
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
    module.MovieAxes.prototype.playtoggle = function() {
        if (this.movie.paused)
            this.movie.play();
        else
            this.movie.pause();
        this.figure.notify("playtoggle", this);
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