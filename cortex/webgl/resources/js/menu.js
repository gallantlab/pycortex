var jsplot = (function (module) {
    module.Menu = function(gui) {
        this._gui = gui;
        this._desc = {};
        this._folders = {};
        this._controls = {};
        this._hidden = false; //the initial state of this folder
        this._bubble_evt = function() {
            this.dispatchEvent({type:"update"});
        }.bind(this);
        this._wheelListeners = {};
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
            if (action[0][action[1]] instanceof Function)
                action[0][action[1]].call(action[0], value)
            else 
                action[0][action[1]] = value;
        } else {
            this[n] = value;
            var action = this._desc[n].action;
            action[0][action[1]](value);
        }
        if (this._controls[n])
            this._controls[n].updateDisplay();
        this.dispatchEvent({type:"update"});
    }
    module.Menu.prototype.get = function(name) {
        var namespace = name.split('.');
        var n = namespace.shift();
        if (namespace.length > 0) {
            return this[n].get(namespace.join('.'));
        } else if (this[n] === undefined) { //object reference
            var action = this._desc[n].action;
            if (action[0][action[1]] instanceof Function)
                return action[0][action[1]].call(action[0]);
            return action[0][action[1]];
        } else { // object function
            var action = this._desc[n].action;
            if (action[0][action[1]] instanceof Function)
                return action[0][action[1]]();
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
            this[name] = desc.action;
            if (!desc.hidden) 
                gui.add(desc, "action").name(name);
        } else if ( desc.action instanceof Array) {
            var obj = desc.action[0][desc.action[1]];
            if (obj instanceof Function) {
                var parent = desc.action[0];
                var method = desc.action[1];
                this[name] = parent[method]();

                if (!desc.hidden) {
                    //copy the args to avoid changing the desc
                    var newargs = [this, name];
                    for (var i = 2; i < desc.action.length; i++)
                        newargs.push(desc.action[i]);

                    var ctrl = gui.add.apply(gui, newargs);
                    ctrl.onChange(function(name) {
                        parent[method](this[name]);
                        this.dispatchEvent({type:"update"});
                    }.bind(this, name));
                    
                    //replace the function so that calling it will update the control
                    ctrl._oldfunc = parent[method]; //store original so we can replace
                    var func = parent[method].bind(parent);
                    parent[method] = function(name, val) {
                        if (val === undefined)
                            return func();
                        
                        func(val);
                        this[name] = val;
                        ctrl.updateDisplay();
                    }.bind(this, name)
                };
            } else if (!desc.hidden) {
                var ctrl = gui.add.apply(gui, desc.action).name(name);
                ctrl.onChange(function() {
                    this.dispatchEvent({type:"update"});
                }.bind(this));
            }
        }
        //setup keyboard shortcuts for commands
        if (desc.key) {
            var key = desc.key;
            var action = desc.action;
            window.addEventListener("keypress", function(event) {

                if (window.colorlegendOpen) {
                    return;
                }

                if (event.target.nodeName === 'INPUT' && (event.target.id !== "" || (event.target.hasOwnProperty('classList') && event.target.classList.index("select2-search__field") ==! -1))) {
                    // note: normally you would want to block on all INPUT target tags. however, if you tab-key away from an input element, INPUT remains the target even if the element has been manually deblurred, but the id *will* be cleared. since it would be nice to be able to use shortcuts after tab-aways, this statement only blocks events from inputs with ids
                    return;
                }
                if (event.defaultPrevented)
                    return;

                if (desc.hasOwnProperty('modKeys')) {
                    for (let modKey of desc.modKeys) {
                        if (!event[modKey]) {
                            return
                        }
                    }
                }

                if (event.key == key) {
                    action();
                    event.preventDefault();
                    event.stopPropagation();
                }
            }.bind(this), true);
        }

        // setup mousewheel shortcuts
        if (desc.wheel && !(desc.help in this._wheelListeners)) {
            this._wheelListeners[desc.help] = window.addEventListener("wheel", function(event) {
                if (desc.hasOwnProperty('modKeys')) {
                    for (let modKey of desc.modKeys) {
                        if (!event[modKey]) {
                            return
                        }
                    }
                }

                event.preventDefault();
                event.stopPropagation();

                desc.action(event.deltaY);

                for (var i in gui.__controllers) {
                    gui.__controllers[i].updateDisplay();
                }
            }.bind(this), true);
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
        } else if (this._controls[name] !== undefined) {
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
    return module;
}(jsplot || {}));