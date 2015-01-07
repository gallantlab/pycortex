var mriview = (function(module) {
    var grid_shapes = [null, [1,1], [2, 1], [3, 1], [2, 2], [2, 2], [3, 2], [3, 2]];
    module.Viewer = function(figure) { 
        jsplot.Axes.call(this, figure);
        //Initialize all the html
        $(this.object).html($("#mriview_html").html())
        //Catalog the available colormaps
        $(this.object).find("#colormap option").each(function() {
            var im = new Image();
            im.src = $(this).data("imagesrc");
            var tex = new THREE.Texture(im);
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            tex.premultiplyAlpha = true;
            tex.flipY = true;
            tex.needsUpdate = true;
            colormaps[$(this).text()] = tex;
        });
        this.canvas = $(this.object).find("#brain");
        jsplot.Axes3D.call(this, figure);

        this.surfs = [];
        this.dataviews = {};
        this.active = null;

        this.loaded = $.Deferred().done(function() {
            // this.schedule();
            $(this.object).find("#ctmload").hide();
            // this.canvas.css("opacity", 1);
        }.bind(this));

        this.ui = new jsplot.Menu();
        this.ui.addEventListener("update", this._schedule);

        this._bindUI();
    }
    module.Viewer.prototype = Object.create(jsplot.Axes3D.prototype);
    THREE.EventDispatcher.prototype.apply(module.Viewer.prototype);
    module.Viewer.prototype.constructor = module.Viewer;

    module.Viewer.prototype.drawView = function(scene, idx) {
        if (this.surfs[idx].prerender !== undefined)
            this.surfs[idx].prerender(this.renderer, scene, this.camera);

        for (var i = 0; i < this.surfs.length; i++)
            this.surfs[i].apply(this.active);
        if (this.oculus)
            this.oculus.render(scene, this.camera);
        else
            this.renderer.render(scene, this.camera);
    }

    module.Viewer.prototype.setOculus = function() {
        if (this.oculus) {
            this.removeEventListener("resize", this.oculus._resize);
            delete this.oculus;
        } else {
            this.oculus = new THREE.OculusRiftEffect(this.renderer, {worldScale:1000});
            this.oculus._resize = function(evt) {
                this.oculus.setSize(evt.width, evt.height);
            }.bind(this);
            this.addEventListener("resize", this.oculus._resize);
            this.resize();
        }
        this.schedule();
    }
    
    module.Viewer.prototype.animate = function(animation) {
        var state = {};
        var anim = [];
        animation.sort(function(a, b) { return a.idx - b.idx});
        for (var i = 0, il = animation.length; i < il; i++) {
            var f = animation[i];
            if (f.idx == 0) {
                this.setState(f.state, f.value);
                state[f.state] = {idx:0, val:f.value};
            } else {
                if (state[f.state] === undefined)
                    state[f.state] = {idx:0, val:this.getState(f.state)};
                var start = {idx:state[f.state].idx, state:f.state, value:state[f.state].val}
                var end = {idx:f.idx, state:f.state, value:f.value};
                state[f.state].idx = f.idx;
                state[f.state].val = f.value;
                if (start.value instanceof Array) {
                    var test = true;
                    for (var j = 0; test && j < start.value.length; j++)
                        test = test && (start.value[j] == end.value[j]);
                    if (!test)
                        anim.push({start:start, end:end, ended:false});
                } else if (start.value != end.value)
                    anim.push({start:start, end:end, ended:false});
            }
        }
        if (this.active.fastshader) {
            this.meshes.left.material = this.active.fastshader;
            this.meshes.right.material = this.active.fastshader;
        }
        this._animation = {anim:anim, start:new Date()};
        this.schedule();
    };
    module.Viewer.prototype._animate = function(sec) {
        var state = false;
        var idx, val, f, i, j;
        for (i = 0, il = this._animation.anim.length; i < il; i++) {
            f = this._animation.anim[i];
            if (!f.ended) {
                if (f.start.idx <= sec && sec < f.end.idx) {
                    idx = (sec - f.start.idx) / (f.end.idx - f.start.idx);
                    if (f.start.value instanceof Array) {
                        val = [];
                        for (j = 0; j < f.start.value.length; j++) {
                            //val.push(f.start.value[j]*(1-idx) + f.end.value[j]*idx);
                            val.push(this._animInterp(f.start.state, f.start.value[j], f.end.value[j], idx));
                        }
                    } else {
                        //val = f.start.value * (1-idx) + f.end.value * idx;
                        val = this._animInterp(f.start.state, f.start.value, f.end.value, idx);
                    }
                    this.setState(f.start.state, val);
                    state = true;
                } else if (sec >= f.end.idx) {
                    this.setState(f.end.state, f.end.value);
                    f.ended = true;
                } else if (sec < f.start.idx) {
                    state = true;
                }
            }
        }
        return state;
    };
    module.Viewer.prototype._animInterp = function(state, startval, endval, idx) {
        switch (state) {
            case 'azimuth':
                // Azimuth is an angle, so we need to choose which direction to interpolate
                if (Math.abs(endval - startval) >= 180) { // wrap
                    if (startval > endval) {
                        return (startval * (1-idx) + (endval+360) * idx + 360) % 360;
                    }
                    else {
                        return (startval * (1-idx) + (endval-360) * idx + 360) % 360;
                    }
                } 
                else {
                    return (startval * (1-idx) + endval * idx);
                }
            default:
                // Everything else can be linearly interpolated
                return startval * (1-idx) + endval * idx;
        }
    };

    module.Viewer.prototype.reset_view = function(center, height) {
        var asp = this.flatlims[1][0] / this.flatlims[1][1];
        var camasp = height !== undefined ? asp : this.camera.cameraP.aspect;
        var size = [module.flatscale*this.flatlims[1][0], module.flatscale*this.flatlims[1][1]];
        var min = [module.flatscale*this.flatlims[0][0], module.flatscale*this.flatlims[0][1]];
        var xoff = center ? 0 : size[0] / 2 - min[0];
        var zoff = center ? 0 : size[1] / 2 - min[1];
        var h = size[0] / 2 / camasp;
        h /= Math.tan(this.camera.fov / 2 * Math.PI / 180);
        this.controls.target.set(xoff, this.flatoff[1], zoff);
        this.controls.setCamera(180, 90, h);
        this.setMix(1);
        this.setShift(0);
    };
    // module.Viewer.prototype.update_volvis = function() {
    //     for (var i = 0; i < 3; i++) {
    //         this.planes[i].update();
    //         this.planes[i].mesh.visible = $("#volvis").prop("checked");
    //     }
    //     this.schedule();
    // };
    // module.Viewer.prototype.update_leftvis = function() {
    //     this.meshes.left.visible = $("#leftvis").prop("checked");
    //     this.schedule();
    // };
    // module.Viewer.prototype.update_rightvis = function() {
    //     this.meshes.right.visible = $("#rightvis").prop("checked");
    //     this.schedule();
    // };
    module.Viewer.prototype.update_projection = function() {
        if ($("#projpersp").prop("checked")) {
            this.setState("projection", "perspective");
        } else {
            this.setState("projection", "orthographic");
        }
        this.schedule();
    };
    // module.Viewer.prototype.saveflat = function(height, posturl) {
    //     var width = height * this.flatlims[1][0] / this.flatlims[1][1];;
    //     var roistate = $(this.object).find("#roishow").attr("checked");
    //     this.screenshot(width, height, function() { 
    //         this.reset_view(false, height); 
    //         $(this.object).find("#roishow").attr("checked", false);
    //         $(this.object).find("#roishow").change();
    //     }.bind(this), function(png) {
    //         $(this.object).find("#roishow").attr("checked", roistate);
    //         $(this.object).find("#roishow").change();
    //         this.controls.target.set(0,0,0);
    //         this.roipack.saveSVG(png, posturl);
    //     }.bind(this));
    // }; 

    module.Viewer.prototype.addData = function(data) {
        if (!(data instanceof Array))
            data = [data];

        var name, view;

        var handle = "<div class='handle'><span class='ui-icon ui-icon-carat-2-n-s'></span></div>";
        for (var i = 0; i < data.length; i++) {
            view = data[i];
            name = view.name;
            this.dataviews[name] = view;

            var found = false;
            $(this.object).find("#datasets li").each(function() {
                found = found || ($(this).text() == name);
            })
            if (!found)
                $(this.object).find("#datasets").append("<li class='ui-corner-all'>"+handle+name+"</li>");
        }
        
        this.setData(data[0].name);
    };

    module.Viewer.prototype.setData = function(name) {
        if (name instanceof Array) {
            if (name.length == 1) {
                name = name[0];
            } else if (name.length == 2) {
                var dv1 = this.dataviews[name[0]];
                var dv2 = this.dataviews[name[1]];
                //Can't create 2D data view when the view is already 2D!
                if (dv1.data.length > 1 || dv2.data.length > 1)
                    return false;
                //Can't create mixed volume/vertex renderer
                if (!(dv1.vertex ^ dv.vertex))
                    return false;

                return this.addData(dataset.makeFrom(dv1, dv2));
            } else {
                return false;
            }
        }

        //unbind the shader update event from the existing surfaces
        for (var i = 0; i < this.surfs.length; i++) {
            this.active.removeEventListener("update", this.surfs[i]._update);
            this.active.removeEventListener("attribute", this.surfs[i]._attrib);
        }
        //set the new active data and update shaders for all surfaces
        this.active = this.dataviews[name];
        if (this.surfs.length < 1) {
            var surf = this.addSurf(module.SurfDelegate);
        } else {
            if (this.active.vertex) {
                if (this.surfs.length > 1)
                    for (var i = 0; i < this.surfs.length; i++) {

                    }//delete all other surfaces, since vertex is tightly coupled to surface
            }
            for (var i = 0; i < this.surfs.length; i++) {
                this.surfs[i].update(this.active);
                this.active.addEventListener("update", this.surfs[i]._update);
                this.active.addEventListener("attribute", this.surfs[i]._attrib);
            }
        }
        this.active.loaded.done(function() { this.active.set(); }.bind(this));

        var surf, scene, grid = grid_shapes[this.active.data.length];
        //cleanup old scene grid for the multiview
        // for (var i = 0; i < this.views.length; i++) {
        //     this.views[i].scene.dispose();
        // }
        //Generate new scenes and re-add the surface objects
        for (var i = 0; i < this.active.data.length; i++) {
            scene = this.setGrid(grid[0], grid[1], i);
        }

        //Show or hide the colormap for raw / non-raw dataviews
        if (this.active.data[0].raw) {
            $("#color_fieldset").fadeTo(0.15, 0);
        } else {
            $("#color_fieldset").fadeTo(0.15, 1);
        }

        var defers = [];
        for (var i = 0; i < this.active.data.length; i++) {
            defers.push(subjects[this.active.data[i].subject].loaded)
        }
        $.when.apply(null, defers).done(function() {
            //unhide the main canvas object
            this.canvas[0].style.opacity = 1;

            // $(this.object).find("#vrange").slider("option", {min: this.active.data[0].min, max:this.active.data[0].max});
            // if (this.active.data.length > 1) {
            //     $(this.object).find("#vrange2").slider("option", {min: this.active.data[1].min, max:this.active.data[1].max});
            //     $(this.object).find("#vminmax2").show();
            // } else {
            //     $(this.object).find("#vminmax2").hide();
            //     this.setVminmax(this.active.vmin[0].value[0], this.active.vmax[0].value[0], 0);
            // }

            // this.setupStim();
            
            $(this.object).find("#datasets li").each(function() {
                if ($(this).text() == name)
                    $(this).addClass("ui-selected");
                else
                    $(this).removeClass("ui-selected");
            })

            $(this.object).find("#datasets").val(name);
            if (typeof(this.active.description) == "string") {
                var html = name+"<div class='datadesc'>"+this.active.description+"</div>";
                $(this.object).find("#dataname").html(html);
                $(this.object).find("#dataopts").show();
            } else {
                $(this.object).find("#dataname").text(name);
                $(this.object).find("#dataopts").show();
            }
            this.schedule();
        }.bind(this));
    };
    module.Viewer.prototype.nextData = function(dir) {
        var i = 0, found = false;
        var datasets = [];
        $(this.object).find("#datasets li").each(function() {
            if (!found) {
                if (this.className.indexOf("ui-selected") > 0)
                    found = true;
                else
                    i++;
            }
            datasets.push($(this).text())
        });
        if (dir === undefined)
            dir = 1

        this.setData([datasets[(i+dir).mod(datasets.length)]]);
    };
    module.Viewer.prototype.rmData = function(name) {
        delete this.datasets[name];
        $(this.object).find("#datasets li").each(function() {
            if ($(this).text() == name)
                $(this).remove();
        })
    };
    module.Viewer.prototype.addSurf = function(surftype, opts) {
        //Sets the slicing surface used to visualize the data
        var surf = new surftype(this.active, opts);
        surf.addEventListener("mix", function(evt){
            this.controls.update(evt.flat);
        }.bind(this));

        this.surfs.push(surf);
        this.root.add(surf.object);

        this.active.addEventListener("update", surf._update);
        this.active.addEventListener("attribute", surf._attrib);
        this.addEventListener("resize", surf._resize);

        if (surf.ui !== undefined) {
            this.ui.addFolder("Surface", surf.ui);
        }

        this.schedule();
        return surf;
    };
    module.Viewer.prototype.rmSurf = function(surftype) {
        for (var i = 0; i < this.surfs.length; i++) {
            if (this.surfs[i].constructor == surftype) {
                this.active.removeEventListener("update", this.surfs[i]._update);
                this.active.removeEventListener("attribute", this.surfs[i]._attrib);
                this.removeEventListener("resize", this.surfs[i]._resize);

                this.root.remove(this.surfs[i].object);
            }
        }
        this.schedule();
    }

    module.Viewer.prototype.pick = function(evt) {
        for (var i = 0; i < this.surfs.length; i++) {
            if (this.surfs[i].pick)
                this.surfs[i].pick(this.renderer, this.camera, evt.x, evt.y);
        }
    }

    module.Viewer.prototype.setupStim = function() {
        if (this.active.data[0].movie) {
            $(this.object).find("#moviecontrols").show();
            $(this.object).find("#bottombar").addClass("bbar_controls");
            $(this.object).find("#movieprogress>div").slider("option", {min:0, max:this.active.length});
            this.active.data[0].loaded.progress(function(idx) {
                var pct = idx / this.active.frames * 100;
                $(this.object).find("#movieprogress div.ui-slider-range").width(pct+"%");
            }.bind(this)).done(function() {
                $(this.object).find("#movieprogress div.ui-slider-range").width("100%");
            }.bind(this));
            
            this.active.loaded.done(function() {
                this.setFrame(0);
            }.bind(this));

            if (this.active.stim && figure) {
                figure.setSize("right", "30%");
                this.movie = figure.add(jsplot.MovieAxes, "right", false, this.active.stim);
                this.movie.setFrame(0);
            }
        } else {
            $(this.object).find("#moviecontrols").hide();
            $(this.object).find("#bottombar").removeClass("bbar_controls");
        }
        this.schedule();
    };

    module.Viewer.prototype.setVminmax = function(vmin, vmax, dim) {
        if (dim === undefined)
            dim = 0;
        var range, min, max;
        if (dim == 0) {
            range = "#vrange"; min = "#vmin"; max = "#vmax";
        } else {
            range = "#vrange2"; min = "#vmin2"; max = "#vmax2";
        }

        if (vmax > $(this.object).find(range).slider("option", "max")) {
            $(this.object).find(range).slider("option", "max", vmax);
            this.active.data[dim].max = vmax;
        } else if (vmin < $(this.object).find(range).slider("option", "min")) {
            $(this.object).find(range).slider("option", "min", vmin);
            this.active.data[dim].min = vmin;
        }
        $(this.object).find(range).slider("values", [vmin, vmax]);
        $(this.object).find(min).val(vmin);
        $(this.object).find(max).val(vmax);

        this.active.setVminmax(vmin, vmax, dim);

        this.schedule();
    };

    module.Viewer.prototype.startCmapSearch = function() {
        var sr = $(this.object).find("#cmapsearchresults"),
        cm = $(this.object).find("#colormap"),
        sb = $(this.object).find("#cmapsearchbox"),
        v = this;
        
        sb.val("");
        sb.css("width", cm.css("width"));
        sb.css("height", cm.css("height"));
        sr.show();
        sb.keyup(function(e) {
            if (e.keyCode == 13) { // enter
                try {this.setColormap($(sr[0]).find(".selected_sr input")[0].value);}
                finally {this.stopCmapSearch();}
            } if (e.keyCode == 27) { // escape
                this.stopCmapSearch();
            }
            var value = sb[0].value.trim();
            sr.empty();
            if (value.length > 0) {
                for (var k in this.cmapnames) {
                    if (k.indexOf(value) > -1) {
                        sr.append($($(this.object).find("#colormap li")[this.cmapnames[k]]).clone());
                    }
                }
                $(sr[0].firstChild).addClass("selected_sr");
                sr.children().mousedown(function() {
                    try {v.setColormap($(this).find("input")[0].value);}
                    finally {v.stopCmapSearch();}
                });
            }
        }.bind(this)).show();
        sb.focus();
        sb.blur(function() {this.stopCmapSearch();}.bind(this));
    };

    module.Viewer.prototype.stopCmapSearch = function() {
        var sr = $(this.object).find("#cmapsearchresults"),
        sb = $(this.object).find("#cmapsearchbox");
        sr.hide().empty();
        sb.hide();
    };

    module.Viewer.prototype.setFrame = function(frame) {
        if (frame > this.active.length) {
            frame -= this.active.length;
            this._startplay += this.active.length;
        }

        this.frame = frame;
        this.active.setFrame(this.uniforms, frame);
        $(this.object).find("#movieprogress div").slider("value", frame);
        $(this.object).find("#movieframe").attr("value", frame);
        this.schedule();
    };

    module.Viewer.prototype.getImage = function(width, height, post) {
        if (width === undefined)
            width = this.canvas.width();
        
        if (height === undefined)
            height = width * this.canvas.height() / this.canvas.width();

        console.log(width, height);
        var renderbuf = new THREE.WebGLRenderTarget(width, height, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format:THREE.RGBAFormat,
            stencilBuffer:false,
        });

        var clearAlpha = this.renderer.getClearAlpha();
        var clearColor = this.renderer.getClearColor();
        var oldw = this.canvas.width(), oldh = this.canvas.height();
        this.camera.setSize(width, height);
        this.camera.updateProjectionMatrix();
        this.controls._zoom(1.0) // To assure orthographic zoom is set correctly
        //this.renderer.setSize(width, height);
        this.renderer.setClearColor(new THREE.Color(0,0,0), 0);
        this.renderer.render(this.scene, this.camera, renderbuf);
        //this.renderer.setSize(oldw, oldh);
        this.renderer.setClearColor(new THREE.Color(0,0,0), 1);
        this.camera.setSize(oldw, oldh);
        this.camera.updateProjectionMatrix();

        var img = mriview.getTexture(this.renderer.context, renderbuf)
        if (post !== undefined)
            $.post(post, {png:img.toDataURL()});
        return img;
    };

    var _bound = false;
    module.Viewer.prototype._bindUI = function() {
        $(window).scrollTop(0);
        $(window).resize(function() { this.resize(); }.bind(this));
        this.canvas.resize(function() { this.resize(); }.bind(this));
        //These are events that should only happen once, regardless of multiple views
        if (!_bound) {
            _bound = true;
            window.addEventListener( 'keydown', function(e) {
                btnspeed = 0.5;
                if (e.target.tagName == "INPUT" && e.target.type == "text")
                    return;
                if (e.keyCode == 32) {         //space
                    if (this.active.data[0].movie)
                        this.playpause();
                    e.preventDefault();
                    e.stopPropagation();
                } else if (e.keyCode == 82) { //r
                    this.animate([{idx:btnspeed, state:"target", value:[0,0,0]},
                                  {idx:btnspeed, state:"mix", value:0.0}]);
                } else if (e.keyCode == 73) { //i
                    this.animate([{idx:btnspeed, state:"mix", value:0.5}]);
                } else if (e.keyCode == 70) { //f
                    this.animate([{idx:btnspeed, state:"target", value:[0,0,0]},
                                  {idx:btnspeed, state:"mix", value:1.0}]);
                } else if (e.keyCode == 37) { //left
		    this.animate([{idx:btnspeed, state:"azimuth", value:(Math.floor(this.getState("azimuth")/90)+1)*90.5}]);
		} else if (e.keyCode == 39) { //right
		    this.animate([{idx:btnspeed, state:"azimuth", value:(Math.floor(this.getState("azimuth")/90)-1)*90.5}]);
		} else if (e.keyCode == 38) { //up
		    this.animate([{idx:btnspeed, state:"altitude", value:(Math.round(this.getState("altitude")/90)-1)*90.5}]);
		} else if (e.keyCode == 40) { //down
		    this.animate([{idx:btnspeed, state:"altitude", value:(Math.round(this.getState("altitude")/90)+1)*90.5}]);
		}
            }.bind(this));
        }
        window.addEventListener( 'keydown', function(e) {
            if (e.target.tagName == "INPUT" && e.target.type == "text")
                return;
            if (e.keyCode == 107 || e.keyCode == 187) { //+
                this.nextData(1);
            } else if (e.keyCode == 109 || e.keyCode == 189) { //-
                this.nextData(-1);
            } else if (e.keyCode == 68) { //d
                if (this.uniforms.dataAlpha.value < 1)
                    this.uniforms.dataAlpha.value = 1;
                else
                    this.uniforms.dataAlpha.value = 0;
                this.schedule();
            } else if (e.keyCode == 76) { //l
                var box = $(this.object).find("#labelshow");
                box.attr("checked", box.attr("checked") == "checked" ? null : "checked");
                this.labelshow = !this.labelshow;
                this.schedule();
                e.stopPropagation();
                e.preventDefault();
            }
        }.bind(this));
        var _this = this;

        if ($(this.object).find("#color_fieldset").length > 0) {
            $(this.object).find("#colormap").ddslick({ width:296, height:350, 
                onSelected: function() { 
                    var name = $(this.object).find("#colormap .dd-selected-text").text();
                    if (this.active) {
                        this.active.setColormap(name);
                        this.schedule();
                    }
                }.bind(this)
            });
            $(this.object).find("#cmapsearch").click(function() {
                this.startCmapSearch();
            }.bind(this));

            $(this.object).find("#vrange").slider({ 
                range:true, width:200, min:0, max:1, step:.001, values:[0,1],
                slide: function(event, ui) { 
                    $(this.object).find("#vmin").val(ui.values[0]);
                    $(this.object).find("#vmax").val(ui.values[1]);
                    this.active.setVminmax(ui.values[0], ui.values[1]);
                    this.schedule();
                }.bind(this)
            });
            $(this.object).find("#vmin").change(function() { 
                this.active.setVminmax(
                    parseFloat($(this.object).find("#vmin").val()), 
                    parseFloat($(this.object).find("#vmax").val())
                ); 
                this.schedule();
            }.bind(this));
            $(this.object).find("#vmax").change(function() { 
                this.active.setVminmax(
                    parseFloat($(this.object).find("#vmin").val()), 
                    parseFloat($(this.object).find("#vmax").val())
                    ); 
                this.schedule();
            }.bind(this));

            $(this.object).find("#vrange2").slider({ 
                range:true, width:200, min:0, max:1, step:.001, values:[0,1], orientation:"vertical",
                slide: function(event, ui) { 
                    $(this.object).find("#vmin2").value(ui.values[0]);
                    $(this.object).find("#vmax2").value(ui.values[1]);
                    this.active.setVminmax(ui.values[0], ui.values[1], 1);
                    this.schedule();
                }.bind(this)
            });
            $(this.object).find("#vmin2").change(function() { 
                this.active.setVminmax(
                    parseFloat($(this.object).find("#vmin2").val()), 
                    parseFloat($(this.object).find("#vmax2").val()),
                    1);
                this.schedule();
            }.bind(this));
            $(this.object).find("#vmax2").change(function() { 
                this.active.setVminmax(
                    parseFloat($(this.object).find("#vmin2").val()), 
                    parseFloat($(this.object).find("#vmax2").val()), 
                    1); 
                this.schedule();
            }.bind(this));            
        }
        
        //Dataset box
        var setdat = function(event, ui) {
            var names = [];
            $(this.object).find("#datasets li.ui-selected").each(function() { names.push($(this).text()); });
            this.setData(names);
        }.bind(this)
        $(this.object).find("#datasets")
            .sortable({ 
                handle: ".handle",
                stop: setdat,
             })
            .selectable({
                selecting: function(event, ui) {
                    var selected = $(this.object).find("#datasets li.ui-selected, #datasets li.ui-selecting");
                    if (selected.length > 2) {
                        $(ui.selecting).removeClass("ui-selecting");
                    }
                }.bind(this),
                unselected: function(event, ui) {
                    var selected = $(this.object).find("#datasets li.ui-selected, #datasets li.ui-selecting");
                    if (selected.length < 1) {
                        $(ui.unselected).addClass("ui-selected");
                    }
                }.bind(this),
                stop: setdat,
            });

        // $(this.object).find("#moviecontrol").click(this.playpause.bind(this));

        // $(this.object).find("#movieprogress>div").slider({min:0, max:1, step:.001,
        //     slide: function(event, ui) { 
        //         this.setFrame(ui.value); 
        //         this.figure.notify("setFrame", this, [ui.value]);
        //     }.bind(this)
        // });
        // $(this.object).find("#movieprogress>div").append("<div class='ui-slider-range ui-widget-header'></div>");

        // $(this.object).find("#movieframe").change(function() { 
        //     _this.setFrame(this.value); 
        //     _this.figure.notify("setFrame", _this, [this.value]);
        // });
    };
    module.Viewer.prototype._makeBtns = function(names) {
        var btnspeed = 0.5; // How long should folding/unfolding animations take?
        var td, btn, name;
        td = document.createElement("td");
        btn = document.createElement("button");
        btn.setAttribute("title", "Reset to fiducial view of the brain (Hotkey: R)");
        btn.innerHTML = "Fiducial";
        td.setAttribute("style", "text-align:left;width:150px;");
        btn.addEventListener("click", function() {
            this.animate([{idx:btnspeed, state:"target", value:[0,0,0]},
                          {idx:btnspeed, state:"mix", value:0.0}]);
        }.bind(this));
        td.appendChild(btn);
        $(this.object).find("#mixbtns").append(td);

        var nameoff = this.flatlims === undefined ? 0 : 1;
        for (var i = 0; i < names.length; i++) {
            name = names[i][0].toUpperCase() + names[i].slice(1);
            td = document.createElement("td");
            btn = document.createElement("button");
            btn.innerHTML = name;
            btn.setAttribute("title", "Switch to the "+name+" view of the brain");

            btn.addEventListener("click", function(j) {
                this.animate([{idx:btnspeed, state:"mix", value: (j+1) / (names.length+nameoff)}]);
            }.bind(this, i));
            td.appendChild(btn);
            $(this.object).find("#mixbtns").append(td);
        }

        if (this.flatlims !== undefined) {
            td = document.createElement("td");
            btn = document.createElement("button");
            btn.innerHTML = "Flat";
            btn.setAttribute("title", "Switch to the flattened view of the brain (Hotkey: F)");
            td.setAttribute("style", "text-align:right;width:150px;");
            btn.addEventListener("click", function() {
                this.animate([{idx:btnspeed, state:"mix", value:1.0}]);
            }.bind(this));
            td.appendChild(btn);
            $(this.object).find("#mixbtns").append(td);
        }

        $(this.object).find("#mix, #pivot, #shifthemis").parent().attr("colspan", names.length+2);
    };

    return module;
}(mriview || {}));
