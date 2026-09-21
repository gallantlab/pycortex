var mriview = (function(module) {
    var grid_shapes = [null, [1,1], [2, 1], [3, 1], [2, 2], [2, 2], [3, 2], [3, 2]];

    // Returns true iff every VolumeData entry in `dataview.data` shares the
    // same shape, mosaic, and numslices, AND every dim's transform matches
    // dim0's. The hover/click value readout for 2D volume views computes
    // mouse_index from dim0's voxel coordinate (via picker.xfm = dim0 xfm)
    // and reuses it across all dims; that is only correct when both shape/
    // mosaic AND the transforms match. Python-side Volume2D enforces matching
    // xfmname (so xfms agree); dataset.makeFrom can pair volumes with
    // matching shape but different transforms — bail out in that case.
    module.volumeGeomsMatch = function (dataview) {
        var data = dataview && dataview.data;
        if (!data || data.length < 2) return true;
        var d0 = data[0];
        for (var i = 1; i < data.length; i++) {
            var di = data[i];
            if (!d0.shape || !di.shape || !d0.mosaic || !di.mosaic) return false;
            if (d0.shape[0] !== di.shape[0] || d0.shape[1] !== di.shape[1]) return false;
            if (d0.mosaic[0] !== di.mosaic[0] || d0.mosaic[1] !== di.mosaic[1]) return false;
            if (d0.numslices !== di.numslices) return false;
        }
        // For 2D volume dataviews, dataview.xfm is [xfm_dim0, xfm_dim1, ...];
        // each entry is a flat 16-element array. (For 1D the xfm is itself a
        // flat 16-element array, no per-dim entries to compare — handled by
        // the early return above.)
        var xfm = dataview.xfm;
        if (Array.isArray(xfm) && xfm.length === data.length && Array.isArray(xfm[0])) {
            var x0 = xfm[0];
            for (var j = 1; j < xfm.length; j++) {
                var xj = xfm[j];
                if (!Array.isArray(xj) || xj.length !== x0.length) return false;
                for (var k = 0; k < x0.length; k++) {
                    if (x0[k] !== xj[k]) return false;
                }
            }
        }
        return true;
    };

    // Returns true iff every entry in `data` has the per-frame buffers the
    // hover/click handlers need (verts for vertex data, textures for
    // volume data). Loading is async — the dataview's `loaded` deferred
    // can resolve a tick before all child VertexData/VolumeData buffers
    // are populated, and the setData refire fires inside that window.
    module.dataBuffersReady = function (data) {
        if (!data || data.length === 0) return false;
        for (var i = 0; i < data.length; i++) {
            var d = data[i];
            if (d.mosaic !== undefined) {
                if (!d.textures || d.textures.length === 0) return false;
            } else {
                if (!d.verts || d.verts.length === 0) return false;
            }
        }
        return true;
    };

    module.Viewer = function(figure) {
        jsplot.Axes.call(this, figure);

        //mix function to attach to surface when it's added
        this._mix = function(evt){
            this.controls.setMix(evt.flat);
        }.bind(this);

        //allowTilt function to attach to surface when it's added
        this._allowTilt = function(evt){
            this.controls.allowTilt = evt.value;
        }.bind(this);

        //the lights are children of the camera, so the surface has to ask the
        //viewer to switch lighting modes for it
        this._lighting = function(evt){
            this.setTopLeftLighting(evt.topleft);
        }.bind(this);

        //Initialize all the html
        $(this.object).html($("#mriview_html").html())

        //Catalog the available colormaps
        $(this.object).find(".cmap img").each(function() {
            var tex = new THREE.Texture(this);
            tex.minFilter = THREE.LinearFilter;
            tex.magFilter = THREE.LinearFilter;
            // Premultiply alpha on upload so the shader's premultiplied-over
            // composite (gl_FragColor = vColor + (1-α)·cColor in shaderlib.js)
            // produces the correct α·rgb + (1-α)·bg result when sampling
            // alpha-bearing colormaps (e.g. RdBu_r_alpha, fire_alpha). For
            // non-alpha cmaps every row has α=255, so premultiplication is a
            // no-op. Without this, alpha-cmap-rendered Vertex2D/Volume2D
            // foregrounds clip toward white instead of fading to the
            // curvature underlay (companion fix to issue #631 for the LUT
            // texture path).
            tex.premultiplyAlpha = true;
            tex.flipY = true;
            tex.needsUpdate = true;
            colormaps[this.parentNode.id] = tex;
        });
        window.colormaps = colormaps

        this.canvas = $(this.object).find("#brain");
        jsplot.Axes3D.call(this, figure);

        this.surfs = [];
        this.dataviews = {};
        this.active = null;

        this.loaded = $.Deferred().done(function() {
            //this.schedule();
            this.resize();
            $(this.object).find("#ctmload").hide();
            this.canvas.css("opacity", 1);
            this.object.appendChild(this.controls.twodbutton[0]);
            this.controls.twodbutton.click(function(){
                this.animate([
                    {state:'camera.azimuth', idx:0.5, value:180},
                    {state:'camera.altitude', idx:0.5, value:0.1},
                ]);
            }.bind(this));
        }.bind(this));

        this.sliceplanes = {
            x: new sliceplane.Plane(this, 0),
            y: new sliceplane.Plane(this, 1),
            z: new sliceplane.Plane(this, 2),
        };
        this._clipx = false;
        this._clipy = false;
        this._clipz = false;
        //whether the canvas is split between the three slice views and the
        //3D one, rather than the 3D one alone
        this._sliceviews = false;

        this.ui = new jsplot.Menu();
        this.ui.addEventListener("update", this.schedule.bind(this));
        this.ui.add({
            mix: {action:[this, "setMix"], hidden:true},
            frame:{action:[this, "setFrame"], hidden:true},
        });
        // this.dataui = this.ui.addFolder("datasets", false);

        this._bindUI();
    }
    module.Viewer.prototype = Object.create(jsplot.Axes3D.prototype);
    THREE.EventDispatcher.prototype.apply(module.Viewer.prototype);
    module.Viewer.prototype.constructor = module.Viewer;

    module.Viewer.prototype.drawView = function(scene, idx, camera) {
        camera = camera === undefined ? this.camera : camera;
        if (this.surfs[idx] !== undefined && this.surfs[idx].prerender !== undefined)
            this.surfs[idx].prerender(this.renderer, scene, camera);

        for (var i = 0; i < this.surfs.length; i++)
            this.surfs[i].apply(this.active);
        if (this.oculus)
            this.oculus.render(scene, camera);
        else
            this.renderer.render(scene, camera);
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
                this.ui.set(f.state, f.value);
                state[f.state] = {idx:0, val:f.value};
            } else {
                if (state[f.state] === undefined)
                    state[f.state] = {idx:0, val:this.ui.get(f.state)};
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
                    this.ui.set(f.start.state, val);
                    state = true;
                } else if (sec >= f.end.idx) {
                    this.ui.set(f.end.state, f.end.value);
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
            case 'camera.azimuth':
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
        //Accept the raw metadata package sent by the python interface
        //(see JSMixer.addData in cortex/webgl/view.py). It is recognizable
        //by its "images" key, and has to be turned into DataView objects
        //(which also registers the new BrainData in dataset.brains).
        if (!(data instanceof Array) && data.images !== undefined)
            data = dataset.fromJSON(data);
        if (!(data instanceof Array))
            data = [data];

        var name, view, ui;

        var setDataFun = function(name) {
            return function() {this.setData(name)}.bind(this);
        }.bind(this);

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


            // view.ui.add({show: {action: setDataFun(name), key: i+1}});
            // this.dataui.addFolder(name, true, view.ui);
        }

        this.setData(data[0].name);
    };

    module.Viewer.prototype.fitDataname = function() {
        var dn = document.getElementById("dataname");
        if (!dn || !dn.offsetWidth) return;
        dn.style.fontSize = "";
        var available = Math.floor(window.innerWidth * 0.9) - 60;
        var width = dn.scrollWidth;
        if (width <= available) return;
        var maxPx = 32, minPx = 14;
        var fitted = Math.max(minPx, Math.floor(maxPx * available / width));
        dn.style.fontSize = fitted + "px";
    };

    module.Viewer.prototype.setData = function(name) {

        // Hide any previously displayed hover/click values immediately. They
        // refer to the OLD dataview; we will re-fire them against the NEW
        // dataview once it has loaded (see this.active.loaded.done() below).
        // Without this transient hide, a stale number is briefly visible
        // during the (sometimes slow) load of the new dataset.
        $('#mouseover_value').css('display', 'none')
        $('#picked_value').css('display', 'none')

        // blur any selected input elements
        let ids = [
            ['#vmin', '#vmin-input'],
            ['#vmax', '#vmax-input'],
            ['#xd-vmin', '#xd-vmin-input'],
            ['#xd-vmax', '#xd-vmax-input'],
            ['#yd-vmin', '#yd-vmin-input'],
            ['#yd-vmax', '#yd-vmax-input'],
        ]
        for (let displayIdInputId of ids) {
            $(displayIdInputId[0]).css('display', 'block')
            $(displayIdInputId[1]).css('display', 'none')
            document.getElementById(displayIdInputId[1].slice(1)).blur()
        }

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
                if (dv1.vertex ^ dv2.vertex) {
                    return false;
                }
                return this.addData(dataset.makeFrom(dv1, dv2));
            } else {
                return false;
            }
        }
        this.frame = 0;

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
        this.active.loaded.done(function() {
            this.active.set();
            // Register event for dataset switching
            this.dispatchEvent({type:"setData", name:this.active.name});
            // Push the new dataview into each surface (shaders + picker xfm)
            // before re-firing hover/click. drawView normally does this on
            // the next render frame, but pick() needs the picker's xfm to
            // already match the new dataview — otherwise a vertex→volume
            // switch picks with the prior (vertex-view, NaN) transform.
            for (var i = 0; i < this.surfs.length; i++) {
                if (this.surfs[i].apply !== undefined)
                    this.surfs[i].apply(this.active);
            }
            // Re-fire hover and click indicators so the displayed values
            // reflect the newly-active dataview at the same screen positions
            // (see _lastHoverEvent / _lastPickEvt cached by the handlers).
            if (this._lastHoverEvent) {
                $("#brain").trigger($.Event('mousemove', {
                    clientX: this._lastHoverEvent.clientX,
                    clientY: this._lastHoverEvent.clientY,
                }));
            }
            if (this._lastPickEvt) {
                this.pick({x: this._lastPickEvt.x, y: this._lastPickEvt.y});
            }
        }.bind(this));

        // var surf, scene, grid = grid_shapes[this.active.data.length];
        // //cleanup old scene grid for the multiview
        // // for (var i = 0; i < this.views.length; i++) {
        // //     this.views[i].scene.dispose();
        // // }
        // //Generate new scenes and re-add the surface objects
        // for (var i = 0; i < this.active.data.length; i++) {
        //     scene = this.setGrid(grid[0], grid[1], i);
        // }
        scene = this.setGrid(1,1,0);

        //Show or hide the colormap for raw / non-raw dataviews
        if (this.active.data[0].raw) {
            $("#color_fieldset").fadeTo(0.15, 0);
        } else {
            $("#color_fieldset").fadeTo(0.15, 1);
        }

        // // // // // // // // // // // // // // // // // // // // // // // //
        // start colorlegend code

        function get1dColormaps () {
            let colormaps1d = {}
            for (let colormap of Object.keys(window.colormaps)) {
                if (colormaps[colormap].image.height == 1) {
                    colormaps1d[colormap] = colormaps[colormap]
                }
            }
            return colormaps1d
        }

        function get2dColormaps () {
            let colormaps2d = {}
            for (let colormap of Object.keys(window.colormaps)) {
                if (colormaps[colormap].image.height > 1) {
                    colormaps2d[colormap] = colormaps[colormap]
                }
            }
            return colormaps2d
        }

        function setColorOptions (colormaps) {

            // clear current options
            let options = $('.colorlegend-select option')
            for (let key in Object.keys(options)) {
                let option = options[key]
                if (option) {
                    option.remove()
                }
            }

            // add new options
            let selectElement = $('.colorlegend-select')[0]
            for (let colormap of Object.keys(colormaps).sort()) {
                let optionElement = document.createElement('option')
                optionElement.setAttribute('value', colormap)
                optionElement.innerText = colormap
                selectElement.appendChild(optionElement)
            }
        }


        // adjust display and options according to dimensionality
        let dims = this.active.data.length
        if (this.active.data[0].raw) {
            dims = 3
        }

        function setColorOptionsByDim(dims) {
            if (dims === 1) {
                $('#colorlegend').removeClass('colorlegend-2d')
                $('#colorlegend').removeClass('colorlegend-3d')
                setColorOptions(get1dColormaps())
            } else if (dims === 2) {
                $('#colorlegend').removeClass('colorlegend-3d')
                $('#colorlegend').addClass('colorlegend-2d')
                setColorOptions(get2dColormaps())
            } else if (dims === 3) {
                console.log('rgb detected')
                $('#colorlegend').removeClass('colorlegend-2d')
                $('#colorlegend').addClass('colorlegend-3d')
            } else {
                console.log('unknown case: dims=' + dims)
            }
        }

        setColorOptionsByDim(dims)

        // clean numbers for display in colorbar
        function cleanNumber (number, decimals, exponential_if_beyond) {
            if (!decimals) {
                decimals = 3;
            }
            cleaned = parseFloat(number).toPrecision(decimals);

            // remove trailing zeros
            if (cleaned.indexOf('e') === -1 && cleaned.indexOf('.') !== -1) {
                cleaned = cleaned.replace(/0+$/, '')
            }

            // remove trailing decimal place
            if (cleaned[cleaned.length - 1] == '.') {
                cleaned = cleaned.substring(0, cleaned.length - 1)
            }

            //  convert to scientific notation if too long
            if (cleaned.length > 7) {
                cleaned = parseFloat(cleaned).toExponential();
            }

            return cleaned
        }

        var viewer = this

        let imageData = $('#' + this.active.cmapName + ' img')[0].src

        $('#colorlegend-colorbar').attr('src', imageData);
        $('.colorlegend-select').val(this.active.cmapName).trigger('change');

        // displaycolor limits
        if (dims === 1) {
            $('#vmin').text(cleanNumber(viewer.active.vmin[0]['value'][0]));
            $('#vmax').text(cleanNumber(viewer.active.vmax[0]['value'][0]));
        } else if (dims === 2) {
            $('#xd-vmin').text(cleanNumber(viewer.active.vmin[0]['value'][0], 3, true));
            $('#xd-vmax').text(cleanNumber(viewer.active.vmax[0]['value'][0], 3, true));
            $('#yd-vmin').text(cleanNumber(viewer.active.vmin[0]['value'][1], 3, true));
            $('#yd-vmax').text(cleanNumber(viewer.active.vmax[0]['value'][1], 3, true));
        }

        $('.colorlegend-select').off('select2:open')
        $('.colorlegend-select').on('select2:open', function (e) {
            window.colorlegendOpen = true
        });

        $('.colorlegend-select').off('select2:opening')
        $('.colorlegend-select').on('select2:opening', function (e) {
            setColorOptionsByDim(dims)
        });
        $('.colorlegend-select').off('select2:close')
        $('.colorlegend-select').on('select2:close', function (e) {
            window.colorlegendOpen = false
        });

        $('.colorlegend-select').off('select2:select')
        $('.colorlegend-select').on('select2:select', function (e) {
            var cmapName = e.params.data.id;
            viewer.active.cmapName = cmapName;
            viewer.active.setColormap(cmapName);
            viewer.schedule();
            $('#colorlegend-colorbar').attr('src', colormaps[cmapName].image.currentSrc);
        });

        function submitVmin(newVal, dim, textId, decimals) {
            if (!decimals) {
                decimals = 3
            }

            if (!dim) {
                dim = 0
            }

            if ($.isNumeric(newVal)) {
                viewer.active.setvmin(newVal, dim);
                $(textId).text(cleanNumber(newVal, decimals--));
                viewer.schedule();
            }
        }

        function submitVmax(newVal, dim, textId, decimals) {
            if (!decimals) {
                decimals = 3
            }

            if (!dim) {
                dim = 0
            }

            if ($.isNumeric(newVal)) {
                viewer.active.setvmax(newVal, dim);
                $(textId).text(cleanNumber(newVal, decimals));
                viewer.schedule();
            }
        }

        let submitFunctions = {
            'vmin': submitVmin,
            'vmax': submitVmax,
        }

        // new wheel functions
        function setWheelFunctions (side, id, dim) {
            $(id).off('wheel')
            $(id).on('wheel', function (e) {
                let currentVal = parseFloat(viewer.active[side][0]['value'][dim]);
                let newVal;
                let step = 0.25;
                if (e.altKey) {
                    step *= 0.1;
                }

                if (e.shiftKey) {
                    // logarithmic step
                    let range = viewer.active.vmax[0]['value'][dim] - viewer.active.vmin[0]['value'][dim];

                    if (e.originalEvent.deltaY <= 0) {
                        newVal = currentVal + step * range;
                    } else {
                        newVal = currentVal - step * range;
                    }

                } else {
                    // linear step
                    if (e.originalEvent.deltaY <= 0) {
                        newVal = currentVal + step;
                    } else {
                        newVal = currentVal - step;
                    }
                }
                submitFunctions[side](newVal, dim, id);
            });
        }

        setWheelFunctions('vmin', '#vmin', 0)
        setWheelFunctions('vmax', '#vmax', 0)
        setWheelFunctions('vmin', '#xd-vmin', 0)
        setWheelFunctions('vmax', '#xd-vmax', 0)
        setWheelFunctions('vmin', '#yd-vmin', 1)
        setWheelFunctions('vmax', '#yd-vmax', 1)

        function setClickFunctions (side, textId, inputId, dim, decimals) {
            function enterFunction () {
                $(textId).css('display', 'none');
                $(inputId).val(cleanNumber(viewer.active[side][0]['value'][dim], decimals, true));
                $(inputId).css('display', 'block');
                $(inputId).focus();
            }

            function leaveFunction () {
                $(inputId).css('display', 'none');
                $(textId).css('display', 'block');
                submitFunctions[side]($(inputId).val(), dim, textId, decimals);
            }

            $(textId).on('click', enterFunction);
            $(inputId).off();
            $(inputId).on('blur', leaveFunction);
            $(inputId).on('keyup', function (e) { if (event.keyCode === 13) leaveFunction() });
        }

        setClickFunctions('vmin', '#vmin', '#vmin-input', 0)
        setClickFunctions('vmax', '#vmax', '#vmax-input', 0)
        setClickFunctions('vmin', '#xd-vmin', '#xd-vmin-input', 0, 3)
        setClickFunctions('vmax', '#xd-vmax', '#xd-vmax-input', 0, 3)
        setClickFunctions('vmin', '#yd-vmin', '#yd-vmin-input', 1, 3)
        setClickFunctions('vmax', '#yd-vmax', '#yd-vmax-input', 1, 3)

        // end colorlegend code
        // // // // // // // // // // // // // // // // // // // // // // // //



        var defers = [];
        for (var i = 0; i < this.active.data.length; i++) {
            defers.push(subjects[this.active.data[i].subject].loaded)
        }
        $.when.apply(null, defers).done(function() {
            // $(this.object).find("#vrange").slider("option", {min: this.active.data[0].min, max:this.active.data[0].max});
            // if (this.active.data.length > 1) {
            //     $(this.object).find("#vrange2").slider("option", {min: this.active.data[1].min, max:this.active.data[1].max});
            //     $(this.object).find("#vminmax2").show();
            // } else {
            //     $(this.object).find("#vminmax2").hide();
            //     this.setVminmax(this.active.vmin[0].value[0], this.active.vmax[0].value[0], 0);
            // }

            this.setupStim();

            $(this.object).find("#datasets li").each(function() {
                if ($(this).text() == name)
                    $(this).addClass("ui-selected");
                else
                    $(this).removeClass("ui-selected");
            })

            $(this.object).find("#datasets").val(name);
            if (typeof(this.active.description) == "string") {
                var html = name+"<div class='datadesc'>"+this.active.description+"</div>";
                $("#dataname").html(html);
                $("#dataopts").show();
            } else {
                $("#dataname").text(name);
                $("#dataopts").show();
            }
            this.fitDataname();
            this.schedule();
            this.loaded.resolve();

            //set data for sliceplanes
            this.sliceplanes.x.setData(this.active);
            this.sliceplanes.y.setData(this.active);
            this.sliceplanes.z.setData(this.active);
            //setGrid above builds the views again, so the slice views, which
            //are views of its scene, are built again on top of it
            this.setSliceViews(this._sliceviews);

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

        if (this.state != "pause")
            this.playpause();
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
        surf.addEventListener("mix", this._mix);
        surf.addEventListener("allowTilt", this._allowTilt);
        surf.addEventListener("lighting", this._lighting);
        // The SVG overlay (ROIs/sulci/labels) re-bakes its texture asynchronously, then dispatches
        // "update" on the surface once the new texture is committed. Redraw when that lands --
        // otherwise a toggle (or a freshly-loaded label texture) is not visible until the next
        // interaction, because the menu's immediate schedule() races and beats the async bake.
        surf.addEventListener("update", this.schedule.bind(this));

        this.surfs.push(surf);
        this.root.add(surf.object);

        this.active.addEventListener("update", surf._update);
        this.active.addEventListener("attribute", surf._attrib);
        this.addEventListener("resize", surf._resize);

        if (surf.ui !== undefined) {
            this.ui.addFolder("surface", true, surf.ui);
        }

        $("#brain").on('mousemove',
            function (event) {
                // Cache last mouse position so setData() can recompute the
                // hover indicator for the newly-active dataset at the same
                // screen coordinate.
                this._lastHoverEvent = event;
                // Length check first so the short-circuit guards us against
                // an empty data array before we read data[0]. Then skip RGB
                // (no underlying scalars), then ensure all child buffers
                // have populated (the setData refire can briefly precede
                // buffer fill).
                if ((this.active.data.length !== 1 && this.active.data.length !== 2) ||
                    this.active.data[0].raw ||
                    !module.dataBuffersReady(this.active.data)) {
                    $('#mouseover_value').css('display', 'none')
                    return
                }
                // We need to use a different logic if we have a VolumeData or a VertexData object
                let values = null;
                if (this.active.vertex) {
                    let coords = this.getCoords(event)
                    if (coords !== -1) {
                        let hemiIdx = (coords.hemi == 'left') ? 0 : 1
                        // Map the picked vertex through the subject's indexMap.
                        // dim1 and dim2 share subject for 2D views (server-side).
                        let subject = this.active.data[0].subject
                        let indexMap = subjects[subject].hemis[coords.hemi].indexMap
                        let vertex = indexMap[coords.vertex]
                        // Now access the data for each channel (1 for 1D, 2 for 2D)
                        values = this.active.data.map(function (d) {
                            return d.verts[0][hemiIdx].array[vertex]
                        })
                    }
                } else {
                    // Volume branch. For 2D views we reuse data[0]'s mouse_index
                    // across all dims; that is safe iff every dim shares the
                    // same shape/mosaic/numslices. Python-side Volume2D enforces
                    // matching xfmname → matching geometry. Client-side
                    // dataset.makeFrom (mriview.js:283) can pair arbitrary 1D
                    // volumes; if their geometries diverge, hide the readout
                    // rather than display a wrong-but-plausible value.
                    if (!module.volumeGeomsMatch(this.active)) {
                        $('#mouseover_value').css('display', 'none')
                        return
                    }
                    let mouse_index = this.getMouseIndex(event)
                    if (mouse_index !== -1) {
                        values = this.active.data.map(function (d) {
                            return d.textures[0].image.data[mouse_index]
                        })
                    }
                }
                if (values !== null) {
                    let formatted = values.map(function (v) {
                        return parseFloat(v).toPrecision(3)
                    }).join(', ')
                    if (values.length > 1) {
                        formatted = '(' + formatted + ')'
                    }
                    $('#mouseover_value').text(formatted)
                    $('#mouseover_value').css('display', 'block')
                } else {
                    $('#mouseover_value').css('display', 'none')
                }
            }.bind(this)
        )

        this.schedule();
        return surf;
    };

    module.Viewer.prototype.getCoords = function(event) {
        if (this.surfs[0].pick && this.surfs[0].surf.picker) {
            this.svgOff(this.surfs)
            let coords = this.surfs[0].surf.picker.get_vertex_index(
                this.renderer,
                this.camera,
                event.clientX,
                event.clientY,
            )
            this.svgOn(this.surfs)
            // console.log("coords: " + JSON.stringify(coords));
            return coords
        }
        return -1
    };

    module.Viewer.prototype.getMouseIndex = function(event) {
        if (this.surfs[0].pick && this.surfs[0].surf.picker) {
            let coords = this.getCoords(event)
            // console.log(coords)
            if (coords !== -1) {
                return this.xyxToI(coords.voxel.x, coords.voxel.y, coords.voxel.z)
            } else {
                return -1
            }
        }
        return -1
    };

    module.Viewer.prototype.xyxToI = function(x, y, z) {
        // xyz is 3d coordinate in voxel space ie (100, 100, 20)
        // uv is 2d coordinate in mosaic slice space  ie (6, 5)
        // jk is 2d coordinate in mosaic texture space ie (600, 500)
        // i is linear index in unraveled image ie (123213)
        // all zero-indexed

        let n_x = this.active.data[0].shape[0]
        let n_y = this.active.data[0].shape[1]
        let n_z = this.active.data[0].numslices
        
        let n_u = this.active.data[0].mosaic[0]
        let n_v = this.active.data[0].mosaic[1]

        if (typeof this.active.data[0].textures[0] === "undefined") {
            return -1
        }

        let n_j = this.active.data[0].textures[0].image.width
        let n_k = this.active.data[0].textures[0].image.height

        let u = z % n_u
        let v = Math.floor(z / n_u)

        let j_origin = 1 + u * (n_x + 1)
        let k_origin = 1 + v * (n_y + 1)

        let j = j_origin + x
        let k = k_origin + y

        let i = k * n_j + j

        // console.log('n_x: ', n_x)
        // console.log('n_y: ', n_y)
        // console.log('n_z: ', n_z)
        // console.log('n_u: ', n_u)
        // console.log('n_v: ', n_v)
        // console.log('n_j: ', n_j)
        // console.log('n_k: ', n_k)
        // console.log('x: ', x)
        // console.log('y: ', y)
        // console.log('z: ', z)
        // console.log('u: ', u)
        // console.log('v: ', v)
        // console.log('j_origin: ', j_origin)
        // console.log('k_origin: ', k_origin)
        // console.log('j: ', j)
        // console.log('k: ', k)
        // console.log('i: ', i)

        return i
    };

    module.Viewer.prototype.svgOff = function(surfs) {
        for (surf of surfs) {
            if (surf.surf.svg !== undefined) {
                surf.surf.svg.labels.left.visible = false;
                surf.surf.svg.labels.right.visible = false;
            }
        }
    }
    module.Viewer.prototype.svgOn = function(surfs) {
        for (surf of surfs) {
            if (surf.surf.svg !== undefined) {
                surf.surf.svg.labels.left.visible = true;
                surf.surf.svg.labels.right.visible = true;
            }
        }
    }

    module.Viewer.prototype.rmSurf = function(surftype) {
        var newsurfs = [];
        for (var i = 0; i < this.surfs.length; i++) {
            if (this.surfs[i].constructor == surftype) {
                this.active.removeEventListener("update", this.surfs[i]._update);
                this.active.removeEventListener("attribute", this.surfs[i]._attrib);
                this.removeEventListener("resize", this.surfs[i]._resize);
                this.surfs[i].removeEventListener("mix", this._mix);
                this.surfs[i].removeEventListener("allowTilt", this._allowTilt);
                this.surfs[i].removeEventListener("lighting", this._lighting);

                this.root.remove(this.surfs[i].object);
            } else
                newsurfs.push(this.surfs[i]);
        }
        this.surfs = newsurfs;
        this.schedule();
    }

    module.Viewer.prototype.setMix = function(mix) {
        if (mix === undefined) {
            return this.surfs.length == 0 ? 0 : this.surfs[0].setMix();
        }
        for (var i = 0; i < this.surfs.length; i++)
            if (this.surfs[i].setMix !== undefined)
                this.surfs[i].setMix(mix);
    }

    module.Viewer.prototype.pick = function(evt) {
        // Cache last pick position so setData() can refresh the picked
        // indicator for the newly-active dataset at the same screen point.
        this._lastPickEvt = {x: evt.x, y: evt.y};
        var x = evt.x, y = evt.y;
        if (this._sliceviews) {
            // The picker draws the 3D view over the whole canvas, while the
            // split shows it in the corner. Both have the canvas's aspect, so
            // they are the same image at half the scale and a click in that
            // corner lands here; a click in a slice view picks nothing.
            if (x < this.width / 2 || y < this.height / 2)
                return;
            x = (x - this.width / 2) * 2;
            y = (y - this.height / 2) * 2;
        }
        let coords
        for (var i = 0; i < this.surfs.length; i++) {
            if (this.surfs[i].pick)
                coords = this.surfs[i].pick(this.renderer, this.camera, x, y);
        }
        this.setCursor(coords);
        // set the picked value display
        // Length check first so we don't index data[0] on an empty array.
        // Skip RGB, then ensure all child buffers have populated.
        if ((this.active.data.length !== 1 && this.active.data.length !== 2) ||
            this.active.data[0].raw ||
            !module.dataBuffersReady(this.active.data)) {
            $('#picked_value').css('display', 'none')
            return
        }

        // We need to use a different logic if we have a VolumeData or a VertexData object
        let values = null;
        if (this.active.vertex) {
            if (coords !== -1) {
                let hemiIdx = (coords.hemi == 'left') ? 0 : 1
                // Map the picked vertex through the subject's indexMap.
                // dim1 and dim2 share subject for 2D views (server-side).
                let subject = this.active.data[0].subject
                let indexMap = subjects[subject].hemis[coords.hemi].indexMap
                let vertex = indexMap[coords.vertex]
                // Now access the data for each channel (1 for 1D, 2 for 2D)
                values = this.active.data.map(function (d) {
                    return d.verts[0][hemiIdx].array[vertex]
                })
            }
        } else {
            if (coords !== -1) {
                // Volume branch. See the matching note in the mousemove handler
                // for why we hide on geometry mismatch.
                if (!module.volumeGeomsMatch(this.active)) {
                    $('#picked_value').css('display', 'none')
                    return
                }
                let mouse_index = this.xyxToI(coords.voxel.x, coords.voxel.y, coords.voxel.z)
                if (mouse_index !== -1) {
                    values = this.active.data.map(function (d) {
                        return d.textures[0].image.data[mouse_index]
                    })
                }
            }
        }
        if (values !== null) {
            let formatted = values.map(function (v) {
                return parseFloat(v).toPrecision(3)
            }).join(', ')
            if (values.length > 1) {
                formatted = '(' + formatted + ')'
            }
            $('#picked_value').text(formatted)
            $('#picked_value').css('display', 'block')
        } else {
            $('#picked_value').css('display', 'none')
        }
    }

    var movie_ui;
    module.Viewer.prototype.setupStim = function() {
        if (this.active.data[0].movie) {
            if ("movie" in this.ui._folders) {
                // nothing?
            } else {
                movie_ui = this.ui.addFolder("movie", true);
                movie_ui.add({play_pause: {action: this.playpause.bind(this), key:' '}});
                movie_ui.add({frame: {action:[this, "setFrame", 0, this.active.frames-1]}});
            }

            if (this.movie) {
                this.movie.destroy();
            }
            if (this.active.stim && figure) {
                figure.setSize("right", "30%");
                this.movie = figure.add(jsplot.MovieAxes, "right", false, this.active.stim);
                this.movie.setFrame(this.active.frame);
                setTimeout(this.resize.bind(this), 1000);
            }
            this.dispatchEvent({type:"stimulus", object:this.movie});
        } else {
            // if movie was left playing, pause before removing menu with pause button
            if (this.state == "play") {
                this.playpause();
            }
            this.ui.remove("movie");
        }
        this.schedule();
        if (this.movie) {
            // I don't know why I had to do this twice,
            // but it makes the stimulus movie buffer better...
            this.movie.destroy();
            this.movie.destroy();
        }
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
        if (frame === undefined)
            return this.frame;
        if (frame >= this.active.frames) {
            frame %= this.active.frames;
            this._startplay += this.active.frames;
            this.playpause();
        }
        var replay = false;
        if (this.state == "play") {
            this.playpause();
            replay = true;
        }
        this.frame = frame;
        this.active.setFrame(frame);
        if (this.movie) {
            this.movie.setFrame(frame);
        }
        // $(this.object).find("#movieprogress div").slider("value", frame);
        // $(this.object).find("#movieframe").attr("value", frame);
        if (replay == true) {
            this.playpause();
        }
        this.schedule();
    };

    module.Viewer.prototype.close = function() {
        window.close();
    };

    var _bound = false;
    module.Viewer.prototype._bindUI = function() {
        $(window).scrollTop(0);
        $(window).resize(function() { this.resize(); this.fitDataname(); }.bind(this));
        this.canvas.resize(function() { this.resize(); }.bind(this));

        var cam_ui = this.ui.addFolder("camera", true);
        cam_ui.add({
            azimuth: {action:[this.controls, 'setAzimuth', 0, 360]},
            altitude: {action:[this.controls, 'setAltitude', 0, 180]},
            radius: {action:[this.controls, 'setRadius', 10, 1000]},
            target: {action:[this.controls, 'setTarget'], hidden:true},
        });

        var fold_brain = function() {
            this.animate([
                {state:'mix', idx:parseFloat(viewopts.anim_speed), value:0},
            ]);
        }.bind(this);
        this.reset_view = function() {
            this.animate([
                {state:'camera.target', idx:parseFloat(viewopts.anim_speed), value:[0,0,0]},
                {state:'camera.azimuth', idx:parseFloat(viewopts.anim_speed), value:45},
                {state:'camera.altitude', idx:parseFloat(viewopts.anim_speed), value:75},
                {state:'camera.radius', idx:parseFloat(viewopts.anim_speed), value:400},
                {state:'mix', idx:parseFloat(viewopts.anim_speed), value:0},
            ]);
        }.bind(this);
        var inflate = function() {
            this.animate([{state:'mix', idx:parseFloat(viewopts.anim_speed), value:.5}]);
        }.bind(this);
        var inflate_to_cuts = function() {
            this.animate([{state:'mix', idx:parseFloat(viewopts.anim_speed), value:.501}]);
        }.bind(this);
        var flatten = function() {
            this.animate([ {state:'mix', idx:parseFloat(viewopts.anim_speed), value:1}]);
        }.bind(this);

        // saves out current view as a png
        this.imageWidth = 2400;
        this.imageHeight = 1200;
        var saveImage = function() {
            var image = viewer.getImage(this.imageWidth.toFixed(), this.imageHeight.toFixed());
            var a = document.createElement('a');
            a.download = 'image.png';
            a.href = image.toDataURL();
            a.click();
        }.bind(this);

        cam_ui.add({
            fold: {action:fold_brain, key:'r', help:'Fold brain'},
            reset: {action:this.reset_view, key:'t', help:'Reset view'},
            inflate: {action:inflate, key:'i', help:'Inflate'},
            "inflate to cuts": {action:inflate_to_cuts, key:'k', help:'Inflate to cuts'},
            flatten: {action:flatten, key:'f', help:'Flatten'},
        });

        // UI controls for saving image and selecting image size
        var imageFolder = cam_ui.addFolder('Save image');
        imageFolder.add({
            "Save": {action:saveImage, key:'S', modKeys: ['shiftKey'], help:'Save current view as a png'},
            "Width": {action:[this, 'imageWidth', 500, 4000]},
            "Height": {action:[this, 'imageHeight', 500, 4000]}
        });

        // keyboard shortcut menu
        var _show_help = false;
        var helpmenu = function() {
            var helpmenu = $(this.object).find('#helpmenu');
            if (!_show_help){
                var new_html = '<table>'
                list = [this.ui._desc,
                        this.ui._desc.camera._desc,
                        this.ui._desc.sliceplanes._desc,
                        this.ui._desc.sliceplanes._desc.move._desc,
                        imageFolder._desc];

                // add surface items to list
                surface_names = Object.keys(this.ui._desc.surface)
                    .filter((key) => key[0] != '_')
                    .forEach((key) => list.push(this.ui._desc.surface[key]._desc))

                for (var i = 0; i < list.length; i++){
                    for (var name in list[i]){
                        if ('help' in list[i][name]){
                            var diplay_name = list[i][name]['help']
                        }
                        else{
                            var diplay_name = name
                        }
                        
                        if ('key' in list[i][name]){
                            new_html += '<tr><td style="text-align: center;">'
                            if ('modKeys' in list[i][name]){
                                let modKeys = list[i][name]['modKeys'];
                                modKeys = modKeys.map((modKey) => modKey.charAt(0).toUpperCase() + modKey.substring(1, modKey.length - 3));
                                modKeys = modKeys.join(' + ');
                                new_html += modKeys + ' + ';
                            }
                            new_html += list[i][name]['key'] + '</td><td>' + diplay_name + '</td></tr>'
                        }
                        if ('wheel' in list[i][name]){
                            new_html += '<tr><td style="text-align: center;">'
                            var modKeys = list[i][name]['modKeys']
                            modKeys = modKeys.map((modKey) => modKey.charAt(0).toUpperCase() + modKey.substring(1, modKey.length - 3))
                            modKeys = modKeys.join(' + ')
                            new_html += modKeys + ' + wheel  </td><td>' + diplay_name + '</td></tr>'
                        }
                    }
                }
                new_html += '</table>'
                helpmenu.html(new_html);
                helpmenu.show()
            }
            else{
                helpmenu.hide()
            }
            _show_help = !_show_help;
        }.bind(this);

        var help_ui = this.ui.addFolder("help", true)
        help_ui.add({
            shortcuts: {action:helpmenu, key:'h', help:'Toggle this help'},
        });

        var _hidelabels = false;
        var hidelabels = function() {
            for (var i = 0; i < this.surfs.length; i++) {
                if (this.surfs[i].surf) { //only do this for surfdelegate objects
                    var svg = this.surfs[i].surf.svg;

                    if (_hidelabels) {
                        // if layers were hidden
                        var toggledLayers = this.toggledLayers || Object.keys(svg.layers)
                        for (var name of toggledLayers) {
                            svg.layers[name].labels.showhide(_hidelabels);
                        }
                    } else {
                        // if layers were not hidden
                        var toggledLayers = []
                        for (var name of Object.keys(svg.layers)) {
                            if (!svg.layers[name]._hidden) {
                                toggledLayers.push(name)
                            }
                            svg.layers[name].labels.showhide(_hidelabels);
                        }
                        this.toggledLayers = toggledLayers
                    }
                    
                }
            }
            _hidelabels = !_hidelabels;
            this.schedule();
        }.bind(this);
        this.ui.add({
            hide_labels: {action:hidelabels, key:'l', hidden:true, help:'Toggle labels'},
        });

        //add sliceplane gui
        var sliceplane_ui = this.ui.addFolder("sliceplanes", true)
        sliceplane_ui.add({
            "Show ortho views": {action:[this, "setSliceViews"], toggle:true},
            orthoToggle: {action: this.toggleSliceViews.bind(this), key: 'v', hidden: true,
                          help:'Show ortho views along with 3D'},
            x: {action:[this.sliceplanes.x, "setVisible"]},
            xToggle: {action: this.toggleXVis.bind(this), key: 'e', hidden: true, help:'Toggle X slice'},
            y: {action:[this.sliceplanes.y, "setVisible"]},
            yToggle: {action: this.toggleYVis.bind(this), key: 'd', hidden: true, help:'Toggle Y slice'},
            z: {action:[this.sliceplanes.z, "setVisible"]},
            zToggle: {action: this.toggleZVis.bind(this), key: 'c', hidden: true, help:'Toggle Z slice'},
        });
        var sliceplane_move = sliceplane_ui.addFolder("move", true);
        sliceplane_move.add({
            x_up: {action:this.sliceplanes.x.next.bind(this.sliceplanes.x), key:'q', hidden:true, help:'Next X slice'},
            x_down: {action:this.sliceplanes.x.prev.bind(this.sliceplanes.x), key:'w', hidden:true, help:'Previous X slice'},
            y_up: {action:this.sliceplanes.y.next.bind(this.sliceplanes.y), key:'a', hidden:true, help:'Next Y slice'},
            y_down: {action:this.sliceplanes.y.prev.bind(this.sliceplanes.y), key:'s', hidden:true, help:'Previous Y slice'},
            z_up: {action:this.sliceplanes.z.next.bind(this.sliceplanes.z), key:'z', hidden:true, help:'Next Z slice'},
            z_down: {action:this.sliceplanes.z.prev.bind(this.sliceplanes.z), key:'x', hidden:true, help:'Previous Z slice'},
        });
        sliceplane_move.add({
            move_x: {action:[this.sliceplanes.x, 'setSmoothSlice', 0, 1, 0.001]},
            move_y: {action:[this.sliceplanes.y, 'setSmoothSlice', 0, 1, 0.001]},
            move_z: {action:[this.sliceplanes.z, 'setSmoothSlice', 0, 1, 0.001]},
            rotate_x: {action:[this.sliceplanes.x, 'setAngle', -89, 89]},
            rotate_y: {action:[this.sliceplanes.y, 'setAngle', -89, 89]},
            rotate_z: {action:[this.sliceplanes.z, 'setAngle', -89, 89]}
        });

        var sliceplane_clip = sliceplane_ui.addFolder("clip", true);
        sliceplane_clip.add({
            clip_x: {action:[this, "setClippingX"]},
            flip_x: {action:[this.sliceplanes.x, "setFlip"]},
            clip_y: {action:[this, "setClippingY"]},
            flip_y: {action:[this.sliceplanes.y, "setFlip"]},
            clip_z: {action:[this, "setClippingZ"]},
            flip_z: {action:[this.sliceplanes.z, "setFlip"]},
        })

        //

        if ($(this.object).find("#colormap_category").length > 0) {
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
        var dataopts = $(this.object).find('#dataopts');
        var dataset_cat = $(dataopts).find('#dataset_category');
        dataset_cat.hide();
        $(dataopts).find('#dataname').click(function(e) {
          dataset_cat.slideToggle();
        });

        var setdat = function(event, ui) {
            var names = [];
            $(this.object).find("#datasets li.ui-selected").each(function() { names.push($(this).text()); });
            this.setData(names);
        }.bind(this);

        // add keybindings for cycling through datasets
        this.ui.add({
          next_dset: {action: this.nextData.bind(this), key: '+', hidden: true, help:'Next dataset'},
          next_dset_alt: {action: this.nextData.bind(this), key: '=', hidden: true, help:'Next dataset (alt)'},
          prev_dset: {action: this.nextData.bind(this, -1), key: '-', hidden: true, help:'Previous dataset'}});

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
    module.Viewer.prototype.toggleXVis = function() {
        this.sliceplanes.x.setVisible(!this.sliceplanes.x._visible);
        viewer.schedule();
    };
    module.Viewer.prototype.toggleYVis = function() {
        this.sliceplanes.y.setVisible(!this.sliceplanes.y._visible);
        viewer.schedule();
    };
    module.Viewer.prototype.toggleZVis = function() {
        this.sliceplanes.z.setVisible(!this.sliceplanes.z._visible);
        viewer.schedule();
    };
    module.Viewer.prototype.toggleSliceViews = function() {
        this.setSliceViews(!this._sliceviews);
    };
    //-------------------------------------------------------------------------
    // Orthogonal slice views
    //-------------------------------------------------------------------------
    //Where each slice view sits in the canvas, in the fractions of it
    //three.js takes (the origin is its bottom left corner). The 3D view
    //keeps the last quarter.
    var SLICE_VIEWS = [
        {plane: "y", axis: 1, left: 0,   bottom: 0.5},
        {plane: "z", axis: 2, left: 0,   bottom: 0},
        {plane: "x", axis: 0, left: 0.5, bottom: 0.5},
    ];

    //Splits the canvas between the three slice planes, each drawn straight
    //down its own axis in a quarter of it, and the 3D view, which keeps the
    //last quarter along with the camera the controls move and the surface as
    //it is always drawn. A slice view shows its plane alone: the surface
    //would stand in front of the data the view is there to show.
    module.Viewer.prototype.setSliceViews = function(val) {
        if (val === undefined)
            return this._sliceviews;
        //A point picked in the 3D view on its own left the planes alone, so
        //the slices are taken to it as the views open; once they are open a
        //pick moves them, and the slice keys are free to move them again.
        var opening = !!val && this._sliceviews !== true;
        this._sliceviews = !!val;
        if (this.views.length == 0)
            return;
        if (opening)
            this._slicesToCursor();

        var scene = this.views[0].scene;
        this._cursorObject();
        var views = [];
        if (this._sliceviews) {
            for (var i = 0; i < SLICE_VIEWS.length; i++) {
                var spec = SLICE_VIEWS[i];
                var view = {
                    left: spec.left, bottom: spec.bottom, width: 0.5, height: 0.5,
                    scene: scene, surf: 0, plane: this.sliceplanes[spec.plane],
                    axis: spec.axis,
                    camera: new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 10000),
                };
                view.prepare = this._prepareSlice.bind(this, view);
                view.overlay = this._drawCursor.bind(this);
                views.push(view);
            }
            var solid = {left: 0.5, bottom: 0, width: 0.5, height: 0.5, scene: scene, surf: 0};
            solid.prepare = this._showAllPlanes.bind(this);
            solid.overlay = this._drawCursor.bind(this);
            views.push(solid);
        } else {
            views.push({left: 0, bottom: 0, width: 1, height: 1, scene: scene, surf: 0});
        }
        this.views = views;
        this._orthoLabels(this._sliceviews);
        this._showPickMarkers(!this._sliceviews);
        //the surfaces and the planes are left as the 3D view wants them, so
        //that a single view, a screenshot or a pick sees what it always did
        this._showAllPlanes();
        this.resize();
        this.schedule();
    };

    //The crosshair that marks the picked point in the slice views. It is
    //not the picker's own marker, which rides with the surface as it
    //unfolds: this one stays where the point is in the anatomy, which is
    //where the slices cut it.
    module.Viewer.prototype._cursorObject = function() {
        if (this._cursor === undefined) {
            var far = 500;
            //Three lines along the axes of the view that draws them. In a
            //slice view the third one runs down the line of sight, so it is
            //seen end on and only the cross of the other two shows; in the
            //3D view all three do, which is what marks a point in space.
            var lines = new THREE.Geometry();
            lines.vertices.push(new THREE.Vector3(-far, 0, 0), new THREE.Vector3(far, 0, 0),
                                new THREE.Vector3(0, -far, 0), new THREE.Vector3(0, far, 0),
                                new THREE.Vector3(0, 0, -far), new THREE.Vector3(0, 0, far));
            var material = new THREE.LineBasicMaterial({
                color: 0x44ccff, depthTest: false, depthWrite: false});
            this._cursor = new THREE.Line(lines, material, THREE.LinePieces);
            this._cursor.frustumCulled = false;
            this._cursor.visible = false;
            this._cursorAt = false;
            //kept out of the scene the views draw, so that it can be laid
            //over each of them once the rest of it has been drawn
            this._cursorScene = new THREE.Scene();
            this._cursorScene.add(this._cursor);
        }
        return this._cursor;
    };

    //Puts the crosshair on a picked point and takes the slice views to the
    //slices through it, so that the four views agree on where it is. Picking
    //nothing leaves them where they are.
    module.Viewer.prototype.setCursor = function(coords) {
        var cursor = this._cursorObject();
        var voxel = (coords === undefined || coords === -1) ? undefined : coords.voxel;
        var xfm = (this.active === null || this.active === undefined)
                ? undefined : this.active.uniforms.volxfm;
        //A click that picked nothing leaves the crosshair where it is: it
        //marks a place, and clicking beside the brain is not a way of saying
        //to forget it.
        if (voxel === undefined)
            return;
        //Data on the vertices has no volume to point into, so there is no
        //place for the crosshair to be.
        if (!isFinite(voxel.x) || !isFinite(voxel.y) || !isFinite(voxel.z) ||
            xfm === undefined || xfm.value[0] === undefined) {
            this._cursorAt = false;
            cursor.visible = false;
            this.schedule();
            return;
        }
        var toWorld = new THREE.Matrix4().getInverse(xfm.value[0]);
        cursor.position.copy(voxel.clone().applyMatrix4(toWorld));
        this._cursorAt = true;
        this._cursorVoxel = voxel.clone();

        //The slice views take the slices the point is on, so that it is on
        //screen in each of them. The 3D view on its own leaves its planes
        //where they were put, and is given them when it is split.
        if (this._sliceviews)
            this._slicesToCursor();
        this.schedule();
    };
    module.Viewer.prototype._slicesToCursor = function() {
        var voxel = this._cursorVoxel;
        if (this._cursorAt !== true || voxel === undefined)
            return;
        var slices = {x: voxel.x, y: voxel.y, z: voxel.z};
        for (var name in this.sliceplanes) {
            if (this.sliceplanes[name].mesh !== undefined)
                this.sliceplanes[name].update(slices[name]);
        }
    };

    //Draws one slice view: the plane of this view alone, seen from straight
    //down its normal and framed on the slice it cuts. Data that has no
    //volume to slice, such as data on the vertices, leaves the surface in
    //view instead, seen down the axis this view stands for.
    module.Viewer.prototype._prepareSlice = function(view, width, height) {
        var plane = view.plane;
        var sliced = plane.mesh !== undefined && plane.geometry !== undefined;
        this.root.visible = !sliced;
        //the 3D view has the picker's own marker, so this one is for these
        var cursor = this._cursorObject();
        cursor.visible = this._cursorAt === true;
        for (var name in this.sliceplanes) {
            var mesh = this.sliceplanes[name].mesh;
            if (mesh !== undefined)
                mesh.visible = this.sliceplanes[name] === plane;
        }

        var look = new THREE.Vector3(), up, right, center, halfUp = 0, halfRight = 0;
        if (sliced) {
            //the face normal rather than the clipping one, which the flip
            //controls turn around
            look.copy(plane.geometry.faces[0].normal).applyEuler(plane.mesh.rotation).normalize();
            //seen from the subject's right, front or top, whichever of those
            //the plane faces, so that the slices are laid out as they are read
            var axis = 0;
            for (var a = 1; a < 3; a++) {
                if (Math.abs(look.getComponent(a)) > Math.abs(look.getComponent(axis)))
                    axis = a;
            }
            if (look.getComponent(axis) < 0)
                look.negate();

            //Up is one of the plane's own edges, not a world axis: the volume
            //sits at an angle to the anatomy, and a world axis would leave
            //the slice tilted on screen. Of the two edges, the one that
            //points most nearly superior is up, or most nearly anterior on
            //the plane that superior is normal to.
            var corners = plane.geometry.vertices;
            var edges = [new THREE.Vector3().subVectors(corners[1], corners[0]).normalize(),
                         new THREE.Vector3().subVectors(corners[2], corners[0]).normalize()];
            var upAxis = axis == 2 ? 1 : 2;
            up = Math.abs(edges[0].getComponent(upAxis)) >= Math.abs(edges[1].getComponent(upAxis))
               ? edges[0] : edges[1];
            if (up.getComponent(upAxis) < 0)
                up.negate();
            right = new THREE.Vector3().crossVectors(up, look).normalize();
            up.crossVectors(look, right).normalize();

            //the quad of the slice, which is centered on the object itself
            for (var v = 0; v < corners.length; v++) {
                halfUp = Math.max(halfUp, Math.abs(corners[v].dot(up)));
                halfRight = Math.max(halfRight, Math.abs(corners[v].dot(right)));
            }
            center = plane.center;
        } else {
            var box = new THREE.Box3().setFromObject(this.root);
            if (box.empty())
                return;
            look.setComponent(view.axis, 1);
            up = view.axis == 2 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
            right = new THREE.Vector3().crossVectors(up, look).normalize();
            var size = box.size();
            halfUp = Math.abs(size.dot(up)) / 2;
            halfRight = Math.abs(size.dot(right)) / 2;
            center = box.center();
        }
        this._aimOrtho(view.camera, center, look, up, halfRight, halfUp, width / height);
        //where the view is aimed, which is what a click in it is read against
        view.look = look.clone();
        view.center = center.clone();
        //the crosshair lies in the plane of the view that draws it, so its
        //two lines cross at the point rather than running off at an angle
        var basis = new THREE.Matrix4();
        basis.set(right.x, up.x, look.x, 0,
                  right.y, up.y, look.y, 0,
                  right.z, up.z, look.z, 0,
                  0, 0, 0, 1);
        cursor.quaternion.setFromRotationMatrix(basis);

        //The lights ride with the camera the controls move, so a view drawn
        //with another one would be lit from wherever that camera happens to
        //point. Turn them to face this view for as long as it is drawn; the
        //3D view, which is drawn last, puts them back.
        if (this._camQuat === undefined)
            this._camQuat = this.camera.quaternion.clone();
        this.camera.quaternion.copy(view.camera.quaternion);
    };

    //Points an orthographic camera at `center` from along `look`, with `up`
    //on the screen and wide enough to hold what is being looked at
    module.Viewer.prototype._aimOrtho = function(camera, center, look, up, halfRight, halfUp, aspect) {
        var half = Math.max(halfUp, halfRight / aspect) * 1.05;
        camera.up.copy(up);
        camera.position.copy(center).add(look.clone().multiplyScalar(1000));
        camera.lookAt(center);
        camera.left = -half * aspect;
        camera.right = half * aspect;
        camera.top = half;
        camera.bottom = -half;
        camera.updateProjectionMatrix();
        camera.updateMatrixWorld();
    };

    //The slice planes are transparent, so they are drawn over everything
    //opaque, the crosshair included. It goes back on top in a pass of its
    //own, over the view that was just drawn.
    module.Viewer.prototype._drawCursor = function(camera) {
        if (!this._cursorObject().visible)
            return;
        //the slices are transparent and the brain is solid, so either would
        //cover the crosshair if it were drawn among them
        this.renderer.autoClear = false;
        this.renderer.render(this._cursorScene, camera);
        this.renderer.autoClear = true;
    };

    //Puts the surfaces and the planes back the way the 3D view shows them
    module.Viewer.prototype._showAllPlanes = function() {
        if (this._camQuat !== undefined) {
            this.camera.quaternion.copy(this._camQuat);
            this._camQuat = undefined;
        }
        this.root.visible = true;
        //the crosshair marks the same point in the 3D view as in the slices,
        //as long as they are there to be marked in
        this._cursorObject().visible =
            this._sliceviews === true && this._cursorAt === true;
        for (var name in this.sliceplanes) {
            var plane = this.sliceplanes[name];
            if (plane.mesh !== undefined)
                plane.mesh.visible = plane.setVisible();
        }
    };

    //The picker marks the vertex it picked on the surface itself. The
    //crosshair marks the point in the volume, which is the same point and is
    //in every view, so the two are never up at once.
    module.Viewer.prototype._showPickMarkers = function(show) {
        for (var i = 0; i < this.surfs.length; i++) {
            var surf = this.surfs[i].surf;
            if (surf === undefined || surf.picker === undefined)
                continue;
            surf.picker.markers.left.visible = show;
            surf.picker.markers.right.visible = show;
        }
        this.schedule();
    };

    //A click in a slice view puts the crosshair where it landed on that
    //slice, and takes the other two views to the slices through it.
    module.Viewer.prototype._pickSlice = function(index, event) {
        var view = this.views[index];
        //the view aims itself as it is drawn, so its own frame says where in
        //the volume the pointer is
        if (view === undefined || view.camera === undefined ||
            view.look === undefined || view.center === undefined)
            return;
        //with data on the vertices these views show the surface instead, and
        //there is no slice for a click to land on
        if (view.plane === undefined || view.plane.mesh === undefined)
            return;
        var xfm = (this.active === null || this.active === undefined)
                ? undefined : this.active.uniforms.volxfm;
        if (xfm === undefined || xfm.value[0] === undefined)
            return;

        var rect = event.currentTarget.getBoundingClientRect();
        var point = new THREE.Vector3(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            1 - ((event.clientY - rect.top) / rect.height) * 2, 0).unproject(view.camera);
        //the camera looks down the normal of the slice, so the point under
        //the pointer is that one carried along the line of sight onto it
        var depth = view.look.dot(new THREE.Vector3().subVectors(view.center, point));
        point.add(view.look.clone().multiplyScalar(depth));

        var voxel = point.applyMatrix4(xfm.value[0]);
        this.setCursor({voxel: new THREE.Vector3(
            Math.round(voxel.x), Math.round(voxel.y), Math.round(voxel.z))});
    };

    //One div over each slice view, which names it in its corner and
    //takes the clicks made in it
    module.Viewer.prototype._orthoLabels = function(show) {
        if (this._orthoDivs === undefined) {
            this._orthoDivs = [];
            for (var i = 0; i < SLICE_VIEWS.length; i++) {
                var div = document.createElement("div");
                div.className = "mriview-ortho";
                div.style.left = (100 * SLICE_VIEWS[i].left) + "%";
                //the views count from the bottom of the canvas, CSS from the top
                div.style.top = (100 * (0.5 - SLICE_VIEWS[i].bottom)) + "%";
                div.appendChild(document.createElement("span")).textContent =
                    SLICE_VIEWS[i].plane + " slice";
                //a drag here would otherwise turn the 3D view, which is not
                //what the pointer is over
                var swallow = function(event) { event.stopPropagation(); };
                div.addEventListener("mousedown", swallow, false);
                div.addEventListener("wheel", swallow, false);
                div.addEventListener("dblclick", swallow, false);
                div.addEventListener("click", this._pickSlice.bind(this, i), false);
                this.object.appendChild(div);
                this._orthoDivs.push(div);
            }
        }
        for (var i = 0; i < this._orthoDivs.length; i++)
            this._orthoDivs[i].style.display = show ? "block" : "none";
    };

    module.Viewer.prototype.setClippingX = function(val) {
        if (val === undefined)
            return this._clipx;

        this._clipx = val;

        if (this.active !== undefined) {
            this.active.uniforms.doslicex.value = this._clipx;
        }
        this.schedule();
    }
    module.Viewer.prototype.setClippingY = function(val) {
        if (val === undefined)
            return this._clipy;

        this._clipy = val;

        if (this.active !== undefined) {
            this.active.uniforms.doslicey.value = this._clipy;
        }
        this.schedule();
    }
    module.Viewer.prototype.setClippingZ = function(val) {
        if (val === undefined)
            return this._clipz;

        this._clipz = val;

        if (this.active !== undefined) {
            this.active.uniforms.doslicez.value = this._clipz;
        }
        this.schedule();
    }

    return module;
}(mriview || {}));
