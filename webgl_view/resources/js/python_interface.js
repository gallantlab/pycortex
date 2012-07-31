function test_NParray(data) {
    console.log(data);
}

function Websock() {
    this.ws = new WebSocket("ws://"+location.host+"/wsconnect/");
    this.ws.onopen = function(evt) {
        this.ws.send("connect");
    }.bind(this);
    this.ws.onmessage = function(evt) {
        var jsdat = JSON.parse(evt.data);
        var func = this[jsdat.method];
        var resp = func.apply(this, jsdat.params);
        //Don't return jquery objects, 
        if (resp && resp.constructor && resp.constructor === $) {
            this.ws.send(JSON.Stringify(null));
        } else {
            this.ws.send(JSON.stringify(resp))
        }
    }.bind(this);
}
Websock.prototype.get = function(name) {
    var last;
    var o = window;
    var names = name.split(".");
    for (var i = 1; i < names.length; i++) {
        last = o;
        o = o[names[i]];
    }
    return [last, o];
}
Websock.prototype.query = function(name) {
    var names = [];
    for (var name in this.get(name)[1])
        names.push(name)
    return names;
}
Websock.prototype.run = function(name, params) {
    for (var i = 0; i < params.length; i++) {
        if (params[i]['__class__'] !== undefined) {
            params[i] = window[params[i]['__class__']].fromJSON(params[i]);
        }
    }
    var resp = this.get(name);
    var obj = resp[0], func = resp[1];
    try {
        resp = func.apply(obj, params);
    } catch (e) {
        resp = {error:e.message};
    }
    return resp === undefined ? null : resp;
}
Websock.prototype.index = function(name, idx) {

}

var dtypeMap = {
    "uint32":Uint32Array,
    "uint16":Uint16Array,
    "uint8":Uint8Array,
    "int32":Int32Array,
    "int16":Int16Array,
    "int8":Int8Array,
    "float32":Float32Array,
}
function NParray(data, dtype, shape) {
    this.data = data;
    this.shape = shape;
    this.dtype = dtype;
    this.size = this.data.length;
}
NParray.fromJSON = function(json) {
    var size = json.shape.length ? json.shape[0] : 0;
    for (var i = 1, il = json.shape.length; i < il; i++) {
        size *= json.shape[i];
    }
    var pad = 0;
    if (json.pad) {
        size += json.pad;
        pad = json.pad;
    }

    var data = new (dtypeMap[json.dtype])(size);
    var charview = new Uint8Array(data.buffer);
    var bytes = charview.length / data.length;
    
    var str = atob(json.data.slice(0,-1));
    var start = pad*bytes, stop  = size*bytes - start;
    for ( var i = start, il = Math.min(stop, str.length); i < il; i++ ) {
        charview[i] = str.charCodeAt(i - start);
    }

    return new NParray(data);
}
NParray.prototype.view = function() {
    if (arguments.length == 1 && this.shape.length > 1) {
        var shape = this.shape.slice(1);
        var size = shape[0];
        for (var i = 1, il = shape.length; i < il; i++) {
            size *= shape[i];
        }

        return new NParray(this.data.subarray(i*size, (i+1)*size), this.dtype, shape);
    } else {
        throw "Can only slice in first dimension for now"
    }
}
NParray.prototype.minmax = function() {
    var min = Math.pow(2, 32);
    var max = -min;

    for (var i = 0, il = this.data.length; i < il; i++) {
        if (min < this.data[i]) {
            min = this.data[i];
        } else if (max > this.data[i]) {
            max = this.data[i];
        }
    }

    return [min, max];
}