var allStims = [];
function classify(data) {
    if (data['__class__'] !== undefined) {
        data = window[data['__class__']].fromJSON(data);
    } else {
        for (var name in data) {
            if (data[name] instanceof Object){
                data[name] = classify(data[name])
            }
        }
    }
    return data
}

var dtypeNames = {
    "uint32":Uint32Array,
    "uint16":Uint16Array,
    "uint8":Uint8Array,
    "int32":Int32Array,
    "int16":Int16Array,
    "int8":Int8Array,
    "float32":Float32Array,
}
var dtypeMap = [Uint32Array, Uint16Array, Uint8Array, Int32Array, Int16Array, Int8Array, Float32Array]
var loadCount = 0;
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

    var data = new (dtypeNames[json.dtype])(size);
    var charview = new Uint8Array(data.buffer);
    var bytes = charview.length / data.length;
    
    var str = atob(json.data.slice(0,-1));
    var start = pad*bytes, stop  = size*bytes - start;
    for ( var i = start, il = Math.min(stop, str.length); i < il; i++ ) {
        charview[i] = str.charCodeAt(i - start);
    }

    return new NParray(data, json.dtype, json.shape);
}
NParray.fromURL = function(url, callback) {
    loadCount++;
    $("#dataload").show();
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';
    xhr.onload = function(e) {
        if (this.readyState == 4 && this.status == 200) {
            loadCount--;
            var data = new Uint32Array(this.response, 0, 2);
            var dtype = dtypeMap[data[0]];
            var ndim = data[1];
            var shape = new Uint32Array(this.response, 8, ndim);
            var array = new dtype(this.response, (ndim+2)*4);
            callback(new NParray(array, dtype, shape));
            if (loadCount == 0)
                $("#dataload").hide();
        }
    };
    xhr.send()
}
NParray.prototype.view = function() {
    if (arguments.length == 1 && this.shape.length > 1) {
        var shape;
        var slice = arguments[0];
        if (this.shape.slice)
            shape = this.shape.slice(1);
        else
            shape = this.shape.subarray(1);
        var size = shape[0];
        for (var i = 1, il = shape.length; i < il; i++) {
            size *= shape[i];
        }
        return new NParray(this.data.subarray(slice*size, (slice+1)*size), this.dtype, shape);
    } else {
        throw "Can only slice in first dimension for now"
    }
}
NParray.prototype.minmax = function() {
    var min = Math.pow(2, 32);
    var max = -min;

    for (var i = 0, il = this.data.length; i < il; i++) {
        if (this.data[i] < min) {
            min = this.data[i];
        }
        if (this.data[i] > max) {
            max = this.data[i];
        }
    }

    return [min, max];
}
