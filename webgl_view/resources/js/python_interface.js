function Websock(obj) {
    this.obj = obj;
    this.methods = [];
    for (var name in this.obj) {
        if (typeof(this.obj[name]) == "function")
            this.methods.push(name);
    }
    this.ws = new WebSocket("ws://"+location.hostname+"/wsconnect/");
    this.ws.binaryType = 'arraybuffer';
    this.ws.onopen = function() {
        this.ws.send(this.methods);
    }.bind(this);
    this.ws.onmessage = function(evt) {

    }
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
    if (json.pad) 
        size += json.pad;

    var data = new (dtypeMap[json.dtype])(size);
    var charview = new Uint8Array(data.buffer);
    var bytes = charview.length / data.length;
    
    var str = atob(json.data);
    var start = pad ? pad*bytes : 0;
    var stop  = pad ? size*bytes - start : size*bytes;
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