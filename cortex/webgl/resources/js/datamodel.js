/*
 *
 * Implements a limited parser for a .npy file. Only supports the listed dtypes, no complex ones
 * Supports slicing in the first dimension only (no strided slicing)
 * 
 *
 */
var dtypeNames = {
    "|u4":Uint32Array,
    "|u2":Uint16Array,
    "|u1":Uint8Array,
    "|i4":Int32Array,
    "|i2":Int16Array,
    "|i1":Int8Array,
    "<f4":Float32Array,
}

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

function parse_dict(dict) {
    dict = dict.trim();
    if (dict[0] != '{' || dict[dict.length-1] != '}')
        throw 'Invalid dictionary string';

    var parse = /['"]([^'"]*)['"]\s*:\s*['"]?((?:\([^'"\)]*\))|[^'",]*)['"]?,/g;
    var str, obj = {};
    while ((str = parse.exec(dict)) !== null) {
        obj[str[1]] = str[2];
    }
    return obj;
}

var loadCount = 0;
function NParray(dtype, shape) {
    this.dtype = dtype;
    this.shape = shape;
    this.size = shape[0];
    if (this.shape.length > 1) {
        this._slice = shape[1];
        for (var i = 2, il = shape.length; i < il; i++)
            this._slice *= shape[i];
        this.size *= this._slice;
        this.available = 0;
    }
    this.loaded = $.Deferred();
}
NParray.fromData = function(data, dtype, shape) {
    var arr = new NParray(dtype, shape);
    arr.data = data;
    return arr;
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

    return NParray.fromData(data, json.dtype, json.shape);
}
NParray.fromURL = function(url, callback) {
    console.log("Loading numpy array from " + url);
    loadCount++;
    $("#dataload").show();
    var hideload = function() {
        loadCount--;
        if (loadCount == 0)
            $("#dataload").hide();
    };

    // With thanks to https://gist.github.com/nvictus/88b3b5bfe587d32ac1ab519fd0009607
    var xhr = new XMLHttpRequest();
    xhr.open("GET", url, true);
    xhr.setRequestHeader("Range", "bytes=0-1023");
    xhr.responseType = "arraybuffer";

    xhr.onload = function() {
        if (xhr.status !== 200 && xhr.status !== 206) {
            console.error("Failed to fetch file: status " + xhr.status);
            hideload();
            return;
        }

        function asciiDecode(buf) {
            return String.fromCharCode.apply(null, new Uint8Array(buf));
        }

        function readUint16LE(buffer) {
            var view = new DataView(buffer);
            var val = view.getUint8(0);
            val |= view.getUint8(1) << 8;
            return val;
        }

        var buffer = xhr.response;
        // console.log("Got buffer: " + buffer);
        // console.log("Buffer length: " + buffer.byteLength);

        // Check we got a valid npy file
        var magic = asciiDecode(buffer.slice(0, 6));
        if (magic.slice(1, 6) != 'NUMPY')
            throw "Invalid npy file"
        // console.log("Magic: "+magic);

        // OK, now let's parse the header and get some info
        var version = new Uint8Array(buffer.slice(6, 8));
        // console.log("npy version " + version);

        var headerLength = readUint16LE(buffer.slice(8, 10));
        // console.log("Header length: " + headerLength);

        var headerStr = asciiDecode(buffer.slice(10, 10 + headerLength));
        // console.log("Header: " + headerStr);

        // Create the array object
        var info = parse_dict(asciiDecode(buffer.slice(10, 10 + headerLength)));
        var shape = info.shape.slice(1, info.shape.length-1).split(',');
        shape = shape.map(function(num) { return parseInt(num.trim()) });
        var array = new NParray(dtypeNames[info.descr], shape);

        // console.log("Array shape: " + array.shape);
        // console.log("Array size: " + array.size);
        // console.log("Array dtype: " + array.dtype);

        if (xhr.status == 206) {
            // We got partial data, we're streaming it
            var length = array.dtype.BYTES_PER_ELEMENT * array.size;
            var increment = array._slice * array.dtype.BYTES_PER_ELEMENT || length;
            var offset = 10 + headerLength;
            Stream(url, length, increment, offset).progress(array.update.bind(array)).done(hideload);
        } else if (xhr.status == 200) { 
            // Server doesn't support partial data, returned the whole thing
            array.update(buffer.slice(10 + headerLength));
            hideload();
            // console.log("Data length: " + array.data.length);
            // console.log("Data: " + array.data);
        }

        // Finish up
        hideload();
        callback(array);
    }

    xhr.onerror = function() {
        console.error("Failed to fetch file");
        hideload();
    };

    xhr.send();
}
NParray.prototype.update = function(buffer) {
    try {
        if (this.shape.length > 1) {
            this.data = new this.dtype(buffer, 0, (++this.available)*this._slice);
            this.loaded.notify(this.available);
        } else {
            this.data = new this.dtype(buffer);
            this.loaded.resolve();
        }
    } catch (e) {
        console.error("Error updating array:", e);
    }
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
        return NParray.fromData(this.data.subarray(slice*size, (slice+1)*size), this.dtype, shape);
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

function Stream(url, length, increment, offset) {
    //Implements a streaming interface via 206 requests
    var inc = increment || 8192;
    var off = offset || 0;

    var data = new Uint8Array(length);
    var chunk = 0, loaded = $.Deferred();

    var next = function() {
        var start = off + chunk*inc;
        var stop = start + inc - 1;
        var xhr = new XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.setRequestHeader("Range", "bytes="+start+"-"+stop);
        xhr.responseType = 'arraybuffer';
        xhr.addEventListener("load", function() {
            if (this.status == 206 && this.readyState == 4) {
                loaded.notify(this.response);
                chunk++;
                if (chunk*inc < length)
                    next();
                else
                    loaded.resolve(data.buffer);
            }
        }, false);
        xhr.send();
    }
    next();

    return loaded.then(null, null, function(buffer) {
        data.subarray(chunk*inc, chunk*inc+inc).set(new Uint8Array(buffer));
        return data.buffer;
    });
}