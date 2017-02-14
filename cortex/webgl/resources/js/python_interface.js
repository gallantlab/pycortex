function Websock() {
    this.ws = new WebSocket("ws://"+location.host+"/wsconnect/");
    this.ws.onopen = function(evt) {
        this.ws.send("connect");
    }.bind(this);
    this.ws.onmessage = function(evt) {
        var jsdat = classify(JSON.parse(evt.data));
        var func = this[jsdat.method];
        var resp = func.apply(this, jsdat.params);
        //Don't return jquery objects, 
        if (resp instanceof $) {
            this.ws.send(JSON.stringify(null));
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
        if (!(o[names[i]] instanceof Object || o[names[i]] instanceof Function))
            o = names[i]
        else 
            o = o[names[i]];
    }
    return [last, o];
}
Websock.prototype.query = function(name) {
    var names = {};
    var obj = this.get(name)[1];
    for (var name in obj) {
        names[name] = [typeof(obj[name])];
        if (names[name] != "object" && names[name] != "function")
            names[name].push(obj[name])
    }
    return names;
}
Websock.prototype.set = function(name, value) {
    var resp = this.get(name);
    var obj = resp[0], val = resp[1];
    try {
        obj[val] = value;
        resp = null;
    } catch (e) {
        resp = {error:e.message};
    }
    return resp;
}
Websock.prototype.run = function(name, params) {
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

