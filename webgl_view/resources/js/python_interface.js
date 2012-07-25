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