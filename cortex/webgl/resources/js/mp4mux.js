// An MP4 muxer for one H.264 video track, fed by the browser's VideoEncoder.
//
// WebCodecs encodes frames but does not write files: VideoEncoder hands back a
// stream of encoded chunks, plus (with avc: {format: "avc"}) the decoder
// configuration record the MP4 container needs. This packs those into a
// playable .mp4 -- a single constant-frame-rate track with no audio, which is
// all the animation panel renders, so none of the container's generality is
// needed.
//
// The file is written "fast start": ftyp, then moov (the index), then mdat (the
// frames), so it can play before it has finished loading. Encoded video is
// small, so every chunk is held in memory until finish() lays the file out;
// the samples go into one chunk of the mdat, which is why stco has a single
// entry. Frames must arrive in presentation order -- no B-frame reordering --
// and addChunk() says so if they do not.
//
// Box layout follows ISO/IEC 14496-12 (the base media file format) and
// 14496-15 (the avc1 sample entry and its avcC record).

var jsplot = (function (module) {
    module.mp4mux = (function (mm) {

    var MAX_SIZE = 0xFFFFFFFF;

    // ------------------------------------------------------------------
    // Choosing an H.264 codec string
    // ------------------------------------------------------------------

    // H.264 levels as [level_idc, max frame size in macroblocks, max
    // macroblocks per second], from Table A-1 of the standard. 4.1 is left
    // out: it differs from 4.0 only in bitrate.
    var LEVELS = [
        [0x1E, 1620, 40500],       // 3.0
        [0x1F, 3600, 108000],      // 3.1
        [0x20, 5120, 216000],      // 3.2
        [0x28, 8192, 245760],      // 4.0
        [0x2A, 8704, 522240],      // 4.2
        [0x32, 22080, 589824],     // 5.0
        [0x33, 36864, 983040],     // 5.1
        [0x34, 36864, 2073600],    // 5.2
        [0x3C, 139264, 4177920],   // 6.0
        [0x3D, 139264, 8355840],   // 6.1
        [0x3E, 139264, 16711680],  // 6.2
    ];

    function hex2(n) {
        return (n < 16 ? "0" : "") + n.toString(16);
    }

    // Codec strings for H.264 High profile at every level that can carry
    // width x height at fps, lowest first. The lowest is the most widely
    // playable, but an encoder may only accept some of them, so the caller
    // tries them in turn (see mm.encoderConfig).
    mm.avcCodecs = function(width, height, fps) {
        var wmb = Math.ceil(width / 16), hmb = Math.ceil(height / 16);
        var frame = wmb * hmb, codecs = [];
        for (var i = 0; i < LEVELS.length; i++) {
            var maxfs = LEVELS[i][1], maxmbps = LEVELS[i][2];
            // Besides the frame size, each dimension is capped at
            // sqrt(8 * MaxFS) macroblocks.
            var side = Math.sqrt(8 * maxfs);
            if (frame <= maxfs && frame * fps <= maxmbps &&
                    wmb <= side && hmb <= side)
                codecs.push("avc1.6400" + hex2(LEVELS[i][0]));
        }
        return codecs;
    };

    // A bitrate that leaves renders of the brain clean: 0.2 bits per pixel
    // per frame, within 2-50 Mb/s.
    mm.bitrate = function(width, height, fps) {
        return Math.round(Math.min(Math.max(width * height * fps * 0.2, 2e6), 50e6));
    };

    // The first VideoEncoder configuration the browser will accept for an
    // H.264 video of this size, as a promise; it rejects with a message that
    // says so when there is none. `width` and `height` must be even, since
    // H.264 stores colour at half resolution.
    mm.encoderConfig = function(width, height, fps) {
        if (typeof VideoEncoder === "undefined")
            return Promise.reject(new Error(
                "This browser cannot encode video (no WebCodecs VideoEncoder)"));

        var codecs = mm.avcCodecs(width, height, fps);
        var bitrate = mm.bitrate(width, height, fps);

        function attempt(i) {
            if (i >= codecs.length)
                return Promise.reject(new Error(
                    "This browser's H.264 encoder cannot make a " + width +
                    " × " + height + " video; choose a smaller size, or " +
                    "render PNG frames instead"));
            var config = {codec: codecs[i], width: width, height: height,
                          bitrate: bitrate, framerate: fps,
                          avc: {format: "avc"}};
            return VideoEncoder.isConfigSupported(config).then(function(s) {
                return s.supported ? config : attempt(i + 1);
            }, function() {
                return attempt(i + 1);
            });
        }
        return attempt(0);
    };

    // ------------------------------------------------------------------
    // Writing boxes
    // ------------------------------------------------------------------

    // A growable big-endian byte buffer.
    function Bytes() {
        this.parts = [];
        this.length = 0;
    }
    Bytes.prototype.push = function(u8) {
        this.parts.push(u8);
        this.length += u8.length;
        return this;
    };
    Bytes.prototype.u8 = function(v) {
        return this.push(new Uint8Array([v & 0xFF]));
    };
    Bytes.prototype.u16 = function(v) {
        var b = new Uint8Array(2);
        new DataView(b.buffer).setUint16(0, v);
        return this.push(b);
    };
    Bytes.prototype.u32 = function(v) {
        var b = new Uint8Array(4);
        new DataView(b.buffer).setUint32(0, v >>> 0);
        return this.push(b);
    };
    Bytes.prototype.zeros = function(n) {
        return this.push(new Uint8Array(n));
    };
    Bytes.prototype.fourcc = function(s) {
        var b = new Uint8Array(4);
        for (var i = 0; i < 4; i++)
            b[i] = s.charCodeAt(i);
        return this.push(b);
    };
    Bytes.prototype.bytes = function() {
        var out = new Uint8Array(this.length), at = 0;
        for (var i = 0; i < this.parts.length; i++) {
            out.set(this.parts[i], at);
            at += this.parts[i].length;
        }
        return out;
    };

    // A box: 32-bit size, four-character type, then `body` (a Bytes).
    function box(type, body) {
        var out = new Bytes();
        out.u32(8 + body.length).fourcc(type);
        for (var i = 0; i < body.parts.length; i++)
            out.push(body.parts[i]);
        return out;
    }

    // A full box: a box whose body starts with a version byte and 24 bits of
    // flags.
    function fullBox(type, version, flags, body) {
        var head = new Bytes();
        head.u8(version).u8(flags >> 16).u8(flags >> 8).u8(flags);
        for (var i = 0; i < body.parts.length; i++)
            head.push(body.parts[i]);
        return box(type, head);
    }

    // Boxes with several children: concatenate them into one body.
    function container(type, children) {
        var body = new Bytes();
        for (var i = 0; i < children.length; i++)
            for (var j = 0; j < children[i].parts.length; j++)
                body.push(children[i].parts[j]);
        return box(type, body);
    }

    // The unity transform every track and movie header carries.
    function matrix(b) {
        return b.u32(0x00010000).u32(0).u32(0)
                .u32(0).u32(0x00010000).u32(0)
                .u32(0).u32(0).u32(0x40000000);
    }

    // ------------------------------------------------------------------
    // The muxer
    // ------------------------------------------------------------------

    // Collects VideoEncoder output for a width x height video at `fps` frames
    // per second. width and height are the encoded size, which must match what
    // the encoder was configured with.
    mm.Mp4Muxer = function(width, height, fps) {
        this.width = width;
        this.height = height;
        // Every sample lasts exactly 1000 ticks at this timescale, so a whole
        // number of frames per second never accumulates rounding error.
        this.timescale = Math.round(fps * 1000);
        this.delta = 1000;
        this.samples = [];     // {data: Uint8Array, key: bool}
        this.avcC = null;      // the decoder configuration record
        this._lastTimestamp = -Infinity;
    };

    // Feed one EncodedVideoChunk, with the metadata VideoEncoder passed along
    // with it -- the first carries the avcC record.
    mm.Mp4Muxer.prototype.addChunk = function(chunk, metadata) {
        if (metadata && metadata.decoderConfig &&
                metadata.decoderConfig.description !== undefined) {
            var d = metadata.decoderConfig.description;
            this.avcC = d instanceof ArrayBuffer ? new Uint8Array(d.slice(0)) :
                new Uint8Array(d.buffer.slice(d.byteOffset,
                                              d.byteOffset + d.byteLength));
        }
        if (chunk.timestamp < this._lastTimestamp)
            throw new Error("The encoder reordered frames (B-frames), which " +
                            "this muxer does not support");
        this._lastTimestamp = chunk.timestamp;

        var data = new Uint8Array(chunk.byteLength);
        chunk.copyTo(data);
        this.samples.push({data: data, key: chunk.type === "key"});
    };

    mm.Mp4Muxer.prototype._moov = function(mdatOffset) {
        var n = this.samples.length;
        var duration = n * this.delta;
        var i;

        var mvhd = new Bytes();
        mvhd.u32(0).u32(0)                        // creation, modification time
            .u32(this.timescale).u32(duration)
            .u32(0x00010000).u16(0x0100)          // rate 1.0, volume 1.0
            .zeros(10);                           // reserved
        matrix(mvhd).zeros(24)                    // pre_defined
            .u32(2);                              // next track ID

        var tkhd = new Bytes();
        tkhd.u32(0).u32(0)                        // creation, modification time
            .u32(1).u32(0)                        // track ID, reserved
            .u32(duration).zeros(8)
            .u16(0).u16(0)                        // layer, alternate group
            .u16(0).u16(0);                       // volume (video: 0), reserved
        matrix(tkhd).u32(this.width << 16).u32(this.height << 16);

        var mdhd = new Bytes();
        mdhd.u32(0).u32(0).u32(this.timescale).u32(duration)
            .u16(0x55C4)                          // language: "und"
            .u16(0);

        var hdlr = new Bytes();
        hdlr.u32(0).fourcc("vide").zeros(12);
        hdlr.push(new TextEncoder().encode("VideoHandler\u0000"));

        var vmhd = new Bytes();
        vmhd.u16(0).zeros(6);                     // graphics mode, opcolor

        var dref = new Bytes();
        dref.u32(1);
        var url = fullBox("url ", 0, 1, new Bytes());   // 1: data is in this file
        for (i = 0; i < url.parts.length; i++)
            dref.push(url.parts[i]);

        var avcC = new Bytes();
        avcC.push(this.avcC);
        var avc1 = new Bytes();
        avc1.zeros(6).u16(1)                      // reserved, data reference index
            .u16(0).u16(0).zeros(12)              // pre_defined, reserved
            .u16(this.width).u16(this.height)
            .u32(0x00480000).u32(0x00480000)      // 72 dpi
            .u32(0).u16(1)                        // reserved, frame count
            .zeros(32)                            // compressor name
            .u16(0x0018).u16(0xFFFF);             // depth, pre_defined -1
        var avcCBox = box("avcC", avcC);
        for (i = 0; i < avcCBox.parts.length; i++)
            avc1.push(avcCBox.parts[i]);

        var stsd = new Bytes();
        stsd.u32(1);
        var avc1Box = box("avc1", avc1);
        for (i = 0; i < avc1Box.parts.length; i++)
            stsd.push(avc1Box.parts[i]);

        var stts = new Bytes();
        stts.u32(1).u32(n).u32(this.delta);

        var keys = [];
        for (i = 0; i < n; i++)
            if (this.samples[i].key)
                keys.push(i + 1);                 // sample numbers are 1-based
        var stss = new Bytes();
        stss.u32(keys.length);
        for (i = 0; i < keys.length; i++)
            stss.u32(keys[i]);

        var stsc = new Bytes();
        stsc.u32(1).u32(1).u32(n).u32(1);         // one chunk holds every sample

        var stsz = new Bytes();
        stsz.u32(0).u32(n);
        for (i = 0; i < n; i++)
            stsz.u32(this.samples[i].data.length);

        var stco = new Bytes();
        stco.u32(1).u32(mdatOffset);

        var stbl = container("stbl", [
            fullBox("stsd", 0, 0, stsd), fullBox("stts", 0, 0, stts),
            fullBox("stss", 0, 0, stss), fullBox("stsc", 0, 0, stsc),
            fullBox("stsz", 0, 0, stsz), fullBox("stco", 0, 0, stco)]);
        var minf = container("minf", [
            fullBox("vmhd", 0, 1, vmhd),
            container("dinf", [fullBox("dref", 0, 0, dref)]), stbl]);
        var mdia = container("mdia", [
            fullBox("mdhd", 0, 0, mdhd), fullBox("hdlr", 0, 0, hdlr), minf]);
        var trak = container("trak", [fullBox("tkhd", 0, 3, tkhd), mdia]);
        return container("moov", [fullBox("mvhd", 0, 0, mvhd), trak]);
    };

    // The finished file, as a video/mp4 Blob.
    mm.Mp4Muxer.prototype.finish = function() {
        if (this.samples.length === 0)
            throw new Error("No frames were encoded");
        if (this.avcC === null)
            throw new Error("The encoder never supplied its configuration " +
                            "(configure it with avc: {format: \"avc\"})");

        var ftyp = new Bytes();
        ftyp.fourcc("isom").u32(0x200)
            .fourcc("isom").fourcc("iso2").fourcc("avc1").fourcc("mp41");
        ftyp = box("ftyp", ftyp);

        var payload = 0, i;
        for (i = 0; i < this.samples.length; i++)
            payload += this.samples[i].data.length;

        // moov's size does not depend on the offset it records, so lay it out
        // once to measure it, then again with the real offset.
        var moovSize = this._moov(0).length;
        var mdatOffset = ftyp.length + moovSize + 8;
        if (mdatOffset + payload > MAX_SIZE)
            throw new Error("The video would be larger than 4 GiB; render a " +
                            "shorter range of frames");

        var mdatHead = new Bytes();
        mdatHead.u32(8 + payload).fourcc("mdat");

        var parts = [ftyp.bytes(), this._moov(mdatOffset).bytes(),
                     mdatHead.bytes()];
        for (i = 0; i < this.samples.length; i++)
            parts.push(this.samples[i].data);
        return new Blob(parts, {type: "video/mp4"});
    };

    return mm;
    }(module.mp4mux || {}));

    return module;
}(jsplot || {}));
