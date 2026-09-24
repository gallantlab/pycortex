// A ZIP writer for files the browser has already compressed.
//
// The animation panel renders a movie as one PNG per frame, and a page cannot
// hand the person hundreds of separate downloads: browsers throttle them, or
// ask whether to allow them. So the frames are packed into a single .zip in the
// page, and that is downloaded instead.
//
// PNG is compressed already, so every entry is *stored* (compression method 0)
// rather than deflated. That keeps this small: a stored entry is its bytes
// behind a fixed header, and the only computation is a CRC-32 of each one.
// The archive is assembled as a Blob of parts -- headers interleaved with the
// frames' own Blobs -- so the image bytes are never copied into one buffer,
// and a browser that backs large Blobs with disk (as Chrome does) need not
// hold the whole movie in memory.
//
// Classic ZIP only: at most 65535 entries and 4 GiB. Past either, add() throws
// rather than writing ZIP64.

var jsplot = (function (module) {
    module.zipstore = (function (zs) {

    var MAX_ENTRIES = 0xFFFF;
    var MAX_OFFSET = 0xFFFFFFFF;

    var CRC_TABLE = (function() {
        var table = new Uint32Array(256);
        for (var n = 0; n < 256; n++) {
            var c = n;
            for (var k = 0; k < 8; k++)
                c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
            table[n] = c >>> 0;
        }
        return table;
    }());

    // CRC-32 (the one ZIP, gzip and PNG use) of a Uint8Array.
    zs.crc32 = function(bytes) {
        var crc = 0xFFFFFFFF;
        for (var i = 0; i < bytes.length; i++)
            crc = CRC_TABLE[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
        return (crc ^ 0xFFFFFFFF) >>> 0;
    };

    // The MS-DOS time and date fields ZIP stores, for `when`.
    function dosDateTime(when) {
        return {
            time: (when.getHours() << 11) | (when.getMinutes() << 5) |
                  (when.getSeconds() >> 1),
            date: ((when.getFullYear() - 1980) << 9) |
                  ((when.getMonth() + 1) << 5) | when.getDate(),
        };
    }

    // Collects stored entries and turns them into one application/zip Blob.
    zs.ZipWriter = function() {
        this._parts = [];      // local headers and file data, in order
        this._central = [];    // one central directory record per entry
        this._offset = 0;      // where the next local header starts
        this._stamp = dosDateTime(new Date());
        this._encoder = new TextEncoder();
    };

    // Add `blob` under `name`. Returns a promise, since reading the blob to
    // work out its CRC is asynchronous; add entries one at a time.
    zs.ZipWriter.prototype.add = function(name, blob) {
        var self = this;
        if (this.count() >= MAX_ENTRIES)
            return Promise.reject(new Error(
                "A zip holds at most " + MAX_ENTRIES + " files; render a " +
                "shorter range of frames"));

        return blob.arrayBuffer().then(function(buffer) {
            var data = new Uint8Array(buffer);
            var crc = zs.crc32(data);
            var nameBytes = self._encoder.encode(name);

            if (self._offset + 30 + nameBytes.length + data.length > MAX_OFFSET)
                throw new Error("A zip holds at most 4 GiB; render a shorter " +
                                "range of frames, or a smaller size");

            // Bit 11 of the flags: the name is UTF-8.
            var local = new DataView(new ArrayBuffer(30));
            local.setUint32(0, 0x04034b50, true);   // local file header
            local.setUint16(4, 20, true);           // version needed: 2.0
            local.setUint16(6, 0x0800, true);       // flags
            local.setUint16(8, 0, true);            // method: stored
            local.setUint16(10, self._stamp.time, true);
            local.setUint16(12, self._stamp.date, true);
            local.setUint32(14, crc, true);
            local.setUint32(18, data.length, true); // compressed size
            local.setUint32(22, data.length, true); // uncompressed size
            local.setUint16(26, nameBytes.length, true);
            local.setUint16(28, 0, true);           // extra field length

            var central = new DataView(new ArrayBuffer(46));
            central.setUint32(0, 0x02014b50, true); // central directory header
            central.setUint16(4, 20, true);         // version made by
            central.setUint16(6, 20, true);         // version needed
            central.setUint16(8, 0x0800, true);
            central.setUint16(10, 0, true);
            central.setUint16(12, self._stamp.time, true);
            central.setUint16(14, self._stamp.date, true);
            central.setUint32(16, crc, true);
            central.setUint32(20, data.length, true);
            central.setUint32(24, data.length, true);
            central.setUint16(28, nameBytes.length, true);
            // extra, comment, disk, internal and external attributes: all 0
            central.setUint32(42, self._offset, true);

            self._parts.push(local.buffer, nameBytes, blob);
            self._central.push(central.buffer, nameBytes);
            self._offset += 30 + nameBytes.length + data.length;
        });
    };

    // The finished archive.
    zs.ZipWriter.prototype.finish = function() {
        var size = 0;
        for (var i = 0; i < this._central.length; i++)
            size += this._central[i].byteLength;

        var end = new DataView(new ArrayBuffer(22));
        var count = this._central.length / 2;
        end.setUint32(0, 0x06054b50, true);         // end of central directory
        end.setUint16(8, count, true);              // entries on this disk
        end.setUint16(10, count, true);             // entries in total
        end.setUint32(12, size, true);
        end.setUint32(16, this._offset, true);      // where the directory starts

        return new Blob(this._parts.concat(this._central, [end.buffer]),
                        {type: "application/zip"});
    };

    // How many files have been added.
    zs.ZipWriter.prototype.count = function() {
        return this._central.length / 2;
    };

    return zs;
    }(module.zipstore || {}));

    return module;
}(jsplot || {}));
