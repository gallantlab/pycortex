import os
import re
import time
import json
import stat
import email
import struct
import socket
import logging
import binascii
import datetime
import mimetypes
import multiprocessing as mp

import numpy as np
import tornado.web
import tornado.ioloop
from tornado import websocket
from tornado.web import HTTPError

cwd = os.path.split(os.path.abspath(__file__))[0]
hostname = socket.gethostname()

def make_base64(imgfile):
    with open(imgfile) as img:
        mtype = mimetypes.guess_type(imgfile)[0]
        data = binascii.b2a_base64(img.read()).strip()
        return "data:{mtype};base64,{data}".format(mtype=mtype, data=data)

class NPEncode(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float:
                obj = obj.astype(np.float32)
            elif obj.dtype == np.int:
                obj = obj.astype(np.int32)

            return dict(
                __class__="NParray",
                dtype=obj.dtype.descr[0][1], 
                shape=obj.shape, 
                data=binascii.b2a_base64(obj.tostring()))
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return super(NPEncode, self).default(obj)

class StaticFileHandler(tornado.web.RequestHandler):
    """StaticFileHandler from https://github.com/kzahel"""
    CACHE_MAX_AGE = 86400*365*10 #10 years

    _static_hashes = {}

    def initialize(self, path, default_filename=None):
        self.root = os.path.abspath(path) + os.path.sep
        self.default_filename = default_filename

    def head(self, path):
        self.get(path, include_body=False)

    @tornado.web.asynchronous
    def get(self, path, include_body=True):
        #logging.info('static request %s, %s' % (self.request.uri,  self.request.headers))
        if os.path.sep != "/":
            path = path.replace("/", os.path.sep)
        abspath = os.path.abspath(os.path.join(self.root, path))
        # os.path.abspath strips a trailing /
        # it needs to be temporarily added back for requests to root/
        if not (abspath + os.path.sep).startswith(self.root):
            raise HTTPError(403, "%s is not in root static directory", path)
        if os.path.isdir(abspath) and self.default_filename is not None:
            # need to look at the request.path here for when path is empty
            # but there is some prefix to the path that was already
            # trimmed by the routing
            if not self.request.path.endswith("/"):
                self.redirect(self.request.path + "/")
                return
            abspath = os.path.join(abspath, self.default_filename)
        if not os.path.exists(abspath):
            raise HTTPError(404)
        if not os.path.isfile(abspath):
            raise HTTPError(403, "%s is not a file", path)
        self.set_extra_headers(path)

        stat_result = os.stat(abspath)
        
        mime_type, encoding = mimetypes.guess_type(abspath)
        if mime_type:
            self.set_header("Content-Type", mime_type)

        self.set_header('Accept-Ranges','bytes')

        self.file = open(abspath, "rb")
        self._transforms = []

        if 'Range' not in self.request.headers:
            modified = datetime.datetime.fromtimestamp(stat_result[stat.ST_MTIME])
            self.set_header("Last-Modified", modified)

            cache_time = self.get_cache_time(path, modified, mime_type)
            if cache_time > 0:
                self.set_header("Expires", datetime.datetime.utcnow() + \
                                           datetime.timedelta(seconds=cache_time))
                self.set_header("Cache-Control", "max-age=" + str(cache_time))
            else:
                self.set_header("Cache-Control", "public")

            # Check the If-Modified-Since, and don't send the result if the
            # content has not been modified
            ims_value = self.request.headers.get("If-Modified-Since")
            if ims_value is not None:
                date_tuple = email.utils.parsedate(ims_value)
                if_since = datetime.datetime.fromtimestamp(time.mktime(date_tuple))
                if if_since >= modified:
                    self.set_status(304)
                    self.finish()
                    return

            self.bytes_start = 0
            self.bytes_end = stat_result.st_size - 1
            if not include_body:
                self.file.close()
                self.finish()
                return
        else:
            logging.info('got range string %s' % self.request.headers['Range'])
            self.set_status(206)
            rangestr = self.request.headers['Range'].split('=')[1]
            start, end = rangestr.split('-')
            logging.info('seeking to start %s' % start)
            self.bytes_start = int(start)
            self.file.seek(self.bytes_start)
            if not end:
                self.bytes_end = stat_result.st_size - 1
            else:
                self.bytes_end = int(end)

            clenheader = 'bytes %s-%s/%s' % (self.bytes_start, self.bytes_end, stat_result.st_size)
            self.set_header('Content-Range', clenheader)
            self.set_header('Content-Length', self.bytes_end-self.bytes_start+1)
            logging.info('set content range header %s' % clenheader)

        if 'If-Range' in self.request.headers:
            logging.debug('staticfilehandler had if-range header %s' % self.request.headers['If-Range'])


        self.bytes_remaining = self.bytes_end - self.bytes_start + 1
        self.set_header('Content-Length', str(self.bytes_remaining))
        self.bufsize = 4096 * 16
        #logging.info('writing to frontend: %s' % self._generate_headers())
        self.flush() # flush out the headers
        self.stream_one()

    def stream_one(self):
        if self.request.connection.stream.closed():
            self.file.close()
            return

        if self.bytes_remaining == 0:
            self.file.close()
            self.finish()
        else:
            data = self.file.read(min(self.bytes_remaining, self.bufsize))
            self.bytes_remaining -= len(data)
            #logging.info('read from disk %s, remaining %s' % (len(data), self.bytes_remaining))
            self.request.connection.stream.write( data, self.stream_one )

    def set_extra_headers(self, path):
        """For subclass to add extra headers to the response"""
        pass

    def get_cache_time(self, path, modified, mime_type):
        """Override to customize cache control behavior.

        Return a positive number of seconds to trigger aggressive caching or 0
        to mark resource as cacheable, only.

        By default returns cache expiry of 10 years for resources requested
        with "v" argument.
        """
        return self.CACHE_MAX_AGE if "v" in self.request.arguments else 0


class ClientSocket(websocket.WebSocketHandler):
    def initialize(self, parent):
        self.parent = parent

    def open(self):
        self.parent.sockets.append(self)

    def on_close(self):
        self.parent.sockets.remove(self)
        self.parent.clients.value -= 1
        if self.parent.clients.value == 0 and self.parent.disconnect_on_close:
            self.parent.stop()

    def on_message(self, message):
        if (message == "connect"):
            self.parent.clients.value += 1
            self.parent.c_evt.set()
        else:
            self.parent.lastmsg = message
            self.parent._response.send(message)

class WebApp(mp.Process):
    disconnect_on_close = True
    def __init__(self, handlers, port):
        super(WebApp, self).__init__()
        self.handlers = handlers + [
            (r"/wsconnect/", ClientSocket, dict(parent=self)),
            (r"/(.*)", StaticFileHandler, dict(path=cwd)),
        ]
        self._pipe, self.pipe = os.pipe()
        self._response, self.response = mp.Pipe()
        self.port = port
        self.clients = mp.Value('i', 0)
        self.c_evt = mp.Event()

    def run(self):
        self.sockets = []
        application = tornado.web.Application(self.handlers, gzip=True)
        application.listen(self.port)
        self.ioloop = tornado.ioloop.IOLoop.instance()
        self.ioloop.add_handler(self._pipe, self._send, self.ioloop.READ)
        self.ioloop.start()

    def stop(self):
        print("Stopping server")
        self.ioloop.stop()

    def _send(self, fd, event):
        nbytes, = struct.unpack('I', os.read(fd, 4))
        msg = os.read(fd, nbytes)
        if msg == "stop":
            self.ioloop.stop()
        else:
            for sock in self.sockets:
                sock.write_message(msg)

    def srvsend(self, **msg):
        if not isinstance(msg, str):
            msg = json.dumps(msg, cls=NPEncode)

        for sock in self.sockets:
            sock.write_message(msg)

    def srvresp(self):
        if self.response.poll():
            json.loads(self.response.recv())
        return self.lastmsg

    def send(self, **msg):
        if not isinstance(msg, str):
            msg = json.dumps(msg, cls=NPEncode)

        os.write(self.pipe, struct.pack('I', len(msg))+msg)        
        return [json.loads(self.response.recv()) for _ in range(self.clients.value)]

    def get_client(self):
        self.c_evt.wait()
        self.c_evt.clear()
        return JSProxy(self.send)

class JSProxy(object):
    def __init__(self, sendfunc, name="window"):
        self.send = sendfunc
        self.name = name
        self.attrs = self.send(method='query', params=[name])[0]
    
    def __getattr__(self, attr):
        assert attr in self.attrs
        if self.attrs[attr][0] in ["object", "function"]:
            return JSProxy(self.send, "%s.%s"%(self.name, attr))
        else:
            return self.attrs[attr][1]

    def __setattr__(self, attr, value):
        if not hasattr(self, "attrs") or attr not in self.attrs:
            return super(JSProxy, self).__setattr__(attr, value)

        assert self.attrs[attr] not in ["object", "function"]
        resp = self.send(method='set', params=["%s.%s"%(self.name, attr), value])
        if isinstance(resp[0], dict) and "error" in resp[0]:
            raise Exception(resp[0]['error'])
        else:
            return resp

    def __repr__(self):
        return "<JS: %s>"%self.name

    def __dir__(self):
        return list(self.attrs)

    def __call__(self, *args):
        resp = self.send(method='run', params=[self.name, args])
        if isinstance(resp[0], dict) and "error" in resp[0]:
            raise Exception(resp[0]['error'])
        else:
            return resp

    def __getitem__(self, idx):
        assert not isinstance(idx, (slice, list, tuple, np.ndarray))
        resp = self.send(method='index', params=[self.name, args])
        if isinstance(resp[0], dict) and "error" in resp[0]:
            raise Exception(resp[0]['error'])
        else:
            return resp

class JSLocal(JSProxy):
    def __init__(self, sendfunc, getfunc, **kwargs):
        import time
        def sendget(**msg):
            sendfunc(**msg)
            return getfunc()
        super(JSLocal, self).__init__(sendget, **kwargs)

if __name__ == "__main__":
    app = WebApp([], 8888)
    app.start()
    app.join()