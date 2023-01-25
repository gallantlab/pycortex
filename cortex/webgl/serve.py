import os
import re
import time
import json
import stat
import email
try:  # python 2
    from Queue import Queue
except ImportError:  # python 3
    from queue import Queue
import struct
import socket
import logging
import binascii
import base64
import datetime
import mimetypes
import functools
import threading

import numpy as np
import tornado.web
import tornado.ioloop
import tornado.httpserver
from tornado import websocket
from tornado.web import HTTPError

cwd = os.path.split(os.path.abspath(__file__))[0]
hostname = socket.gethostname()

def make_base64(imgfile):
    with open(imgfile, 'rb') as img:
        mtype = mimetypes.guess_type(imgfile)[0]
        imbytes = base64.encodebytes(img.read())
        data = imbytes.decode('utf-8').strip()
        return u"data:{mtype};base64,{data}".format(mtype=mtype, data=data)

class NPEncode(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == float:
                obj = obj.astype(np.float32)
            elif obj.dtype == int:
                obj = obj.astype(np.int32)

            return dict(
                __class__="NParray",
                dtype=obj.dtype.descr[0][1], 
                shape=obj.shape, 
                data=binascii.b2a_base64(obj.tostring()).decode('utf-8'))
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8,
                              np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return super(NPEncode, self).default(obj)

class StaticFileHandler(tornado.web.RequestHandler):
    """A simple handler that can serve static content from a directory.

    To map a path to this handler for a static data directory /var/www,
    you would add a line to your application like::

        application = web.Application([
            (r"/static/(.*)", web.StaticFileHandler, {"path": "/var/www"}),
        ])

    The local root directory of the content should be passed as the "path"
    argument to the handler.

    To support aggressive browser caching, if the argument "v" is given
    with the path, we set an infinite HTTP expiration header. So, if you
    want browsers to cache a file indefinitely, send them to, e.g.,
    /static/images/myimage.png?v=xxx. Override ``get_cache_time`` method for
    more fine-grained cache control.
    """
    CACHE_MAX_AGE = 86400*365*10 #10 years

    _static_hashes = {}

    def initialize(self, path, default_filename=None):
        self.root = os.path.abspath(path) + os.path.sep
        self.default_filename = default_filename

    def head(self, path):
        self.get(path, include_body=False)

    def get(self, path, include_body=True):
        self._auto_finish = False

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

    @classmethod
    def make_static_url(cls, settings, path):
        """Constructs a versioned url for the given path.

        This method may be overridden in subclasses (but note that it is
        a class method rather than an instance method).
        
        ``settings`` is the `Application.settings` dictionary.  ``path``
        is the static path being requested.  The url returned should be
        relative to the current host.
        """
        hashes = cls._static_hashes
        abs_path = os.path.join(settings["static_path"], path)
        if abs_path not in hashes:
            try:
                f = open(abs_path, "rb")
                hashes[abs_path] = hashlib.md5(f.read()).hexdigest()
                f.close()
            except Exception:
                logging.error("Could not open static file %r", path)
                hashes[abs_path] = None
        static_url_prefix = settings.get('static_url_prefix', '/static/')
        if hashes.get(abs_path):
            return static_url_prefix + path + "?v=" + hashes[abs_path][:5]
        else:
            return static_url_prefix + path


class ClientSocket(websocket.WebSocketHandler):
    def initialize(self, parent):
        self.parent = parent

    def open(self):
        self.parent.sockets.append(self)

    def on_close(self):
        self.parent.sockets.remove(self)
        if self.parent.n_clients == 0 and self.parent.disconnect_on_close:
            self.parent.stop()

    def on_message(self, message):
        if (message == "connect"):
            self.parent.connect.set()
        else:
            self.parent.response.put(message)

class WebApp(threading.Thread):
    daemon = True
    disconnect_on_close = True

    def __init__(self, handlers, port):
        super(WebApp, self).__init__()
        self.handlers = handlers + [
            (r"/wsconnect/", ClientSocket, dict(parent=self)),
            (r"/(.*)", tornado.web.StaticFileHandler, dict(path=cwd)),
        ]
        self.port = port
        self.response = Queue()
        self.connect = threading.Event()
        self.sockets = []

    @property
    def n_clients(self):
        num = len(self.sockets)
        return num

    def run(self):
        ioloop = tornado.ioloop.IOLoop()
        ioloop.clear_current()
        ioloop.make_current()
        self.ioloop = ioloop
        application = tornado.web.Application(self.handlers, gzip=True)
        # If tornado version is 5.0 or greater, io_loop arg does not exist
        if tornado.version_info[0] < 5:
            self.server = tornado.httpserver.HTTPServer(application, io_loop=ioloop)
        else:
            self.server = tornado.httpserver.HTTPServer(application)
        self.server.listen(self.port)
        ioloop.start()

    def stop(self):
        print("Stopping server")
        self.server.stop()
        tornado.ioloop.IOLoop.current().stop()

    def send(self, **msg):
        if not isinstance(msg, str):
            msg = json.dumps(msg, cls=NPEncode, ensure_ascii=False)

        async def _send(sockets, msg):
            for sock in sockets:
                await sock.write_message(msg)

        self.ioloop.add_callback(_send, self.sockets, msg)

        try:
            return [json.loads(self.response.get(timeout=2)) for _ in range(self.n_clients)]
        except:
            return [None for _ in range(self.n_clients)]

    def get_client(self):
        self.connect.wait()
        self.connect.clear()
        return JSProxy(self.send)

class JSProxy(object):
    def __init__(self, sendfunc, name="window"):
        super(JSProxy, self).__setattr__('send', sendfunc)
        super(JSProxy, self).__setattr__('name', name)
        
        # self.attrs = self.send(method='query', params=[self.name])[0]
        self.max_time_retry = 10.  # in seconds

    @property
    def attrs(self):
        return_value = self.send(method='query', params=[self.name])[0]
        # Sometimes the return value can be None or an int (I assume an error value).
        # This can be caused by the delay in updating the JS viewer.
        # Waiting for 0.1 s should be enough.
        if return_value is None or not isinstance(return_value, dict):
            time.sleep(0.1)
            return_value = self.send(method='query', params=[self.name])[0]
        return return_value

    def __getattr__(self, attr):
        # if attr == 'attrs':
        #    return self.send(method='query', params=[self.name])[0]
        tstart = time.time()
        # To avoid querying too many times, assign self.attrs to attrs
        attrs = self.attrs
        while attr not in attrs and time.time() - tstart < self.max_time_retry:
            time.sleep(0.1)
            attrs = self.attrs
        if attr not in attrs:
            raise KeyError(f"Attribute '{attr}' not found in {self}")

        if attrs[attr][0] in ["object", "function"]:
            return JSProxy(self.send, "%s.%s"%(self.name, attr))
        else:
            return attrs[attr][1]

    def __setattr__(self, attr, value):
        if hasattr(self, "attrs") and self.attrs is None:
            return super(JSProxy, self).__setattr__(attr, value)
        if not hasattr(self, "attrs") or attr not in self.attrs:
            return super(JSProxy, self).__setattr__(attr, value)

        assert self.attrs[attr] not in ["object", "function"]
        resp = self.send(method='set',
                         params=["%s.%s"%(self.name, attr), value])
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
        return JSProxy(self.send, "%s.%d"%(self.name, idx))

if __name__ == "__main__":
    app = WebApp([], 8888)
    app.start()
    app.join()
