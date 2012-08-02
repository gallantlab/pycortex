import random
import webbrowser
import tornado.web
import numpy as np

from .. import utils, vtkctm

import serve

#Raw volume movie:      [[3, 4], t, z, y, x]
#Raw volume image:      [[3, 4], z, y, x]
#Raw cortical movie:    [[3, 4], t, vox]
#Raw cortical image:    [[3, 4], vox]
#Regular volume movie:  [t, z, y, x]
#Regular volume image:  [z, y, x]
#Regular cortical movie: [t, vox]
#Regular cortical image: [vox]

def _normalize_data(data, mask):
    if isinstance(data, str):
        import nibabel
        data = nibabel.load(data).get_data().T

    raw = data.dtype.type == np.uint8
    assert raw and data.shape[0] in [3, 4] or not raw
    shape = data.shape[1:] if raw else data.shape
    movie = len(shape) in [2, 4]
    volume = len(shape) in [3, 4]

    if raw:
        if volume:
            if movie:
                return data[:, :, mask].transpose(1, 2, 0)
            else:
                return data[:, mask].T
        else: #cortical
            if movie:
                return data.transpose(1,2,0)
            else:
                return data.T
    else: #regular
        if volume:
            if movie:
                return data[:, mask]
            else:
                return data[mask]
        else: #cortical
            return data

def show(data, subject, xfmname, types=("inflated",)):
    ctm = vtkctm.get_pack(subject, xfmname, types, method='raw', level=0)
    mask = utils.get_cortical_mask(subject, xfmname)

    class JSMixer(serve.JSProxy):
        def setData(self, data, name="dataset"):
            Proxy = serve.JSProxy(self.send, "window.viewer.setData")
            return Proxy(_normalize_data(data, mask), name)

    class WebApp(serve.WebApp):
        def get_client(self):
            self.c_evt.wait()
            self.c_evt.clear()
            return JSMixer(self.send, "window.viewer")

    class CTMHandler(tornado.web.RequestHandler):
        def get(self):
            self.set_header("Content-Type", "application/octet-stream")
            self.write(open(ctm).read())

    server = WebApp([(r'/ctm/', CTMHandler)], random.randint(1024, 65536))
    server.start()
    webbrowser.open("http://localhost:%d/mixer.html"%server.port)
    client = server.get_client()
    client.server = server
    client.load("/ctm/")
    client.setData(data)
    return client