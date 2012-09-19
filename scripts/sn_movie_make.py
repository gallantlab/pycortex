import os
import numpy as np
import tempfile
from scipy.interpolate import InterpolatedUnivariateSpline as Spline, interp1d


def _zoom(cam, distance):
    vec = cam.position - cam.focal_point
    d = np.sqrt((vec**2).sum())
    cam.position = (distance / d * vec) + cam.focal_point

def rotate(mixer, h5, stops=[0, 0.33, 0.66], rot_frames=15*8, n_cycles=1, grow_frames=15*3):
    tmp = tempfile.mkdtemp()
    mixer.reset_view(False)

    dist = Spline([0,0.3,0.6,1], [320, 350, 350, 690])
    cam = mixer.figure.camera
    
    i = 0
    for s, e in zip(stops, stops[1:]+[1]):
        mix = iter(np.linspace(s, e, grow_frames, endpoint=False))
        mixer.mix = mix.next()
        
        for angle in np.linspace(0, 2*n_cycles*np.pi, rot_frames) - np.pi/2:
            x, y, z = cam.focal_point
            d = dist(s)
            cam.position = d * np.cos(angle) + x, d*np.sin(angle) + y, cam.position[-1]
            fm = np.floor(h5.root.idx[i])
            dm = h5.root.idx[i]%fm
            print i
            mixer.data = (1-dm)*h5.root.zmap[fm-1] + dm*h5.root.zmap[fm]
            mixer.figure.renderer.reset_camera_clipping_range()
            mixer.figure.save_png(os.path.join(tmp, "im%07d.png"%i))
            i += 1

        for m in mix:
            mixer.mix = m
            _zoom(cam, dist(m))
            fm = np.floor(h5.root.idx[i])
            dm = h5.root.idx[i]%fm
            print i
            mixer.data = (1-dm)*h5.root.zmap[fm-1] + dm*h5.root.zmap[fm]
            mixer.figure.renderer.reset_camera_clipping_range()
            mixer.figure.save_png(os.path.join(tmp, "im%07d.png"%i))
            i += 1

    mixer.mix = 1
    _zoom(cam, dist(1))
    mixer.figure.renderer.reset_camera_clipping_range()
    mixer.figure.save_png(os.path.join(tmp, "im%07d.png"%i))

if __name__ == '__main__':
    import tables
    from mritools import view
    mixer = view.show(np.random.randn(32, 100, 100), "SN", "SN_shinji", ("inflated", "veryinflated"))
    mixer.showrois = True
    mixer.showlabels = True
    mixer.labelsize = 24
    mixer.texres = 2048
    h5 = tables.openFile("/home/james/sn_movies.hdf")