import os
import shlex
import tempfile
import subprocess as sp

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline, interp1d

import view

def _zoom(cam, distance):
    vec = cam.position - cam.focal_point
    d = np.sqrt((vec**2).sum())
    cam.position = (distance / d * vec) + cam.focal_point

def _save(folder, outfile, fps):
    cmd = "/usr/bin/ffmpeg -qscale 5 -r {fps} -b 9600 -i {folder}/im%07d.png {outfile}"
    sp.call(shlex.split(cmd.format(folder=folder, fps=fps, outfile=outfile)))

def rotate(mixer, outfile, stops=[0, 0.5], rot_frames=24*5, n_cycles=1, grow_frames=24*2, fps=24):
    tmp = tempfile.mkdtemp()
    mixer.figure.off_screen_rendering = True
    mixer.reset_view(False)

    dist = Spline([0,0.3,0.6,1], [320, 350, 350, 690])
    
    i = 0
    for s, e in zip(stops, stops[1:]+[1]):
        mix = iter(np.linspace(s, e, grow_frames, endpoint=False))
        mixer.mix = mix.next()
        cam = mixer.figure.camera
        for angle in np.linspace(0, 2*n_cycles*np.pi, 60) - np.pi/2:
            x, y, z = cam.focal_point
            d = dist(s)
            cam.position = d * np.cos(angle) + x, d*np.sin(angle) + y, cam.position[-1]
            mixer.figure.renderer.reset_camera_clipping_range()
            mixer.figure.save_png(os.path.join(tmp, "im%07d.png"%i))
            i += 1

        for m in mix:
            mixer.mix = m
            _zoom(cam, dist(m))
            mixer.figure.renderer.reset_camera_clipping_range()
            mixer.figure.save_png(os.path.join(tmp, "im%07d.png"%i))
            i += 1
    mixer.mix = 1
    _zoom(cam, dist(1))
    mixer.figure.renderer.reset_camera_clipping_range()
    mixer.figure.save_png(os.path.join(tmp, "im%07d.png"%i))
    _save(tmp, outfile, fps)
    mixer.figure.off_screen_rendering = False

def inflate(mixer, outfile, pos_interp=0.5, fps=24, frames=30*20):
    tmp = tempfile.mkdtemp()
    mixer.figure.off_screen_rendering = True
    cam = mixer.figure.camera
    ipos = cam.position, cam.focal_point
    mixer.reset_view(False)
    fpos = cam.position, cam.focal_point
    
    dist = Spline([0,0.3,0.6,1], [320, 350, 350, 690])
    pinterp = [ interp1d([0,pos_interp,1], [ipos[0][j], ipos[0][j], fpos[0][j]]) for j in [0,1,2] ]
    finterp = [ interp1d([0,pos_interp,1], [ipos[1][j], ipos[1][j], fpos[1][j]]) for j in [0,1,2] ]
    
    cam.position, cam.focal_point = ipos
    for frame, mix in enumerate(np.linspace(0,1,frames)):
        mixer.mix = mix
        if mix >= pos_interp:
            cam.position = [p(mix) for p in pinterp]
            cam.focal_point = [f(mix) for f in finterp]
        else:
            _zoom(cam, dist(mix))
        mixer.figure.renderer.reset_camera_clipping_range()
        mixer.figure.save_png(os.path.join(tmp, "im%07d.png"%frame))
    
    _save(tmp, outfile, fps)
    mixer.figure.off_screen_rendering = False