import os
import numpy as np
from cortex import freesurfer

cwd = os.path.split(os.path.abspath(__file__))[0]
npz = np.load(os.path.join(cwd, "decimate.npz"))
pts, polys, dpts = npz['flat'][:,:2], npz['polys'], npz['pts']

layout = freesurfer.SpringLayout(pts, polys, dpts)
