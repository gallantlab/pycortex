import numpy as np
import Image
import scipy.io as sio

def makeImage(cmap, fname):
    cmarr = (cmap*255).astype(np.uint8)
    im = Image.fromarray(cmarr[np.newaxis])
    im.save(fname)

def cmList(additional):
    cmaps = {}
    values = np.linspace(0, 1, 256)
    from matplotlib import cm, colors
    for cmname in dir(cm):
        cmap = getattr(cm, cmname)
        if isinstance(cmap, colors.Colormap):
            cmaps[cmname] = cmap(values)

    for name, cmap in additional.items():
        cmaps[name] = colors.LinearSegmentedColormap.from_list(name, cmap)(values)

    return cmaps    

if __name__ == "__main__":
    import os
    import sys
    path = sys.argv[1]

    matfile = sio.loadmat("/auto/k2/share/mritools_store/colormaps.mat")
    del matfile['__globals__']
    del matfile['__header__']
    del matfile['__version__']

    for name, cm in cmList(matfile).items():
        fname = os.path.join(path, "%s.png"%name)
        makeImage(cm, fname)