import numpy as np
import Image

def makeImage(cmap, fname):
    cmarr = (cmap*255).astype(np.uint8)
    im = Image.fromarray(cmarr[np.newaxis])
    im.save(fname)

def cmList():
    cmaps = {}
    values = np.linspace(0, 1, 256)
    from matplotlib import cm, colors
    for cmname in dir(cm):
        cmap = getattr(cm, cmname)
        if isinstance(cmap, colors.Colormap):
            cmaps[cmname] = cmap(values)

    return cmaps

if __name__ == "__main__":
    import os
    import sys
    path = sys.argv[1]
    for name, cm in cmList().items():
        fname = os.path.join(path, "%s.png"%name)
        makeImage(cm, fname)