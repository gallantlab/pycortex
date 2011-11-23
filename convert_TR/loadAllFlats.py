import os
import glob

from utils.mri.db import surfs

def loadVTKdir(vtkdir, subject):
    types = ['raw','fiducial','inflated','veryinflated', 'superinflated',
             'hyperinflated','ellipsoid','flat'];
    anat = glob.glob(os.path.join(vtkdir, "anatomical*"))
    if len(anat) > 0:
        surfs.loadVTK(anat[-1], subject, "anatomical", "both", None)
    
    # coords = None
    # if os.path.exists(os.path.join(flatdir, "coords")):
    #     coords = open(os.path.join(flatdir, "coords")).read()
    
    for d in ['lh', 'rh']:
        for t in types:
            fname = "{lr}_{type}.vtk".format(lr=d, type=t)
            fpath = os.path.join(vtkdir, fname)
            if os.path.exists(fpath):
                print fpath
                if t == "fiducial":
                    offset = "0 0 0"
                else:
                    offset = None
                surfs.loadVTK(fpath, subject, t, d, offset)
            else:
                print "couldn't find %s"%fpath

if __name__ == "__main__":
#    cwd = os.path.split(os.path.abspath(__file__))[0]
#    fpath = os.path.join(cwd, "flats.sql")
#    if os.path.exists(fpath):
#        os.unlink(fpath)
    
    names = "AH,AV,DS,JG,ML,MO,NB,SN,TC,TN,WH".split(",")
    path = "/auto/data/archive/mri_flats/%s/"
    for n in names:
        loadVTKdir(path%n, n)
