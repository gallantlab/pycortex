import os
import glob
import shutil
import tables
import numpy as np

cwd = os.path.split(os.path.abspath(__file__))[0]
flatsfile = os.path.join(cwd, "flats.hf5")
flats = tables.openFile(flatsfile, "r")

from vtk import vtkread

class DBwrite(object):
    def __init__(self):
        flats.flush()
        self.backup = os.path.join(cwd, "_flats.hf5")
        shutil.copy(flatsfile, self.backup)
        self.flats = tables.openFile(self.backup, "a")
    
    def commit(self):
        global flats
        flats.close()
        self.flats.close()
        shutil.copy(self.backup, flatsfile)
        os.unlink(self.backup)
        flats = tables.openFile(flatsfile, "r")
    
    def loadVTKdir(self, flatdir, subject):
        types = ['raw','fiducial','inflated','veryinflated', 'superinflated',
                 'hyperinflated','ellipsoid','flat'];
        filefmt = "{lr}_{type}.vtk"
        atom = tables.Float32Atom()
        filters = tables.Filters(complevel=5, shuffle=True)

        self.flats.createGroup("/", subject)
        self.flats.createGroup("/{subj}".format(subj=subject), "surfaces")
        self.flats.createGroup("/{subj}".format(subj=subject), "anatomical")
        anat = glob.glob(os.path.join(flatdir, "anatomical*"))
        if len(anat) > 0:
            self.flats.createArray('/{subj}/anatomical'.format(subj=subject), 'filename', anat[-1])

        self.flats.createGroup("/{subj}".format(subj=subject), "transforms")
        
        for d in ['lh', 'rh']:
            self.flats.createGroup("/{subj}/surfaces/".format(subj=subject), d)
            for t in types:
                nodename = "/{subj}/surfaces/{lr}".format(subj=subject, lr=d)
                self.flats.createGroup(nodename, t.lower())
                fname = filefmt.format(lr=d, type=t)
                fpath = os.path.join(flatdir, fname)
                if os.path.exists(fpath):
                    print fpath
                    h5root = "/{subj}/surfaces/{lr}/{t}".format(subj=subject, lr=d, t=t.lower())
                    self.flats.createArray(h5root, "filename", fpath)

                    '''
                    #There's no need to preload, now that I fixed vtkread
                    pts, polys, normals = vtkread([fpath])
                    p = self.flats.createCArray(h5root, "points", atom, 
                        pts.shape, filters=filters)
                    p[:] = pts
                    p = self.flats.createCArray(h5root, "polys", atom, 
                        polys.shape, filters=filters)
                    p[:] = polys
                    p = self.flats.createCArray(h5root, "normals", atom, 
                        normals.shape, filters=filters)
                    p[:] = normals
                    '''
                else:
                    print "couldn't find %s"%fpath

        return self
    
    def loadXfm(self, subject, name, xfm, shortcut=None):
        self.flats.createGroup("/{subj}/transforms/".format(subj=subject), name)
        self.flats.createArray("/{subj}/transforms/{name}", "magnet", )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load a directory of flatmaps into the database")
    parser.add_argument("subject", type=str, help="Subject name (two letter abbreviation)")
    parser.add_argument("vtkdir", type=str, help="Directory with VTK's")
    args = parser.parse_args()

    try:
        DBwrite().loadVTKdir(args.vtkdir, args.subject).commit()
        print "Success!"
    except Exception, e:
        print "Error with processing: ", e