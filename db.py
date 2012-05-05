import os
import glob
import time
import json
import shutil
import numpy as np

cwd = os.path.split(os.path.abspath(__file__))[0]
options = json.load(open(os.path.join(cwd, "defaults.json")))
filestore = options['file_store']
#dbfile = os.path.join(cwd, "database.sql")

class SubjectDB(object):
    def __init__(self, subj):
        self.transforms = XfmDB(subj)
        self.surfaces = SurfaceDB(subj)
        self.anatfile = None
        anatname = "{subj}_anatomical_both.*".format(subj=subj)
        anatname = glob.glob(os.path.join(filestore, "surfaces", anatname))
        if len(anatname) > 0:
            self.anatfile = anatname[0]
    
    def __dir__(self):
        names = ["transforms", "surfaces"]
        if self.anatfile is not None:
            names.append("anatomical")
        return names
    
    def __getattr__(self, attr):
        if attr == "anatomical" and self.anatfile is not None:
            import nibabel
            return nibabel.load(os.path.join(filestore, "surfaces", self.anatfile))
        raise AttributeError

class SurfaceDB(object):
    def __init__(self, subj):
        self.subject = subj
        self.types = {}
        pname = os.path.join(filestore, "surfaces", "{subj}_*.vtk").format(subj=subj)
        for fname in glob.glob(pname):
            fname = os.path.splitext(os.path.split(fname)[1])[0].split('_') 
            subj = fname.pop(0)
            hemi = fname.pop()
            name = "_".join(fname)
            self.types[name] = Surf(subj, name)
    
    def __repr__(self):
        return "Surfaces: [{surfs}]".format(surfs=', '.join(self.types.keys()))
    
    def __dir__(self):
        return self.types.keys()

    def __getattr__(self, attr):
        if attr in self.types:
            return self.types[attr]
        raise AttributeError(attr)

class Surf(object):
    def __init__(self, subject, surftype):
        self.subject, self.surftype = subject, surftype
        self.fname = os.path.join(filestore, "surfaces", "{subj}_{name}_{hemi}.vtk")

    def get(self, hemisphere="both"):
        return surfs.getVTK(self.subject, self.surftype, hemisphere)
    
    def show(self, hemisphere="both"):
        import vtkutils
        lh = self.fname.format(subj=self.subject, name=self.surftype, hemi="lh")
        rh = self.fname.format(subj=self.subject, name=self.surftype, hemi="rh")
        if hemisphere == "both":
            return vtkutils.show([lh, rh])
        elif hemisphere.lower() in ["l", "lh", "left"]:
            return vtkutils.show([lh])
        elif hemisphere.lower() in ["r", "rh", "right"]:
            return vtkutils.show([rh])

class XfmDB(object):
    def __init__(self, subj):
        self.subj = subj
        xfms = glob.glob(os.path.join(filestore, "transforms", "{subj}_*.xfm".format(subj=subj)))
        self.xfms = ['_'.join(os.path.splitext(os.path.split(x)[1])[0].split('_')[1:]) for x in xfms]

    def __getitem__(self, name):
        if name in self.xfms:
            return XfmSet(self.subj, name)
        raise AttributeError
    
    def __repr__(self):
        return "Transforms: [{xfms}]".format(xfms=",".join(self.xfms))

class XfmSet(object):
    def __init__(self, subj, name):
        self.subject = subj
        self.name = name
        fname = "{subj}_{name}.xfm".format(subj=subj, name=name)
        self.jsdat = json.load(open(os.path.join(filestore, "transforms", fname)))
        self.filename = self.jsdat['epifile']
    
    def __getattr__(self, attr):
        if attr in self.jsdat:
            return np.array(self.jsdat[attr])
        raise AttributeError
    
    def __repr__(self):
        names = set(self.jsdat.keys())
        names -= set(["epifile", "subject"])
        return "Types: {types}".format(types=", ".join(names))


class Database(object):
    def __init__(self):
        vtks = glob.glob(os.path.join(filestore, "surfaces", "*.vtk"))
        subjs = set([os.path.split(vtk)[1].split('_')[0] for vtk in vtks])
        xfms = glob.glob(os.path.join(filestore, "transforms", "*.xfm"))

        self.subjects = dict([(sname, SubjectDB(sname)) for sname in subjs])
        self.xfms = [os.path.splitext(os.path.split(xfm)[1])[0].split('_') for xfm in xfms]
        self.xfms = [(s[0], '_'.join(s[1:])) for s in self.xfms]
    
    def __repr__(self):
        subjs = ", ".join(sorted(self.subjects.keys()))
        xfms = "[%s]"%", ".join('(%s, %s)'% p for p in set(self.xfms))
        return """Flatmapping database
        Subjects:   {subjs}
        Transforms: {xfms}""".format(subjs=subjs, xfms=xfms)
    
    def __getattr__(self, attr):
        if attr in self.subjects:
            return self.subjects[attr]
        else:
            raise AttributeError
    
    def __dir__(self):
        return ["loadXfm","getXfm", "loadVTK", "getVTK"] + self.subjects.keys()
    
    def loadXfm(self, subject, name, xfm, xfmtype="magnet", epifile=None, override=False):
        """Load a transform into the surface database. If the transform exists already, update it
        If it does not exist, copy the reference epi into the filestore and insert."""
        assert xfmtype in ["magnet", "coord", "base"], "Unknown transform type"
        result = glob.glob(os.path.join(filestore, "transforms", "{name}.xfm".format(name=name)))
        if len(result) > 0:
            if epifile is not None:
                raise ValueError("Cannot change reference epi for existing transform")

            jsdict = json.load(open(result[0]))
            if xfmtype in jsdict:
                prompt = 'There is already a transform for this subject by the name of "%s". Overwrite? (Y/N)'%subject
                if not override and raw_input(prompt).lower().strip() not in ("y", "yes"):
                    print "Not saving..."
                    return
        else:
            assert epifile is not None, "Please specify a reference epi"
            assert os.path.splitext(epifile)[1].lower() == ".nii", "Reference epi must be a nifti"
            filename = "{subj}_{name}_refepi.nii".format(subj=subject, name=name)
            fpath = os.path.join(filestore, "references", filename)
            if not os.path.exists(fpath):
                shutil.copy2(epifile, fpath)

            jsdict = dict(epifile=filename, subject=subject)

        jsdict[xfmtype] = xfm.tolist()
        json.dump(jsdict, open(result[0], "w"), sort_keys=True, indent=4)
    
    def getXfm(self, subject, name, xfmtype="coord"):
        fname = os.path.join(filestore, "transforms", "{subj}_{name}.xfm".format(subj=subject, name=name))
        xfmdict = json.load(open(fname))
        assert xfmdict['subject'] == subject, "Incorrect subject for the name"
        return np.array(xfmdict[xfmtype]), os.path.join(filestore, "references", xfmdict['epifile'])

    def getVTK(self, subject, type, hemisphere="both"):
        import vtkutils
        fname = os.path.join(filestore, "surfaces", "{subj}_{type}_{hemi}.vtk")

        if hemisphere == "both":
            lh = fname.format(subj=subject, type=type, hemi="lh")
            rh = fname.format(subj=subject, type=type, hemi="rh")
            lpts, lpolys, lnorms = vtkutils.read(os.path.join(filestore, "surfaces", lh))
            rpts, rpolys, rnorms = vtkutils.read(os.path.join(filestore, "surfaces", rh))
            
            lpts[:,0] -= lpts.max(0)[0]
            rpts[:,0] -= rpts.min(0)[0]
            
            rpolys += len(lpts)
            return np.vstack([lpts, rpts]), np.vstack([lpolys, rpolys]), np.vstack([lnorms, rnorms])
        else:
            if hemisphere.lower() in ("lh", "left"):
                hemi = "lh"
            elif hemisphere.lower() in ("rh", "right"):
                hemi = "rh"
            else:
                raise TypeError("Not a valid hemisphere name")

            return vtkutils.read(fname.format(subj=subject, type=type, hemi=hemi))


surfs = Database()