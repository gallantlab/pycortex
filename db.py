import os
import glob
import time
import json
import shutil
import sqlite3
import numpy as np

cwd = os.path.split(os.path.abspath(__file__))[0]
options = json.load(open(os.path.join(cwd, "defaults.json")))
filestore = options['file_store']
dbfile = os.path.join(cwd, "database.sql")

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
    def __init__(self, subj, conn, cur):
        self.subject = subj
        self.conn, self.cur = conn, cur
        query = "SELECT type, hemisphere, filename FROM surfaces WHERE subject=?"
        results = cur.execute(query, (subj,))
        types = {}
        for row in results:
            if row[0] not in types:
                types[row[0]] = {}
            types[row[0]][row[1]] = row[2]
        self.types = {}
        for k, v in types.items():
            if "lh" in v and "rh" in v:
                self.types[k] = Surf(subj, k, v['lh'], v['rh'], conn, cur)
    
    def __repr__(self):
        return "Surfaces: [{surfs}]".format(surfs=', '.join(self.types.keys()))
    
    def __dir__(self):
        return self.types.keys()

    def __getattr__(self, attr):
        if attr in self.types:
            return self.types[attr]
        raise AttributeError(attr)

class Surf(object):
    def __init__(self, subject, surftype, lh, rh, conn, cur):
        self.subject, self.surftype = subject, surftype
        self.lh, self.rh = lh, rh
        self.conn, self.cur = conn, cur

    def get(self, hemisphere="both"):
        if hemisphere == "both":
            return surfs.getVTK(self.subject, self.surftype, "both")
        elif hemisphere.lower() in ["l", "lh", "left"]:
            return surfs.getVTK(self.subject, self.surftype, "lh")
        elif hemisphere.lower() in ["r", "rh", "right"]:
            return surfs.getVTK(self.subject, self.surftype, "rh")
        raise AttributeError
    
    def show(self, hemisphere="both"):
        import vtkutils
        lh, rh = map(lambda x:os.path.join(filestore, "surfaces", x), [self.lh, self.rh])
        if hemisphere == "both":
            return vtkutils.show([lh, rh])
        elif hemisphere.lower() in ["l", "lh", "left"]:
            return vtkutils.show([lh])
        elif hemisphere.lower() in ["r", "rh", "right"]:
            return vtkutils.show([rh])
    
    def __dir__(self):
        return self.__dict__.keys() + ["loffset", "roffset"]
    
    def __getattr__(self, attr):
        if attr == "loffset":
            query = "SELECT offset FROM surfaces WHERE hemisphere='lh' and filename=?"
            return self.cur.execute(query, (self.lh,)).fetchone()[0]
        elif attr == "roffset":
            query = "SELECT offset FROM surfaces WHERE hemisphere='rh' and filename=?"
            return self.cur.execute(query, (self.rh,)).fetchone()[0]
    
    def __setattr__(self, attr, val):
        if attr == "loffset":
            query = "UPDATE surfaces SET offset=? WHERE hemisphere='lh' and filename=?"
            self.cur.execute(query, (val, self.lh))
            self.conn.commit()
        elif attr == "roffset":
            query = "UPDATE surfaces SET offset=? WHERE hemisphere='rh' and filename=?"
            self.cur.execute(query, (val, self.rh))
            self.conn.commit()
        else:
            super(Surf, self).__setattr__(attr, val)
            
class XfmDB(object):
    def __init__(self, subj, conn, cur):
        self.conn, self.cur = conn, cur
        self.subj = subj

        query = "SELECT name FROM transforms WHERE subject=?"
        results = cur.execute(query, (subj,)).fetchall()
        self.xfms = set([r[0] for r in results])
    
    def __getitem__(self, name):
        if name in self.xfms:
            return XfmSet(self.subj, name, self.conn, self.cur)
        raise AttributeError
    
    def __repr__(self):
        return "Transforms: [{xfms}]".format(xfms=",".join(self.xfms))

class XfmSet(object):
    def __init__(self, subj, name, conn, cur):
        self.conn, self.cur = conn, cur
        self.subject = subj
        self.name = name
        query = "SELECT type, xfm FROM transforms WHERE subject=? and name=?"
        self.data = dict(cur.execute(query, (subj, name)).fetchall())
        query = "SELECT filename FROM transforms WHERE subject=? and name=?"
        self.filename, = cur.execute(query, (subj, name)).fetchone()
        self.filename = os.path.join(filestore, "references", self.filename)
    
    def __getattr__(self, attr):
        if attr in self.data:
            return np.fromstring(self.data[attr]).reshape(4,4)
        raise AttributeError
    
    def __repr__(self):
        return "Types: {types}".format(types=", ".join(self.data.keys()))
    
    def remove(self): 
        if raw_input("Are you sure? (Y/N) ").lower().strip() in ["y", "yes"]:
            query = "DELETE FROM transforms WHERE subject=? AND name=?"
            self.cur.execute(query, (self.subject, self.name))
            self.conn.commit()

class Database(object):
    def __init__(self):
        vtks = glob.glob(os.path.join(filestore, "surfaces", "*.vtk"))
        subjs = set([os.path.split(vtk)[1].split('_')[0] for vtk in vtks])
        self.subjects = []
    
    def __repr__(self):
        subjs = ", ".join(sorted(self.subjects.keys()))
        pairs = self.cur.execute("SELECT subject, name from transforms").fetchall()
        xfms = "[%s]"%", ".join('(%s, %s)'% p for p in set(pairs))
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
        fname = os.path.join(filestore, "transforms", "{name}.xfm".format(name=name))
        xfmdict = json.load(open(fname))
        assert xfmdict['subject'] == subject, "Incorrect subject for the name"
        return xfmdict[xfmtype], xfmdict['epifile']

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