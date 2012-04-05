import os
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
    def __init__(self, subj, conn, cur):
        self.transforms = XfmDB(subj, conn, cur)
        self.surfaces = SurfaceDB(subj, conn, cur)
        self.anatfile = None
        query = "SELECT filename FROM surfaces WHERE subject=? and type=?"
        data = cur.execute(query, (subj, "anatomical")).fetchone()
        if data is not None:
            self.anatfile = data[0]
    
    def __dir__(self):
        names = ["transforms", "surfaces"]
        if self.anatfile is not None:
            names.append("anatomical")
        return names
    
    def __getattr__(self, attr):
        if attr == "anatomical" and self.anatfile is not None:
            import nibabel
            return nibabel.load(os.path.join(filestore, "surfaces",self.anatfile))
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
        self.conn = sqlite3.connect(dbfile)
        self.conn.text_factory = str
        self.cur = self.conn.cursor()
        self._setup()
        subjects = self.cur.execute("SELECT subject FROM surfaces").fetchall()
        self.subjects = dict([(n[0], SubjectDB(n[0], self.conn, self.cur)) for n in subjects])
    
    def _setup(self):
        schema = dict(surfaces='subject, type, hemisphere, filename, offset',
                    transforms='subject, name, date, type, filename, xfm BLOB')
        for table, types in schema.items():
            c = self.cur.execute("select name from sqlite_master where name=?", (table,))
            if c.fetchone() is None:
                self.cur.execute("create table {0} ({1})".format(table, types))
        self.conn.commit()
    
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

    def loadVTK(self, vtkfile, subject, surftype, hemisphere, offset=None):
        '''Load a vtk file into the database and copy it into the filestore'''
        #assert os.path.splitext(vtkfile)[1] == ".vtk", "Not a VTK file"
        assert hemisphere in ["lh", "rh", "both"], "Invalid hemisphere name, must be 'lh' or 'rh'"
        if surftype == "fiducial":
            offset = "0 0 0"
        #Let's delete any possible duplicates
        query = "DELETE FROM surfaces WHERE subject=? and type=? and hemisphere=?"
        self.cur.execute(query, (subject, surftype, hemisphere))
        self.conn.commit()

        query = "INSERT into surfaces (subject, type, hemisphere, filename, offset) VALUES (?,?,?,?,?)"
        ext = os.path.splitext(vtkfile)[1].lower()
        filename = "{subj}_{type}_{hemi}{ext}".format(subj=subject, type=surftype, hemi=hemisphere, ext=ext)

        self.cur.execute(query, (subject, surftype, hemisphere, filename, offset))
        self.conn.commit()

        #Copy the vtk file into the filestore
        shutil.copy2(vtkfile, os.path.join(filestore, "surfaces", filename))
    
    def loadXfm(self, subject, name, xfm, xfmtype="magnet", epifile=None, override=None):
        """Load a transform into the surface database. If the transform exists already, update it
        If it does not exist, copy the reference epi into the filestore and insert."""
        assert xfmtype in ["magnet", "coord", "base"], "Unknown transform type"
        query = "SELECT name FROM transforms WHERE subject=? and name=? and type=?"
        result = self.cur.execute(query, (subject, name, xfmtype)).fetchone()
        if result is not None:
            #assert epifile is None, 
            if epifile is not None:
                print "Cannot change reference epi for existing transform"
            prompt = 'There is already a transform for this subject by the name of "%s". Overwrite? (Y/N)'%subject
            if override is not None and override or \
                override is None and raw_input(prompt).lower().strip() in ("y", "yes"):

                query = "UPDATE transforms SET xfm=? WHERE subject=? AND name=? and type=?"
                data = (sqlite3.Binary(xfm.tostring()), subject, name, xfmtype)

                self.cur.execute(query, data)
                self.conn.commit()
            else:
                print "Override: skipping %s"%name
        else:
            assert epifile is not None, "Please specify a reference epi"
            assert os.path.splitext(epifile)[1].lower() == ".nii", "Reference epi must be a nifti"
            filename = "{subj}_{name}_refepi.nii".format(subj=subject, name=name)
            fpath = os.path.join(filestore, "references", filename)
            if not os.path.exists(fpath):
                shutil.copy2(epifile, fpath)

            fields = "subject,name,date,type,xfm,filename"
            data = (subject, name, time.time(), xfmtype, sqlite3.Binary(xfm.tostring()), filename)
            query = "INSERT into transforms ({fields}) values (?,?,?,?,?,?)".format(fields=fields)
            self.cur.execute(query, data)
            self.conn.commit()
    
    def getXfm(self, subject, name, xfmtype="coord"):
        query = "SELECT xfm, filename FROM transforms WHERE subject=? AND name=? and type=?"
        data = self.cur.execute(query, (subject, name, xfmtype)).fetchone()
        if data is None:
            return
        else:
            xfm, filename = data
            return np.fromstring(xfm).reshape(4,4), os.path.join(filestore, "references", filename)

    def getVTK(self, subject, type, hemisphere="both"):
        import vtkutils
        query = "SELECT filename, offset FROM surfaces WHERE subject=? AND type=? AND hemisphere=?"
        if self.cur.execute(query, (subject, type, "lh")).fetchone() is None:
            #Subject / type does not exist in the database
            raise ValueError("Cannot find subject/type in the database")

        if hemisphere == "both":
            lh, loff = self.cur.execute(query, (subject, type, 'lh')).fetchone()
            rh, roff = self.cur.execute(query, (subject, type, 'rh')).fetchone()
            loff, roff = map(_convert_offset, [loff, roff])
            lpts, lpolys, lnorms = vtkutils.read(os.path.join(filestore, "surfaces", lh))
            rpts, rpolys, rnorms = vtkutils.read(os.path.join(filestore, "surfaces", rh))
            
            if loff is None and roff is None:
                lpts[:,0] -= lpts.max(0)[0]
                rpts[:,0] -= rpts.min(0)[0]
            if loff is not None:
                lpts += loff
            if roff is not None:
                rpts += roff
            
            rpolys += len(lpts)
            return np.vstack([lpts, rpts]), np.vstack([lpolys, rpolys]), np.vstack([lnorms, rnorms])
        else:
            d, offset = self.cur.execute(query, (subject, type, hemisphere)).fetchone()
            pts, polys, norms = vtkutils.read(os.path.join(filestore, "surfaces", d))
            if offset is not None:
                pts += _convert_offset(offset)
            return pts, polys, norms

def _convert_offset(offset):
    if offset is not None and len(offset) > 0:
        return np.array([float(d) for d in offset.split()])

surfs = Database()

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Load a directory of flatmaps into the database")
#     parser.add_argument("subject", type=str, help="Subject name (two letter abbreviation)")
#     parser.add_argument("vtkdir", type=str, help="Directory with VTK's")
#     args = parser.parse_args()


#     #surfs.loadXfm(subject, xfmname, magnet, xfmtype='magnet', filename=epi, override=True)
#     #surfs.loadXfm(subject, xfmname, shortcut, xfmtype='coord', filename=epi, override=True)
    
#     try:
#         surfs.loadVTKdir(args.vtkdir, args.subject)
#         print "Success!"
#     except Exception, e:
#         print "Error with processing: ", e
