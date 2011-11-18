import os
import glob
import sqlite3
import time
import numpy as np

cwd = os.path.split(os.path.abspath(__file__))[0]
flatsfile = os.path.join(cwd, "flats.sql")

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
            return nibabel.load(self.anatfile)
        raise AttributeError

class SurfaceDB(object):
    def __init__(self, subj, conn, cur):
        self.subject = subj
        self.conn, self.cur = conn, cur
        query = "SELECT type, hemisphere, filename, offset FROM surfaces WHERE subject=?"
        results = cur.execute(query, (subj,))
        types = {}
        for row in results:
            if row[0] not in types:
                types[row[0]] = {}
            types[row[0]][row[1]] = row[2]
            offset = row[3]
            if offset is not None and len(offset) > 0:
                offset = [float(r) for r in offset.split()]
            types[row[0]]['offset'] = offset
        self.types = types
    
    def __repr__(self):
        return "Surfaces: [{surfs}]".format(surfs=', '.join(self.types.keys()))
    
    def __dir__(self):
        return self.types.keys()

    def __getattr__(self, attr):
        if attr in self.types:
            return Surf(
                self.types[attr]['lh'], 
                self.types[attr]['rh'], 
                self.conn, self.cur,
                self.types[attr]['offset'])
        raise AttributeError

class Surf(object):
    def __init__(self, lh, rh, conn, cur, offset=None):
        self.lh, self.rh = lh, rh
        self.conn, self.cur = conn, cur
        self.offset = None if offset is None else np.array(offset)

    def get(self, hemisphere="both"):
        import vtkutils
        if hemisphere == "both":
            return vtkutils.read([self.lh, self.rh], offset=self.offset)
        elif hemisphere.lower() in ["l", "lh", "left"]:
            return vtkutils.read([self.lh])
        elif hemisphere.lower() in ["r", "rh", "right"]:
            return vtkutils.read([self.rh])
        raise AttributeError
    
    def show(self, hemisphere="both"):
        import vtkutils
        if hemisphere == "both":
            return vtkutils.show([self.lh, self.rh], offset=self.offset)
        elif hemisphere.lower() in ["l", "lh", "left"]:
            return vtkutils.show([self.lh])
        elif hemisphere.lower() in ["r", "rh", "right"]:
            return vtkutils.show([self.rh])
    
    def __setattr__(self, attr, val):
        if not hasattr(self, attr):
            super(Surf, self).__setattr__(attr, val)
            return
        
        if attr == "offset":
            query = "UPDATE surfaces SET offset=? WHERE lh=? OR rh=?"
            self.cur.execute(query, (val, self.lh, self.rh))
            self.conn.commit()
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
    
    def __getattr__(self, attr):
        if attr in self.data:
            return np.fromstring(self.data[attr]).reshape(4,4)
        raise AttributeError
    
    def __repr__(self):
        return "Types: {types}".format(types=", ".join(self.data.keys()))
    
    def setFile(self, filename):
        assert os.path.exists(filename)
        filename = os.path.abspath(filename)
        query = "UPDATE transforms SET filename=? WHERE subject=? and name=?"
        self.cur.execute(query, (filename, self.subject, self.name))
        self.conn.commit()
        self.filename = filename
    
    def remove(self): 
        if raw_input("Are you sure? (Y/N) ").lower().strip() in ["y", "yes"]:
            query = "DELETE FROM transforms WHERE subject=? AND name=?"
            self.cur.execute(query, (self.subject, self.name))
            self.conn.commit()

class Database(object):
    def __init__(self):
        self.conn = sqlite3.connect(flatsfile)
        self.conn.text_factory = str
        self.cur = self.conn.cursor()
        self._setup()
        subjects = self.cur.execute("SELECT subject FROM surfaces").fetchall()
        self.subjects = set([n[0] for n in subjects])
    
    def _setup(self):
        schema = dict(surfaces='subject, type, hemisphere, filename, offset',
                    transforms='subject, name, date, type, filename, xfm BLOB')
        for table, types in schema.items():
            c = self.cur.execute("select name from sqlite_master where name=?", (table,))
            if c.fetchone() is None:
                self.cur.execute("create table {0} ({1})".format(table, types))
        self.conn.commit()
    
    def __repr__(self):
        subjs = ", ".join(sorted(list(self.subjects)))
        pairs = self.cur.execute("SELECT subject, name from transforms").fetchall()
        xfms = "[%s]"%", ".join('(%s, %s)'% p for p in set(pairs))
        return """Flatmapping database
        Subjects:   {subjs}
        Transforms: {xfms}""".format(subjs=subjs, xfms=xfms)
    
    def __getattr__(self, attr):
        if attr in self.subjects:
            return SubjectDB(attr, self.conn, self.cur)
        else:
            raise AttributeError
    
    def __dir__(self):
        return ["loadXfm","getXfm", "loadVTKdir", "getVTK"] + list(self.subjects)

    def loadVTKdir(self, flatdir, subject):
        types = ['raw','fiducial','inflated','veryinflated', 'superinflated',
                 'hyperinflated','ellipsoid','flat'];
        query = "INSERT into surfaces (subject, type, hemisphere, filename, offset) VALUES (?,?,?,?,?)"

        anat = glob.glob(os.path.join(flatdir, "anatomical*"))
        if len(anat) > 0:
            data = subject, "anatomical", "both", anat[-1], "0 0 0"
            self.cur.execute(query, data)
        
        coords = None
        if os.path.exists(os.path.join(flatdir, "coords")):
            coords = open(os.path.join(flatdir, "coords")).read()
        
        for d in ['lh', 'rh']:
            for t in types:
                fname = "{lr}_{type}.vtk".format(lr=d, type=t)
                fpath = os.path.join(flatdir, fname)
                if os.path.exists(fpath):
                    print fpath
                    if t == "fiducial":
                        coord = "0 0 0"
                    data = [subject, t, d, fpath, coords]
                    self.cur.execute(query, data)
                else:
                    print "couldn't find %s"%fpath

        self.conn.commit()
    
    def loadXfm(self, subject, name, xfm, xfmtype="magnet", filename=None, override=None):
        """Load a transform into the surface database"""
        assert xfmtype in ["magnet", "coord", "base"], "Unknown transform type"
        query = "SELECT name FROM transforms WHERE subject=? and name=? and type=?"
        result = self.cur.execute(query, (subject, name, xfmtype)).fetchone()
        if result is not None:
            prompt = 'There is already a transform for this subject by the name of "%s". Overwrite? (Y/N)'%subject
            if override is not None and override or (
                override is None and raw_input(prompt).lower().strip() in ("y", "yes")):

                query = "UPDATE transforms SET xfm=?%s WHERE subject=? AND name=? and type=?"
                if filename is not None:
                    query = query%", filename=? "
                    data = (sqlite3.Binary(xfm.tostring()), filename, subject, name, xfmtype)
                else:
                    query = query%""
                    data = (sqlite3.Binary(xfm.tostring()), subject, name, xfmtype)

                self.cur.execute(query, data)
                self.conn.commit()
            else:
                print "Override: skipping %s"%name
        else:
            fields = "subject,name,date,type,xfm".split(",")
            data = (subject, name, time.time(), xfmtype, sqlite3.Binary(xfm.tostring()))
            if filename is not None:
                fields.append('filename')
                data = data + (filename,)

            query = "INSERT into transforms ({fields}) values ({qs})".format(fields=",".join(fields), qs=",".join("?"*len(fields)))
            self.cur.execute(query, data)
            self.conn.commit()
    
    def getXfm(self, subject, name, xfmtype="coord"):
        query = "SELECT xfm, filename FROM transforms WHERE subject=? AND name=? and type=?"
        data = self.cur.execute(query, (subject, name, xfmtype)).fetchone()
        if data is None:
            return
        else:
            xfm, filename = data
            return np.fromstring(xfm).reshape(4,4), filename

    def getVTK(self, subject, type, hemisphere="both", date=None):
        import vtkutils
        query = "SELECT filename, offset FROM surfaces WHERE subject=? AND type=? AND hemisphere=?"
        if self.cur.execute(query, (subject, type, "lh")).fetchone() is None:
            #Subject / type does not exist in the database
            return None

        if hemisphere == "both":
            lh, offset = self.cur.execute(query, (subject, type, 'lh')).fetchone()
            rh, offset = self.cur.execute(query, (subject, type, 'rh')).fetchone()
            if offset is not None and len(offset) > 0:
                offset = np.array([float(d) for d in offset.split()])
            return vtkutils.read([lh, rh], offset=offset)
        else:
            d, offset = self.cur.execute(query, (subject, type, hemisphere)).fetchone()
            return vtkutils.read([d])

surfs = Database()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load a directory of flatmaps into the database")
    parser.add_argument("subject", type=str, help="Subject name (two letter abbreviation)")
    parser.add_argument("vtkdir", type=str, help="Directory with VTK's")
    args = parser.parse_args()


    #surfs.loadXfm(subject, xfmname, magnet, xfmtype='magnet', filename=epi, override=True)
    #surfs.loadXfm(subject, xfmname, shortcut, xfmtype='coord', filename=epi, override=True)
    
    try:
        surfs.loadVTKdir(args.vtkdir, args.subject)
        print "Success!"
    except Exception, e:
        print "Error with processing: ", e