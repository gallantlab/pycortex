import os
import json
import cStringIO
import sqlite3

import numpy as np
import db

def main():
    names = dict()
    buf = cStringIO.StringIO()
    conn = sqlite3.connect("database.sql")
    cur = conn.cursor()
    fetched = cur.execute("SELECT subject, name, xfm, filename, type from transforms")
    for subj, xfmname, xfm, filename, tname in fetched.fetchall():
        xfm = np.fromstring(xfm).reshape(4,4)
        if xfmname not in names:
            names[xfmname] = dict(epifile=filename, subject=subj)

        #buf.seek(0)
        #np.savetxt(buf, xfm)
        #buf.seek(0)
        names[xfmname][tname] = xfm.tolist()

    for name, jsdict in names.items():
        fname = "{subj}_{name}.xfm".format(subj=jsdict['subject'], name=name)
        fname = os.path.join(db.filestore, "transforms", fname)
        json.dump(jsdict, open(fname, "w"), indent=4, sort_keys=True)


def simulate_loaderr(xfm, cycles=10000):
    start = xfm.copy()
    buf = cStringIO.StringIO()
    err = np.zeros(cycles)
    for i in range(cycles):
        buf.seek(0)
        json.dump(dict(value=xfm.tolist()), buf)
        buf.seek(0)
        xfm = np.array(json.load(buf)['value'])
        err[i] = ((xfm - start)**2).sum()

    return err