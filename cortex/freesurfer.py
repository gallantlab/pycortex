import os
import shutil
import struct
import tempfile
import warnings
import shlex
import subprocess as sp

import numpy as np

from . import database
from . import anat

def get_paths(subject, hemi, type="patch", freesurfer_subject_dir=None):
    """Retrive paths for all surfaces for a subject processed by freesurfer

    Parameters
    ----------
    subject : string
        Subject ID
    hem : string ['lh'|'rh']
        Left ('lh') or right ('rh') hemisphere
    type : string ['patch'|'surf'|'curv']
        Which type of files to return
    freesurfer_subject_dir : string | None
        Directory of freesurfer subjects. Defaults to the value for
        the environment variable 'SUBJECTS_DIR' (which should be set
        by freesurfer)
    """
    if freesurfer_subject_dir is None:
        freesurfer_subject_dir = os.environ['SUBJECTS_DIR']
    base = os.path.join(freesurfer_subject_dir,subject)
    if type == "patch":
        return os.path.join(base, "surf", hemi+".{name}.patch.3d")
    elif type == "surf":
        return os.path.join(base, "surf", hemi+".{name}")
    elif type == "curv":
        return os.path.join(base, "surf", hemi+".curv{name}")

def autorecon(subject, type="all"):
    types = {
        'all':'autorecon-all',
        '1':"autorecon1",
        '2':"autorecon2",
        '3':"autorecon3",
        'cp':"autorecon2-cp",
        'wm':"autorecon2-wm",
        'pia':"autorecon2-pial"}

    times = {
        'all':"12 hours",
        '2':"6 hours",
        'cp':"8 hours",
        'wm':"4 hours"
        }
    if str(type) in times:
        resp = raw_input("recon-all will take approximately %s to run! Continue? "%times[str(type)])
        if resp.lower() not in ("yes", "y"):
            return

    cmd = "recon-all -s {subj} -{cmd}".format(subj=subject, cmd=types[str(type)])
    sp.check_call(shlex.split(cmd))

def flatten(subject, hemi, patch, freesurfer_subject_dir=None):
    resp = raw_input('Flattening takes approximately 2 hours! Continue? ')
    if resp.lower() in ('y', 'yes'):
        inpath = get_paths(subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir).format(name=patch)
        outpath = get_paths(subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir).format(name=patch+".flat")
        cmd = "mris_flatten -O fiducial {inpath} {outpath}".format(inpath=inpath, outpath=outpath)
        sp.check_call(shlex.split(cmd))
    else:
        print("Not going to flatten...")

def import_subj(subject, sname=None, freesurfer_subject_dir=None):
    """
    Parameters
    ----------
    subject : string
        Freesurfer subject name
    sname : string
        Pycortex subject name (These variable names should be changed)
    """
    if sname is None:
        sname = subject
    database.db.make_subj(sname)

    import nibabel
    surfs = os.path.join(database.default_filestore, sname, "surfaces", "{name}_{hemi}.gii")
    anats = os.path.join(database.default_filestore, sname, "anatomicals", "{name}.nii.gz")
    surfinfo = os.path.join(database.default_filestore, sname, "surface-info", "{name}.npz")
    if freesurfer_subject_dir is None:
        freesurfer_subject_dir = os.environ['SUBJECTS_DIR']
    fspath = os.path.join(freesurfer_subject_dir, subject, 'mri')
    curvs = os.path.join(freesurfer_subject_dir, subject, 'surf', '{hemi}.{name}')

    #import anatomicals
    for fsname, name in dict(rawavg="raw", aseg="aseg").items():
        path_pattern = os.path.join(fspath, "{fsname}.mgz")
        if fsname == 'rawavg' and (not os.path.exists(path_pattern.format(fsname=fsname))):
            # no `rawavg` available, default to T1
            fsname = 'T1'

        path = path_pattern.format(fsname=fsname)
        out = anats.format(subj=sname, name=name)
        cmd = "mri_convert {path} {out}".format(path=path, out=out)
        sp.call(shlex.split(cmd))

    if not os.path.exists(curvs.format(hemi="lh", name="fiducial")):
        make_fiducial(subject)

    #Freesurfer uses FOV/2 for center, let's set the surfaces to use the magnet isocenter
    trans = nibabel.load(out).get_affine()[:3, -1]
    surfmove = trans - np.sign(trans) * [128, 128, 128]

    from . import formats
    #import surfaces
    for fsname, name in [('smoothwm',"wm"), ('pial',"pia"), ('inflated',"inflated")]:
        for hemi in ("lh", "rh"):
            pts, polys, _ = get_surf(subject, hemi, fsname, freesurfer_subject_dir=freesurfer_subject_dir)
            fname = surfs.format(subj=sname, name=name, hemi=hemi)
            formats.write_gii(fname, pts=pts + surfmove, polys=polys)

    #import surfinfo
    for curv, info in dict(sulc="sulcaldepth", thickness="thickness", curv="curvature").items():
        lh, rh = [parse_curv(curvs.format(hemi=hemi, name=curv)) for hemi in ['lh', 'rh']]
        np.savez(surfinfo.format(subj=sname, name=info), left=-lh, right=-rh)

def import_flat(subject, patch, sname=None, freesurfer_subject_dir=None):
    if sname is None:
        sname = subject
    surfs = os.path.join(database.default_filestore, sname, "surfaces", "flat_{hemi}.gii")

    from . import formats
    for hemi in ['lh', 'rh']:
        pts, polys, _ = get_surf(subject, hemi, "patch", patch+".flat", freesurfer_subject_dir=freesurfer_subject_dir)
        flat = pts[:,[1, 0, 2]]
        flat[:,1] = -flat[:,1]
        fname = surfs.format(hemi=hemi)
        print("saving to %s"%fname)
        formats.write_gii(fname, pts=flat, polys=polys)

    #clear the cache, per #81
    cache = os.path.join(database.default_filestore, sname, "cache")
    shutil.rmtree(cache)
    os.makedirs(cache)

def make_fiducial(subject, freesurfer_subject_dir=None):
    for hemi in ['lh', 'rh']:
        spts, polys, _ = get_surf(subject, hemi, "smoothwm", freesurfer_subject_dir=freesurfer_subject_dir)
        ppts, _, _ = get_surf(subject, hemi, "pial", freesurfer_subject_dir=freesurfer_subject_dir)
        fname = get_paths(subject, hemi, "surf", freesurfer_subject_dir=freesurfer_subject_dir).format(name="fiducial")
        write_surf(fname, (spts + ppts) / 2, polys)

def parse_surf(filename):
    with open(filename, 'rb') as fp:
        #skip magic
        fp.seek(3)
        comment = fp.readline()
        fp.readline()
        print(comment)
        verts, faces = struct.unpack('>2I', fp.read(8))
        pts = np.fromstring(fp.read(4*3*verts), dtype='f4').byteswap()
        polys = np.fromstring(fp.read(4*3*faces), dtype='i4').byteswap()

        return pts.reshape(-1, 3), polys.reshape(-1, 3)

def write_surf(filename, pts, polys, comment=''):
    with open(filename, 'wb') as fp:
        fp.write('\xff\xff\xfe')
        fp.write(comment+'\n\n')
        fp.write(struct.pack('>2I', len(pts), len(polys)))
        fp.write(pts.astype(np.float32).byteswap().tostring())
        fp.write(polys.astype(np.uint32).byteswap().tostring())
        fp.write('\n')

def parse_curv(filename):
    with open(filename, 'rb') as fp:
        fp.seek(15)
        return np.fromstring(fp.read(), dtype='>f4').byteswap().newbyteorder()

def parse_patch(filename):
    with open(filename, 'rb') as fp:
        header, = struct.unpack('>i', fp.read(4))
        nverts, = struct.unpack('>i', fp.read(4))
        data = np.fromstring(fp.read(), dtype=[('vert', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
        assert len(data) == nverts
        return data

def get_surf(subject, hemi, type, patch=None, freesurfer_subject_dir=None):
    if type == "patch":
        assert patch is not None
        surf_file = get_paths(subject, hemi, 'surf', freesurfer_subject_dir=freesurfer_subject_dir).format(name='smoothwm')
    else:
        surf_file = get_paths(subject, hemi, 'surf', freesurfer_subject_dir=freesurfer_subject_dir).format(name=type)

    pts, polys = parse_surf(surf_file)

    if patch is not None:
        patch_file = get_paths(subject, hemi, 'patch', freesurfer_subject_dir=freesurfer_subject_dir).format(name=patch)
        patch = parse_patch(patch_file)
        verts = patch[patch['vert'] > 0]['vert'] - 1
        edges = -patch[patch['vert'] < 0]['vert'] - 1

        idx = np.zeros((len(pts),), dtype=bool)
        idx[verts] = True
        idx[edges] = True
        valid = idx[polys.ravel()].reshape(-1, 3).all(1)
        polys = polys[valid]
        idx = np.zeros((len(pts),))
        idx[verts] = 1
        idx[edges] = -1

    if type == "patch":
        for i, x in enumerate(['x', 'y', 'z']):
            pts[verts, i] = patch[patch['vert'] > 0][x]
            pts[edges, i] = patch[patch['vert'] < 0][x]
        return pts, polys, idx

    return pts, polys, get_curv(subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir)

def get_curv(subject, hemi, type='wm', freesurfer_subject_dir=None):
    if type == "wm":
        curv_file = get_paths(subject, hemi, 'curv', freesurfer_subject_dir=freesurfer_subject_dir).format(name='')
    else:
        curv_file = get_paths(subject, hemi, 'curv', freesurfer_subject_dir=freesurfer_subject_dir).format(name='.'+type)

    return parse_curv(curv_file)

def show_surf(subject, hemi, type, patch=None, curv=True, freesurfer_subject_dir=None):
    """Show a surface from a Freesurfer subject directory
    """
    from mayavi import mlab
    from tvtk.api import tvtk

    pts, polys, idx = get_surf(subject, hemi, type, patch, freesurfer_subject_dir=freesurfer_subject_dir)
    if curv:
        curv = get_curv(subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir)
    else:
        curv = idx

    fig = mlab.figure()
    src = mlab.pipeline.triangular_mesh_source(pts[:,0], pts[:,1], pts[:,2], polys, scalars=curv, figure=fig)
    norms = mlab.pipeline.poly_data_normals(src, figure=fig)
    norms.filter.splitting = False
    surf = mlab.pipeline.surface(norms, figure=fig)
    surf.parent.scalar_lut_manager.set(lut_mode='RdBu', data_range=[-1,1], use_default_range=False)

    cursors = mlab.pipeline.scalar_scatter([0], [0], [0])
    glyphs = mlab.pipeline.glyph(cursors, figure=fig)
    glyphs.glyph.glyph_source.glyph_source = glyphs.glyph.glyph_source.glyph_dict['axes']

    fig.scene.background = (0,0,0)
    fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

    path = os.path.join(os.environ['SUBJECTS_DIR'], subject)
    def picker_callback(picker):
        if picker.actor in surf.actor.actors:
            npts = np.append(cursors.data.points.to_array(), [pts[picker.point_id]], axis=0)
            cursors.data.points = npts
            print(picker.point_id)
            x, y, z = pts[picker.point_id]
            with open(os.path.join(path, 'tmp', 'edit.dat'), 'w') as fp:
                fp.write('%f %f %f\n'%(x, y, z))

    picker = fig.on_mouse_pick(picker_callback)
    picker.tolerance = 0.01
    mlab.show()
    return fig, surf

def write_dot(fname, pts, polys, name="test"):
    import networkx as nx
    def iter_surfedges(tris):
        for a,b,c in tris:
            yield a,b
            yield b,c
            yield a,c
    graph = nx.Graph()
    graph.add_edges_from(iter_surfedges(polys))
    lengths = []
    with open(fname, "w") as fp:
        fp.write("graph %s {\n"%name)
        fp.write('node [shape=point,label=""];\n')
        for a, b in graph.edges_iter():
            l = np.sqrt(((pts[a] - pts[b])**2).sum(-1))
            lengths.append(l)
            fp.write("%s -- %s [len=%f];\n"%(a, b, l))
        fp.write("maxiter=1000000;\n");
        fp.write("}")

def read_dot(fname, pts):
    import re
    parse = re.compile(r'\s(\d+)\s\[label="", pos="([\d\.]+),([\d\.]+)".*];')
    data = np.zeros((len(pts), 2))
    with open(fname) as fp:
        fp.readline()
        fp.readline()
        fp.readline()
        fp.readline()
        el = fp.readline().split(' ')
        while el[1] != '--':
            x, y = el[2][5:-2].split(',')
            data[int(el[0][1:])] = float(x), float(y)
            el = fp.readline().split(' ')
    return data

def write_decimated(path, pts, polys):
    from .polyutils import decimate, boundary_edges
    dpts, dpolys = decimate(pts, polys)
    write_surf(path+'.smoothwm', dpts, dpolys)
    edges = boundary_edges(dpolys)
    data = np.zeros((len(dpts),), dtype=[('vert', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
    data['vert'] = np.arange(len(dpts))+1
    data['vert'][edges] *= -1
    data['x'] = dpts[:,0]
    data['y'] = dpts[:,1]
    data['z'] = dpts[:,2]
    with open(path+'.full.patch.3d', 'wb') as fp:
        fp.write(struct.pack('>i', -1))
        fp.write(struct.pack('>i', len(dpts)))
        fp.write(data.tostring())

import copy
class SpringLayout(object):
    def __init__(self, pts, polys, dpts=None, pins=None, stepsize=1, neighborhood=0):
        self.pts = pts
        self.polys = polys
        self.stepsize = stepsize
        pinmask = np.zeros((len(pts),), dtype=bool)
        if isinstance(pins, (list, set, np.ndarray)):
            pinmask[pins] = True
        self.pins = pinmask
        self.neighbors = [set() for _ in range(len(pts))]

        for i, j, k in polys:
            self.neighbors[i].add(j)
            self.neighbors[i].add(k)
            self.neighbors[j].add(i)
            self.neighbors[j].add(k)
            self.neighbors[k].add(i)
            self.neighbors[k].add(j)

        for _ in range(neighborhood):
            _neighbors = copy.deepcopy(self.neighbors)
            for v, neighbors in enumerate(self.neighbors):
                for n in neighbors:
                    _neighbors[v] |= self.neighbors[n]
            self.neighbors = _neighbors

        for i in range(len(self.neighbors)):
            self.neighbors[i] = list(self.neighbors[i] - set([i]))

        if dpts is None:
            dpts = pts

        #self.kdt = cKDTree(self.pts)
        self._next = self.pts.copy()

        width = max(len(n) for n in self.neighbors)
        self._mask = np.zeros((len(pts), width), dtype=bool)
        self._move = np.zeros((len(pts), width, 3))
        #self._mean = np.zeros((len(pts), width))
        self._num = np.zeros((len(pts),))
        self._dists = []
        self._idx = []
        for i, n in enumerate(self.neighbors):
            self._mask[i, :len(n)] = True
            self._dists.append(np.sqrt(((dpts[n] - dpts[i])**2).sum(-1)))
            self._idx.append(np.ones((len(n),))*i)
            self._num[i] = len(n)
        self._dists = np.hstack(self._dists)
        self._idx = np.hstack(self._idx).astype(np.uint)
        self._neigh = np.hstack(self.neighbors).astype(np.uint)
        self.figure = None

    def _spring(self):
        svec = self.pts[self._neigh] - self.pts[self._idx]
        slen = np.sqrt((svec**2).sum(-1))
        force = (slen - self._dists) # / self._dists
        svec /= slen[:,np.newaxis]
        fvec = force[:, np.newaxis] * svec
        self._move[self._mask] = self.stepsize * fvec
        return self._move.sum(1) / self._num[:, np.newaxis]

    def _estatic(self, idx):
        dist, neighbors = self.kdt.query(self.pts[idx], k=20)
        valid = dist > 0
        mag = self.stepsize * (1 / dist)
        diff = self.pts[neighbors] - self.pts[idx]
        return (mag[valid] * diff[valid].T).T.mean(0)

    def step(self):
        move = self._spring()[~self.pins]
        self._next[~self.pins] += move #+ self._estatic(i)
        self.pts = self._next.copy()
        return dict(x=self.pts[:,0],y=self.pts[:, 1], z=self.pts[:,2]), move
        #self.kdt = cKDTree(self.pts)

    def run(self, n=1000):
        for _ in range(n):
            self.step()
            print(_)

    def view_step(self):
        from mayavi import mlab
        if self.figure is None:
            self.figure = mlab.triangular_mesh(self.pts[:,0], self.pts[:,1], self.pts[:,2], self.polys, representation='wireframe')
        self.step()
        self.figure.mlab_source.set(x=self.pts[:,0], y=self.pts[:,1], z=self.pts[:,2])

def stretch_mwall(pts, polys, mwall):
    inflated = pts.copy()
    center = pts[mwall].mean(0)
    radius = max((pts.max(0) - pts.min(0))[1:])
    angles = np.arctan2(pts[mwall][:,2], pts[mwall][:,1])
    pts[mwall, 0] = center[0]
    pts[mwall, 1] = radius * np.cos(angles) + center[1]
    pts[mwall, 2] = radius * np.sin(angles) + center[2]
    return SpringLayout(pts, polys, inflated, pins=mwall)


if __name__ == "__main__":
    import sys
    show_surf(sys.argv[1], sys.argv[2], sys.argv[3])
