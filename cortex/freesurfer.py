"""Contains functions for interfacing with freesurfer
"""
import os
import shutil
import struct
import tempfile
import warnings
import shlex
import subprocess as sp
from builtins import input

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
    """Run Freesurfer's autorecon-all command for a given freesurfer subject
    
    Parameters
    ----------
    subject : string
        Freesurfer subject ID (should be a folder in your freesurfer $SUBJECTS_DIR)
    type : string
        Which steps of autorecon-all to perform. {'all', '1','2','3','cp','wm', 'pia'}

    """
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
        resp = input("recon-all will take approximately %s to run! Continue? "%times[str(type)])
        if resp.lower() not in ("yes", "y"):
            return
            
    cmd = "recon-all -s {subj} -{cmd}".format(subj=subject, cmd=types[str(type)])
    sp.check_call(shlex.split(cmd))

def flatten(subject, hemi, patch, freesurfer_subject_dir=None):
    """Perform flattening of a brain using freesurfer
    
    Parameters
    ----------
    subject : 
    
    hemi : 
    
    patch : 
    
    freesurfer_subject_dir :
    
    Returns
    -------
    """
    resp = input('Flattening takes approximately 2 hours! Continue? ')
    if resp.lower() in ('y', 'yes'):
        inpath = get_paths(subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir).format(name=patch)
        outpath = get_paths(subject, hemi, freesurfer_subject_dir=freesurfer_subject_dir).format(name=patch+".flat")
        cmd = "mris_flatten -O fiducial {inpath} {outpath}".format(inpath=inpath, outpath=outpath)
        sp.check_call(shlex.split(cmd))
    else:
        print("Not going to flatten...")

def import_subj(subject, sname=None, freesurfer_subject_dir=None, whitematter_surf='smoothwm'):
    """Imports a subject from freesurfer
    
    Parameters
    ----------
    subject : string
        Freesurfer subject name
    sname : string, optional
        Pycortex subject name (These variable names should be changed). By default uses
        the same name as the freesurfer subject.
    freesurfer_subject_dir : string, optional
        Freesurfer subject directory to pull data from. By default uses the directory
        given by the environment variable $SUBJECTS_DIR.
    whitematter_surf : string, optional
        Which whitematter surface to import as 'wm'. By default uses 'smoothwm', but that
        surface is smoothed and may not be appropriate. A good alternative is 'white'.
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
    for fsname, name in dict(T1="raw", aseg="aseg").items():
        path = os.path.join(fspath, "{fsname}.mgz").format(fsname=fsname)
        out = anats.format(subj=sname, name=name)
        cmd = "mri_convert {path} {out}".format(path=path, out=out)
        sp.call(shlex.split(cmd))

    if not os.path.exists(curvs.format(hemi="lh", name="fiducial")):
        make_fiducial(subject, freesurfer_subject_dir=freesurfer_subject_dir)

    #Freesurfer uses FOV/2 for center, let's set the surfaces to use the magnet isocenter
    trans = nibabel.load(out).get_affine()[:3, -1]
    surfmove = trans - np.sign(trans) * [128, 128, 128]

    from . import formats
    #import surfaces
    for fsname, name in [(whitematter_surf,"wm"), ('pial',"pia"), ('inflated',"inflated")]:
        for hemi in ("lh", "rh"):
            pts, polys, _ = get_surf(subject, hemi, fsname, freesurfer_subject_dir=freesurfer_subject_dir)
            fname = str(surfs.format(subj=sname, name=name, hemi=hemi))
            formats.write_gii(fname, pts=pts + surfmove, polys=polys)

    #import surfinfo
    for curv, info in dict(sulc="sulcaldepth", thickness="thickness", curv="curvature").items():
        lh, rh = [parse_curv(curvs.format(hemi=hemi, name=curv)) for hemi in ['lh', 'rh']]
        np.savez(surfinfo.format(subj=sname, name=info), left=-lh, right=-rh)

    database.db = database.Database()

def import_flat(subject, patch, sname=None, freesurfer_subject_dir=None):
    """Imports a flat brain from freesurfer
    
    Parameters
    ----------
    subject : str
        Freesurfer subject name
    patch : 
    
    sname : str
        Pycortex subject name3
    freesurfer_subject_dir : str
    
    Returns
    -------
    """
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
    """  
    """
    for hemi in ['lh', 'rh']:
        spts, polys, _ = get_surf(subject, hemi, "smoothwm", freesurfer_subject_dir=freesurfer_subject_dir)
        ppts, _, _ = get_surf(subject, hemi, "pial", freesurfer_subject_dir=freesurfer_subject_dir)
        fname = get_paths(subject, hemi, "surf", freesurfer_subject_dir=freesurfer_subject_dir).format(name="fiducial")
        write_surf(fname, (spts + ppts) / 2, polys)

def parse_surf(filename):
    """  
    """
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
    """  
    """
    with open(filename, 'wb') as fp:
        fp.write(b'\xff\xff\xfe')
        fp.write((comment+'\n\n').encode())
        fp.write(struct.pack('>2I', len(pts), len(polys)))
        fp.write(pts.astype(np.float32).byteswap().tostring())
        fp.write(polys.astype(np.uint32).byteswap().tostring())
        fp.write(b'\n')

def parse_curv(filename):
    """  
    """
    with open(filename, 'rb') as fp:
        fp.seek(15)
        return np.fromstring(fp.read(), dtype='>f4').byteswap().newbyteorder()

def parse_patch(filename):
    """  
    """
    with open(filename, 'rb') as fp:
        header, = struct.unpack('>i', fp.read(4))
        nverts, = struct.unpack('>i', fp.read(4))
        data = np.fromstring(fp.read(), dtype=[('vert', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
        assert len(data) == nverts
        return data

def get_surf(subject, hemi, type, patch=None, freesurfer_subject_dir=None):
    """  
    """
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

def _move_labels(subject, label, hemisphere=('lh','rh'), fs_dir=None, src_subject='fsaverage'):
    """subject is a freesurfer subject"""
    if fs_dir is None:
        fs_dir = os.environ['SUBJECTS_DIR']
    for hemi in hemisphere:
        srclabel = os.path.join(fs_dir, src_subject, 'label', 
                                '{hemi}.{label}.label'.format(hemi=hemi, label=label))
        trglabel = os.path.join(fs_dir, subject, 'label', 
                                '{hemi}.{label}.label'.format(hemi=hemi, label=label))
        if not os.path.exists(srclabel):
            raise ValueError("Label {} doesn't exist!".format(srclabel))
        fs_sub_dir = os.path.join(fs_dir, subject, 'label')
        if not os.path.exists(fs_sub_dir):
            raise ValueError("Freesurfer subject directory for subject ({}) does not exist!".format(fs_sub_dir))
        cmd = ("mri_label2label --srcsubject {src_subject} --trgsubject {subject} "
               "--srclabel {srclabel} --trglabel {trglabel} "
               "--regmethod surface --hemi {hemi}")
        cmd_f = cmd.format(hemi=hemi, subject=subject, src_subject=src_subject, 
                           srclabel=srclabel, trglabel=trglabel)
        print("Calling: ")
        print(cmd_f)
        to_call = shlex.split(cmd_f)
        proc = sp.Popen(to_call,
                           stdin=sp.PIPE,
                           stdout=sp.PIPE,
                           stderr=sp.PIPE)
        stdout, stderr = proc.communicate()
        if stderr not in ('', b''):
            raise Exception("Error in freesurfer function call:\n{}".format(stderr))
    print("Labels transferred")

def _parse_labels(label_files, subject):
    """Extract values from freesurfer label file(s) and map to vertices

    Parameters
    ----------
    label_files : str or list
        full paths to label file or files to load
    subject : str
        pycortex subject ID
    """
    if not isinstance(label_files, (list, tuple)):
        label_files = [label_files]
    verts = []
    values = []
    lh_surf, _ = database.db.get_surf(subject, 'fiducial', 'left')
    for fname in label_files:
        with open(fname) as fid:
            lines = fid.readlines()
            lines = [[float(xx.strip()) for xx in x.split(' ') if xx.strip()] for x in lines[2:]]
            vals = np.array(lines)
            if '/lh.' in fname:
                verts.append(vals[:,0])
            elif '/rh.' in fname:
                verts.append(vals[:,0] + lh_surf.shape[0])
            values.append(vals[:,-1])
    verts = np.hstack(verts)
    values = np.hstack(values)
    return verts, values

def get_label(subject, label, fs_subject=None, fs_dir=None, src_subject='fsaverage', hemisphere=('lh', 'rh'), **kwargs):
    """Get data from a label file for fsaverage subject

    Parameters
    ----------
    subject : str
        A pycortex subject ID
    label : str
        Label name
    fs_subject : str
        Freesurfer subject ID, if different from pycortex subject ID
    src_subject : str
        Freesurfer subject ID from which to transfer the label. 
    fs_dir : str
        Freesurfer subject directory; None defaults to OS environment variable 
    hemisphere : list | tuple

    """
    if fs_dir is None:
        fs_dir = os.environ['SUBJECTS_DIR']
    else:
        os.environ['SUBJECTS_DIR'] = fs_dir
    if fs_subject is None:
        fs_subject = subject
    label_files = [os.path.join(fs_dir, fs_subject, 'label', '{}.{}.label'.format(h, label)) for h in hemisphere]
    # If label file doesn't exist, try to move it there
    print('looking for {}'.format(label_files))
    if not all([os.path.exists(f) for f in label_files]):
        print("Transforming label file to subject's freesurfer directory...")
        _move_labels(fs_subject, label, hemisphere=hemisphere, fs_dir=fs_dir, src_subject=src_subject)
    verts, values = _parse_labels(label_files, subject)
    idx = verts.astype(np.int)
    return idx, values

def get_curv(subject, hemi, type='wm', freesurfer_subject_dir=None):
    """  
    """
    if type == "wm":
        curv_file = get_paths(subject, hemi, 'curv', freesurfer_subject_dir=freesurfer_subject_dir).format(name='')
    else:
        curv_file = get_paths(subject, hemi, 'curv', freesurfer_subject_dir=freesurfer_subject_dir).format(name='.'+type)

    return parse_curv(curv_file)

def show_surf(subject, hemi, type, patch=None, curv=True, freesurfer_subject_dir=None):
    """Show a surface from a Freesurfer subject directory
    
    Parameters
    ----------
    subject : str
        Freesurfer subject name
    hemi : 
    
    type : 
    
    patch : 
    
    curv : bool
    
    freesurfer_subject_dir :
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
    """  
    """
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
    """  
    """
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
    """  
    """
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
    with open(path+'.full.patch.3d', 'w') as fp:
        fp.write(struct.pack('>i', -1))
        fp.write(struct.pack('>i', len(dpts)))
        fp.write(data.tostring())

import copy
class SpringLayout(object):
    """  
    """
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
    """  
    """
    inflated = pts.copy()
    center = pts[mwall].mean(0)
    radius = max((pts.max(0) - pts.min(0))[1:])
    angles = np.arctan2(pts[mwall][:,2], pts[mwall][:,1])
    pts[mwall, 0] = center[0]
    pts[mwall, 1] = radius * np.cos(angles) + center[1]
    pts[mwall, 2] = radius * np.sin(angles) + center[2]
    return SpringLayout(pts, polys, inflated, pins=mwall)

# aseg partition labels (up to 256 only)
fs_aseg_dict = {'Unknown' : 0,
    'Left-Cerebral-Exterior' : 1,
    'Left-Cerebral-White-Matter' : 2,
    'Left-Cerebral-Cortex' : 3,
    'Left-Lateral-Ventricle' : 4,
    'Left-Inf-Lat-Vent' : 5,
    'Left-Cerebellum-Exterior' : 6,
    'Left-Cerebellum-White-Matter' : 7,
    'Left-Cerebellum-Cortex' : 8,
    'Left-Thalamus' : 9,
    'Left-Thalamus-Proper' : 10,
    'Left-Caudate' : 11,
    'Left-Putamen' : 12,
    'Left-Pallidum' : 13,
    '3rd-Ventricle' : 14,
    '4th-Ventricle' : 15,
    'Brain-Stem' : 16,
    'Left-Hippocampus' : 17,
    'Left-Amygdala' : 18,
    'Left-Insula' : 19,
    'Left-Operculum' : 20,
    'Line-1' : 21,
    'Line-2' : 22,
    'Line-3' : 23,
    'CSF' : 24,
    'Left-Lesion' : 25,
    'Left-Accumbens-area' : 26,
    'Left-Substancia-Nigra' : 27,
    'Left-VentralDC' : 28,
    'Left-undetermined' : 29,
    'Left-vessel' : 30,
    'Left-choroid-plexus' : 31,
    'Left-F3orb' : 32,
    'Left-lOg' : 33,
    'Left-aOg' : 34,
    'Left-mOg' : 35,
    'Left-pOg' : 36,
    'Left-Stellate' : 37,
    'Left-Porg' : 38,
    'Left-Aorg' : 39,
    'Right-Cerebral-Exterior' : 40,
    'Right-Cerebral-White-Matter' : 41,
    'Right-Cerebral-Cortex' : 42,
    'Right-Lateral-Ventricle' : 43,
    'Right-Inf-Lat-Vent' : 44,
    'Right-Cerebellum-Exterior' : 45,
    'Right-Cerebellum-White-Matter' : 46,
    'Right-Cerebellum-Cortex' : 47,
    'Right-Thalamus' : 48,
    'Right-Thalamus-Proper' : 49,
    'Right-Caudate' : 50,
    'Right-Putamen' : 51,
    'Right-Pallidum' : 52,
    'Right-Hippocampus' : 53,
    'Right-Amygdala' : 54,
    'Right-Insula' : 55,
    'Right-Operculum' : 56,
    'Right-Lesion' : 57,
    'Right-Accumbens-area' : 58,
    'Right-Substancia-Nigra' : 59,
    'Right-VentralDC' : 60,
    'Right-undetermined' : 61,
    'Right-vessel' : 62,
    'Right-choroid-plexus' : 63,
    'Right-F3orb' : 64,
    'Right-lOg' : 65,
    'Right-aOg' : 66,
    'Right-mOg' : 67,
    'Right-pOg' : 68,
    'Right-Stellate' : 69,
    'Right-Porg' : 70,
    'Right-Aorg' : 71,
    '5th-Ventricle' : 72,
    'Left-Interior' : 73,
    'Right-Interior' : 74,
    'Left-Lateral-Ventricles' : 75,
    'Right-Lateral-Ventricles' : 76,
    'WM-hypointensities' : 77,
    'Left-WM-hypointensities' : 78,
    'Right-WM-hypointensities' : 79,
    'non-WM-hypointensities' : 80,
    'Left-non-WM-hypointensities' : 81,
    'Right-non-WM-hypointensities' : 82,
    'Left-F1' : 83,
    'Right-F1' : 84,
    'Optic-Chiasm' : 85,
    'Corpus_Callosum' : 86,
    'Left-Amygdala-Anterior' : 96,
    'Right-Amygdala-Anterior' : 97,
    'Dura' : 98,
    'Left-wm-intensity-abnormality' : 100,
    'Left-caudate-intensity-abnormality' : 101,
    'Left-putamen-intensity-abnormality' : 102,
    'Left-accumbens-intensity-abnormality' : 103,
    'Left-pallidum-intensity-abnormality' : 104,
    'Left-amygdala-intensity-abnormality' : 105,
    'Left-hippocampus-intensity-abnormality' : 106,
    'Left-thalamus-intensity-abnormality' : 107,
    'Left-VDC-intensity-abnormality' : 108,
    'Right-wm-intensity-abnormality' : 109,
    'Right-caudate-intensity-abnormality' : 110,
    'Right-putamen-intensity-abnormality' : 111,
    'Right-accumbens-intensity-abnormality' : 112,
    'Right-pallidum-intensity-abnormality' : 113,
    'Right-amygdala-intensity-abnormality' : 114,
    'Right-hippocampus-intensity-abnormality' : 115,
    'Right-thalamus-intensity-abnormality' : 116,
    'Right-VDC-intensity-abnormality' : 117,
    'Epidermis' : 118,
    'Conn-Tissue' : 119,
    'SC-Fat/Muscle' : 120,
    'Cranium' : 121,
    'CSF-SA' : 122,
    'Muscle' : 123,
    'Ear' : 124,
    'Adipose' : 125,
    'Spinal-Cord' : 126,
    'Soft-Tissue' : 127,
    'Nerve' : 128,
    'Bone' : 129,
    'Air' : 130,
    'Orbital-Fat' : 131,
    'Tongue' : 132,
    'Nasal-Structures' : 133,
    'Globe' : 134,
    'Teeth' : 135,
    'Left-Caudate/Putamen' : 136,
    'Right-Caudate/Putamen' : 137,
    'Left-Claustrum' : 138,
    'Right-Claustrum' : 139,
    'Cornea' : 140,
    'Diploe' : 142,
    'Vitreous-Humor' : 143,
    'Lens' : 144,
    'Aqueous-Humor' : 145,
    'Outer-Table' : 146,
    'Inner-Table' : 147,
    'Periosteum' : 148,
    'Endosteum' : 149,
    'R/C/S' : 150,
    'Iris' : 151,
    'SC-Adipose/Muscle' : 152,
    'SC-Tissue' : 153,
    'Orbital-Adipose' : 154,
    'Left-IntCapsule-Ant' : 155,
    'Right-IntCapsule-Ant' : 156,
    'Left-IntCapsule-Pos' : 157,
    'Right-IntCapsule-Pos' : 158,
    'Left-Cerebral-WM-unmyelinated' : 159,
    'Right-Cerebral-WM-unmyelinated' : 160,
    'Left-Cerebral-WM-myelinated' : 161,
    'Right-Cerebral-WM-myelinated' : 162,
    'Left-Subcortical-Gray-Matter' : 163,
    'Right-Subcortical-Gray-Matter' : 164,
    'Skull' : 165,
    'Posterior-fossa' : 166,
    'Scalp' : 167,
    'Hematoma' : 168,
    'Left-Cortical-Dysplasia' : 180,
    'Right-Cortical-Dysplasia' : 181,
    'Left-hippocampal_fissure' : 193,
    'Left-CADG-head' : 194,
    'Left-subiculum' : 195,
    'Left-fimbria' : 196,
    'Right-hippocampal_fissure' : 197,
    'Right-CADG-head' : 198,
    'Right-subiculum' : 199,
    'Right-fimbria' : 200,
    'alveus' : 201,
    'perforant_pathway' : 202,
    'parasubiculum' : 203,
    'presubiculum' : 204,
    'subiculum' : 205,
    'CA1' : 206,
    'CA2' : 207,
    'CA3' : 208,
    'CA4' : 209,
    'GC-DG' : 210,
    'HATA' : 211,
    'fimbria' : 212,
    'lateral_ventricle' : 213,
    'molecular_layer_HP' : 214,
    'hippocampal_fissure' : 215,
    'entorhinal_cortex' : 216,
    'molecular_layer_subiculum' : 217,
    'Amygdala' : 218,
    'Cerebral_White_Matter' : 219,
    'Cerebral_Cortex' : 220,
    'Inf_Lat_Vent' : 221,
    'Perirhinal' : 222,
    'Cerebral_White_Matter_Edge' : 223,
    'Background' : 224,
    'Ectorhinal' : 225,
    'Fornix' : 250,
    'CC_Posterior' : 251,
    'CC_Mid_Posterior' : 252,
    'CC_Central' : 253,
    'CC_Mid_Anterior' : 254,
    'CC_Anterior' : 255}

if __name__ == "__main__":
    import sys
    show_surf(sys.argv[1], sys.argv[2], sys.argv[3])
