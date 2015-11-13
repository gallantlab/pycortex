import os
import shlex
import shutil
import tempfile
import subprocess as sp

import numpy as np

from . import utils
from . import polyutils
from .database import db
from .xfm import Transform

def curvature(outfile, subject, smooth=20, **kwargs):
    curvs = []
    for pts, polys in db.get_surf(subject, "fiducial"):
        surf = polyutils.Surface(pts, polys)
        curv = surf.smooth(surf.mean_curvature(), smooth)
        curvs.append(curv)
    np.savez(outfile, left=curvs[0], right=curvs[1])

def distortion(outfile, subject, type='areal', smooth=20):
    """Computes distortion of flatmap relative to fiducial surface. Several different
    types of distortion are available:
    
    'areal': computes the areal distortion for each triangle in the flatmap, defined as the
    log ratio of the area in the fiducial mesh to the area in the flat mesh. Returns
    a per-vertex value that is the average of the neighboring triangles.
    See: http://brainvis.wustl.edu/wiki/index.php/Caret:Operations/Morphing
    
    'metric': computes the linear distortion for each vertex in the flatmap, defined as
    the mean squared difference between distances in the fiducial map and distances in
    the flatmap, for each pair of neighboring vertices. See Fishl, Sereno, and Dale, 1999.
    """
    distortions = []
    for hem in ["lh", "rh"]:
        fidvert, fidtri = db.get_surf(subject, "fiducial", hem)
        flatvert, flattri = db.get_surf(subject, "flat", hem)
        surf = polyutils.Surface(fidvert, fidtri)

        dist = getattr(polyutils.Distortion(flatvert, fidvert, flattri), type)
        smdist = surf.smooth(dist, smooth)
        distortions.append(smdist)

    np.savez(outfile, left=distortions[0], right=distortions[1])

def thickness(outfile, subject):
    pl, pr = db.get_surf(subject, "pia")
    wl, wr = db.get_surf(subject, "wm")
    left = np.sqrt(((pl[0] - wl[0])**2).sum(1))
    right = np.sqrt(((pr[0] - wr[0])**2).sum(1))
    np.savez(outfile, left=left, right=right)

def tissots_indicatrix(outfile, sub, radius=10, spacing=50, maxfails=100): 
    tissots = []
    allcenters = []
    for hem in ["lh", "rh"]:
        fidpts, fidpolys = db.get_surf(sub, "fiducial", hem)
        #G = make_surface_graph(fidtri)
        surf = polyutils.Surface(fidpts, fidpolys)
        nvert = fidpts.shape[0]
        tissot_array = np.zeros((nvert,))

        centers = [np.random.randint(nvert)]
        cdists = [surf.geodesic_distance(centers)]
        while True:
            ## Find possible vertices
            mcdist = np.vstack(cdists).min(0)
            possverts = np.nonzero(mcdist > spacing)[0]
            #possverts = np.nonzero(surf.geodesic_distance(centers) > spacing)[0]
            if not len(possverts):
                break
            ## Pick random vertex
            centervert = possverts[np.random.randint(len(possverts))]
            centers.append(centervert)
            print("Adding vertex %d.." % centervert)
            dists = surf.geodesic_distance([centervert])
            cdists.append(dists)

            ## Find appropriate set of vertices
            selverts = dists < radius
            tissot_array[selverts] = 1

        tissots.append(tissot_array)
        allcenters.append(np.array(centers))

    np.savez(outfile, left=tissots[0], right=tissots[1], centers=allcenters)

def flat_border(outfile, subject):
    flatpts, flatpolys = db.get_surf(subject, "flat", merge=True, nudge=True)
    flatpolyset = set(map(tuple, flatpolys))
    
    fidpts, fidpolys = db.get_surf(subject, "fiducial", merge=True, nudge=True)
    fidpolyset = set(map(tuple, fidpolys))
    fidonlypolys = fidpolyset - flatpolyset
    fidonlypolyverts = np.unique(np.array(list(fidonlypolys)).ravel())
    
    fidonlyverts = np.setdiff1d(fidpolys.ravel(), flatpolys.ravel())
    
    import networkx as nx
    def iter_surfedges(tris):
        for a,b,c in tris:
            yield a,b
            yield b,c
            yield a,c

    def make_surface_graph(tris):
        graph = nx.Graph()
        graph.add_edges_from(iter_surfedges(tris))
        return graph

    bounds = [p for p in polyutils.trace_poly(polyutils.boundary_edges(flatpolys))]
    allbounds = np.hstack(bounds)
    
    g = make_surface_graph(fidonlypolys)
    fog = g.subgraph(fidonlyverts)
    badverts = np.array([v for v,d in fog.degree().iteritems() if d<2])
    g.remove_nodes_from(badverts)
    fog.remove_nodes_from(badverts)
    mwallset = set.union(*(set(g[v]) for v in fog.nodes())) & set(allbounds)
    #cutset = (set(g.nodes()) - mwallset) & set(allbounds)

    mwallbounds = [np.in1d(b, mwallset) for b in bounds]
    changes = [np.nonzero(np.diff(b.astype(float))!=0)[0]+1 for b in mwallbounds]
    
    #splitbounds = [np.split(b, c) for b,c in zip(bounds, changes)]
    splitbounds = []
    for b,c in zip(bounds, changes):
        sb = []
        rb = [b[-1]] + b
        rc = [1] + (c + 1).tolist() + [len(b)]
        for ii in range(len(rc)-1):
            sb.append(rb[rc[ii]-1 : rc[ii+1]])
        splitbounds.append(sb)
    
    ismwall = [[s.mean()>0.5 for s in np.split(mwb, c)] for mwb,c in zip(mwallbounds, changes)]
    
    aspect = (height / (flatpts.max(0) - flatpts.min(0))[1])
    lpts = (flatpts - flatpts.min(0)) * aspect
    rpts = (flatpts - flatpts.min(0)) * aspect
    
    #im = Image.new('RGBA', (int(aspect * (flatpts.max(0) - flatpts.min(0))[0]), height))
    #draw = ImageDraw.Draw(im)

    ismwalls = []
    lines = []
    
    for bnds, mw, pts in zip(splitbounds, ismwall, [lpts, rpts]):
        for pbnd, pmw in zip(bnds, mw):
            #color = {True:(0,0,255,255), False:(255,0,0,255)}[pmw]
            #draw.line(pts[pbnd,:2].ravel().tolist(), fill=color, width=2)
            ismwalls.append(pmw)
            lines.append(pts[pbnd,:2])
    
    np.savez(outfile, lines=lines, ismwalls=ismwalls)

def mni_nl(subject, do=True):
    """Create an automatic alignment of an anatomical image to the MNI standard.

    This function does the following:
    1) Re-orders orientation labels on anatomical images using fslreorient2std (without modifying the existing files)
    2) Calls FLIRT
    3) Calls FNIRT with the transform estimated by FLIRT as the specified
    4) Gets the resulting warp field, samples it at each vertex location, and calculates MNI coordinates.
    5) Saves these coordinates as a surfinfo file in the db.

    Parameters
    ----------
    subject : str
        Subject identifier.
    do : bool
        Actually execute the commands (True), or just print them (False, useful for debugging).
    """

    import nibabel as nib

    from .options import config
    from .dataset import Volume

    fsl_prefix = config.get("basic", "fsl_prefix")
    cache = tempfile.mkdtemp()

    print('anat_to_mni, subject: %s' % subject)
    
    raw_anat = db.get_anat(subject, type='raw').get_filename()
    bet_anat = db.get_anat(subject, type='brainmask').get_filename()
    betmask_anat = db.get_anat(subject, type='brainmask_mask').get_filename()
    anat_dir = os.path.dirname(raw_anat)
    odir = cache

    # stem for the reoriented-into-MNI anatomical images (required by FLIRT/FNIRT)
    reorient_anat = 'reorient_anat'
    reorient_cmd = '{fslpre}fslreorient2std {raw_anat} {adir}/{ra_raw}'.format(fslpre=fsl_prefix,raw_anat=raw_anat, adir=odir, ra_raw=reorient_anat)
    print('Reorienting anatomicals using fslreorient2std, cmd like: \n%s' % reorient_cmd)
    if do and sp.call(reorient_cmd, shell=True) != 0:
        raise IOError('Error calling fslreorient2std on raw anatomical')
    
    reorient_cmd = '{fslpre}fslreorient2std {bet_anat} {adir}/{ra_raw}_brain'.format(fslpre=fsl_prefix,bet_anat=bet_anat, adir=odir, ra_raw=reorient_anat)
    if do and sp.call(reorient_cmd, shell=True) != 0:
        raise IOError('Error calling fslreorient2std on brain-extracted anatomical')

    ra_betmask = reorient_anat + "_brainmask"
    reorient_cmd = '{fslpre}fslreorient2std {bet_anat} {adir}/{ra_betmask}'.format(fslpre=fsl_prefix,bet_anat=betmask_anat, adir=odir, ra_betmask=ra_betmask)
    
    if do and sp.call(reorient_cmd, shell=True) != 0:
        raise IOError('Error calling fslreorient2std on brain-extracted mask')
            
    fsldir = os.environ['FSLDIR']
    standard = '%s/data/standard/MNI152_T1_1mm'%fsldir
    bet_standard = '%s_brain'%standard
    standardmask = '%s_mask_dil'%bet_standard
    cout = 'mni2anat' #stem of the filenames of the transform estimates

    # initial affine anatomical-to-standard registration using FLIRT. required, as the output xfm is used as a start by FNIRT.
    flirt_cmd = '{fslpre}flirt -in {bet_standard} -ref {adir}/{ra_raw}_brain -dof 6 -omat {adir}/{cout}_flirt'
    flirt_cmd = flirt_cmd.format(fslpre=fsl_prefix, ra_raw=reorient_anat, bet_standard=bet_standard, adir=odir, cout=cout)
    print('Running FLIRT to estimate initial affine transform with command:\n%s'%flirt_cmd)
    if do and sp.call(flirt_cmd, shell=True) != 0:
        raise IOError('Error calling FLIRT with command: %s' % flirt_cmd)

    # FNIRT mni-to-anat transform estimation cmd (does not apply any transform, but generates estimate [cout])
    # the MNI152 2mm config is used even though we're referencing 1mm, per this FSL list post:
    # https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=FSL;d14e5a9d.1105
    cmd = '{fslpre}fnirt --in={standard} --ref={ad}/{ra_raw} --refmask={ad}/{refmask} --aff={ad}/{cout}_flirt --cout={ad}/{cout}_fnirt --fout={ad}/{cout}_field --iout={ad}/{cout}_iout --config=T1_2_MNI152_2mm'
    cmd = cmd.format(fslpre=fsl_prefix, ra_raw=reorient_anat, standard=standard, refmask=ra_betmask, ad=odir, anat_dir=anat_dir, cout=cout)
    print('Running FNIRT to estimate transform, using the following command... this can take a while:\n%s'%cmd)
    if do and sp.call(cmd, shell=True) != 0:
        raise IOError('Error calling fnirt with cmd: %s'%cmd)

        pts, polys = db.get_surf(subject,"fiducial",merge="True")

    #print('raw anatomical: %s\nbet anatomical: %s\nflirt cmd:%s\nfnirt cmd: %s\npts: %s' % (raw_anat,bet_anat,flirt_cmd,cmd,pts))

    # take the reoriented anatomical, get its affine coord transform, invert this, and save it
    reo_xfmnm = 'reorient_inv'
    # need to change this line, as the reoriented anatomical is not in the db but in /tmp now
    # re_anat = db.get_anat(subject,reorient_anat)
    reo_anat_fn = '{odir}/{reorient_anat}.nii.gz'.format(odir=odir,reorient_anat=reorient_anat)
    # print(reo_anat_fn)
    # since the reoriented anatomicals aren't stored in the db anymore, db.get_anat() will not work (?)
    re_anat = nib.load(reo_anat_fn)
    reo_xfm = Transform(np.linalg.inv(re_anat.get_affine()),re_anat)
    reo_xfm.save(subject,reo_xfmnm,"coord")

    # get the reoriented anatomical's qform and its inverse, they will be needed later
    aqf = re_anat.get_qform()
    aqfinv = np.linalg.inv(aqf)

    # load the warp field data as a volume
    # since it's not in the db anymore but in /tmp instead of:
    # warp = db.get_anat(subject,'%s_field'%cout)
    # it's this:
    warp_fn = '{ad}/{cout}_field.nii.gz'.format(ad=odir,cout=cout)
    # print warp_fn
    warp = nib.load(warp_fn)
    wd = warp.get_data()
    # need in (t,z,y,x) order
    wd = np.swapaxes(wd,0,3) # x <--> t
    wd = np.swapaxes(wd,1,2) # y <--> z
    wv = Volume(wd,subject,reo_xfmnm)

    # now do the mapping! this gets the warp field values at the corresponding points
    # (uses fiducial surface by default)
    warpvd = wv.map(projection="lanczos")

    # reshape into something sensible
    warpverts_L = [vs for vs in np.swapaxes(warpvd.left,0,1)]
    warpverts_R = [vs for vs in np.swapaxes(warpvd.right,0,1)]
    warpverts_ordered = np.concatenate((warpverts_L, warpverts_R))

    # append 1s for matrix multiplication (coordinate transformation)
    o = np.ones((len(pts),1))
    pad_pts = np.append(pts, o, axis=1)

    # print pts, len(pts), len(pts[0]), warpverts_ordered, len(warpverts_ordered), pad_pts, len(pad_pts), pad_pts[0]

    # transform vertex coords from mm to vox using the anat's qform
    voxcoords = [aqfinv.dot(padpt) for padpt in pad_pts]
    # add the offsets specified in the warp at those locations (ignoring the 1s here)
    mnivoxcoords = [voxcoords[n][:-1] + warpverts_ordered[n] for n in range(len(voxcoords))]
    # re-pad for matrix multiplication
    pad_mnivox = np.append(mnivoxcoords, o, axis=1)

    # multiply by the standard's qform to recover mm coords
    std = nib.load('%s.nii.gz'%standard)
    stdqf = std.get_qform()
    mni_coords = np.array([stdqf.dot(padmni)[:-1] for padmni in pad_mnivox])

    # some debug output
    # print pts, mni_coords
    # print pts[0], mni_coords[0]
    # print len(pts), len(mni_coords)
    # print type(pts), type(pts[0][0]), type(mni_coords)

    # now split mni_coords into left and right arrays for saving
    nverts_L = len(warpverts_L)
    #print nverts_L
    left = mni_coords[:nverts_L]
    right = mni_coords[nverts_L:]
    #print len(left), len(right)

    mni_surfinfo_fn = db.get_paths(subject)['surfinfo'].format(type='mnicoords',opts='')
    print('Saving mni coordinates as a surfinfo...')
    np.savez(mni_surfinfo_fn,leftpts=left,rightpts=right)

