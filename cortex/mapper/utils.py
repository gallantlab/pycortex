import cortex
import numpy as np
import scipy


def nanproject(data, mapper, reweigh=True):
    """Project data using the passed mapper while dealing with NaNs.

    Parameters
    ----------
    data : array-like (n_voxels,) or (n_samples, n_voxels)
        The data to be projected.
    mapper : sparse matrix (n_vertices, n_voxels)
    reweigh : bool
        Whether to reweigh the mapper after dealing with NaNs. (Mostly for debugging,
        this should be left to True).

    Returns
    -------
    data_projected : array-like (n_vertices, ) or (n_samples, n_vertices)
        The projected data.
    """
    is_1d = False
    if data.ndim == 1:
        data = np.atleast_2d(data)
        is_1d = True
    if data.ndim > 2:
        raise ValueError("Only one-dimensional or two-dimensional data are allowed")
    # First zero-out nans
    good = ~(np.any(np.isnan(data), axis=0))
    n_voxels = data.shape[1]
    # make diagonal sparse matrix with mask
    good_sparse = scipy.sparse.csr_matrix(
        (good.astype(float), (np.arange(n_voxels), np.arange(n_voxels)))
    )
    # zero-out voxels with nans
    good_mapper = mapper.dot(good_sparse)
    # change data in rows
    if reweigh:
        # now convert to lil to reweigh everything
        good_mapper = good_mapper.tolil()
        # take only rows with data to be used
        rows_to_change = np.where(good_mapper.sum(1) > 0.0)[0]
        for row in rows_to_change:
            sum_row = sum(good_mapper.data[row])
            good_mapper.data[row] = [dt / sum_row for dt in good_mapper.data[row]]
        # convert back to csr
        good_mapper = good_mapper.tocsr()
    # project -- mapper is (n_vertices, n_voxels), data is (n_samples, n_voxels)
    data_projected = good_mapper.dot(data.T).T
    # set vertices receiving only nans to nan
    bad = (~good).astype(float)
    bad_vertex = np.abs(mapper.dot(bad) - 1.0) < 1e-09
    data_projected[:, bad_vertex] = np.nan
    if is_1d:
        data_projected = data_projected[0]
    return data_projected


def vol2surf(
    data,
    subject,
    xfm_name,
    target_surface="native",
    mask_name="thick",
    mapper="line_nearest",
    subject_freesurfer=None,
):
    """Project a subject's volumetric data to a target surface.

    Parameters
    ----------
    data : array (n_voxels,) or (n_samples, n_voxels)
        The flattened volumetric data.
    subject : str
        Subject name.
    xfm_name : str
        Transform name.
    mask_name : str
        Mask to use for the projection. Default is "thick". It should match the
        `n_voxels`.
    target_surface : str
        Surface to project the data to. Default is "native", corresponding to the 
        participant's surface. Alternatives are "fsaverage", "fsaverage6", "fsaverage5",
        or other freesurfer participant codes.
    mapper : str
        Type of mapper to go from volume to the native surface of the subject.
        Just use `line_nearest` if in doubt.
    subject_freesurfer : str or None
        Freesurfer's subject name. If None, it will be the same as `subject`.

    Returns
    -------
    data_projected : array (n_vertices,) or (n_samples, n_vertices)
        The projected data.

    Notes
    -----
    This function averages only non-NaN values. It should be equivalent to nanmean=True
    in pycortex's quickflat.
    """
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1
    else:
        raise ValueError(
            "This function works only with 1-dimensional or 2-dimensional arrays."
        )
    if subject_freesurfer is None:
        subject_freesurfer = subject
    # Get pycortex's mapper to go from volume to fsnative
    voxel2fsnative = cortex.get_mapper(subject, xfm_name, mapper).masks
    mask = cortex.db.get_mask(subject, xfm_name, type=mask_name).ravel()
    assert mask.sum() == data.shape[axis]
    # Select only voxels in the thick mask
    voxel2fsnative = [vfs[:, mask] for vfs in voxel2fsnative]
    if target_surface == "native":
        data_projected = nanproject(data, scipy.sparse.vstack(voxel2fsnative))
    else:
        fsnative2fsaverage = cortex.db.get_mri_surf2surf_matrix(
            subject, "fiducial", fs_subj=subject_freesurfer, target_subj=target_surface
        )
        # Compute projection from volume to fsaverage by combining
        # voxel2fsnative -> fsnative2fsaverage
        voxel2fsaverage = scipy.sparse.vstack(
            [m1.dot(m2) for m1, m2 in zip(fsnative2fsaverage, voxel2fsnative)]
        )
        # Project data
        data_projected = nanproject(data, voxel2fsaverage)
    return data_projected
