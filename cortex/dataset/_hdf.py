"""Low-level HDF5 and hashing helpers for :mod:`cortex.dataset`.

Split out of ``braindata.py`` so that module can become a pure back-compatibility
shim without creating an import cycle: ``views.py`` needs these helpers, and
``braindata.py`` needs to import the classes defined in ``views.py``.
"""

from __future__ import annotations

import hashlib
import os
from typing import Union, cast

import h5py
import numpy as np
import numpy.typing as npt

from ..database import db


def _hash(array: npt.ArrayLike) -> str:
    """A simple numpy hash function"""
    array = np.asarray(array)
    return hashlib.sha1(array.tobytes()).hexdigest()


def _hdf_write(
    h5: Union[h5py.File, h5py.Group],
    data: npt.NDArray,
    name: str = "data",
    group: str = "/data",
) -> h5py.Dataset:
    try:
        node = h5.require_dataset(
            "%s/%s" % (group, name), data.shape, data.dtype, exact=True
        )
    except TypeError:
        del h5[group][name]
        node = h5.create_dataset(
            "%s/%s" % (group, name), data.shape, data.dtype, exact=True
        )

    node[:] = data
    return node


def _find_mask(
    nvox: int, subject: str, xfmname: str
) -> tuple[str, npt.NDArray[np.bool_]]:
    import glob
    import re

    import nibabel

    files = db.get_paths(subject)["masks"].format(xfmname=xfmname, type="*")
    for fname in glob.glob(files):
        nib = cast(nibabel.Nifti1Image, nibabel.load(fname))
        mask = cast(npt.NDArray[np.bool_], nib.get_fdata().T != 0)
        if nvox == np.sum(mask):
            fname = os.path.split(fname)[1]
            name = re.compile(r"mask_(.+).nii.gz").search(fname)
            assert name is not None, (
                f"Mask filename {fname} does not match expected format"
            )
            return name.group(1), mask

    raise ValueError("Cannot find a valid mask")
