"""
==================
Import fmriprep output
==================

Recently, many people have start to use fmriprep as a complete preprocessing
workflow of anatomical and functional data. Pycortex has a convenience function to import
the output of this workflow.

This example is based on the fmriprep 1.0.15 output of openfmri ds000164 that can be found
on openneuro.org:
https://openneuro.org/datasets/ds000164/versions/00001
"""

import cortex
from cortex import fmriprep

fmriprep.import_subj('001', '/derivatives/ds000164')

# We can use the identity transform to visualize the T1-weighted image
t1w = '/derivatives/ds000164/fmriprep/sub-001/anat/sub-001_T1w_preproc.nii.gz'

# Now we can make a volume
t1w_volume = cortex.Volume(t1w, '001', 'identity')

# And show the result.
ds = cortex.Dataset(t1w=t1w_volume)
cortex.webgl.show(ds)
