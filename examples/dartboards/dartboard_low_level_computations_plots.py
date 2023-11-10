# This demo shows the steps of computation of dartboard plots (radial bin averaging and breakdown of plots with low-level functions)
import cortex
import numpy as np

subject = 'S1'
# Get curvature information for this subject
# Note that this is just an easy example of vertex data;
# Vertex data can always be created by creating a Volume
# and mapping it to vertices, like this:
# vol = cx.Volume(<inputs>)
# vx = vol.map()
vx = cortex.db.get_surfinfo(subject, type='curvature')

# Create a dartboard centered on M1H (primary motor cortex hand representation),
# anchored to FEF (frontal eye fields above) anterior,
# M1F (primary motor cortex foot representation) superior,
# S1H (sensorymotor cortex hand representation) posterior,
# and M1M (primary motor cortex mouth representation) inferior.
# This should show the central sulcus going through the radial grid.
dartboard_spec = dict(
    center='M1H',
    anchors=[('FEF', 'centroid'),
             ('M1F', 'centroid'),
             ('S1H', 'centroid'),
             ('M1M', 'centroid'),
             ],
    n_angles=16,
    n_eccentricities=8,
    max_radii=(50, 50),
)

# define vertex-averaging function
def mean_nonan(x, axis=None, threshold=0.8):
    if np.mean(np.isnan(x), axis=axis) > threshold:
        return np.nan
    else:
        return np.nanmean(x)

# Get vertex-wise masks for each dartboard bin (`masks`) and masked data
masks, data = cortex.dartboards.get_dartboard_data(vx, **dartboard_spec)
