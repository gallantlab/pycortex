"""
================
Plot Vertex Data
================

This plots example vertex data onto an example subject, S1, onto a flatmap
using quickflat. In order for this to run, you have to have a flatmap for
this subject in the pycortex filestore.

The cortex.Vertex object is instantiated with a numpy array of the same size
as the total number of vertices in that subject's flatmap. Each pixel is 
colored according to the value given for the nearest vertex in the flatmap.

Instead of the random test data, you can replace this with any array that is
the length of all of the vertices in the subject.

Additionally, if you create a Vertex object using only the number of vertices
that exists in the left hemisphere of the brain, the right hemisphere is 
filled in with zeros.
"""

import cortex
import cortex.polyutils
import numpy as np
import matplotlib.pyplot as plt

subject = 'S1'

surfs = [cortex.polyutils.Surface(*d) 
         for d in cortex.db.get_surf(subject, "fiducial")]

num_verts = surfs[0].pts.shape[0] + surfs[1].pts.shape[0]
test_data = np.random.randn(num_verts)

dv = cortex.Vertex(test_data, subject)
cortex.quickshow(dv)
plt.show()

# Now we can plot just the left hemisphere data
numl = surfs[0].pts.shape[0]
dv_left = cortex.Vertex(test_data[:numl], subject)
cortex.quickshow(dv_left)
plt.show()

