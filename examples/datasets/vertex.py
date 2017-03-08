"""
================
Plot Vertex Data
================

This plots example vertex data onto an example subject, S1

Instead of the random test data, you can replace this with any array that is
the length of all of the vertices in the subject
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

