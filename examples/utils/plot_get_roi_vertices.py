"""
=======================
Get Vertices for an ROI
=======================

In this example we show how to get the vertices that are inside an ROI that was
defined in the SVG ROI file (see :doc:`/rois.rst`).

"""
import cortex

# get vertices for fusiform face area FFA in subject S1
roi_verts = cortex.get_roi_verts('S1', 'FFA')

# roi_verts is a dictionary (in this case with only one entry)
ffa_verts = roi_verts['FFA']

# this includes indices from both hemispheres
# let's create an empty Vertex object and fill FFA

ffa_map = cortex.Vertex.empty('S1', cmap='plasma')
ffa_map.data[ffa_verts] = 1.0

cortex.quickshow(ffa_map)
