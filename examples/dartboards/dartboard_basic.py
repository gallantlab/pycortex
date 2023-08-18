# Basic dartboard pair
import cortex

subject = 'S1'
# Get curvature information for this subject
# Note that this is just an easy example of vertex data; 
# Vertex data can always be created by creating a Volume
# and mapping it to vertices, like this: 
# vol = cx.Volume(<inputs>)
# vx = vol.map()
vx = cortex.db.get_surfinfo(subject, type='curvature')

center = 'M1H'

anchors = [('S1H', 'centroid'),
           ('M1F', 'centroid'),
           ('FEF', 'centroid'),
           ('M1M', 'centroid'),
           ]

out_m1h = cortex.dartboards.get_dartboard_data(vx, center, anchors)
