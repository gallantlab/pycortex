"""
====================
Multi-panels figures
====================

The function `cortex.export.plot_panels` plots a number of 3d views of a given
volume, in the same matplotlib figure. It does that by saving a temporary image
for each view, and then aggregating them in the same figure.

The function needs to be run on a system with a display, since it will launch
a webgl viewer. The best way to get the expected results is to keep the webgl
viewer visible during the process.

The selection of views and the aggregation is controled by a list of "panels".
Examples of panels can be imported with:

    from cortex.export import params_flatmap_lateral_medial
    from cortex.export import params_occipital_triple_view

"""
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

import cortex

subject = 'S1'

###############################################################################
# create some artificial data

shape = cortex.db.get_xfm(subject, 'identity').shape
data = np.arange(np.product(shape)).reshape(shape)
volume = cortex.Volume(data, subject=subject, xfmname='identity')

###############################################################################
# Show examples of multi-panels figures

params = cortex.export.params_flatmap_lateral_medial
cortex.export.plot_panels(volume, **params)
plt.show()

params = cortex.export.params_occipital_triple_view
cortex.export.plot_panels(volume, **params)
plt.show()

###############################################################################
# List all predefined angles

base_name = os.path.join(tempfile.mkdtemp(), 'fig')
list_angles = list(cortex.export.save_views.angle_view_params.keys())

filenames = cortex.export.save_3d_views(
    volume, base_name=base_name, list_angles=list_angles,
    list_surfaces=['inflated'] * len(list_angles))

for filename, angle in zip(filenames, list_angles):
    plt.imshow(plt.imread(filename))
    plt.axis('off')
    plt.title(angle)
    plt.show()

###############################################################################
# List all predefined surfaces

base_name = os.path.join(tempfile.mkdtemp(), 'fig')
list_surfaces = list(cortex.export.save_views.unfold_view_params.keys())

filenames = cortex.export.save_3d_views(
    volume, base_name=base_name,
    list_angles=['lateral_pivot'] * len(list_surfaces),
    list_surfaces=list_surfaces)

for filename, surface in zip(filenames, list_surfaces):
    plt.imshow(plt.imread(filename))
    plt.axis('off')
    plt.title(surface)
    plt.show()
