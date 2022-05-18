import os
import errno
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from .save_views import save_3d_views
from ._default_params import (
    params_inflatedless_lateral_medial_ventral,
    params_occipital_triple_view,
    params_flatmap_lateral_medial,
    params_inflated_dorsal_lateral_medial_ventral,
    params_flatmap_inflated_lateral_medial_ventral
)


def plot_panels(
    volume,
    panels,
    figsize=(16, 9),
    windowsize=(1600 * 4, 900 * 4),
    save_name=None,
    sleep=10,
    viewer_params=dict(labels_visible=[], overlays_visible=["rois"]),
    interpolation="nearest",
    layers=1,
):
    """Plot on the same figure a number of views, as defined by a list of panel

    Parameters
    ----------
    volume : cortex.Volume
        The data to plot.

    panels : list of dict
        List of parameters for each panel. An example of panel is:
            {
                'extent': [0.000, 0.000, 0.300, 0.300],
                'view': {
                    'hemisphere': 'left',
                    'angle': 'lateral_pivot',
                    'surface': 'inflated',
                }
            }
        The `extent` and `zoom` entries are ordered as
        [left, bottom, width, height] with values in [0, 1].

    figsize : tuple of float
        Size of the figure.

    windowsize : tuple of float
        Size of the browser window. Larger values provide higher resolution,
        but they might fail if the screen size is not large enough (this is
        only a working hypothesis). If this function fails, try reducing the 
        windowsize.

    save_name : str or None
        Name of the file where the figure is saved. None to not save.
        Can end with different extensions, such as '.png' of '.pdf'.

    sleep: float > 0
        Time in seconds, to let the viewer open.

    viewer_params: dict
        Parameters passed to the viewer.

    interpolation: str
        Interpolation used to visualize the data. Possible choices are "nearest",
        "trilinear". (Default: "nearest").

    layers: int
        Number of layers between the white and pial surfaces to average prior to
        plotting the data. (Default: 1).

    Returns
    -------
    fig : matplotlib.Figure
        Created figure. Can be used for instance for custom save functions.

    Example
    -------
    >>> from cortex.export import plot_panels, params_flatmap_lateral_medial
    >>> plot_panels(volume, **params_flatmap_lateral_medial)

    """
    # list of couple of angles and surfaces
    angles_and_surfaces = [
        (panel["view"]["angle"], panel["view"]["surface"]) for panel in panels
    ]
    # remove redundant couples, e.g. left and right
    angles_and_surfaces = list(set(angles_and_surfaces))
    list_angles, list_surfaces = list(zip(*angles_and_surfaces))

    # create all images
    temp_dir = tempfile.mkdtemp()
    base_name = os.path.join(temp_dir, "fig")
    filenames = save_3d_views(
        volume,
        base_name,
        list_angles=list_angles,
        list_surfaces=list_surfaces,
        trim=True,
        size=windowsize,
        sleep=sleep,
        viewer_params=viewer_params,
        interpolation=interpolation,
        layers=layers,
    )

    fig = plt.figure(figsize=figsize)
    for panel in panels:

        # load image
        angle_and_surface = (panel["view"]["angle"], panel["view"]["surface"])
        index = angles_and_surfaces.index(angle_and_surface)
        image = plt.imread(filenames[index])

        # chose hemisphere
        if "hemisphere" in panel["view"]:
            left, right = np.split(image, [image.shape[1] // 2], axis=1)

            if panel["view"]["angle"] == "medial_pivot":
                # inverted view, we need to swap left and right
                left, right = right, left

            if panel["view"]["hemisphere"] == "left":
                image = left
            else:
                image = right

        # trim white borders
        image = image[image.sum(axis=1).sum(axis=1) > 0]
        image = image[:, image.sum(axis=0).sum(axis=1) > 0]

        # zoom
        if "zoom" in panel["view"]:
            left, bottom, width, height = panel["view"]["zoom"]
            left = int(left * image.shape[1])
            width = int(width * image.shape[1])
            bottom = int(bottom * image.shape[0])
            height = int(height * image.shape[0])
            image = image[bottom : bottom + height]
            image = image[:, left : left + width]

        # add ax and image
        ax = plt.axes(panel["extent"])
        ax.axis("off")
        ax.imshow(image)

    # note that you might get a slightly different layout with `plt.show()`
    # since it might use a different backend
    if save_name is not None:
        fig.savefig(save_name, bbox_inches="tight", dpi=100)

    # delete temporary directory
    try:
        shutil.rmtree(temp_dir)
    except OSError as e:
        # reraise if the directory has not already been deleted
        if e.errno != errno.ENOENT:
            raise

    return fig
