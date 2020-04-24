import os
import time

import cortex

file_pattern = "{base}_{view}_{surface}.png"


def save_3d_views(volume, base_name='fig', list_angles=['lateral_pivot'],
                  list_surfaces=['inflated'],
                  viewer_params=dict(labels_visible=[],
                                     overlays_visible=['rois']),
                  size=(1024 * 4, 768 * 4), trim=True, sleep=10):
    """Saves 3D views of `volume` under multiple specifications.

    Needs to be run on a system with a display (will launch webgl viewer).
    The best way to get the expected results is to keep the webgl viewer
    visible during the process.

    Parameters
    ----------
    volume: pycortex.Volume
        Data to be displayed.

    base_name: str
        Base name for images.

    list_angles: list of (str or dict)
        Views to be used. Should be of length one, or of the same length as
        `list_surfaces`. Choices are:
            'left', 'right', 'front', 'back', 'top', 'bottom', 'flatmap',
            'medial_pivot', 'lateral_pivot', 'bottom_pivot',
            or a custom dictionary of parameters.

    list_surfaces: list of (str or dict)
        Surfaces to be used. Should be of length one, or of the same length as
        `list_angles`. Choices are:
            'inflated', 'flatmap', 'fiducial', 'inflated_cut',
            or a custom dictionary of parameters.

    viewer_params: dict
        Parameters passed to the viewer.

    size: tuple of int
        Size of produced image (before trimming).

    trim: bool
        Whether to trim the white borders of the image.

    sleep: float > 0
        Time in seconds, to let the viewer open.

    Returns
    -------
    file_names: list of str
        Image paths.
    """
    msg = "list_angles and list_surfaces should have the same length."
    assert len(list_angles) == len(list_surfaces), msg

    # Create viewer
    handle = cortex.webshow(volume, **viewer_params)
    # Wait for the viewer to be loaded
    time.sleep(sleep)

    file_names = []
    for view, surface in zip(list_angles, list_surfaces):
        if isinstance(view, str):
            if view == 'flatmap' or surface == 'flatmap':
                # force flatmap correspondence
                view = surface = 'flatmap'
            view_params = angle_view_params[view]
        else:
            view_params = view
        if isinstance(surface, str):
            surface_params = unfold_view_params[surface]
        else:
            surface_params = surface

        # Combine view parameters
        this_view_params = default_view_params.copy()
        this_view_params.update(view_params)
        this_view_params.update(surface_params)
        print(this_view_params)

        # apply params
        handle._set_view(**this_view_params)

        # wait for the view to have changed
        for _ in range(100):
            for k, v in this_view_params.items():
                k = k.format(subject=volume.subject) if '{subject}' in k else k
                if handle.ui.get(k)[0] != v:
                    print('waiting for', k, handle.ui.get(k)[0], '->', v)
                    time.sleep(0.1)
                    continue
            break
        time.sleep(0.1)

        # Save image, store file_name
        file_name = file_pattern.format(base=base_name, view=view,
                                        surface=surface)
        file_names.append(file_name)
        handle.getImage(file_name, size)

        # Wait for browser to dump file, before applying new view parameters
        while not os.path.exists(file_name):
            pass
        time.sleep(1)

        # Trim white edges
        if trim:
            try:
                import subprocess
                subprocess.call(["convert", "-trim", file_name, file_name])
            except Exception as e:
                print(str(e))
                pass

    # Try to close the window
    try:
        handle.close()
        handle.server.stop()
    except Exception as e:
        print(str(e))
        print('Could not close viewer.')

    return file_names


default_view_params = {
    'camera.azimuth': 45,
    'camera.altitude': 75,
    'camera.target': [0, 0, 0],
    'surface.{subject}.unfold': 0,
    'surface.{subject}.pivot': 0,
    'surface.{subject}.shift': 0,
    'surface.{subject}.specularity': 0,
}

angle_view_params = {
    'left': {
        'camera.azimuth': 90,
        'camera.altitude': 90,
    },
    'right': {
        'camera.azimuth': 270,
        'camera.altitude': 90,
    },
    'front': {
        'camera.azimuth': 0,
        'camera.altitude': 90,
    },
    'back': {
        'camera.azimuth': 180,
        'camera.altitude': 90,
    },
    'top': {
        'camera.azimuth': 180,
        'camera.altitude': 0,
    },
    'bottom': {
        'camera.azimuth': 0,
        'camera.altitude': 180,
    },
    'flatmap': {
        'camera.azimuth': 180,
        'camera.altitude': 0,
        'surface.{subject}.pivot': 180,
        'surface.{subject}.shift': 0,
    },
    'medial_pivot': {
        'camera.azimuth': 0,
        'camera.altitude': 90,
        'surface.{subject}.pivot': 180,
        'surface.{subject}.shift': 10,
    },
    'lateral_pivot': {
        'camera.azimuth': 180,
        'camera.altitude': 90,
        'surface.{subject}.pivot': 180,
        'surface.{subject}.shift': 10,
    },
    'bottom_pivot': {
        'camera.azimuth': 180,
        'camera.altitude': 180,
        'camera.target': [0, -100, 0],
        'surface.{subject}.pivot': 180,
        'surface.{subject}.shift': 10,
    },
}

unfold_view_params = {
    'fiducial': {
        'surface.{subject}.unfold': 0,
    },
    'inflated_less': {
        'surface.{subject}.unfold': 0.25,
    },
    'inflated': {
        'surface.{subject}.unfold': 0.5,
    },
    'inflated_cut': {
        'surface.{subject}.unfold': 0.501,
    },
    'flatmap': {
        'surface.{subject}.unfold': 1,
    },
}
