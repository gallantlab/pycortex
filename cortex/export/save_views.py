import os
import time
from typing import Any, TypedDict, Union

import cortex

from ..dataset import Dataview

file_pattern = "{base}_{view}_{surface}.png"

ViewParams = TypedDict('ViewParams', {
    "camera.azimuth": float,
    "camera.altitude": float,
    "camera.target": list[float],
    "surface.{subject}.unfold": float,
    "surface.{subject}.pivot": float,
    "surface.{subject}.shift": float,
    "surface.{subject}.specularity": float,
}, total=False)

def save_3d_views(
    volume: Dataview,
    base_name: str="fig",
    list_angles: list[Union[str, tuple[str, ViewParams]]]=["lateral_pivot"],
    list_surfaces: list[Union[str, ViewParams]]=["inflated"],
    viewer_params: dict[str, Any]=dict(labels_visible=[], overlays_visible=["rois"]),
    interpolation: str="nearest",
    layers: int=1,
    size: tuple[int, int]=(1024 * 4, 768 * 4),
    trim: bool=True,
    sleep: float=10,
    headless: bool=False,
) -> list[str]:
    """Saves 3D views of `volume` under multiple specifications.

    Needs to be run on a system with a display (will launch webgl viewer).
    The best way to get the expected results is to keep the webgl viewer
    visible during the process.

    Parameters
    ----------
    volume: pycortex.Volume or pycortex.Vertex object
        Data to be displayed.

    base_name: str
        Base name for images.

    list_angles: list of (str or dict)
        Views to be used. Should be of length one, or of the same length as
        `list_surfaces`. Choices are:
            'left', 'right', 'front', 'back', 'top', 'bottom', 'flatmap',
            'medial_pivot', 'lateral_pivot', 'bottom_pivot',
            or tuple of (view_name, custom dictionary of parameters).
            See `angle_view_params` in this file for parameter dict examples.

    list_surfaces: list of (str or dict)
        Surfaces to be used. Should be of length one, or of the same length as
        `list_angles`. Choices are:
            'inflated', 'flatmap', 'fiducial', 'inflated_cut',
            or a custom dictionary of parameters.

    viewer_params: dict
        Parameters passed to the viewer.

    interpolation: str
        Interpolation used to visualize the data. Possible choices are "nearest",
        "trilinear". (Default: "nearest").

    layers: int
        Number of layers between the white and pial surfaces to average prior to
        plotting the data. (Default: 1).

    size: tuple of int
        Size of produced image (before trimming).

    trim: bool
        Whether to trim the white borders of the image.

    sleep: float > 0
        Time in seconds, to let the viewer open. Ignored when ``headless=True``
        because the Playwright browser connects synchronously.

    headless: bool
        If True, render using a headless Chromium browser via Playwright instead
        of requiring the user to manually open a browser window.  This allows
        the function to run fully autonomously without any user interaction.
        Requires ``playwright`` to be installed (``pip install playwright``) and
        Chromium to be available (``playwright install chromium``).
        Software WebGL (SwiftShader) is used, so no GPU or display server is
        needed.  (Default: False)

    Returns
    -------
    file_names: list of str
        Image paths.
    """
    msg = "list_angles and list_surfaces should have the same length."
    assert len(list_angles) == len(list_surfaces), msg

    # Create viewer
    if headless:
        from cortex.export.headless import headless_viewer as _headless_viewer
        _headless_ctx = _headless_viewer(volume, viewer_params)
        handle = _headless_ctx.__enter__()
    else:
        _headless_ctx = None
        handle = cortex.webshow(volume, **viewer_params)
        # Wait for the viewer to be loaded

    time.sleep(sleep)

    # Add interpolation and layers params only if we have a volume
    if isinstance(volume, (cortex.Volume, cortex.Volume2D, cortex.VolumeRGB)):
        interpolation_params = {
            "surface.{subject}.sampler": interpolation,
            "surface.{subject}.layers": layers
        }
    else:
        interpolation_params = dict()

    has_flatmap = hasattr(getattr(cortex.db, volume.subject).surfaces, "flat")
    file_names: list[str] = []
    for view, surface in zip(list_angles, list_surfaces):
        if isinstance(view, str):
            if view == "flatmap" or surface == "flatmap":
                # force flatmap correspondence
                view = surface = "flatmap"
            view_params = angle_view_params[view]
            view_name = view
        else:
            view_name, view_params = view

        if isinstance(surface, str):
            surface_params = unfold_view_params[surface].copy()
            # Fix unfold parameters if this subject doesn't have a flatmap
            # Without a flatmap, the inflated surf corresponds to an unfold value of 1
            # With a flatmap, the inflated surf corresponds to an unfold value of 0.5
            if not has_flatmap:
                surface_params["surface.{subject}.unfold"] = min(
                    surface_params["surface.{subject}.unfold"] * 2, 1
                )
        else:
            surface_params = surface

        # Combine view parameters
        this_view_params = default_view_params.copy()
        this_view_params.update(interpolation_params)
        this_view_params.update(view_params)
        this_view_params.update(surface_params)
        print(this_view_params)

        # apply params
        handle._set_view(**this_view_params)

        # wait for the view to have changed
        # TODO: what is going on here?
        for _ in range(100):
            all_ready = True
            for k, v in this_view_params.items():
                k_resolved = k.format(subject=volume.subject) if "{subject}" in k else k
                current = handle.ui.get(k_resolved)[0]
                if current != v:
                    print("waiting for", k_resolved, current, "->", v)
                    all_ready = False
            if all_ready:
                break
            time.sleep(0.1)
        time.sleep(0.1)

        # Save image, store file_name
        file_name = file_pattern.format(base=base_name, view=view_name, surface=surface)
        file_names.append(file_name)
        handle.getImage(file_name, size)

        # Wait for browser to dump file, before applying new view parameters
        for _wait in range(200):
            if os.path.exists(file_name):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(
                # TODO: what is this f-string syntax?
                f"Image {file_name!r} was not written within 20 seconds. "
                "The browser may have failed to POST the screenshot."
            )
        time.sleep(1)

        # Trim white edges
        if trim:
            try:
                import subprocess

                subprocess.call(["convert", "-trim", file_name, file_name]) # TODO: check return code
            except Exception as e:
                print(str(e))
                pass

    # Try to close the window
    if _headless_ctx is not None:
        # headless mode: delegate teardown to the context manager
        try:
            _headless_ctx.__exit__(None, None, None)
        except Exception as e:
            # TODO: proper exception handling
            print(str(e))
            print("Could not close headless viewer.")
    else:
        try:
            handle.close()
            handle.server.stop()
        except Exception as e:
            print(str(e))
            print("Could not close viewer.")

    return file_names

default_view_params: ViewParams = {
    "camera.azimuth": 45,
    "camera.altitude": 75,
    "camera.target": [0, 0, 0],
    "surface.{subject}.unfold": 0,
    "surface.{subject}.pivot": 0,
    "surface.{subject}.shift": 0,
    "surface.{subject}.specularity": 0,
}

angle_view_params: dict[str, ViewParams] = {
    "left": {"camera.azimuth": 90, "camera.altitude": 90,},
    "right": {"camera.azimuth": 270, "camera.altitude": 90,},
    "left_atl": {"camera.azimuth": 65, "camera.altitude": 100,},
    "right_atl": {"camera.azimuth": 300, "camera.altitude": 100,},
    "front": {"camera.azimuth": 0, "camera.altitude": 90,},
    "back": {"camera.azimuth": 180, "camera.altitude": 90,},
    "top": {"camera.azimuth": 180, "camera.altitude": 0,},
    "bottom": {"camera.azimuth": 0, "camera.altitude": 180,},
    "flatmap": {
        "camera.azimuth": 180,
        "camera.altitude": 0,
        "surface.{subject}.pivot": 180,
        "surface.{subject}.shift": 0,
    },
    "medial_pivot": {
        "camera.azimuth": 0,
        "camera.altitude": 90,
        "surface.{subject}.pivot": 180,
        "surface.{subject}.shift": 10,
    },
    "lateral_pivot": {
        "camera.azimuth": 180,
        "camera.altitude": 90,
        "surface.{subject}.pivot": 180,
        "surface.{subject}.shift": 10,
    },
    "bottom_pivot": {
        "camera.azimuth": 180,
        "camera.altitude": 180,
        "camera.target": [0, -100, 0],
        "surface.{subject}.pivot": 180,
        "surface.{subject}.shift": 10,
    },
    "top_pivot": {
        "camera.azimuth": 180,
        "camera.altitude": 0,
        "camera.target": [0, -100, 0],
        "surface.{subject}.pivot": 180,
        "surface.{subject}.shift": 10,
    },
}

unfold_view_params: dict[str, ViewParams] = {
    "fiducial": {"surface.{subject}.unfold": 0,},
    "inflated_less": {"surface.{subject}.unfold": 0.25,},
    "inflated": {"surface.{subject}.unfold": 0.5,},
    "inflated_cut": {"surface.{subject}.unfold": 0.501,},
    "flatmap": {"surface.{subject}.unfold": 1,},
}
