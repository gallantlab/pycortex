import contextlib
import os
import time
from typing import Any, Mapping, Optional, Sequence, TypedDict, Union

import cortex

from ..dataset import Dataview

file_pattern = "{base}_{view}_{surface}.png"

ViewParams = TypedDict(
    "ViewParams",
    {
        "camera.azimuth": float,
        "camera.altitude": float,
        "camera.target": list[float],
        "surface.{subject}.unfold": float,
        "surface.{subject}.pivot": float,
        "surface.{subject}.shift": float,
        "surface.{subject}.specularity": float,
    },
    total=False,
)


def save_3d_views(
    volume: Dataview,
    base_name: str = "fig",
    list_angles: Sequence[Union[str, tuple[str, ViewParams]]] = ["lateral_pivot"],
    list_surfaces: Sequence[Union[str, ViewParams]] = ["inflated"],
    viewer_params: Mapping[str, Any] = dict(
        labels_visible=[], overlays_visible=["rois"]
    ),
    interpolation: str = "nearest",
    layers: int = 1,
    size: tuple[int, int] = (1024 * 4, 768 * 4),
    trim: bool = True,
    sleep: float = 10,
    headless: bool = False,
    contour_overlay: Optional[Union[str, Dataview]] = None,
    contour_mode: str = "contours+fill",
) -> list[str]:
    """Saves 3D views of `volume` under multiple specifications.

    By default (``headless=False``), a webgl viewer is launched and a display
    server is required.  With ``headless=True``, a headless Chromium browser
    is used instead, so no display server or GPU is needed.

    Parameters
    ----------
    volume : pycortex.Volume, pycortex.Vertex, or pycortex.Dataset
        Data to be displayed.

    base_name : str
        Base name for images.

    list_angles : list of (str or dict)
        Views to be used. Should be of length one, or of the same length as
        `list_surfaces`. Choices are:
            'left', 'right', 'front', 'back', 'top', 'bottom', 'flatmap',
            'medial_pivot', 'lateral_pivot', 'bottom_pivot',
            or tuple of (view_name, custom dictionary of parameters).
            See `angle_view_params` in this file for parameter dict examples.

    list_surfaces : list of (str or dict)
        Surfaces to be used. Should be of length one, or of the same length as
        `list_angles`. Choices are:
            'inflated', 'flatmap', 'fiducial', 'inflated_cut',
            or a custom dictionary of parameters.

    viewer_params : dict
        Parameters passed to the viewer.

    interpolation : str
        Interpolation used to visualize the data. Possible choices are "nearest",
        "trilinear". (Default: "nearest").

    layers : int
        Number of layers between the white and pial surfaces to average prior to
        plotting the data. (Default: 1).

    size : tuple of int
        Size of produced image (before trimming).

    trim : bool
        Whether to trim the white borders of the image.

    sleep : float > 0
        Time in seconds, to let the viewer open.

    headless : bool
        If True, render using a headless Chromium browser via Playwright instead
        of requiring the user to manually open a browser window.  This allows
        the function to run fully autonomously without any user interaction.
        Requires ``playwright`` to be installed (``pip install playwright``) and
        Chromium to be available (``playwright install chromium``).
        Software WebGL (SwiftShader) is used, so no GPU or display server is
        needed.  (Default: False)

    contour_overlay : Dataview, str, or None
        Parcellation data whose borders will be drawn as contour lines.
        Can be a Vertex/Dataview object (automatically bundled into a Dataset
        with ``volume``), or a string naming a view within an existing Dataset
        passed as ``volume``.  (Default: None)

    contour_mode : str
        Contour rendering mode when ``contour_overlay`` is set.
        Options: "contours", "contours+fill", "colored", "colored+fill".
        (Default: "contours+fill")

    Returns
    -------
    file_names : list of str
        Image paths.
    """
    msg = "list_angles and list_surfaces should have the same length."
    assert len(list_angles) == len(list_surfaces), msg

    # If contour_overlay is a Dataview, bundle volume + overlay into a Dataset.
    # Preserve the original volume reference for isinstance checks below.
    _contour_overlay_name = None
    _original_volume = volume
    if contour_overlay is not None:
        if isinstance(contour_overlay, str):
            _contour_overlay_name = contour_overlay
        else:
            # contour_overlay is a Dataview — wrap into Dataset
            _contour_overlay_name = "__contour_overlay__"
            volume = cortex.Dataset(
                data=volume, **{_contour_overlay_name: contour_overlay}
            )

    # Create viewer — use a proper context manager so that cleanup always
    # runs, even if an exception occurs during rendering.
    if headless:
        from cortex.export.headless import headless_viewer as _headless_viewer

        cm = _headless_viewer(volume, viewer_params)
    else:
        cm = contextlib.nullcontext(cortex.webshow(volume, **viewer_params))

    with cm as handle:
        # Wait for the viewer to be loaded
        time.sleep(sleep)

        # Set up contour overlay if requested
        if _contour_overlay_name is not None:
            _contour_mode_map = {
                "contours": 1,
                "contours+fill": 2,
                "colored": 3,
                "colored+fill": 4,
            }
            if contour_mode not in _contour_mode_map:
                raise ValueError(
                    f"Unknown contour_mode {contour_mode!r}. "
                    f"Valid options: {list(_contour_mode_map.keys())}"
                )
            _contour_mode_int = _contour_mode_map[contour_mode]
            handle._set_view(
                **{
                    "surface.{subject}.contours.overlay": _contour_overlay_name,
                }
            )
            # Wait for overlay data to load
            time.sleep(sleep)
            handle._set_view(
                **{
                    "surface.{subject}.contours.mode": _contour_mode_int,
                }
            )
            time.sleep(1)

        # Add interpolation and layers params only if the primary data is a volume.
        # Use _original_volume (before Dataset wrapping) for the type check.
        if isinstance(
            _original_volume, (cortex.Volume, cortex.Volume2D, cortex.VolumeRGB)
        ):
            interpolation_params = {
                "surface.{subject}.sampler": interpolation,
                "surface.{subject}.layers": layers,
            }
        else:
            interpolation_params = dict()

        # Get subject name — handle both Dataview and Dataset
        if hasattr(_original_volume, "subject"):
            _subject = _original_volume.subject
        else:
            # Dataset: get subject from first view
            _subject = next(iter(volume))[1].subject
        has_flatmap = hasattr(getattr(cortex.db, _subject).surfaces, "flat")
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
            for _ in range(100):
                for k, v in this_view_params.items():
                    k = k.format(subject=_subject) if "{subject}" in k else k
                    if handle.ui.get(k)[0] != v:
                        print("waiting for", k, handle.ui.get(k)[0], "->", v)
                        time.sleep(0.1)
                        continue
                break
            time.sleep(0.1)

            # Save image, store file_name
            file_name = file_pattern.format(
                base=base_name, view=view_name, surface=surface
            )
            file_names.append(file_name)
            handle.getImage(file_name, size)

            # Wait for browser to dump file, before applying new view parameters
            for _wait in range(200):
                if os.path.exists(file_name):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(
                    f"Image {file_name!r} was not written within 20 seconds. "
                    "The browser may have failed to POST the screenshot."
                )
            time.sleep(1)

            # Trim transparent edges
            if trim:
                try:
                    from PIL import Image

                    img = Image.open(file_name)
                    bbox = img.getbbox()
                    if bbox:
                        img = img.crop(bbox)
                        img.save(file_name)
                except Exception as e:
                    print(f"Could not trim {file_name}: {e}")

        # For non-headless mode, close the viewer handle explicitly
        # (the headless context manager handles its own teardown)
        if not headless:
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
    "left": {
        "camera.azimuth": 90,
        "camera.altitude": 90,
    },
    "right": {
        "camera.azimuth": 270,
        "camera.altitude": 90,
    },
    "left_atl": {
        "camera.azimuth": 65,
        "camera.altitude": 100,
    },
    "right_atl": {
        "camera.azimuth": 300,
        "camera.altitude": 100,
    },
    "front": {
        "camera.azimuth": 0,
        "camera.altitude": 90,
    },
    "back": {
        "camera.azimuth": 180,
        "camera.altitude": 90,
    },
    "top": {
        "camera.azimuth": 180,
        "camera.altitude": 0,
    },
    "bottom": {
        "camera.azimuth": 0,
        "camera.altitude": 180,
    },
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
    "fiducial": {
        "surface.{subject}.unfold": 0,
    },
    "inflated_less": {
        "surface.{subject}.unfold": 0.25,
    },
    "inflated": {
        "surface.{subject}.unfold": 0.5,
    },
    "inflated_cut": {
        "surface.{subject}.unfold": 0.501,
    },
    "flatmap": {
        "surface.{subject}.unfold": 1,
    },
}
