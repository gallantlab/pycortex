import contextlib
import os
import time
from typing import Any, Mapping, Sequence, TypedDict, Union

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
        "surface.{subject}.lighting.specularity": float,
        "surface.{subject}.lighting.uniform_illumination": float,
        "surface.{subject}.lighting.topleft_lighting": float,
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
) -> list[str]:
    """Saves 3D views of `volume` under multiple specifications.

    By default (``headless=False``), a webgl viewer is launched and a display
    server is required.  With ``headless=True``, a headless Chromium browser
    is used instead, so no display server or GPU is needed.

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
        Time in seconds, to let the viewer open.

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

    # Create viewer — use a proper context manager so that cleanup always
    # runs, even if an exception occurs during rendering.
    if headless:
        from cortex.export.headless import headless_viewer as _headless_viewer

        cm = _headless_viewer(volume, viewer_params)
    else:
        cm = contextlib.nullcontext(cortex.webshow(volume, **viewer_params))

    with cm as handle:
        # Wait for the viewer to be loaded. The headless context manager
        # already blocks on ``viewer.loaded`` before yielding, so we only
        # need this fixed sleep for the interactive (real-browser) path
        # where the user is opening the page manually.
        if not headless:
            time.sleep(sleep)

        # Add interpolation and layers params only if we have a volume
        if isinstance(volume, (cortex.Volume, cortex.Volume2D, cortex.VolumeRGB)):
            interpolation_params = {
                "surface.{subject}.sampler": interpolation,
                "surface.{subject}.layers": layers,
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

            # A flattened surface pins the camera square-on and ignores the
            # angle it is given, so asking for one only makes the settle loop
            # below report a view that never arrives. Kept when the surface is
            # tilt-enabled, where the angle does steer the camera.
            if (this_view_params.get("surface.{subject}.unfold", 0) >= 0.999
                    and not this_view_params.get("surface.{subject}.allow_tilt")):
                for prop in FLAT_INERT_PROPS:
                    this_view_params.pop(prop, None)   # type: ignore[misc]

            print(this_view_params)

            # apply params
            handle._set_view(**this_view_params)

            # wait for the view to have changed
            for _ in range(100):
                for k, v in this_view_params.items():
                    k = k.format(subject=volume.subject) if "{subject}" in k else k
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

            if headless:
                # Only check for WebGL failures in headless mode, since we don't
                # capture console output in the interactive mode.
                pw_thread = handle._pw_thread  # `handle` is a `JSMixer`
                from cortex.export.headless import filter_webgl_failures

                failures = filter_webgl_failures(pw_thread.browser_errors)
                if failures:
                    raise RuntimeError(
                        f"WebGL failed while rendering {view_name!r}/{surface!r}; "
                        f"{file_name!r} is likely blank.\n  "
                        + "\n  ".join(sorted(set(failures)))
                    )

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
    "surface.{subject}.lighting.specularity": 0,
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
    # No camera angle: once the surface is flat the controls hold the camera
    # square-on to it and discard whatever azimuth and altitude they are given
    # (see setAzimuth/setAltitude in resources/js/movement.js), unless the
    # surface's allow_tilt is on. Naming them here would do nothing to a flat
    # view, while making an animation interpolate towards them on the way in --
    # which spins the brain as it flattens, and leaves the folded camera angle
    # overwritten when it unfolds again.
    "flatmap": {
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


# ---------------------------------------------------------------------------
# Views every subject gets
# ---------------------------------------------------------------------------
#
# Offered by every viewer, so that a subject with nothing in its filestore
# views/ directory still has the standard anatomical orientations one click
# away. A view saved under one of these names takes precedence; see
# cortex.webgl.view._load_saved_views.

#: Anatomical name to the entry in `angle_view_params` that produces it.
#:
#: The viewer's camera sits at
#: ``radius * (sin(alt)cos(azi+90), sin(alt)sin(azi+90), cos(alt))`` looking at
#: the target, with up fixed at +z (LandscapeControls.js, axes3d.js). Surfaces
#: are in surface RAS, so +x is right, +y anterior, +z superior. That puts the
#: camera left of the brain at azimuth 90 and right of it at 270, above it at
#: altitude 0 and below at 180; and because `lookAt` resolves the degenerate
#: straight-up/straight-down cases through the azimuth, anterior ends up at the
#: top of the image at azimuth 180 seen from above and at azimuth 0 seen from
#: below. Those are exactly the four angles named here.
DEFAULT_VIEW_ANGLES: dict[str, str] = {
    "dorsal": "top",            # from above, frontal lobe up
    "ventral": "bottom",        # from below, frontal lobe up
    "lateral_left": "left",     # from the left, brain upright
    "lateral_right": "right",   # from the right, brain upright
}

#: Suffix added to the inflated counterpart of each of the above.
INFLATED_SUFFIX = "_inflated"

#: Name of the flattened view.
FLAT_VIEW_NAME = "flat"

#: View properties a flattened surface ignores, and which a flat view or
#: keyframe therefore leaves out. The controls pin the camera square-on to the
#: flatmap and discard both (resources/js/movement.js), so carrying them would
#: only give an animation something spurious to interpolate towards. They do
#: steer the camera when the surface's ``allow_tilt`` is on, so a tilted flat
#: pose keeps them.
FLAT_INERT_PROPS = ("camera.azimuth", "camera.altitude")


def default_subject_views(has_flatmap: bool = True) -> dict[str, ViewParams]:
    """The views offered for every subject, whether or not any are saved.

    Nine views: the four orientations in `DEFAULT_VIEW_ANGLES` on the fiducial
    surface, the same four inflated (suffixed `INFLATED_SUFFIX`), and `flat`.
    They are assembled from `default_view_params`, `angle_view_params` and
    `unfold_view_params` rather than spelled out, so the camera conventions stay
    in one place.

    Parameters
    ----------
    has_flatmap : bool, optional
        Whether the subject has a flat surface. Without one there is no `flat`
        view to offer, and the inflated surface sits at an unfold of 1 rather
        than 0.5 -- the same correction `save_3d_views` makes. Default True.

    Returns
    -------
    dict
        ``{view_name: view_params}``, with the literal ``{subject}`` placeholder
        left in the keys so one view works in a multi-subject viewer.
    """
    def build(*overrides: ViewParams) -> ViewParams:
        params: ViewParams = default_view_params.copy()
        for override in overrides:
            params.update(override)
        return params

    inflated = unfold_view_params["inflated"].copy()
    if not has_flatmap:
        inflated["surface.{subject}.unfold"] = min(
            inflated["surface.{subject}.unfold"] * 2, 1)

    views: dict[str, ViewParams] = {}
    for name, angle in DEFAULT_VIEW_ANGLES.items():
        views[name] = build(angle_view_params[angle],
                            unfold_view_params["fiducial"])
        views[name + INFLATED_SUFFIX] = build(angle_view_params[angle], inflated)

    if has_flatmap:
        # The established flatmap preset, the one save_3d_views renders
        # flatmaps with. It carries neither a camera angle (see
        # angle_view_params["flatmap"]) nor a camera.radius: a flat surface
        # ignores the angle, and the zoom is left to whoever renders it --
        # cortex.webgl.show's handle frames the flatmap the way make_png does
        # on request, with fit_flat_view.
        flat = build(angle_view_params["flatmap"],
                     unfold_view_params["flatmap"])
        for angle in FLAT_INERT_PROPS:
            flat.pop(angle, None)          # type: ignore[misc]
        views[FLAT_VIEW_NAME] = flat
    return views
