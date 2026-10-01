import contextlib
import json
import math
import os
import time
import warnings
from typing import Any, Mapping, Optional, Sequence, TypedDict, Union

import numpy as np

import cortex

from ..dataset import Dataview

file_pattern = "{base}_{view}_{surface}.png"

ViewParams = TypedDict(
    "ViewParams",
    {
        "camera.azimuth": float,
        "camera.altitude": float,
        "camera.target": list[float],
        "camera.radius": float,
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


def default_subject_views(has_flatmap: bool = True,
                          subject: Optional[str] = None) -> dict[str, ViewParams]:
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
    subject : str or None, optional
        The subject the views are for. Given one, every view but `flat` also
        gets the camera target and radius that frame that subject's brain (see
        `default_view_framing`), so a view always returns the same scene.
        Default None: angles and unfolding only, the same for every subject.

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
        # Nor a target: default_view_params' origin is a folded target, which
        # a flat view has no business setting -- and, applied flat without a
        # camera.flat_target, it would be read as the flat target (see
        # JSMixer._set_view) and put the flatmap back where it used to sit,
        # sixty units low. The viewer starts the flat target at the middle of
        # the flatmap on its own.
        flat.pop("camera.target", None)    # type: ignore[misc]
        views[FLAT_VIEW_NAME] = flat

    if subject is not None:
        for name, framing in default_view_framing(subject).items():
            if name in views:
                views[name].update(framing)
    return views


# ---------------------------------------------------------------------------
# Framing the default views for a subject
# ---------------------------------------------------------------------------
#
# A view is only reproducible if it fixes the whole camera, zoom included, so
# every default view but `flat` carries a camera target and radius. They cannot
# be constants: brains differ in size (and pycortex serves macaque as well as
# human data), and the inflated surface is a different shape from the
# fiducial. So they are fitted to each subject's own surfaces, as the viewer
# draws them (see _viewer_points), and cached in the subject's cache directory,
# since reading the surfaces takes about half a second. `flat` frames itself, the way quickflat frames the image it writes
# (Viewer.fitFlatView in resources/js/mriview.js).

#: The viewer camera's vertical field of view, in degrees
#: (``THREE.PerspectiveCamera(45, ...)`` in resources/js/axes3d.js).
VIEWER_FOV = 45.0

#: The frame the default views are fitted to: the brain fills `FRAMING_FILL`
#: of the frame's limiting dimension, in a frame `FRAMING_ASPECT` wide for each
#: unit of height. The lateral views are wider than they are tall, so they are
#: the ones the aspect ratio decides; 4:3 frames them without clipping in any
#: window at least that wide.
FRAMING_ASPECT = 4 / 3
FRAMING_FILL = 0.85

#: The radii LandscapeControls.setRadius accepts (resources/js/movement.js).
RADIUS_LIMITS = (10.0, 600.0)

#: Bump when the cached framing changes meaning, so stale caches are refitted.
_FRAMING_VERSION = 3
_FRAMING_CACHE = "default_view_framing.json"

# LandscapeControls keeps the altitude strictly inside (0, 180): exactly at a
# pole the view direction is parallel to the up vector and the image
# orientation is undefined.
_POLE_EPSILON = 1e-4


def camera_basis(azimuth: float, altitude: float) -> tuple[
        tuple[float, float, float], tuple[float, float, float],
        tuple[float, float, float]]:
    """The viewer camera's axes in world space, for an azimuth and altitude.

    Reproduces the eye position from ``LandscapeControls.update``
    (resources/js/movement.js), with ``camera.up`` fixed at +z by ``axes3d.js``,
    then three.js's ``Matrix4.lookAt``.

    Returns
    -------
    tuple
        ``(right, up, back)`` unit vectors: where the image's right edge, its
        top edge, and the direction from the target towards the camera point in
        world space.
    """
    altitude = min(max(altitude, _POLE_EPSILON), 180 - _POLE_EPSILON)
    altrad = math.radians(altitude)
    azirad = math.radians(azimuth + 90)
    eye = (math.sin(altrad) * math.cos(azirad),
           math.sin(altrad) * math.sin(azirad),
           math.cos(altrad))

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def unit(v):
        length = math.sqrt(sum(c * c for c in v))
        return tuple(c / length for c in v)

    back = unit(eye)                       # the camera looks along -back
    right = unit(cross((0.0, 0.0, 1.0), back))
    up = cross(back, right)
    return right, up, back


def _viewer_points(subject: str, kind: str) -> np.ndarray:
    """The vertices of the surface `kind`, where the viewer draws them.

    Follows brainctm.BrainCTM, which builds the surface packs the viewer loads,
    and the vertex shader in resources/js/shaderlib.js:

    - A subject with pial and white-matter surfaces is packed on the pial one,
      with the white matter alongside, and the folded brain is drawn between
      the two at the cortical depth slider, 0.5 by default -- their midpoint,
      which is also how pycortex derives a fiducial surface. Without them, the
      pack and the folded brain are the fiducial surface itself.
    - Every other folded surface the viewer morphs to, such as the inflated
      one, is rescaled one hemisphere at a time, axis by axis, into the pack's
      base surface's bounding box (brainctm.Hemi.addSurf) -- so an inflated
      brain fills the same box as the pial one, whatever size its file is.
    """
    try:
        pia = cortex.db.get_surf(subject, "pia")
        wm = cortex.db.get_surf(subject, "wm")
        base = [hemi[0] for hemi in pia]
        folded = [(p[0] + w[0]) / 2 for p, w in zip(pia, wm)]
    except IOError:
        base = folded = [hemi[0] for hemi in cortex.db.get_surf(subject, "fiducial")]

    if kind == "fiducial":
        return np.vstack(folded)
    rescaled = []
    for (pts, _), box in zip(cortex.db.get_surf(subject, kind), base):
        low, high = pts.min(0), pts.max(0)
        rescaled.append((pts - low) / (high - low) * (box.max(0) - box.min(0))
                        + box.min(0))
    return np.vstack(rescaled)


def _fit_view(points: np.ndarray, view: Mapping[str, Any]) -> tuple[list[float], float]:
    """The target and radius that frame `points` for `view`'s camera angles.

    The target is the middle of the points' bounding box. The radius is the
    smallest distance from it at which every point projects inside
    `FRAMING_FILL` of a `FRAMING_ASPECT` frame, through the viewer's
    perspective camera. For a point p relative to the target, seen from a
    camera at distance r along `back`, that takes
    ``r >= p.back + |p.up| / (fill * tan(fov/2))`` vertically and
    ``r >= p.back + |p.right| / (fill * tan(fov/2) * aspect)`` horizontally.
    """
    right, up, back = camera_basis(view["camera.azimuth"], view["camera.altitude"])
    centre = (points.min(0) + points.max(0)) / 2
    depth, vertical, horizontal = ((points - centre) @ np.array([back, up, right]).T).T
    tan = math.tan(math.radians(VIEWER_FOV / 2)) * FRAMING_FILL
    radius = max((depth + np.abs(vertical) / tan).max(),
                 (depth + np.abs(horizontal) / (tan * FRAMING_ASPECT)).max())
    return [float(c) for c in centre], float(radius)


def _fit_default_views(subject: str, kinds: Mapping[str, str]) -> dict[str, ViewParams]:
    """Fit every default view but `flat` to `subject`'s surfaces."""
    points: dict[str, np.ndarray] = {}
    framing: dict[str, ViewParams] = {}
    for name, view in default_subject_views(has_flatmap=True).items():
        if name == FLAT_VIEW_NAME:
            continue
        kind = kinds["fiducial" if view["surface.{subject}.unfold"] == 0
                     else "inflated"]
        if kind not in points:
            points[kind] = _viewer_points(subject, kind)
        target, radius = _fit_view(points[kind], view)
        low, high = RADIUS_LIMITS
        if not low <= radius <= high:
            warnings.warn("The %s view of %s needs a camera radius of %.0f, "
                          "outside the viewer's %g-%g; using the nearest"
                          % (name, subject, radius, low, high))
            radius = min(max(radius, low), high)
        framing[name] = {"camera.target": target, "camera.radius": radius}
    return framing


def _surface_sources(surfs: Mapping[str, Mapping[str, str]],
                     kind: str) -> dict[str, Mapping[str, str]]:
    """The files `_viewer_points` reads for `kind`, as ``{name: {hemi: path}}``.

    The folded brain comes from the pial and white-matter surfaces when there
    are both, and from the fiducial surface otherwise; anything else is its own
    file, rescaled into the folded brain's box. These decide whether a cached
    framing is still current. Raises KeyError if a file is missing.
    """
    folded = ({"pia": surfs["pia"], "wm": surfs["wm"]}
              if "pia" in surfs and "wm" in surfs else {"fiducial": surfs["fiducial"]})
    if kind == "fiducial":
        return folded
    return {**folded, kind: surfs[kind]}


def default_view_framing(subject: str) -> dict[str, ViewParams]:
    """The camera target and radius that frame each default view of `subject`.

    Each view but `flat` aims at the middle of the surface it shows (fiducial,
    or inflated) from the distance at which that surface fills `FRAMING_FILL`
    of a `FRAMING_ASPECT` frame. Fitted to the subject's own surfaces and
    cached in its cache directory, so only the first viewer opened for a
    subject reads them; the cache is refitted when a surface file changes, or
    when the framing constants do. A cache directory that cannot be written
    just means no caching.

    Returns
    -------
    dict
        ``{view_name: {"camera.target": [x, y, z], "camera.radius": r}}``. Empty,
        with a warning, if the subject's surfaces cannot be read -- the views
        then keep whatever zoom the viewer has, as they did before.
    """
    try:
        surfs = cortex.db.get_paths(subject)["surfs"]
        # A subject without an inflated surface shows its inflated views on
        # the fiducial one.
        kinds = {"fiducial": "fiducial",
                 "inflated": "inflated" if "inflated" in surfs else "fiducial"}
        depends = {
            "version": _FRAMING_VERSION, "fov": VIEWER_FOV,
            "aspect": FRAMING_ASPECT, "fill": FRAMING_FILL,
            "surfaces": {kind: {source: {hemi: os.path.getmtime(path)
                                         for hemi, path in sorted(paths.items())}
                                for source, paths in
                                _surface_sources(surfs, kind).items()}
                         for kind in sorted(set(kinds.values()))},
        }
    except Exception as err:
        warnings.warn("Cannot frame the default views of %s: %s" % (subject, err))
        return {}

    try:
        cache: Optional[str] = os.path.join(cortex.db.get_cache(subject),
                                            _FRAMING_CACHE)
    except OSError:
        cache = None

    if cache is not None and os.path.exists(cache):
        try:
            with open(cache) as fp:
                stored = json.load(fp)
            if stored.get("depends") == depends:
                return stored["framing"]
        except (OSError, ValueError, KeyError, AttributeError):
            pass                            # unreadable: refit and overwrite

    try:
        framing = _fit_default_views(subject, kinds)
    except Exception as err:
        warnings.warn("Cannot frame the default views of %s: %s" % (subject, err))
        return {}

    if cache is not None:
        try:
            with open(cache, "w") as fp:
                json.dump({"depends": depends, "framing": framing}, fp)
        except OSError:
            pass                            # e.g. a read-only shared filestore
    return framing
