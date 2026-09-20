"""Browser-based manual aligner.

Moves the anatomical surfaces (pial and white matter) in the space of a
functional reference volume. The volume
stays on its own voxel grid, so its slices are displayed without
resampling, and the surfaces are cut off at the displayed slices, which
draws their outline on the anatomy in the image. A second view mode paints
the volume onto the surfaces instead. Rendering happens in the browser
through the WebGL viewer's machinery (``cortex/webgl/resources/js/aligner.js``);
the transform is served, edited and saved through a tornado server in this
process, like ``cortex.webgl.show``.

Saving an edited alignment deletes the masks cached for the transform,
since they were cut through the alignment it replaces.

The entry point for users is :func:`cortex.align.webgl_manual`.
"""
import base64
import glob
import json
import mimetypes
import os
import queue
import re
import time
import uuid
import warnings
import webbrowser
from typing import Any, Optional, Union, cast

import numpy as np
import numpy.typing as npt
from tornado import web

from .. import options, utils, volume
from ..database import db
from . import serve
from .data import _pack_png
from .FallbackLoader import FallbackLoader
from .serve import P
from .view import colormaps

#: Name under which the reference volume is served to the page
REFERENCE_NAME = "reference"

#: What the page shows, as its `display` control names it
DISPLAYS = dict(
    slices="3 ortho + 3D slices",
    brain="3 ortho + 3D brain",
    surface="data on the surface",
)

#: A transform name has to serve as a directory name in the filestore
XFM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def check_xfm_name(name: str) -> str:
    """Return `name` if it can be used as the name of a transform.

    The name becomes a directory in the filestore, so anything else is
    refused here rather than reaching the filesystem.

    Parameters
    ----------
    name : str
        The name to check.

    Returns
    -------
    name : str
        The name, unchanged.

    Raises
    ------
    ValueError
        If the name is empty or holds anything but letters, digits, '.',
        '_' and '-', or does not start with a letter or a digit.
    """
    if not isinstance(name, str) or XFM_NAME.match(name) is None:
        raise ValueError(
            "%r is not a usable transform name: use letters, digits, '.', '_' and "
            "'-', starting with a letter or a digit" % (name,)
        )
    return name


def reference_frame(nii) -> npt.NDArray[np.float64]:
    """The voxel-to-world matrix the aligner works in for a reference image.

    The world frame keeps the voxel grid of the reference image axis aligned,
    scaled to millimeters by the voxel sizes, and permuted and flipped so that
    its x, y and z axes point to the subject's right, anterior and superior.
    Slices of the reference image are therefore drawn without resampling, and
    the surfaces are moved by rigid transforms expressed in millimeters.

    Parameters
    ----------
    nii : nibabel.Nifti1Image
        The reference image.

    Returns
    -------
    world : (4, 4) ndarray
        Affine mapping voxel indices of `nii` to world coordinates.
    """
    import nibabel

    zooms = np.asarray(nii.header.get_zooms()[:3], dtype=float)
    ornt = nibabel.io_orientation(nii.affine)
    world = np.zeros((4, 4))
    world[3, 3] = 1.0
    for voxel_axis, (world_axis, direction) in enumerate(ornt):
        world[int(world_axis), voxel_axis] = float(direction) * zooms[voxel_axis]
    return world


def load_reference(nii) -> npt.NDArray[np.float32]:
    """The reference data as a 3D float32 array in (x, y, z) voxel order.

    A 4D image contributes its first volume; NaNs are replaced by zeros.
    """
    data = np.asarray(nii.get_fdata())
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError("The reference image must have three dimensions, got shape %s" % (data.shape,))
    return np.nan_to_num(data).astype(np.float32)


def _color_hex(color: str) -> str:
    from matplotlib.colors import to_hex

    return to_hex(color)


def cached_masks(subject: str, xfmname: str) -> list[str]:
    """The paths of the masks cached for a transform.

    Masks are cut out of the reference volume through the transform, so
    editing the alignment makes every one of them wrong.

    Parameters
    ----------
    subject : str
        Subject identifier.
    xfmname : str
        Name of the transform.

    Returns
    -------
    paths : list of str
        Paths of the cached mask files, empty when the transform has none.
    """
    pattern = db.get_paths(subject)["masks"].format(xfmname=xfmname, type="*")
    return sorted(glob.glob(pattern))


def clear_masks(subject: str, xfmname: str) -> list[str]:
    """Delete the masks cached for a transform, and return their names.

    The aligner calls this when it saves an edited alignment: the masks it
    deletes were cut with the previous alignment, and nothing else
    invalidates them. ``db.save_xfm`` also refuses to write over a transform
    that still has masks.

    Parameters
    ----------
    subject : str
        Subject identifier.
    xfmname : str
        Name of the transform.

    Returns
    -------
    names : list of str
        Names of the deleted masks, as `db.get_mask` takes them (the `thick`
        of `mask_thick.nii.gz`). Empty when the transform had no masks.
    """
    names = []
    for path in cached_masks(subject, xfmname):
        name = os.path.split(path)[1]
        if name.startswith("mask_") and name.endswith(".nii.gz"):
            name = name[len("mask_"):-len(".nii.gz")]
        os.unlink(path)
        names.append(name)
    return names


class JSAligner(serve.JSProxy[P]):
    """Handle to an aligner running in the browser.

    Besides the generic attribute access of :class:`cortex.webgl.serve.JSProxy`,
    this exposes the transform being edited and the controls of the page.
    Its methods tag their requests, so that replies delayed by a busy page
    (while it parses the surfaces, or draws a slow frame) are matched to the
    right request; the generic attribute access does not have this protection.
    """

    #: Seconds to wait for the tagged reply of a call
    call_timeout = 120.0

    def _call(self, name: str, *args: Any) -> Any:
        token = uuid.uuid4().hex
        resp = self.send(method="run", params=["window.viewer.call", [token, name, list(args)]])
        # plain attribute lookups on a JSProxy go to the page; read the
        # bookkeeping kept on the python object from its dict instead
        server = vars(self).get("server")
        if len(resp) == 0:
            return None
        deadline = time.monotonic() + self.call_timeout
        reply = resp[0]
        while True:
            if isinstance(reply, dict) and reply.get("token") == token:
                # a redraw pending at the time of the reply lands in the next
                # frame; wait_for_frame waits for it
                if reply.get("scheduled", False):
                    target = int(cast(float, reply.get("frames", 0))) + 1
                    object.__setattr__(self, "_frame_target", max(target, vars(self).get("_frame_target", 1)))
                if "error" in reply:
                    raise RuntimeError("%s: %s" % (name, reply["error"]))
                return reply.get("value")
            if server is None:
                # without the server's queue, stale replies cannot be skipped
                if reply is None:
                    return None
                raise RuntimeError("Unexpected reply to %s: %r" % (name, reply))
            if time.monotonic() > deadline:
                raise TimeoutError("No reply to %s within %.0f s" % (name, self.call_timeout))
            try:
                reply = json.loads(server.response.get(timeout=1))
            except queue.Empty:
                reply = None

    def wait_for_frame(self, timeout: float = 120.0) -> int:
        """Block until the page has drawn the effect of the last call.

        Returns the number of frames drawn so far. Drawing happens in the
        page's own animation frames, which can be slow without a GPU, so a
        snapshot taken right after a change may otherwise show the previous
        state.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # the reply to this call also records a redraw still pending
            frames = self._call("getFrames")
            target = max(vars(self).get("_frame_target", 1), 1)
            if frames is not None and int(frames) >= target:
                return int(frames)
            time.sleep(0.2)
        raise TimeoutError("The aligner did not draw a new frame within %.0f s" % timeout)

    def get_control(self, name: str) -> Any:
        """The value of a control of the page by its dotted path, such as
        ``"image.vmin"`` or ``"mesh.color"``."""
        return self._call("getControl", name)

    def set_control(self, name: str, value: Any) -> None:
        """Set a control of the page by its dotted path, such as
        ``"image.colormap"`` or ``"display"``."""
        self._call("setControl", name, value)

    def get_xfm(self) -> npt.NDArray[np.float64]:
        """The current transform, as a (4, 4) pycortex 'coord' matrix
        (anatomical coordinates to voxel indices of the reference)."""
        return np.asarray(self._call("getXfm"), dtype=float)

    def set_xfm(self, xfm: npt.ArrayLike) -> None:
        """Replace the current transform with a (4, 4) 'coord' matrix."""
        matrix = np.asarray(xfm, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("The transform must be a 4x4 matrix")
        self._call("setXfm", matrix.tolist())

    def translate(self, vector: npt.ArrayLike) -> None:
        """Move the surfaces by `vector`, in millimeters along the world axes
        (right, anterior, superior)."""
        self._call("translate", [float(v) for v in np.asarray(vector).ravel()])

    def rotate(self, axis: npt.ArrayLike, angle: float) -> None:
        """Rotate the surfaces by `angle` degrees about the world `axis`
        through the cursor."""
        self._call("rotate", [float(v) for v in np.asarray(axis).ravel()], float(angle))

    def undo(self) -> None:
        """Undo the last change to the transform."""
        self._call("undo")

    def save(self, timeout: float = 60.0) -> str:
        """Save the current transform into the database, as the Save button does.

        Returns the message the server answered with, once the transform is
        written; the page posts the save, so this waits for that request to
        come back rather than returning while it is still out.

        Parameters
        ----------
        timeout : float, optional
            Seconds to wait for the save to land.

        Raises
        ------
        RuntimeError
            If the server refused the save, with the reason it gave.
        TimeoutError
            If no answer arrived within `timeout`.
        """
        self._call("save")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._call("getSaveState")
            if isinstance(state, dict) and state.get("status") != "saving":
                message = str(state.get("message", ""))
                if state.get("status") != "ok":
                    raise RuntimeError(message or "the transform was not saved")
                return message
            time.sleep(0.1)
        raise TimeoutError("The aligner did not finish saving within %.0f s" % timeout)

    def snapshot(self, filename: Optional[str] = None) -> bytes:
        """The current rendering of the four views as PNG bytes, also written
        to `filename` when given. Waits for the frame showing the last change
        first."""
        self.wait_for_frame()
        data_url = self._call("snapshot")
        png = base64.b64decode(data_url.split(",", 1)[1])
        if filename is not None:
            with open(filename, "wb") as fp:
                fp.write(png)
        return png


def show(
    subject: str,
    xfmname: str,
    reference: Optional[str] = None,
    view_only: bool = False,
    cmap: Optional[str] = None,
    mesh_color: Optional[str] = None,
    mesh_opacity: Optional[float] = None,
    open_browser: Optional[bool] = None,
    autoclose: Optional[bool] = None,
    port: Optional[int] = None,
    recache: bool = False,
    types: tuple[str, ...] = ("inflated",),
    title: Optional[str] = None,
    display_url: bool = True,
    token: Optional[str] = None,
    template: str = "aligner.html",
) -> Union[JSAligner, serve.WebApp]:
    """Open the browser-based aligner for a transform of `subject`.

    The functional reference volume is shown on its own voxel grid in three
    slice views and a 3D view, with the pial and white matter surfaces of
    `subject` cut to the displayed slices. Rotating and translating the
    surfaces edits the transform; the Save button writes it into the
    database as `xfmname`. See :func:`cortex.align.webgl_manual` for the
    controls.

    Saving deletes the masks cached for `xfmname` (the page warns when it
    opens a transform that has some): they were cut through the alignment
    being replaced. Data already masked with them has to be masked again
    from the volumes.

    Parameters
    ----------
    subject : str
        Subject identifier.
    xfmname : str
        Name of the transform to create or modify.
    reference : str, optional
        Path to a nibabel-readable functional volume, required when `xfmname`
        does not exist yet. For an existing transform, leave it None: the
        stored reference is loaded, and the transform is used as the starting
        point.
    view_only : bool, optional
        Open the aligner without the possibility to save, to inspect an
        alignment.
    cmap : str, optional
        Initial colormap for the reference volume, one of the 1D pycortex
        colormaps. Defaults to the `colormap` option of the `webgl_aligner`
        section of the config file.
    mesh_color : str, optional
        Initial color of the surface outlines, as a matplotlib color. Defaults
        to the `mesh_color` config option.
    mesh_opacity : float, optional
        Initial opacity of the whole surfaces in the 3D view (0 shows only
        their outlines on the slices). Defaults to the `mesh_opacity` config
        option.
    open_browser : bool, optional
        Open the aligner in the default browser. Defaults to the
        `open_browser` option of the `webshow` config section.
    autoclose : bool, optional
        Stop the server when the last browser window disconnects. Defaults to
        the `autoclose` option of the `webshow` config section.
    port : int, optional
        Port of the server; a free port is picked when None.
    recache : bool, optional
        Regenerate the cached surface (CTM) files.
    types : tuple of str, optional
        Surface types included in the CTM pack, to share the cache with the
        viewer. Default ("inflated",).
    title : str, optional
        Title of the browser window.
    display_url : bool, optional
        When `open_browser` is False, display an IPython link to the aligner.
    token : str, optional
        The session token the server demands, which the address it prints
        carries and the page then keeps in a cookie. A new one is made for
        each aligner; pass '' to take requests from anything that reaches
        the port, which a script talking to the server itself may want.
    template : str, optional
        Name of the tornado template of the page. Default 'aligner.html'.

    Returns
    -------
    handle : JSAligner or WebApp
        With `open_browser`, a handle to the running aligner (its ``server``
        attribute is the tornado server); otherwise the server itself, whose
        ``get_client()`` returns the handle once a browser has connected.
    """
    import nibabel

    close_on_disconnect: bool = (
        options.config.get("webshow", "autoclose", fallback="true") == "true"
        if autoclose is None
        else autoclose
    )
    if open_browser is None:
        open_browser = options.config.get("webshow", "open_browser", fallback="true") == "true"

    # The transform to start from: the stored one, or the header alignment
    # of a new reference (anatomical and functional scanner spaces coincide)
    try:
        dbxfm = db.get_xfm(subject, xfmname, xfmtype="coord")
    except IOError:
        dbxfm = None

    if dbxfm is not None:
        if reference is not None:
            raise ValueError(
                "Refusing to overwrite the reference of the existing transform %s; "
                "pass reference=None to load the stored reference" % xfmname
            )
        nii = dbxfm.reference_nifti
        reference = nii.get_filename()
        coord = np.asarray(dbxfm.xfm, dtype=float)
    else:
        if reference is None or not os.path.exists(reference):
            raise ValueError("Reference image file (%s) does not exist" % reference)
        nii = cast("nibabel.Nifti1Image", nibabel.load(reference))
        coord = np.linalg.inv(nii.affine)

    epi = load_reference(nii)
    world = reference_frame(nii)
    percentiles = np.percentile(epi, [1, 99])
    vmin, vmax = float(percentiles[0]), float(percentiles[1])
    if vmin == vmax:
        vmin, vmax = float(epi.min()), float(epi.max())

    # The reference is served as the float mosaic PNG the viewer uses
    mosaic, mosaic_shape = volume.mosaic(epi.T, show=False)
    png = _pack_png(np.ascontiguousarray(mosaic, dtype=np.float32))

    # The surfaces come from the CTM pack the viewer uses: base positions
    # are the pial surface and the `wm` attribute is the white matter
    ctmfile = utils.get_ctmpack(subject, types, method="mg2", level=9, recache=recache)

    cmap_names = [name for name, _ in cast(list[tuple[str, str]], colormaps)]
    if cmap is None:
        cmap = options.config.get("webgl_aligner", "colormap", fallback="gray")
    if cmap not in cmap_names:
        warnings.warn("Colormap %s is not available, using gray" % cmap)
        cmap = "gray"
    if mesh_color is None:
        mesh_color = options.config.get("webgl_aligner", "mesh_color", fallback="white")
    if mesh_opacity is None:
        mesh_opacity = float(options.config.get("webgl_aligner", "mesh_opacity", fallback="0"))
    if title is None:
        title = "Aligner: %s %s" % (subject, xfmname)

    # Handed to the page and demanded back on a save. The session token says
    # the request comes from this computer; this one says it comes from the
    # page itself, which a site the browser is also on cannot read, so a form
    # it posts carries the session cookie but no save token.
    save_token = uuid.uuid4().hex

    config: dict[str, Any] = dict(
        subject=subject,
        xfmname=xfmname,
        ctm="ctm/%s/" % subject,
        view_only=view_only,
        save_token=save_token,
        # shown as a warning on the page: saving deletes these
        masks=[os.path.split(path)[1] for path in cached_masks(subject, xfmname)],
        volume=dict(
            name=REFERENCE_NAME,
            subject=subject,
            raw=False,
            min=float(epi.min()),
            max=float(epi.max()),
            mosaic=[int(v) for v in mosaic_shape],
            shape=[int(v) for v in epi.T.shape],
        ),
        images={REFERENCE_NAME: ["data/%s.png" % REFERENCE_NAME]},
        world=world.tolist(),
        xfm=coord.tolist(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        mesh_color=_color_hex(mesh_color),
        mesh_opacity=mesh_opacity,
    )

    html = FallbackLoader([os.path.split(os.path.abspath(template))[0], serve.cwd]).load(template)

    class CTMHandler(web.RequestHandler):
        def get(self, path: str):
            subj, path = path.split("/")
            if subj != subject:
                raise web.HTTPError(404)
            if path == "":
                self.set_header("Content-Type", "application/json")
                with open(ctmfile) as fp:
                    self.write(fp.read())
            else:
                fpath = os.path.join(os.path.split(ctmfile)[0], path)
                mtype = mimetypes.guess_type(fpath)[0]
                self.set_header("Content-Type", mtype if mtype is not None else "application/octet-stream")
                with open(fpath, "rb") as fp:
                    self.write(fp.read())

    class DataHandler(web.RequestHandler):
        def get(self, path: str):
            self.set_header("Content-Type", "image/png")
            self.write(png)

    class AlignerHandler(web.RequestHandler):
        def get(self):
            self.set_header("Content-Type", "text/html")
            self.write(
                html.generate(
                    config=json.dumps(config),
                    colormaps=colormaps,
                    default_cmap=cmap,
                    python_interface=True,
                    leapmotion=False,
                    title=title,
                )
            )

    class SaveHandler(web.RequestHandler):
        def post(self):
            self.set_header("Content-Type", "application/json")
            # This writes to the filestore, so it only answers the page it was
            # served to. The save token is handed out in that page, which a
            # site the browser is also on cannot read, so a form it posts
            # carries the session cookie but not this.
            if not serve.same_token(self.get_argument("save_token", None), save_token):
                self.set_status(403)
                self.write(json.dumps(dict(
                    status="error", message="not saved: this is not the aligner's own page")))
                return
            if view_only:
                self.write(json.dumps(dict(status="error", message="view only: the transform is not saved")))
                return
            try:
                xfm = np.asarray(json.loads(self.get_argument("xfm")), dtype=float)
                if xfm.shape != (4, 4):
                    raise ValueError("expected a 4x4 matrix, got shape %s" % (xfm.shape,))
                # The page can save the alignment under another name, which
                # creates a transform of that name rather than changing the
                # one that was loaded.
                name = check_xfm_name(self.get_argument("name", xfmname).strip())
                # The masks of the transform being written were cut with the
                # alignment being replaced, so they are wrong from here on;
                # db.save_xfm also refuses to write over a transform that
                # still has them.
                dropped = clear_masks(subject, name)
                db.save_xfm(subject, name, xfm, xfmtype="coord", reference=reference)
            except Exception as exc:
                self.write(json.dumps(dict(status="error", message="not saved: %s" % exc)))
                return
            message = "saved transform %s for %s" % (name, subject)
            if len(dropped) > 0:
                message += "; deleted %d stale mask%s (%s)" % (
                    len(dropped), "" if len(dropped) == 1 else "s", ", ".join(dropped))
            print(message)
            self.write(json.dumps(dict(status="ok", message=message, name=name, masks_deleted=dropped)))

    class WebApp(serve.WebApp):
        disconnect_on_close = close_on_disconnect

        def get_client(self):
            self.connect.wait()
            self.connect.clear()
            client = JSAligner(self.send, "window.viewer")
            # The handle matches replies to its own requests by draining the
            # server's queue, and needs the server to do it. Without it a
            # reply delayed past WebApp.send's two second wait -- which the
            # page takes while it parses the surfaces or draws a slow frame --
            # is lost rather than waited for.
            # (bypasses JSProxy.__setattr__, which would query the page)
            object.__setattr__(client, "server", self)
            return client

    server = WebApp(
        [
            (r"/ctm/(.*)", CTMHandler),
            (r"/data/(.*)", DataHandler),
            (r"/save", SaveHandler),
            (r"/aligner.html", AlignerHandler),
            (r"/", AlignerHandler),
        ],
        0 if port is None else port,
        token=token,
    )
    server.start()
    print("Started aligner server on port %d" % server.port)
    #: under the machine's own name, which is what a port forward from
    #: another computer is set up under, and carrying the session token
    url = server.url("aligner.html")
    if open_browser:
        webbrowser.open(url)
        return server.get_client()
    elif display_url:
        try:
            from IPython.display import HTML, display

            display(HTML('Open aligner: <a href="{0}" target="_blank">{0}</a>'.format(url)))
        except Exception:
            print("Open the aligner at %s" % url)
    return server
