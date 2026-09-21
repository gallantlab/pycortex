"""Consent prompt shown before the webgl viewer's web server is started.

``cortex.webgl.show`` serves the viewer from a Tornado server that binds every
network interface and authenticates nobody (see ``serve.WebApp.__init__``,
which calls ``bind_sockets`` without an address). That is deliberate -- the
viewer is meant to be opened from other machines -- but it means starting a
viewer exposes both the subject data and a handful of write/read endpoints to
anyone who can route to this host. This module puts an explicit acknowledgement
in front of that, once per user unless they choose to be asked every time.
"""

import os
import sys
from typing import Optional, TextIO

from .. import options

CONFIG_SECTION = "webshow"
CONFIG_OPTION = "skip_security_warning"

#: Set this environment variable to any non-empty value to suppress the prompt
#: without touching the user's config file. Intended for CI and batch jobs.
ENV_VAR = "PYCORTEX_SKIP_SECURITY_WARNING"

_RULE = "=" * 78

_PROMPT = "Start the viewer? [y] yes  [n] no  [i] yes, and stop asking: "


class ViewerStartAborted(RuntimeError):
    """Raised when the user declines to start the viewer's web server."""


def warning_text(cwd: Optional[str] = None, movie_root: Optional[str] = None) -> str:
    """The body of the warning, as printed before the prompt.

    Parameters
    ----------
    cwd : str or None, optional
        Directory reported as reachable over the ``/static/`` route. Defaults
        to the process's current working directory, which is what that route
        actually resolves to.
    movie_root : str or None, optional
        Directory reported as writable by the animation panel's frame
        endpoint. Defaults to `cwd`, matching the ``movie_dir=None`` default
        of `cortex.webgl.show`.
    """
    if cwd is None:
        cwd = os.getcwd()
    if movie_root is None:
        movie_root = cwd
    return f"""{_RULE}
 SECURITY WARNING: the pycortex viewer starts a web server on this machine
{_RULE}
 The server listens on all network interfaces -- not just localhost -- and
 has no authentication. For as long as the viewer is open, anyone who can
 reach this machine over the network can:

   * read the data you are plotting: the cortical surfaces, the volume and
     vertex data, and any stimulus images attached to the dataset;

   * read any file underneath the current working directory
       {cwd}
     by requesting it through the viewer's /static/ route;

   * write files on this computer. The endpoint that saves screenshots and
     SVGs accepts unauthenticated POSTs, so a request from elsewhere on the
     network can have its own contents written to disk under the filename
     your session asked to save to -- replacing your image with arbitrary
     bytes -- or stall the server by posting when no save is pending;

   * write PNGs underneath the animation panel's movie directory
       {movie_root}
     Those writes carry a token, but the token is handed to every client
     that loads the viewer page, so it stops other local processes rather
     than anyone who can reach the viewer.

 Start the viewer only on a network you trust, and close it when you are
 done. Answering 'i' records your choice in
   {options.usercfg}
 as [{CONFIG_SECTION}] {CONFIG_OPTION} = true; delete that line to be asked
 again.
{_RULE}"""


def warning_is_disabled() -> bool:
    """Whether the user has already opted out of the warning."""
    if os.environ.get(ENV_VAR):
        return True
    return options.config.getboolean(
        CONFIG_SECTION, CONFIG_OPTION, fallback=False
    )


def confirm_server_start(
    cwd: Optional[str] = None,
    movie_root: Optional[str] = None,
    stream: Optional[TextIO] = None,
) -> None:
    """Warn about the viewer's network exposure and wait for acknowledgement.

    Prints `warning_text` and asks for ``y`` (start), ``n`` (abort) or ``i``
    (start, and persist `CONFIG_OPTION` so the warning is skipped from now on).
    Returns silently if the user has already opted out, or if there is no
    usable stdin to prompt on -- in that case the warning is still printed,
    since a batch job that cannot answer should not hang forever.

    Parameters
    ----------
    cwd : str or None, optional
        Directory named in the warning as reachable over ``/static/``.
        Defaults to the current working directory.
    movie_root : str or None, optional
        Directory named in the warning as writable by the animation panel.
        Defaults to `cwd`. Pass `show`'s ``movie_dir`` when it is set.
    stream : file-like or None, optional
        Where the warning is written. Defaults to ``sys.stderr``.

    Raises
    ------
    ViewerStartAborted
        If the user answers ``n``.
    """
    if warning_is_disabled():
        return

    if stream is None:
        stream = sys.stderr

    print(warning_text(cwd, movie_root), file=stream)
    stream.flush()

    while True:
        # The prompt is written to `stream` rather than passed to input(),
        # which would always send it to stdout and split the message in two
        # when the warning is going somewhere else.
        print(_PROMPT, end="", file=stream)
        stream.flush()
        try:
            answer = input().strip().lower()
        except (EOFError, OSError, RuntimeError):
            # No console attached (batch job, captured stdin, pythonw, ...).
            # The warning above has been printed; do not block on an answer
            # that can never arrive.
            print(
                f"\n(no interactive input available; continuing. Set {ENV_VAR}=1"
                " to silence this warning.)",
                file=stream,
            )
            stream.flush()
            return

        if answer in ("y", "yes"):
            return
        if answer in ("n", "no"):
            raise ViewerStartAborted(
                "Viewer not started: the network security warning was declined."
            )
        if answer in ("i", "ignore"):
            path = options.set_user_option(CONFIG_SECTION, CONFIG_OPTION, "true")
            print(f"Saved [{CONFIG_SECTION}] {CONFIG_OPTION} = true to {path}",
                  file=stream)
            stream.flush()
            return

        print("Please answer 'y', 'n' or 'i'.", file=stream)
        stream.flush()
