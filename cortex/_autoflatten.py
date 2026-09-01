"""Automatic cutting and flattening of cortical surfaces with ``autoflatten``.

`autoflatten <https://gallantlab.org/autoflatten>`_ projects a template set of
cuts onto a freesurfer subject's surfaces and flattens the resulting patches,
which removes the need to cut the surfaces by hand (see
:doc:`/segmentation_guide` for the manual workflow). It is an optional
dependency of pycortex, installed with ``pip install "pycortex[autoflatten]"``.

`cortex.freesurfer.import_subj` calls `autoflatten_subject` at the end of the
import, unless it is called with ``autoflatten=False``. This module is private;
`autoflatten_subject` is re-exported as `cortex.freesurfer.autoflatten_subject`.
"""

import importlib.util
import os
import subprocess as sp
import sys
import warnings
from typing import Optional, Sequence

#: Base name that ``autoflatten`` gives to the patch files it writes into the
#: freesurfer subject's ``surf/`` directory, i.e. ``?h.autoflatten.patch.3d`` for
#: the cut patch and ``?h.autoflatten.flat.patch.3d`` for the flattened patch.
PATCH_NAME = "autoflatten"

#: Warning issued before starting a run, since it takes a long time.
RUNTIME_WARNING = (
    "Flattening the surfaces with autoflatten takes a while, typically 15-30 "
    "minutes for both hemispheres. Pass `autoflatten=False` to skip this step "
    "and run `cortex.freesurfer.autoflatten_subject(...)` later instead."
)

#: Error message used when the optional `autoflatten` package is missing.
MISSING_MESSAGE = (
    "The `autoflatten` package is not installed, so the surfaces cannot be "
    "automatically cut and flattened. Install it with "
    '`pip install "pycortex[autoflatten]"` (or `pip install autoflatten`).'
)


def get_command() -> Optional[list[str]]:
    """Command prefix that invokes the autoflatten command line interface.

    Returns
    -------
    cmd : list of str or None
        The command to run autoflatten with the interpreter that is running
        pycortex, or None if the `autoflatten` package is not installed.
    """
    if importlib.util.find_spec("autoflatten") is None:
        return None
    return [sys.executable, "-m", "autoflatten.cli"]


def is_available() -> bool:
    """Whether the optional `autoflatten` package is installed."""
    return get_command() is not None


def check_autoflatten_available() -> bool:
    """Whether autoflatten can run, warning with install instructions if it cannot.

    Used by `cortex.freesurfer.import_subj` to decide whether to skip the
    automatic flattening step, without failing the whole import.

    Returns
    -------
    available : bool
        True if the `autoflatten` package is installed. If it is not, a warning
        is issued and False is returned.
    """
    if is_available():
        return True
    warnings.warn(
        MISSING_MESSAGE + " The surfaces will be imported without flatmaps; once "
        "autoflatten is installed, create them with "
        "`cortex.freesurfer.autoflatten_subject(...)`. Pass `autoflatten=False` to "
        "silence this warning."
    )
    return False


def autoflatten_subject(
    freesurfer_subject: str,
    pycortex_subject: Optional[str] = None,
    freesurfer_subject_dir: Optional[str] = None,
    autoflatten_args: Optional[Sequence[str]] = None,
    import_flatmaps: bool = True,
) -> dict[str, str]:
    """Automatically cut and flatten a freesurfer subject with ``autoflatten``.

    This runs the `autoflatten <https://gallantlab.org/autoflatten>`_ pipeline on
    the freesurfer subject, which projects a template set of cuts onto the
    subject's surfaces and flattens the resulting patches, and then imports the
    flatmaps into the pycortex database with `cortex.freesurfer.import_flat`.

    ``autoflatten`` is an optional dependency; install it with
    ``pip install "pycortex[autoflatten]"``.

    Parameters
    ----------
    freesurfer_subject : str
        Freesurfer subject name.
    pycortex_subject : str, optional
        Pycortex subject name to import the flatmaps into. By default it uses
        the freesurfer subject name.
    freesurfer_subject_dir : str, optional
        Freesurfer subjects directory to work in. By default uses the directory
        given by the environment variable ``$SUBJECTS_DIR``.
    autoflatten_args : list of str, optional
        Extra command line arguments passed to ``autoflatten run``, for example
        ``["--backend", "freesurfer"]`` or ``["--n-cores", "4"]``. See
        ``autoflatten run --help`` for the available options. Note that
        ``--output-dir`` must not be changed, since the flat patches are
        expected in the freesurfer subject's ``surf/`` directory.
    import_flatmaps : bool, optional
        Whether to import the resulting flatmaps into the pycortex database
        (True by default). Note that importing the flatmaps deletes the
        overlays.svg file and all cached files for the pycortex subject, since
        the flatmaps change.

    Returns
    -------
    flat_files : dict
        Mapping from hemisphere (``lh``, ``rh``) to the flat patch file that
        ``autoflatten`` produced.

    Notes
    -----
    This function requires freesurfer to be sourced, and takes a while to run
    (typically 15-30 minutes for both hemispheres). Patches that already exist
    are not recomputed; pass ``autoflatten_args=["--overwrite"]`` to force
    ``autoflatten`` to flatten the surfaces again.
    """
    # imported here rather than at the module level, since cortex.freesurfer
    # imports this module
    from . import freesurfer

    cmd = get_command()
    if cmd is None:
        raise ImportError(MISSING_MESSAGE)

    freesurfer_subject_dir = freesurfer._get_freesurfer_subject_dir(
        freesurfer_subject_dir
    )
    subject_dir = os.path.join(freesurfer_subject_dir, freesurfer_subject)
    if not os.path.isdir(subject_dir):
        raise IOError(
            "Freesurfer subject directory not found: {}".format(subject_dir)
        )
    if pycortex_subject is None:
        pycortex_subject = freesurfer_subject

    # autoflatten writes its patches into the freesurfer subject's surf/ directory
    patch_template = freesurfer.get_paths(
        freesurfer_subject,
        "{hemi}",
        type="patch",
        freesurfer_subject_dir=freesurfer_subject_dir,
    )
    flat_files = {
        hemi: patch_template.format(hemi=hemi, name=PATCH_NAME + ".flat")
        for hemi in ("lh", "rh")
    }
    # autoflatten skips patches that already exist, so there is nothing slow to
    # warn about if both hemispheres have already been flattened
    if any(not os.path.exists(path) for path in flat_files.values()):
        warnings.warn(RUNTIME_WARNING)

    cmd = cmd + ["run", subject_dir]
    if autoflatten_args is not None:
        cmd = cmd + list(autoflatten_args)
    print("Calling:\n{}".format(" ".join(cmd)))
    # Let autoflatten write its progress directly to stdout/stderr, since this
    # takes a long time and the user will want to see how far along it is.
    sp.check_call(cmd)

    missing = [path for path in flat_files.values() if not os.path.exists(path)]
    if missing:
        raise IOError(
            "autoflatten did not produce the expected flat patch file(s): "
            "{}".format(", ".join(missing))
        )

    if import_flatmaps:
        freesurfer.import_flat(
            freesurfer_subject,
            PATCH_NAME,
            cx_subject=pycortex_subject,
            flat_type="freesurfer",
            auto_overwrite=True,
            freesurfer_subject_dir=freesurfer_subject_dir,
        )
    return flat_files
