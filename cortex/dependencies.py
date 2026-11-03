"""Locate external tools (e.g. Inkscape) configured under ``[dependency_paths]``."""
from __future__ import annotations

import os
import subprocess as sp
from functools import cache
from shutil import which

from . import options
from .options import config


def has_installed(name: str) -> bool:
    binary_file = options.config.get('dependency_paths', name)
    return os.path.exists(binary_file) or which(name) is not None


def inkscape_path() -> str | None:
    """Return the path of the inkscape executable to run, or None.

    The 'inkscape' option of the 'dependency_paths' config section is used
    first. If it is a path to the inkscape exe, that file is used directly.
    Otherwise the name 'inkscape' is looked up on the PATH environment
    variable, so that a mis-configured or unset option still falls back on
    an inkscape that is on PATH.
    """
    configuredPath = config.get('dependency_paths', 'inkscape', fallback = 'inkscape')
    if configuredPath is not None and configuredPath.strip() != '':
        configuredPath = os.path.expanduser(configuredPath.strip())
        if os.path.isfile(configuredPath):
            return configuredPath
    return which('inkscape')


def inkscape_version() -> str | None:
    """Return the version string of the inkscape executable, or None.

    Nothing is cached while inkscape is missing, so installing it after
    pycortex has been imported takes effect on the next call.
    """
    inkscapeExecutable = inkscape_path()
    if inkscapeExecutable is None:
        return None
    return _query_inkscape_version(inkscapeExecutable)


def require_inkscape() -> tuple[str, str]:
    """Return the (path, version) of inkscape, or raise if it is not available."""
    path = inkscape_path()
    version = inkscape_version()
    if path is None or version is None:
        raise RuntimeError(
            "Inkscape doesn't seem to be installed on this system, and is "
            "required here. Please make sure that inkscape is installed and "
            "accessible from the terminal, or set the 'inkscape' option in "
            f"the [dependency_paths] section of {options.usercfg}.")
    return path, version


@cache
def _query_inkscape_version(inkscapeExecutable: str) -> str | None:
    result = sp.run([inkscapeExecutable, '--version'], stdout = sp.PIPE, stderr = sp.PIPE, check = True)
    # Combine stdout and stderr; some systems print diagnostic messages
    # (e.g. "Setting _INKSCAPE_GC=disable …") before the version line.
    raw = result.stdout + result.stderr
    combined = raw.decode('utf-8') if isinstance(raw, bytes) else raw
    # Find the line that starts with 'Inkscape' to get the real version,
    # e.g. 'Inkscape 1.2.2 (b0a8486, 2022-12-01)'
    for line in combined.splitlines():
        if line.strip().startswith('Inkscape'):
            version = line.split()[1]
            return version
    return None
