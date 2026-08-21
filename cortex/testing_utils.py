"""Module containing utils for testing"""
import os
import subprocess as sp
from shutil import which
from . import options
import os

from .options import config


def has_installed(name):
    binary_file = options.config.get('dependency_paths', name)
    return os.path.exists(binary_file) or which(name) is not None


def inkscapePath():
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


def inkscape_version():
    inkscapeExecutable = inkscapePath()
    if inkscapeExecutable is None:
        return None
    result = sp.run([inkscapeExecutable, '--version'], stdout = sp.PIPE, stderr = sp.PIPE, check = True)
    # Combine stdout and stderr; some systems print diagnostic messages
    # (e.g. "Setting _INKSCAPE_GC=disable …") before the version line.
    combined = result.stdout + result.stderr
    if isinstance(combined, bytes):
        combined = combined.decode('utf-8')
    # Find the line that starts with 'Inkscape' to get the real version,
    # e.g. 'Inkscape 1.2.2 (b0a8486, 2022-12-01)'
    for line in combined.splitlines():
        if line.strip().startswith('Inkscape'):
            version = line.split()[1]
            return version
    return None


INKSCAPE_PATH = inkscapePath()
INKSCAPE_VERSION = inkscape_version()
if INKSCAPE_VERSION is None:
    print(("Inkscape not found. Some tests will be skipped. " 
    "If you are on a Mac or Windows machine, please specify "
    "the path to the Inkscape binary in the config file "
    f"(Here: {options.usercfg})."))


