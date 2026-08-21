"""Module containing utils for testing"""
import subprocess as sp
from shutil import which


def has_installed(name):
    return which(name) is not None


def inkscape_version():
    if not has_installed('inkscape'):
        return None
    cmd = 'inkscape --version'
    result = sp.run(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE, check=True)
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


INKSCAPE_VERSION = inkscape_version()


